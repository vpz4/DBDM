# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:08:08 2024

@author: bPezo
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
from sklearn.linear_model import LogisticRegression
from minisom import MiniSom
from sklearn.metrics import davies_bouldin_score


warnings.filterwarnings('always')
os.chdir(os.getcwd())


def plot_db_scores(cluster_counts, db_scores, optimal_clusters=None):
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, db_scores, marker='o', linestyle='-', color='b')
    
    if optimal_clusters is not None:
        optimal_index = cluster_counts.index(optimal_clusters)
        optimal_score = db_scores[optimal_index]
        plt.scatter(optimal_clusters, optimal_score, color='red', s=100, label=f'Optimal ({optimal_clusters} clusters)')
        plt.legend()
    
    plt.title('Davies-Bouldin Scores by Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')
    plt.grid(True)
    plt.xticks(cluster_counts)
    plt.show()


def perform_clustering(data, num_clusters):
    som_dim = int(np.sqrt(num_clusters))
    sigma_value = min(som_dim / 2, 1.0)
    som = MiniSom(som_dim, som_dim, data.shape[1], sigma=sigma_value, learning_rate=0.5)
    som.random_weights_init(data.values)
    som.train_random(data.values, 500)

    labels = np.array([som.winner(d)[0] * som_dim + som.winner(d)[1] for d in data.values])
    cluster_map = {}
    for idx, label in enumerate(labels):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(idx)

    return labels, cluster_map, som


def find_optimal_clusters(data, max_clusters):
    db_scores = []
    cluster_counts = []
    for clusters in range(2, max_clusters + 1):
        labels, cluster_map, _ = perform_clustering(data, clusters)
        unique_labels = np.unique(labels)
        
        if len(unique_labels) > 1:
            score = davies_bouldin_score(data.values, labels)
            db_scores.append(score)
            cluster_counts.append(clusters)
            print(f"DB Score for {clusters} clusters: {score}")
        else:
            print(f"Insufficient clusters ({len(unique_labels)}) for DB score calculation at {clusters} clusters.")
    
    min_score_index = db_scores.index(min(db_scores))
    optimal_clusters = cluster_counts[min_score_index]

    return cluster_counts, db_scores, optimal_clusters


def calculate_generalized_imbalance(df, facet_name):
    facet_counts = df[facet_name].value_counts()
    total = facet_counts.sum()
    proportions = facet_counts / total
    imbalance = proportions.max() - proportions.min()
    return imbalance


def calculate_difference_in_proportions(na1, na, nd1, nd):
    if na == 0 or nd == 0:
        raise ValueError("Total number of members for either facet a or d cannot be zero.")

    qa = na1 / na if na > 0 else 0
    qd = nd1 / nd if nd > 0 else 0

    dpl = qa - qd
    return dpl


def kullback_leibler_divergence(Pa, Pd):
    Pa = np.array(Pa)
    Pd = np.array(Pd)
    Pd = np.where(Pd == 0, 1e-10, Pd)
    nonzero_Pa = Pa > 0
    kl_divergence = np.sum(Pa[nonzero_Pa] * np.log(Pa[nonzero_Pa] / Pd[nonzero_Pa]))
    
    return kl_divergence


def generalized_demographic_disparity(df, facet_name, outcome_name, reference_group=None):
    group_proportions = df.groupby(facet_name)[outcome_name].value_counts(normalize=True).unstack(fill_value=0)
    
    if reference_group:
        reference_proportions = group_proportions.loc[reference_group]
    else:
        reference_proportions = df[outcome_name].value_counts(normalize=True)
    
    disparity = group_proportions - reference_proportions

    return disparity


def generalized_conditional_demographic_disparity(df, facet_name, outcome_name, subgroup_column):
    subgroup_proportions = df.groupby([subgroup_column, facet_name])[outcome_name].value_counts(normalize=True).unstack(fill_value=0)
    overall_proportions = df[outcome_name].value_counts(normalize=True).reindex(subgroup_proportions.columns, fill_value=0)
    aggregated_disparity = pd.DataFrame(0, index=subgroup_proportions.index, columns=subgroup_proportions.columns)

    for subgroup, group_data in subgroup_proportions.groupby(level=0):
        subgroup_disparity = group_data - overall_proportions
        subgroup_weight = len(df[df[subgroup_column] == subgroup]) / len(df)
        aggregated_disparity += subgroup_disparity * subgroup_weight

    return aggregated_disparity


def compute_probability_distributions(df, facet_name, label_column):
    facet_label_counts = df.groupby(facet_name)[label_column].value_counts().unstack(fill_value=0)
    probability_distributions = facet_label_counts.div(facet_label_counts.sum(axis=1), axis=0)
    distributions_dict = {facet: probabilities.values for facet, probabilities in probability_distributions.iterrows()}
    return distributions_dict


def jensen_shannon_divergence(Pa, Pd):
    M = 0.5 * (Pa + Pd)
    kl_pm = kullback_leibler_divergence(Pa, M)
    kl_qm = kullback_leibler_divergence(Pd, M)
    js_divergence = 0.5 * (kl_pm + kl_qm)
    return js_divergence


def lp_norm(Pa, Pd, p=2):
    return np.linalg.norm(Pa - Pd, ord=p)


def generalized_total_variation_distance(df, facet_name, outcome_name):
    outcome_counts = df.groupby([facet_name, outcome_name]).size().unstack(fill_value=0)
    outcome_probabilities = outcome_counts.div(outcome_counts.sum(axis=1), axis=0)

    facets = outcome_probabilities.index.unique()
    n = len(facets)
    
    if n < 2:
        raise ValueError("Not enough groups for comparison (at least two required).")
    
    total_tvd = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            na = outcome_probabilities.loc[facets[i]]
            nb = outcome_probabilities.loc[facets[j]]
            l1_norm = sum(abs(na[k] - nb[k]) for k in na.index)
            tvd = 0.5 * l1_norm
            total_tvd += tvd
            count += 1

    average_tvd = total_tvd / count if count > 0 else 0
    return average_tvd


def kolmogorov_smirnov_metric(Pa, Pd):
    
    if Pa.size == 0 or Pd.size == 0:
        raise ValueError("One or both probability distributions are empty.")
        
    if Pa.size != Pd.size:
        raise ValueError("Distributions must be of the same length.")

    ks_metric = max(abs(Pa - Pd))
    return ks_metric


def normalized_mutual_information(df, facet_name, outcome_name):
    return normalized_mutual_info_score(df[facet_name], df[outcome_name])


def conditional_mutual_information(df, facet_name, outcome_name, conditional_column):
    unique_values = df[conditional_column].unique()
    cond_mi = 0
    for value in unique_values:
        subset = df[df[conditional_column] == value]
        mi = mutual_info_score(subset[facet_name], subset[outcome_name])
        cond_mi += (len(subset) / len(df)) * mi
    return cond_mi / np.log(len(unique_values))


def binary_ratio(df, facet_name, outcome_name):
    outcomes = df.groupby(facet_name)[outcome_name].mean()
    return outcomes[1] / outcomes[0]


def binary_difference(df, facet_name, outcome_name):
    outcomes = df.groupby(facet_name)[outcome_name].mean()
    return outcomes[1] - outcomes[0]


def conditional_binary_difference(df, facet_name, outcome_name, conditional_column):
    unique_values = df[conditional_column].unique()
    cond_diff = 0
    for value in unique_values:
        subset = df[df[conditional_column] == value]
        diff = binary_difference(subset, facet_name, outcome_name)
        cond_diff += (len(subset) / len(df)) * diff
    return cond_diff


def pearson_correlation(df, facet_name, outcome_name):
    return df[facet_name].corr(df[outcome_name])


def logistic_regression_analysis(df, facet_name, outcome_name):
    model = LogisticRegression()
    X = df[facet_name].values.reshape(-1, 1)
    y = df[outcome_name]
    model.fit(X, y)
    return model.coef_, model.intercept_


def get_user_input():
    file_path = input("Enter the path to your dataset (CSV or JSON file): ")
    
    if not file_path.lower().endswith(('.csv', '.json')):
        print("Unsupported file type. Please provide a CSV or JSON file.")
        return None
    
    use_subgroup_analysis = int(input("Enter 1 to apply subgroup analysis, 0 to skip: "))
    facet_name = input("Enter the column name for the facet (e.g., Gender): ")
    outcome_name = input("Enter the column name for the outcome (e.g., Lymphoma): ")
    subgroup_column = input("Enter the column name for subgroup categorization (optional, press Enter to skip): ")
    
    try:
        label_value = int(input("Enter the label value or threshold for positive outcomes (e.g., 1): "))
    except ValueError:
        print("Invalid input! Please enter a valid integer for the label value.")
        return None
    
    return file_path, facet_name, outcome_name, subgroup_column, label_value, use_subgroup_analysis


def load_data(file_path):
    try:
        if file_path.lower().endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.lower().endswith('.json'):
            return pd.read_json(file_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None


def calculate_metrics(D, facet_name, label_values_or_threshold, outcome_name, subgroup_column):
    print("Calculating pre-training data bias metrics...")
    
    #CLASS IMBALANCE (CI)
    ci = calculate_generalized_imbalance(D, facet_name)
    print("- CI for", facet_name, "is", str(ci))
    
    #CI values near either of the extremes values of -1 or 1 are very imbalanced and are at a substantial risk of making biased predictions.
    if ((np.around(ci,2) >= 0.9)|(np.around(ci,2) <= -0.9)):
        print(">> Warning: Significant bias detected based on CI metric!")
    
    #DIFFERENCE IN PROPORTION LABELS (DPL)
    num_facet = D[facet_name].value_counts()
    num_facet_adv = num_facet[1] #favored facet values
    num_facet_disadv = num_facet[0] #disfavored facet values
    num_facet_and_pos_label = D[facet_name].where(D[outcome_name] == label_values_or_threshold).value_counts()
    num_facet_and_pos_label_adv = num_facet_and_pos_label[1]
    num_facet_and_pos_label_disadv = num_facet_and_pos_label[0]
    dpl = calculate_difference_in_proportions(num_facet_and_pos_label_adv,
                                              num_facet_adv,
                                              num_facet_and_pos_label_disadv,
                                              num_facet_disadv)
    print("- DPL for", 
          facet_name, 
          "given the outcome", 
          outcome_name, 
          "=", 
          str(label_values_or_threshold), 
          "is", 
          str(dpl))
    
    if abs(dpl) > 0.1:
        print(">> Warning: Significant bias detected based on DPL metric!")
    
    #DEMOGRAPHIC DISPARITY (DD)
    dd = generalized_demographic_disparity(D, 
                                           facet_name, 
                                           outcome_name, 
                                           reference_group=None)
    print("- Average DD for", 
          facet_name,
          "given the outcome",
          outcome_name,
          "is:", 
          str(dd.mean().mean()))
    
    if abs(dd.mean().mean()) > 0.1:
        print(">> Warning: Significant bias detected based on DD metric!")
    
    #CONDITIONAL DEMOGRAPHIC DISPARITY (CDD)
    if subgroup_column.strip():  # Check if subgroup_column is provided and not just an empty string
        cdd = generalized_conditional_demographic_disparity(D, 
                                                            facet_name, 
                                                            outcome_name, 
                                                            subgroup_column)
        print("- Average CDD for", 
              facet_name,
              "given the outcome",
              outcome_name,
              "conditioned by",
              subgroup_column,
              "is:", 
              str(cdd.mean().mean()))
        
        if (cdd.mean().mean()) > 0.1:
            print(">> Warning: Significant bias detected based on CDD metric!")
    else:
        print("- Average CDD: Subgroup was not provided.")   
    
    #Compute the probability distributions for the facets
    probability_distributions = compute_probability_distributions(D, 
                                                                  facet_name, 
                                                                  outcome_name)
    Pa = probability_distributions.get(1, np.array([]))
    Pd = probability_distributions.get(0, np.array([]))
    
    if Pa.size > 0 and Pd.size > 0 and Pa.size == Pd.size:
        #JENSEN-SHANNON DIVERGENCE
        js_divergence = jensen_shannon_divergence(Pa, Pd)
        print("- Jensen-Shannon Divergence between", 
              facet_name,
              "and",
              outcome_name,
              "is", 
              str(js_divergence))
        
        if js_divergence > 0.1:
            print(">> Warning: Significant bias detected based on JS metric!")
        
        #L2 NORM
        l2_norm_value = lp_norm(Pa, Pd, p=2)
        print("- L2 norm between", 
              facet_name,
              "and",
              outcome_name,
              "is", 
              str(l2_norm_value))
        
        if l2_norm_value > 0.1:
            print(">> Warning: Significant bias detected based on L2 norm metric!")
            
        #TOTAL VARIATION DISTANCE (TVD)
        try:
            tvd = generalized_total_variation_distance(D, facet_name, outcome_name)
            print("- TVD for", 
                  facet_name,
                  "given",
                  outcome_name,
                  "is", 
                  str(tvd))
            
            if tvd > 0.1:
                print(">> Warning: Significant bias detected based on TVD metric!")
             
        except ValueError as e:
            print(e)
            
        #KS METRIC
        ks_value = kolmogorov_smirnov_metric(Pa, Pd)
        print("- KS metric between", 
              facet_name,
              "and",
              outcome_name,
              "is", 
              str(ks_value))
        if abs(ks_value) > 0.1:
            print(">> Warning: Significant bias detected based on KS metric!")
    else:
        print("Cannot compute Jensen-Shannon Divergence, L2 norm, TVD, KS due to data issues.")

   #NORMALIZED MUTUAL INFORMATION (NMI)
    nmi = normalized_mutual_information(D, facet_name, outcome_name)
    print("- NMI between", facet_name, "and", outcome_name, "is", str(nmi))

    #NORMALIZED CONDITIONAL MUTUAL INFORMATION (NCMI)
    if subgroup_column:
        cond_nmi = conditional_mutual_information(D, facet_name, outcome_name, subgroup_column)
        print("- NCMI for", facet_name, "and", outcome_name, "conditioned on", subgroup_column, "is", str(cond_nmi))
    else:
        print("- NCMI: Subgroup was not provided.")

    #BINARY RATIO (BR)
    if D[facet_name].nunique() == 2 and D[outcome_name].nunique() == 2:  # Check binary nature
        ratio = binary_ratio(D, facet_name, outcome_name)
        print("- BR for", 
              facet_name, 
              "and", 
              outcome_name, 
              "is", 
              str(ratio))
    else:
        print("- BR: One or both variables are not binary.")

    #BINARY DIFFERENCE (BD)
    if D[facet_name].nunique() == 2 and D[outcome_name].nunique() == 2:
        diff = binary_difference(D, facet_name, outcome_name)
        print("- BD for", 
              facet_name, 
              "and", outcome_name, 
              "is", 
              str(diff))
    else:
        print("- BD: One or both variables are not binary.")

    #CONDITIONAL BINARY DIFFERENCE (CBD)
    if subgroup_column and D[facet_name].nunique() == 2 and D[outcome_name].nunique() == 2:
        cond_diff = conditional_binary_difference(D, facet_name, outcome_name, subgroup_column)
        print("- CBD for", 
              facet_name, 
              "and", 
              outcome_name, 
              "conditioned on", 
              subgroup_column, 
              "is", 
              str(cond_diff))
    else:
        print("- CBD: Missing conditions for binary conditional difference.")

    #PEARSON CORRELATION (CORR)
    if D[facet_name].dtype in ['int64', 'float64'] and D[outcome_name].dtype in ['int64', 'float64']:
        corr = pearson_correlation(D, facet_name, outcome_name)
        print("- CORR between", 
              facet_name, 
              "and", 
              outcome_name, 
              "is", 
              str(corr))
    else:
        print("- CORR: Variables are not ordinal.")

    #LOGISTIC REGRESSION (LR)
    if D[facet_name].nunique() == 2:
        coeffs, intercept = logistic_regression_analysis(D, facet_name, outcome_name)
        print("- LR coefficients for", 
              facet_name, 
              "predicting", 
              outcome_name, 
              "are", 
              str(coeffs))
        print("  Intercept is", str(intercept))
    else:
        print("- LR: Protected feature is not binary or outcome is not multi-labeled.")


def main():
    user_input = get_user_input()
    if user_input is None:
        return
    
    file_path, facet_name, outcome_name, subgroup_column, label_values_or_threshold, use_subgroup_analysis = user_input
    D = load_data(file_path)
    if D is None:
        return
    
    print("")

    if use_subgroup_analysis == 1:
        max_clusters = 30
        cluster_counts, db_scores, optimal_clusters = find_optimal_clusters(D, max_clusters)

        print(f"The optimal number of clusters is {optimal_clusters}")
        plot_db_scores(cluster_counts, db_scores, optimal_clusters)

        print("Applying the Minisom")
        _, cluster_map, _ = perform_clustering(D, optimal_clusters)
        print("")

        ct = 0
        for cluster_id, indices in cluster_map.items():
            ct+=1
            try:
                print(f"Starting analysis for cluster {ct + 1} of {optimal_clusters} with {len(indices)} samples.")
                Dk = D.iloc[indices]
                print(f"\nAnalyzing cluster {ct + 1} / {optimal_clusters}")

                print(f"Unique outcomes {len(np.unique(Dk[outcome_name]))}")
                print(f"Unique facets {len(np.unique(Dk[facet_name]))}")

                if len(np.unique(Dk[outcome_name])) == 1 or len(np.unique(Dk[facet_name])) == 1:
                    print(f"Skipping cluster {ct + 1}: Not enough diversity in '{outcome_name}' or in '{facet_name}'.")
                else:
                    try:
                        calculate_metrics(Dk, facet_name, label_values_or_threshold, outcome_name, subgroup_column)
                    except Exception as e:
                        print(f"Failed to calculate metrics: {e}")
            except Exception as e:
                print(f"Error processing cluster {ct + 1}: {e}")
        print("Done")
    else:
            try:
                calculate_metrics(D, facet_name, label_values_or_threshold, outcome_name, subgroup_column)
            except Exception as e:
                print(f"Failed to calculate metrics: {e}")


if __name__ == "__main__":
    main()
    
