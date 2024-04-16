# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:08:08 2024

@author: bPezo
"""

import pandas as pd
import numpy as np
import os

os.chdir(os.getcwd())

def calculate_generalized_imbalance(df, facet_column):
    facet_counts = df[facet_column].value_counts()
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


def generalized_demographic_disparity(df, facet_column, outcome_column, reference_group=None):
    group_proportions = df.groupby(facet_column)[outcome_column].value_counts(normalize=True).unstack(fill_value=0)
    
    if reference_group:
        reference_proportions = group_proportions.loc[reference_group]
    else:
        reference_proportions = df[outcome_column].value_counts(normalize=True)
    
    disparity = group_proportions - reference_proportions

    return disparity


def generalized_conditional_demographic_disparity(df, facet_column, outcome_column, subgroup_column):
    subgroup_proportions = df.groupby([subgroup_column, facet_column])[outcome_column].value_counts(normalize=True).unstack(fill_value=0)
    overall_proportions = df[outcome_column].value_counts(normalize=True).reindex(subgroup_proportions.columns, fill_value=0)
    aggregated_disparity = pd.DataFrame(0, index=subgroup_proportions.index, columns=subgroup_proportions.columns)

    for subgroup, group_data in subgroup_proportions.groupby(level=0):
        subgroup_disparity = group_data - overall_proportions
        subgroup_weight = len(df[df[subgroup_column] == subgroup]) / len(df)
        aggregated_disparity += subgroup_disparity * subgroup_weight

    return aggregated_disparity


def compute_probability_distributions(df, facet_column, label_column):
    facet_label_counts = df.groupby(facet_column)[label_column].value_counts().unstack(fill_value=0)
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


def generalized_total_variation_distance(df, facet_column, outcome_column):
    outcome_counts = df.groupby([facet_column, outcome_column]).size().unstack(fill_value=0)
    facets = outcome_counts.index.unique()
    n = len(facets)
    
    if n < 2:
        raise ValueError("Not enough groups for comparison (at least two required).")
    
    total_tvd = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate the L1-norm between each pair of groups
            na = outcome_counts.loc[facets[i]]
            nb = outcome_counts.loc[facets[j]]
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


def get_user_input():
    file_path = input("Enter the path to your dataset (Excel file): ")
    facet_column = input("Enter the column name for the facet (e.g., Gender): ")
    outcome_column = input("Enter the column name for the outcome (e.g., Lymphoma): ")
    subgroup_column = input("Enter the column name for subgroup categorization (optional, press Enter to skip): ")
    
    try:
        label_value = int(input("Enter the label value or threshold for positive outcomes (e.g., 1): "))
    except ValueError:
        print("Invalid input! Please enter a valid integer for the label value.")
        return None
    
    return file_path, facet_column, outcome_column, subgroup_column, label_value


def load_data(file_path):
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None


def main():
    user_input = get_user_input()
    if user_input is None:
        return  # Exit if the user input was invalid
    
    file_path, facet_name, outcome_name, subgroup_column, label_values_or_threshold = user_input
    D = load_data(file_path)
    if D is None:
        return  # Exit if data loading failed
    
    print("")
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


if __name__ == "__main__":
    main()
    
