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
    """
    Calculate a generalized imbalance measure for multiple facets.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - facet_column (str): The column in df that distinguishes between facets.

    Returns:
    - float: A generalized imbalance measure.
    """
    facet_counts = df[facet_column].value_counts()
    total = facet_counts.sum()
    proportions = facet_counts / total
    imbalance = proportions.max() - proportions.min()
    return imbalance


def calculate_difference_in_proportions(na1, na, nd1, nd):
    """
    Calculate the Difference in Proportions of Labels (DPL).

    Parameters:
    - na1 (int): Number of members of facet a with a positive outcome.
    - na (int): Total number of members of facet a.
    - nd1 (int): Number of members of facet d with a positive outcome.
    - nd (int): Total number of members of facet d.

    Returns:
    - float: The Difference in Proportions of Labels (DPL).

    Description:
    - DPL values are normalized between -1 and 1.
    - Positive DPL indicates a higher proportion of positive outcomes in facet a compared to facet d.
    - DPL = 0 indicates demographic parity between the two facets.
    - Negative DPL indicates a higher proportion of positive outcomes in facet d compared to facet a.
    """
    # Ensure that the denominators are not zero to avoid division by zero
    if na == 0 or nd == 0:
        raise ValueError("Total number of members for either facet a or d cannot be zero.")

    # Calculate the proportions for each facet
    qa = na1 / na if na > 0 else 0  # Proportion of positive outcomes in facet a
    qd = nd1 / nd if nd > 0 else 0  # Proportion of positive outcomes in facet d

    # Calculate the Difference in Proportions of Labels (DPL)
    dpl = qa - qd
    return dpl


def kullback_leibler_divergence(Pa, Pd):
    """
    Calculate the Kullback-Leibler divergence between two distributions.

    Parameters:
    - Pa (list or numpy array): Probability distribution of facet a.
    - Pd (list or numpy array): Probability distribution of facet d.

    Returns:
    - float: The Kullback-Leibler divergence (KL) in nats.

    Description:
    - KL(Pa || Pd) is the expectation of the logarithmic difference between the probabilities
      Pa(y) and Pd(y), weighted by the probabilities Pa(y).
    - KL is zero when Pa and Pd are the same.
    - Positive KL values indicate divergence, with larger values showing greater divergence.
    """
    # Convert lists to numpy arrays if they are not already
    Pa = np.array(Pa)
    Pd = np.array(Pd)

    # Where Pd is zero, set it to a small number to avoid division by zero
    Pd = np.where(Pd == 0, 1e-10, Pd)
    
    # Where Pa is zero, the contribution to KL is zero, so we can ignore these terms
    nonzero_Pa = Pa > 0

    # Calculate KL divergence only for nonzero Pa values
    kl_divergence = np.sum(Pa[nonzero_Pa] * np.log(Pa[nonzero_Pa] / Pd[nonzero_Pa]))
    
    return kl_divergence


def generalized_demographic_disparity(df, facet_column, outcome_column, reference_group=None):
    """
    Calculate demographic disparity for each group within a facet compared to a reference group.
    
    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - facet_column (str): Column in df that distinguishes between demographic groups.
    - outcome_column (str): Column containing the outcomes.
    - reference_group (str): The group to use as a reference for comparison. If None, use the overall proportions.
    
    Returns:
    - DataFrame: Disparity measures for each group compared to the reference group.
    """
    # Calculate the proportion of each outcome within each facet group
    group_proportions = df.groupby(facet_column)[outcome_column].value_counts(normalize=True).unstack(fill_value=0)
    
    # Determine the reference proportions
    if reference_group:
        reference_proportions = group_proportions.loc[reference_group]
    else:
        reference_proportions = df[outcome_column].value_counts(normalize=True)
    
    # Calculate disparity as the difference or ratio (customizable based on requirements)
    disparity = group_proportions - reference_proportions  # Difference in proportions
    # disparity = group_proportions / reference_proportions  # Ratio of proportions could also be used

    return disparity


def generalized_conditional_demographic_disparity(df, facet_column, outcome_column, subgroup_column):
    # Compute proportions of each outcome within each facet group for each subgroup
    subgroup_proportions = df.groupby([subgroup_column, facet_column])[outcome_column].value_counts(normalize=True).unstack(fill_value=0)

    # Compute overall proportions in the entire dataset for comparison
    overall_proportions = df[outcome_column].value_counts(normalize=True).reindex(subgroup_proportions.columns, fill_value=0)

    # Initialize a DataFrame to store aggregated results with the same columns and index as subgroup_proportions
    aggregated_disparity = pd.DataFrame(0, index=subgroup_proportions.index, columns=subgroup_proportions.columns)

    # Iterate over each subgroup and calculate disparities
    for subgroup, group_data in subgroup_proportions.groupby(level=0):
        # Calculate disparity
        subgroup_disparity = group_data - overall_proportions
        # Weight of subgroup in dataset
        subgroup_weight = len(df[df[subgroup_column] == subgroup]) / len(df)
        # Add weighted disparity to the aggregated disparity
        aggregated_disparity += subgroup_disparity * subgroup_weight

    return aggregated_disparity


def compute_probability_distributions(df, facet_column, label_column):
    """
    Compute probability distributions for each facet based on the label column and return as numpy arrays.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - facet_column (str): The column in df that distinguishes between facets.
    - label_column (str): The column containing the labels to analyze.

    Returns:
    - dict: Dictionary of numpy arrays containing the probability distributions for each facet.
    """
    facet_label_counts = df.groupby(facet_column)[label_column].value_counts().unstack(fill_value=0)
    probability_distributions = facet_label_counts.div(facet_label_counts.sum(axis=1), axis=0)
    
    # Convert DataFrame rows to numpy arrays and store in a dictionary
    distributions_dict = {facet: probabilities.values for facet, probabilities in probability_distributions.iterrows()}
    return distributions_dict


def jensen_shannon_divergence(Pa, Pd):
    """
    Calculate the Jensen-Shannon divergence between two distributions.
    Both Pa and Pd are numpy arrays of probabilities and must be of the same length.
    """
    # Calculate M as the average of Pa and Pd
    M = 0.5 * (Pa + Pd)
    
    # Calculate the KL divergences from Pa to M and Pd to M
    kl_pm = kullback_leibler_divergence(Pa, M)
    kl_qm = kullback_leibler_divergence(Pd, M)
    
    # Calculate the JS divergence
    js_divergence = 0.5 * (kl_pm + kl_qm)
    return js_divergence


def lp_norm(Pa, Pd, p=2):
    """
    Calculate the Lp-norm between two distributions.

    Parameters:
    - Pa (numpy array): Distribution of outcomes for facet a.
    - Pd (numpy array): Distribution of outcomes for facet d.
    - p (int): The order of the norm (2 for Euclidean).

    Returns:
    - float: The Lp-norm between the two distributions.
    """
    
    return np.linalg.norm(Pa - Pd, ord=p)


def compute_outcome_distributions(df, facet_column, label_column):
    """
    Compute outcome distributions for each facet based on the label column.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - facet_column (str): The column in df that distinguishes between facets.
    - label_column (str): The column containing the labels to analyze.

    Returns:
    - np.array: Distribution of outcomes for facet a.
    - np.array: Distribution of outcomes for facet d.
    """
    # Count outcomes for each label within each facet
    outcome_counts = df.groupby(facet_column)[label_column].value_counts().unstack(fill_value=0)
    # Normalize these counts if needed or leave as counts for Lp-norm calculation
    Pa = outcome_counts.loc[1].values #ATTENTION
    Pd = outcome_counts.loc[0].values #ATTENTION

    return Pa, Pd


def total_variation_distance(df, facet_column, outcome_column):
    """
    Calculate the Total Variation Distance (TVD) based on the L1-norm of the differences in outcome counts.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - facet_column (str): The column in df that distinguishes between facets.
    - outcome_column (str): The column containing the outcomes to analyze.

    Returns:
    - float: The Total Variation Distance (TVD).
    """
    # Count the outcomes for each facet and outcome category
    outcome_counts = df.groupby([facet_column, outcome_column]).size().unstack(fill_value=0)

    # Ensure both facets are present in the outcomes
    if 0 not in outcome_counts.index or 1 not in outcome_counts.index:
        raise ValueError("Both facets 'a' and 'd' must be present in the data")

    # Get the counts for each facet
    na = outcome_counts.loc[0]
    nd = outcome_counts.loc[1]

    # Calculate the L1-norm
    l1_norm = sum(abs(na[i] - nd[i]) for i in na.index)

    # Calculate TVD as half the L1-norm
    tvd = 0.5 * l1_norm
    return tvd


def kolmogorov_smirnov_metric(Pa, Pd):
    """
    Calculate the Kolmogorov-Smirnov metric as the maximum divergence between two distributions.

    Parameters:
    - Pa (numpy array): Probability distribution for facet 'a'.
    - Pd (numpy array): Probability distribution for facet 'd'.

    Returns:
    - float: The KS metric.
    """
    # Validate inputs
    if Pa.size == 0 or Pd.size == 0:
        raise ValueError("One or both probability distributions are empty.")
    if Pa.size != Pd.size:
        raise ValueError("Distributions must be of the same length.")

    # Calculate the maximum divergence
    ks_metric = max(abs(Pa - Pd))
    return ks_metric


def get_user_input():
    # File input
    file_path = input("Enter the path to your dataset (Excel file): ")
    
    # Column names
    facet_column = input("Enter the column name for the facet (e.g., Gender): ")
    outcome_column = input("Enter the column name for the predicted outcomes (e.g., Lymphoma): ")
    subgroup_column = input("Enter the column name for subgroup categorization (optional, press Enter to skip): ")
    
    # Label values or thresholds
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
    
    file_path, facet_name, predicted_column, subgroup_column, label_values_or_threshold = user_input
    D = load_data(file_path)
    if D is None:
        return  # Exit if data loading failed
    
    #CLASS IMBALANCE (CI)
    ci = calculate_generalized_imbalance(D, facet_name)
    print("CI for", facet_name, "is", str(ci))
    
    #DIFFERENCE IN PROPORTION LABELS (DPL)
    num_facet = D[facet_name].value_counts()
    num_facet_adv = num_facet[1] #favored facet values
    num_facet_disadv = num_facet[0] #disfavored facet values
    num_facet_and_pos_label = D[facet_name].where(D[predicted_column] == label_values_or_threshold).value_counts()
    num_facet_and_pos_label_adv = num_facet_and_pos_label[1]
    num_facet_and_pos_label_disadv = num_facet_and_pos_label[0]
    dpl = calculate_difference_in_proportions(num_facet_and_pos_label_adv,
                                              num_facet_adv,
                                              num_facet_and_pos_label_disadv,
                                              num_facet_disadv)
    print("DPL for", facet_name, "given", predicted_column, "=", 
          str(label_values_or_threshold), "is", str(dpl))
    
    #DEMOGRAPHIC DISPARITY (DD)
    dd = generalized_demographic_disparity(D, 
                                           facet_name, 
                                           predicted_column, 
                                           reference_group=None)
    print(f"Demographic Disparity for facet {facet_name} is: {dd}")
    
    #CONDITIONAL DEMOGRAPHIC DISPARITY (CDD)
    cdd = generalized_conditional_demographic_disparity(D, 
                                                        facet_name, 
                                                        predicted_column, 
                                                        subgroup_column)
    print(f"Conditional Demographic Disparity (CDD): {cdd}")
    
    #Compute the probability distributions for the facets
    probability_distributions = compute_probability_distributions(D, 
                                                                  facet_name, 
                                                                  predicted_column)
    
    #Extract Pa and Pd assuming keys are known (e.g., '1' for Pa and '0' for Pd)
    Pa = probability_distributions.get(1, np.array([]))
    Pd = probability_distributions.get(0, np.array([]))
    
    if Pa.size > 0 and Pd.size > 0 and Pa.size == Pd.size:
        #JENSEN-SHANNON DIVERGENCE
        js_divergence = jensen_shannon_divergence(Pa, Pd)
        print(f"Jensen-Shannon Divergence for {facet_name}: {js_divergence}")
        
        #L2 NORM
        l2_norm_value = lp_norm(Pa, Pd, p=2)
        print(f"L2 Norm between facet distributions: {l2_norm_value}")
        
        #TOTAL VARIATION DISTANCE (TVD)
        try:
            tvd = total_variation_distance(D, facet_name, predicted_column)
            print(f"Total Variation Distance (TVD): {tvd}")
        except ValueError as e:
            print(e)
            
        #KS METRIC
        ks_value = kolmogorov_smirnov_metric(Pa, Pd)
        print(f"Kolmogorov-Smirnov metric: {ks_value}")
    else:
        print("Cannot compute Kolmogorov-Smirnov metric due to data issues.")


if __name__ == "__main__":
    main()
    
