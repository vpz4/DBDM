# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:08:08 2024

@author: bPezo
"""

import pandas as pd
import numpy as np
import os

os.chdir(os.getcwd())

def calculate_facet_imbalance(na, nd):
    """
    Calculate the normalized facet imbalance measure (CI).

    Parameters:
    - na (int): Number of members of facet a.
    - nd (int): Number of members of facet d.

    Returns:
    - float: The normalized facet imbalance measure (CI).
    
    Description:
    - CI values are normalized between -1 and 1.
    - CI = 1 indicates only facet a is present.
    - CI = -1 indicates only facet d is present.
    - CI = 0 indicates a perfect balance between facet a and facet d.
    """
    # Calculate the normalized facet imbalance measure
    ci = (na - nd) / (na + nd)
    return ci


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


def demographic_disparity(nd0, n0, nd1, n1):
    """
    Calculate Demographic Disparity (DD) for a facet.

    Parameters:
    - nd0 (int): Number of rejected outcomes for facet d.
    - n0 (int): Total number of rejected outcomes.
    - nd1 (int): Number of accepted outcomes for facet d.
    - n1 (int): Total number of accepted outcomes.

    Returns:
    - float: Demographic Disparity (DD).
    """
    PdR = nd0 / n0 if n0 != 0 else 0  # Avoid division by zero
    PdA = nd1 / n1 if n1 != 0 else 0
    return PdR - PdA


def conditional_demographic_disparity(groups):
    """
    Calculate Conditional Demographic Disparity (CDD) across subgroups.

    Parameters:
    - groups (list of tuples): Each tuple contains (ni0, n0, ni1, n1, ni) for each subgroup i,
                               where ni0 and ni1 are the rejected and accepted counts for subgroup i,
                               n0 and n1 are the total rejected and accepted in the dataset,
                               and ni is the total observations in subgroup i.

    Returns:
    - float: Conditional Demographic Disparity (CDD).
    """
    total_observations = sum(group[4] for group in groups)
    if total_observations == 0:
        return 0  # Avoid division by zero

    weighted_sum = sum(
        group[4] * demographic_disparity(group[0], group[1], group[2], group[3]) for group in groups
    )
    cdd = weighted_sum / total_observations
    return cdd

def compute_probability_distributions(df, facet_column, label_column):
    """
    Compute probability distributions for each facet based on the label column.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - facet_column (str): The column in df that distinguishes between facets a and d.
    - label_column (str): The column containing the labels to analyze.

    Returns:
    - Pa (Series): Probability distribution for facet 'a'.
    - Pd (Series): Probability distribution for facet 'd'.
    """
    # Get the count of each label within each facet
    facet_label_counts = df.groupby(facet_column)[label_column].value_counts().unstack(fill_value=0)
    
    # Normalize these counts to get probability distributions
    probability_distributions = facet_label_counts.div(facet_label_counts.sum(axis=1), axis=0)
    
    # Assuming the facets are labeled as 0 and 1 in your facet_column
    Pa = probability_distributions.loc[1]  # Facet 'a'
    Pd = probability_distributions.loc[0]  # Facet 'd'
    
    return Pa, Pd

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
    - Pa (Series): Probability distribution for facet 'a'.
    - Pd (Series): Probability distribution for facet 'd'.

    Returns:
    - float: The KS metric.
    """
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
    num_facet = D[facet_name].value_counts()
    num_facet_adv = num_facet[1] #favored facet values
    num_facet_disadv = num_facet[0] #disfavored facet values
    ci = calculate_facet_imbalance(num_facet_adv, num_facet_disadv)
    print("CI for", facet_name, "is", str(ci))
    
    #DIFFERENCE IN PROPORTION LABELS (DPL)
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
    n0 = D[D[predicted_column] == 0].shape[0]  # Total rejections
    n1 = D[D[predicted_column] == 1].shape[0]  # Total acceptances
    
    # Counts of outcomes for the disfavored facet
    nd0 = D[(D[facet_name] == 0) & (D[predicted_column] == 0)].shape[0]  # Rejections in disfavored facet
    nd1 = D[(D[facet_name] == 0) & (D[predicted_column] == 1)].shape[0]  # Acceptances in disfavored facet
    
    # Counts of outcomes for the favored facet (for completeness and verification)
    na0 = D[(D[facet_name] == 1) & (D[predicted_column] == 0)].shape[0]  # Rejections in favored facet
    na1 = D[(D[facet_name] == 1) & (D[predicted_column] == 1)].shape[0]  # Acceptances in favored facet
    
    # Now let's assume we are calling the demographic_disparity function
    dd = demographic_disparity(nd0, n0, nd1, n1)
    print(f"Demographic Disparity for facet {facet_name} is: {dd}")
    
    # Verify by calculating DPL for comparison
    dpl = calculate_difference_in_proportions(na1, na0 + na1, nd1, nd0 + nd1)
    print(f"DPL for {facet_name} given outcomes is: {dpl}")
    
    # Group by subgroup_column and outcome_column and count occurrences
    grouped = D.groupby(subgroup_column)[predicted_column].value_counts().unstack(fill_value=0)
    
    # Prepare groups list
    groups = []
    for subgroup, row in grouped.iterrows():
        ni0 = row.get(0, 0)  # Rejections in this subgroup
        ni1 = row.get(1, 0)  # Acceptances in this subgroup
        ni = ni0 + ni1  # Total observations in this subgroup
        groups.append((ni0, n0, ni1, n1, ni))
    
    #Conditional Demographic Disparity (CDD)
    cdd = conditional_demographic_disparity(groups)
    print(f"Conditional Demographic Disparity (CDD): {cdd}")
    
    # Compute the probability distributions for the facets
    Pa, Pd = compute_probability_distributions(D, facet_name, predicted_column)
    
    #Jensen-Shannon divergence
    js_divergence = jensen_shannon_divergence(Pa, Pd)
    print(f"Jensen-Shannon Divergence for {facet_name}: {js_divergence}")
    
    # Assuming D is already loaded and cleaned
    Pa, Pd = compute_outcome_distributions(D, facet_name, predicted_column)
    
    # Calculate the L2 norm
    l2_norm_value = lp_norm(Pa, Pd, p=2)
    print(f"L2 Norm between facet distributions: {l2_norm_value}")
    
    # Calculate TVD
    try:
        tvd = total_variation_distance(D, facet_name, predicted_column)
        print(f"Total Variation Distance (TVD): {tvd}")
    except ValueError as e:
        print(e)
        
    # Compute the probability distributions
    Pa, Pd = compute_outcome_distributions(D, facet_name, predicted_column)
    
    # Calculate the Kolmogorov-Smirnov metric
    ks_value = kolmogorov_smirnov_metric(Pa, Pd)
    print(f"Kolmogorov-Smirnov metric: {ks_value}")
    
if __name__ == "__main__":
    main()
    