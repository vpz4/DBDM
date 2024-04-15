# DBDM
A suite of metrics for data bias detection and mitigation (DBDM)

Input: a tabular data set (rows: patient recods x cols: features)

Output: Data bias metrics
- Class imbalance (CI)
- Difference in proportions of labels (DPL)
- Kullback-Leibler divergence (KL)*
- Jensen-Shannon divergence (JS)*
- Lp-norm (LP)*
- Total variation distance data bias metric (TVD)
- Demographic disparity metric (DD)
- Conditional demographic disparity (CDD)

*Advanced

Note: All these metrics require a "facet_name" (e.g. Gender, Age group, Ethnicity), a "predicted_label" (e.g. An outcome), and a "subgroup_name" (e.g. Country).
