# DBDM
**An open-source toolkit for data bias detection and mitigation**

## Description
This repository hosts a Python-based toolkit which has been designed for the detection and mitigation of data bias in various tabular datasets. It is useful for analyzing facets (e.g., Gender) and outcomes (e.g., disease status or test scores). The toolkit utilizes state of the art metrics to provide a holistic view on the detection of pre-training data bias.

## Requirements
- **Python version:** Python 3.9 or higher.
- **Libraries:** `pandas`, `numpy`
- **Data format:** The input dataset should be in a format readable by `pandas`, typically CSV or JSON.
- **Facet:** A binary or continuous variable indicating the facet (e.g., Gender).
- **Outcome:** A binary variable indicating the outcome (e.g., Lymphoma).
- **Subgroup:** An optional binary or categorical variable for subgroup categorization.
- **Label value or threshold:** A label value or threshold for positive outcomes (e.g. 1).

## Metrics
- **Class Imbalance (CI):** Evaluates the imbalance between the groups in the facet.
- **Difference in Proportions of Labels (DPL):** Measures the disparity in the positive outcomes between the groups in the facet.
- **Demographic Disparity (DD):** Computes the disparity for a specific group.
- **Conditional Demographic Disparity (CDD):** Examines demographic disparities within subgroups.
- **Kullback-Leibler divergences:** Estimates the Kullback-Leibler (KL) divergence between the probability distributions of the facet and the outcome.
- **Jensen-Shannon (JS) divergence:** Estimates the Jensen-Shannon (JS) divergence between the probability distributions of the facet and the outcome.
- **Total Variation Distance (TVD):** Measures the distance between the probability distributions of the facet and the outcome.
- **Kolmogorov-Smirnov (KS) metric:** Assesses the statistical distance between the probability distributions of the facet and the outcome.
- **Normalized Mutual Information (NMI):** Measures the information shared between two categorical variables, normalized to a range of [0, 1] where 1 indicates perfect correlation and 0 indicates no correlation.
- **Normalized Conditional Mutual Information (NCMI):** Measures the mutual information between two categorical variables, conditioned on a third, normalized over the possible outcomes of the conditioning variable.
- **Binary Ratio (BR):** Computes the ratio of positive outcomes between two binary groups.
- **Binary Difference (BD):** Calculates the difference in proportions of positive outcomes between two binary groups to detect disparities.
- **Conditional Binary Difference (CBD):** Computes the binary difference, conditioned on another categorical feature, to analyze disparities within subgroups.
- **Pearson Correlation (CORR):** Determines the linear correlation between two ordinal features, with values ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).
- **Logistic Regression (LR):** Fits a logistic regression model to predict a multi-labeled outcome from a binary protected feature to asssess the influence of the feature on the outcome.

## Installation
To install the toolkit, clone the repository and set up the required environment:

```bash
git clone https://github.com/vpz4/DBDM.git
cd DBDM
pip install pandas numpy
```

## Usage
- **Running the script:** From the command line, navigate to the script's directory and execute it.
```bash
python DBDM.py
```
- **Prompt:** The script will request necessary input such as dataset path and facet, outcome, subgroup names as described in the Requirements section.
```bash
Enter the path to your dataset (CSV or JSON file):
```
```bash
Enter the column name for the facet (e.g., Gender):
```
```bash
Enter the column name for the outcome (e.g., Lymphoma):
```
```bash
Enter the column name for subgroup categorization (optional, press Enter to skip):
```
```bash
Enter the label value or threshold for positive outcomes (e.g., 1):
```
- **Output:** Outputs bias metrics directly to the console for further analysis or integration into reports. Warningns are also provided in the case of threshold violations in any of the above metrics.
```bash
Enter the path to your dataset (CSV or JSON file): test.csv
Enter the column name for the facet (e.g., Gender): Gender
Enter the column name for the outcome (e.g., Lymphoma): Lymphoma
Enter the column name for subgroup categorization (optional, press Enter to skip): 
Enter the label value or threshold for positive outcomes (e.g., 1): 1

Calculating pre-training data bias metrics...
- CI for Gender is 0.897872340425532
>> Warning: Significant bias detected based on CI metric!
- DPL for Gender given the outcome Lymphoma = 1 is 0.005455904334828107
- Average DD for Gender given the outcome Lymphoma is: -1.0408340855860843e-17
- Average CDD: Subgroup was not provided.
- Jensen-Shannon Divergence between Gender and Lymphoma is 4.7313507784679104e-05
- L2 norm between Gender and Lymphoma is 0.007715813905324001
- TVD for Gender given Lymphoma is 0.005455904334828059
- KS metric between Gender and Lymphoma is 0.005455904334828107
- NMI between Gender and Lymphoma is 3.631951353741736e-05
- NCMI: Subgroup was not provided.
- BR for Gender and Lymphoma is 1.0654708520179372
- BD for Gender and Lymphoma is 0.005455904334828107
- CBD: Missing conditions for binary conditional difference.
- CORR between Gender and Lymphoma is 0.004228325385464291
- LR coefficients for Gender predicting Lymphoma are [[0.06110546]]
  Intercept is [-2.39097006]
```


## Contribution
Contributions are welcome. Please fork the repository and submit pull requests with your enhancements. Ensure that new features are accompanied by appropriate tests and documentation.

## License
This project is licensed under the MIT License.<br />

## Additional notes
The script is designed to be modular and easy to extend for additional types of analyses.<br />
Future enhancements could include visualizations of the distributions and biases.<br />
