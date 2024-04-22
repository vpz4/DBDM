# DBDM
**An open-source toolkit for data bias detection and mitigation**

## Description
This repository hosts a Python-based toolkit which has been designed for the detection and mitigation of data bias in various tabular datasets. It is useful for analyzing facets (e.g., Gender) and outcomes (e.g., disease status or test scores). The toolkit utilizes state of the art metrics to provide a holistic view on the detection of pre-training bias, including the Class Imbalance, the Difference in Proportions of Labels, the Demographic Disparity, the Conditional Demographic Disparity, and various statistical metrics for estimating divergence between facets and outcomes, such as, the Kullback-Leibler (KL) divergence, the Jensen-Shannon (JS) divergence, and the Kolmogorov-Smirnov (KS) metric.

## Requirements
- **Python version:** Python 3.9.
- **Libraries:** `pandas`, `numpy`
- **Data format:** The input dataset should be in a format readable by `pandas`, typically CSV or JSON.
- **Facet:** A binary or continuous variable indicating the facet (e.g., Gender).
- **Outcome:** A binary variable indicating the outcome (e.g., Lymphoma).
- **Subgroup:** An optional binary or categorical variable for subgroup categorization (this is used only for the estimation of the CDD metric).
- **Label value or threshold:** A label value or threshold for positive outcomes (e.g. 1).

## Features
- **Class Imbalance (CI):** Evaluates the imbalance between the groups in the facet.
- **Difference in Proportions of Labels (DPL):** Measures the disparity in the positive outcomes between the groups in the facet.
- **Demographic Disparity (DD):** Computes the disparity for a specific group.
- **Conditional Demographic Disparity (CDD):** Examines demographic disparities within subgroups.
- **Kullback-Leibler divergences:** Estimates the Kullback-Leibler (KL) divergence between the probability distributions of the facet and the outcome.
- **Jensen-Shannon (JS) divergence:** Estimates the Jensen-Shannon (JS) divergence between the facet and the outcome.
- **Total Variation Distance (TVD):** Measures the distance between probability distributions. 
- **Kolmogorov-Smirnov (KS) metric:** Assesses the statistical distance between the probability distributions of the facet and the outcome.
- **Normalized Mutual Information (NMI):** Measures the information shared between two categorical variables, normalized to a range of [0, 1] where 1 indicates perfect correlation and 0 indicates no correlation.
- **Normalized Conditional Mutual Information (NCMI):** Measures the mutual information between two categorical variables, conditioned on a third, normalized over the possible outcomes of the conditioning variable.
- **Binary Ratio (BR):** Computes the ratio of positive outcomes between two binary groups.
- **Binary Difference (BD):** Calculates the difference in proportions of positive outcomes between two binary groups to detect disparities.
- **Conditional Binary Difference (CBD):** Computes the binary difference, conditioned on another categorical feature, to analyze disparities within subgroups.
- **Pearson Correlation (CORR):** Determines the linear correlation between two ordinal features, with values ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).
- **Logistic Regression (LR):** Fits a logistic regression model to predict a multi-labeled outcome from a binary protected feature to asssess the influence of the feature on the outcome.

## Usage
- **Running the script:** From the command line, navigate to the script's directory and execute it. The script will request necessary input such as dataset path and facet, outcome, subgroup names as described in the Requirements section.
- **Output:** Outputs bias metrics directly to the console for further analysis or integration into reports. Warningns are also provided in the case of threshold violations in any of the above metrics.

## Installation
To install the toolkit, clone the repository and set up the required environment:

```bash
git clone https://github.com/vpz4/DBDM.git
cd DBDM
pip install pandas numpy
```

## Example of console output
![Capture](https://github.com/vpz4/DBDM/assets/15791743/04135823-500a-43b9-b883-3658bc4488f4)


## Contribution
Contributions are welcome. Please fork the repository and submit pull requests with your enhancements. Ensure that new features are accompanied by appropriate tests and documentation.

## License
This project is licensed under the MIT License - see the LICENSE file for details.<br />

## Additional notes
The script is designed to be modular and easy to extend for additional types of analyses.<br />
Future enhancements could include visualizations of the distributions and biases.<br />
