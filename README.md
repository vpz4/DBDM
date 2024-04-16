# DBDM
**An open-source toolkit for data bias detection and mitigation**

## Description
This repository hosts a Python-based toolkit which has been designed for the detection and mitigation of (pre-training) data bias in various tabular datasets. It is useful for analyzing facets (e.g., Gender) and outcomes (e.g., disease status or test scores). The toolkit provides state of the art metrics to provvide a holistic view on the occurence of pre-training data bias, including Class Imbalance, Difference in Proportions of Labels, Demographic Disparity, Conditional Demographic Disparity, and various statistical metrics for estimating divergences, such as, Kullback-Leibler, Jensen-Shannon, and Kolmogorov-Smirnov.

## Requirements
- **Python Version:** Ensure Python 3.9 is installed.
- **Libraries required:** `pandas`, `numpy`
- **Data format:** The dataset should be in a format readable by `pandas`, typically CSV or Excel.
- **Facet column:** A column indicating binary facets (e.g., Gender with values 0: male and 1: female).
- **Outcome column:** A column with binary or continuous outcomes.
- **Subgroup column:** An optional column for subgroup analysis.

## Features
- **Class Imbalance (CI):** Evaluates the imbalance between two groups.
- **Difference in Proportions of Labels (DPL):** Measures disparity in positive outcomes between groups.
- **Demographic Disparity (DD):** Computes outcome disparity for a specific group.
- **Conditional Demographic Disparity (CDD):** Examines demographic disparities within subgroups.
- **Statistical Divergences:** Includes Kullback-Leibler and Jensen-Shannon divergences to quantify differences between probability distributions.
- **Total Variation Distance (TVD) and Kolmogorov-Smirnov Metric:** Assess the statistical distance between outcome distributions.
- **Interactive User Input:** Enables specification of dataset paths, column names, and values via console input, enhancing flexibility for various datasets.

## Usage
- **Running the script:** From the command line, navigate to the script's directory and execute it. The script will request necessary input such as dataset path and column names.
- **Output:** Outputs bias metrics directly to the console for further analysis or integration into reports.

## Installation
To install the toolkit, clone the repository and set up the required environment:

```bash
git clone https://github.com/vpz4/DBDM.git
cd DBDM
pip install pandas numpy
```

## Contribution
Contributions are welcome. Please fork the repository and submit pull requests with your enhancements. Ensure that new features are accompanied by appropriate tests and documentation.

## License
This project is licensed under the MIT License - see the LICENSE file for details.<br />

## Additional notes
The script is designed to be modular and easy to extend for additional types of analyses.<br />
Future enhancements could include visualizations of the distributions and biases or integration with machine learning models to predict and mitigate biases directly.<br />
