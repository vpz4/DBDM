# DBDM
**Data Bias Detection and Mitigation Toolkit**

## Description
This repository hosts a Python-based toolkit designed for the detection and mitigation of data bias in various datasets. It is especially useful for analyzing binary facets (e.g., Gender) and outcomes (e.g., disease status or test scores). The toolkit provides crucial metrics to assess bias, including Class Imbalance, Difference in Proportions of Labels, Demographic Disparity, Conditional Demographic Disparity, and various statistical divergences such as Kullback-Leibler, Jensen-Shannon, and Kolmogorov-Smirnov.

## Features
- **Class Imbalance (CI):** Evaluates the imbalance between two groups.
- **Difference in Proportions of Labels (DPL):** Measures disparity in positive outcomes between groups.
- **Demographic Disparity (DD):** Computes outcome disparity for a specific group.
- **Conditional Demographic Disparity (CDD):** Examines demographic disparities within subgroups.
- **Statistical Divergences:** Includes Kullback-Leibler and Jensen-Shannon divergences to quantify differences between probability distributions.
- **Total Variation Distance (TVD) and Kolmogorov-Smirnov Metric:** Assess the statistical distance between outcome distributions.
- **Interactive User Input:** Enables specification of dataset paths, column names, and values via console input, enhancing flexibility for various datasets.

## Usage
- **Setup:** Ensure Python 3.9 and necessary libraries (Pandas, NumPy) are installed.
- **Running the Script:** From the command line, navigate to the script's directory and execute it. The script will request necessary input such as dataset path and column names.
- **Output:** Outputs bias metrics directly to the console for further analysis or integration into reports.

## Installation
To install the toolkit, clone the repository and set up the required environment:

``bash
git clone https://github.com/vpz4/DBDM.git
cd data-bias-detection
pip install pandas numpy

## Contribution
Contributions are welcome. Please fork the repository and submit pull requests with your enhancements. Ensure that new features are accompanied by appropriate tests and documentation.

## License
This project is licensed under the MIT License - see the LICENSE file for details.<br />

## Additional notes
The script is designed to be modular and easy to extend for additional types of analyses.<br />
Future enhancements could include visualizations of the distributions and biases or integration with machine learning models to predict and mitigate biases directly.<br />
