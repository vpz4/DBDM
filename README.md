# DBDM
A toolkit for data bias detection and mitigation (DBDM)

Description

This repository contains a Python script designed for the detection and mitigation of data bias in datasets. It is particularly useful in analyzing datasets with binary facets (such as Gender or any dichotomous variable) and binary or continuous outcomes (like disease status or test scores). The script provides several key metrics to evaluate bias, including Class Imbalance (CI), Difference in Proportions of Labels (DPL), Demographic Disparity (DD), Conditional Demographic Disparity (CDD), as well as statistical divergences like Kullback-Leibler, Jensen-Shannon, and Kolmogorov-Smirnov.

1. Features<br />

Class Imbalance (CI): Measures the imbalance between two facets.<br />
Difference in Proportions of Labels (DPL): Quantifies the disparity in positive outcomes between two facets.<br />
Demographic Disparity (DD): Calculates the disparity in outcomes for a specific facet.<br />
Conditional Demographic Disparity (CDD): Assesses demographic disparity across various subgroups.
Statistical Divergences: Includes calculations for Kullback-Leibler divergence and Jensen-Shannon divergence to measure the difference between probability distributions.
Total Variation Distance (TVD) and Kolmogorov-Smirnov Metric: These metrics measure the statistical distance between distributions of outcomes for each facet.
Interactive User Input: Allows users to specify dataset paths, column names, and relevant values through console input, making the script flexible for different datasets and scenarios.

2. Usage

Setup: Ensure Python 3 and required libraries (Pandas, NumPy) are installed.
Running the Script: Navigate to the script's directory and run it via a command line interface (CLI). The script will prompt for necessary inputs such as the path to the dataset, relevant column names, and parameters for analysis.
Output: The script outputs bias metrics to the console, which can be used for further analysis or reporting.

3. Installation

Clone the repository and install required dependencies:
git clone https://github.com/yourusername/data-bias-detection.git
cd data-bias-detection
pip install pandas numpy
Contribution
Contributions are welcome. Please fork the repository and submit pull requests with your enhancements. Ensure that new features are accompanied by appropriate tests and documentation.

4. License

This project is licensed under the MIT License - see the LICENSE file for details.

5. Additional Notes

The script is designed to be modular and easy to extend for additional types of analyses.
Future enhancements could include visualizations of the distributions and biases or integration with machine learning models to predict and mitigate biases directly.
