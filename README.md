# DBDM
A toolkit for data bias detection and mitigation (DBDM)

Description<br />
This repository contains a Python script designed for the detection and mitigation of data bias in datasets. It is particularly useful in analyzing datasets with binary facets (such as Gender or any dichotomous variable) and binary or continuous outcomes (like disease status or test scores). The script provides several key metrics to evaluate bias, including Class Imbalance (CI), Difference in Proportions of Labels (DPL), Demographic Disparity (DD), Conditional Demographic Disparity (CDD), as well as statistical divergences like Kullback-Leibler, Jensen-Shannon, and Kolmogorov-Smirnov.

Features<br />
Class Imbalance (CI): Measures the imbalance between two facets.<br />
Difference in Proportions of Labels (DPL): Quantifies the disparity in positive outcomes between two facets.<br />
Demographic Disparity (DD): Calculates the disparity in outcomes for a specific facet.<br />
Conditional Demographic Disparity (CDD): Assesses demographic disparity across various subgroups.<br />
Statistical Divergences: Includes calculations for Kullback-Leibler divergence and Jensen-Shannon divergence to measure the difference between probability distributions.<br />
Total Variation Distance (TVD) and Kolmogorov-Smirnov Metric: These metrics measure the statistical distance between distributions of outcomes for each facet.<br />
Interactive User Input: Allows users to specify dataset paths, column names, and relevant values through console input, making the script flexible for different datasets and scenarios.<br />

Usage<br />
Setup: Ensure Python 3 and required libraries (Pandas, NumPy) are installed.<br />
Running the Script: Navigate to the script's directory and run it via a command line interface (CLI). The script will prompt for necessary inputs such as the path to the dataset, relevant column names, and parameters for analysis.<br />
Output: The script outputs bias metrics to the console, which can be used for further analysis or reporting.<br />

Installation<br />
Clone the repository and install required dependencies:<br />
Markup :  `code()`
git clone https://github.com/yourusername/data-bias-detection.git<br />
cd data-bias-detection<br />
pip install pandas numpy<br />

Contribution<br />
Contributions are welcome. Please fork the repository and submit pull requests with your enhancements. Ensure that new features are accompanied by appropriate tests and documentation.

License<br />
This project is licensed under the MIT License - see the LICENSE file for details.<br />

Additional notes<br />
The script is designed to be modular and easy to extend for additional types of analyses.<br />
Future enhancements could include visualizations of the distributions and biases or integration with machine learning models to predict and mitigate biases directly.<br />
