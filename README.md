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
- **Statistical Divergences:** Estimates the Kullback-Leibler (KL) and the Jensen-Shannon (JS) divergences to quantify differences between probability distributions.
- **Total Variation Distance (TVD) and Kolmogorov-Smirnov (KS) Metric:** Assesses the statistical distance between the outcome distributions.
- **Interactive User Input:** Enables specification of dataset paths, column names, and values via console input, enhancing flexibility for various datasets.

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
![Capture](https://github.com/vpz4/DBDM/assets/15791743/0f0e39b9-e60c-404a-b3f9-49af9406d5a3)

## Contribution
Contributions are welcome. Please fork the repository and submit pull requests with your enhancements. Ensure that new features are accompanied by appropriate tests and documentation.

## License
This project is licensed under the MIT License - see the LICENSE file for details.<br />

## Additional notes
The script is designed to be modular and easy to extend for additional types of analyses.<br />
Future enhancements could include visualizations of the distributions and biases.<br />
