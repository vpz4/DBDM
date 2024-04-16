![image](https://github.com/vpz4/DBDM/assets/15791743/cd080fb6-135e-4081-9253-e0c54d92574b)# DBDM
**An open-source toolkit for data bias detection and mitigation**

## Description
This repository hosts a Python-based toolkit which has been designed for the detection and mitigation of data bias in various tabular datasets. It is useful for analyzing facets (e.g., Gender) and outcomes (e.g., disease status or test scores). The toolkit utilizes state of the art metrics to provide a holistic view on the detection of pre-training bias, including the Class Imbalance, the Difference in Proportions of Labels, the Demographic Disparity, the Conditional Demographic Disparity, and various statistical metrics for estimating divergence between facets and outcomes, such as, the Kullback-Leibler (KL) divergence, the Jensen-Shannon (JS) divergence, and the Kolmogorov-Smirnov (KS) metric.

## Requirements
- **Python version:** Python 3.9.
- **Libraries:** `pandas`, `numpy`
- **Data format:** The input dataset should be in a format readable by `pandas`, typically CSV or Excel.
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
runfile('C:/DBDM/DBDM_prompt_general_facets_v01.py', wdir='C:/DBDM')
Enter the path to your dataset (Excel file): test.xlsx
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
- TVD for Gender given Lymphoma is 527.5
>> Warning: Significant bias detected based on TVD metric!
- KS metric between Gender and Lymphoma is 0.005455904334828107

## Contribution
Contributions are welcome. Please fork the repository and submit pull requests with your enhancements. Ensure that new features are accompanied by appropriate tests and documentation.

## License
This project is licensed under the MIT License - see the LICENSE file for details.<br />

## Additional notes
The script is designed to be modular and easy to extend for additional types of analyses.<br />
Future enhancements could include visualizations of the distributions and biases.<br />
