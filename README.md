# DBDM
**An open-source toolkit for data bias detection and mitigation**

## Description
This repository hosts a Python-based toolkit which has been designed for the detection and mitigation of data bias in various tabular datasets. It is useful for analyzing the effect of facets (e.g., Gender) and outcomes (e.g., disease status or test scores). The toolkit utilizes state of the art metrics to provide a holistic view on the detection of pre-training data bias. The tool also supports an option for the application of cluster analysis using the MiniSOM algorithm to identify biases in smaller subsets of the input dataset.

## Requirements
- **Python version:** Python 3.9 or higher.
- **Libraries:** `pandas`, `numpy`
- **Data format:** The input dataset should be in a format readable by `pandas`, typically CSV or JSON.
- **Facet:** A binary or continuous variable indicating the facet (e.g., Gender).
- **Outcome:** A binary variable indicating the outcome (e.g., Lymphoma).
- **Subgroup:** An optional binary or categorical variable for subgroup categorization.
- **Label value or threshold:** A label value or threshold for positive outcomes (e.g. 1).

## Metrics
- **Class Imbalance (CI):** Evaluates the imbalance between the groups in the facet. Reference: https://link.springer.com/chapter/10.1007/978-3-642-23166-7_12.
- **Difference in Proportions of Labels (DPL):** Measures the disparity in the positive outcomes between the groups in the facet. Reference: https://books.google.gr/books/about/Applied_Regression_Analysis_and_Other_Mu.html?id=v590AgAAQBAJ&redir_esc=y
- **Demographic Disparity (DD):** Computes the disparity for a specific group. Reference: https://arxiv.org/abs/1412.3756
- **Conditional Demographic Disparity (CDD):** Examines demographic disparities within subgroups. Reference: https://fairmlbook.org/
- **Kullback-Leibler divergence:** Estimates the Kullback-Leibler (KL) divergence between the probability distributions of the facet and the outcome. Reference: https://www.jstor.org/stable/2236703
- **Jensen-Shannon (JS) divergence:** Estimates the Jensen-Shannon (JS) divergence between the probability distributions of the facet and the outcome. Reference: https://ieeexplore.ieee.org/document/61115
- **Total Variation Distance (TVD):** Measures the distance between the probability distributions of the facet and the outcome. Reference: https://ecommons.cornell.edu/items/88a62f81-14bf-443a-9c35-9bf85b32bcab
- **Kolmogorov-Smirnov (KS) metric:** Assesses the statistical distance between the probability distributions of the facet and the outcome. Reference: https://www.tandfonline.com/doi/abs/10.1080/01621459.1951.10500769
- **Normalized Mutual Information (NMI):** Measures the information shared between two categorical variables, normalized to a range of [0, 1] where 1 indicates perfect correlation and 0 indicates no correlation. Reference: https://www.jmlr.org/papers/volume3/strehl02a/strehl02a.pdf
- **Normalized Conditional Mutual Information (NCMI):** Measures the mutual information between two categorical variables, conditioned on a third, normalized over the possible outcomes of the conditioning variable. Reference: https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf
- **Binary Ratio (BR):** Computes the ratio of positive outcomes between two binary groups. Reference: https://www.science.org/doi/10.1126/science.187.4175.398
- **Binary Difference (BD):** Calculates the difference in proportions of positive outcomes between two binary groups to detect disparities. Reference: https://ieeexplore.ieee.org/document/4909197
- **Conditional Binary Difference (CBD):** Computes the binary difference, conditioned on another categorical feature, to analyze disparities within subgroups. Reference: https://arxiv.org/abs/1610.02413
- **Pearson Correlation (CORR):** Determines the linear correlation between two ordinal features, with values ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation). Reference: https://link.springer.com/chapter/10.1007/978-3-642-00296-0_5
- **Logistic Regression (LR):** Fits a logistic regression model to predict a multi-labeled outcome from a binary protected feature to assess the influence of the feature on the outcome. Reference: https://onlinelibrary.wiley.com/doi/book/10.1002/9781118548387

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
Enter the path to your dataset (CSV or JSON file): test.csv
Enter 1 to apply subgroup analysis, 0 to skip: 1
Enter the column name for the facet (e.g., Gender): Gender
Enter the column name for the outcome (e.g., Lymphoma): Lymphoma
Enter the column name for subgroup categorization (optional, press Enter to skip):
Enter the label value or threshold for positive outcomes (e.g., 1): 1
```
- **Output:** Outputs bias metrics directly to the console for further analysis or integration into reports. Warnings are also provided in the case of threshold violations in any of the above metrics.
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

## Cluster analysis
- The tool supports an option for the application of cluster analysis.
- It applies the MiniSOM clustering algorithm to identify clusters with similar profiles.
- The optimal number of clsuters is estimated by calculating the Davies Bouldin (DB) scores across different number of pre-defined clusters (maximum 30 clusters) and extracting the one with the lowest BD score.
- Then it estimates the above measures for each cluster to identify biases that might not be detected in the overall data.
- Some clusters might include homogeneous facets or targets. These clusters are ignored.

```bash
Enter the path to your dataset (CSV or JSON file): test.csv
Enter 1 to apply subgroup analysis, 0 to skip: 1
Enter the column name for the facet (e.g., Gender): Gender
Enter the column name for the outcome (e.g., Lymphoma): Lymphoma
Enter the column name for subgroup categorization (optional, press Enter to skip): 
Enter the label value or threshold for positive outcomes (e.g., 1): 1

DB Score for 2 clusters: 0.8366707826706445
DB Score for 3 clusters: 0.8992724031690805
DB Score for 4 clusters: 0.9481305663368798
DB Score for 5 clusters: 0.9104716780225016
DB Score for 6 clusters: 0.9167542479340357
DB Score for 7 clusters: 1.04261707581957
DB Score for 8 clusters: 0.9826945968326137
DB Score for 9 clusters: 0.95853804899992
DB Score for 10 clusters: 0.9375362591430827
DB Score for 11 clusters: 0.9878578993873958
DB Score for 12 clusters: 0.98957960334394
DB Score for 13 clusters: 1.0664137728508025
DB Score for 14 clusters: 1.0256051464009408
DB Score for 15 clusters: 1.0321304020928317
DB Score for 16 clusters: 1.0100857834920873
DB Score for 17 clusters: 1.0899092989989714
DB Score for 18 clusters: 1.0570953512110735
DB Score for 19 clusters: 1.101991476823526
DB Score for 20 clusters: 1.0271731771154786
DB Score for 21 clusters: 1.1323183373925332
DB Score for 22 clusters: 1.0600443170312999
DB Score for 23 clusters: 1.1752693164274097
DB Score for 24 clusters: 1.1583490978951878
DB Score for 25 clusters: 1.1775202186437215
DB Score for 26 clusters: 1.2405592244428243
DB Score for 27 clusters: 1.1613712803648617
DB Score for 28 clusters: 1.1785194201812899
DB Score for 29 clusters: 1.1672976670962065
DB Score for 30 clusters: 1.1835632671263867
The optimal number of clusters is 2

Starting analysis for cluster 1 of 2 with 525 samples.
Calculating pre-training data bias metrics...
- CI for Gender is 0.9047619047619047
>> Warning: Significant bias detected based on CI metric!
- DPL for Gender given the outcome Lymphoma = 1 is 0.0040000000000000036
- Average DD for Gender given the outcome Lymphoma is: 6.938893903907228e-18
- Average CDD: Subgroup was not provided.
- Jensen-Shannon Divergence between Gender and Lymphoma is 2.6571313666603948e-05
- L2 norm between Gender and Lymphoma is 0.005656854249492385
- TVD for Gender given Lymphoma is 0.0040000000000000036
- KS metric between Gender and Lymphoma is 0.0040000000000000036
- NMI between Gender and Lymphoma is 1.9975183644356853e-05
- NCMI: Subgroup was not provided.
- BR for Gender and Lymphoma is 1.05
- BD for Gender and Lymphoma is 0.0040000000000000036
- CBD: Missing conditions for binary conditional difference.
- CORR between Gender and Lymphoma is 0.0030740867668317016
- LR coefficients for Gender predicting Lymphoma are [[0.03741296]]
  Intercept is [-2.42780331]

Starting analysis for cluster 2 of 2 with 650 samples.
Calculating pre-training data bias metrics...
- CI for Gender is 0.8923076923076922
- DPL for Gender given the outcome Lymphoma = 1 is 0.006968641114982577
- Average DD for Gender given the outcome Lymphoma is: 0.0
- Average CDD: Subgroup was not provided.
- Jensen-Shannon Divergence between Gender and Lymphoma is 7.473519897611557e-05
- L2 norm between Gender and Lymphoma is 0.009855146776119088
- TVD for Gender given Lymphoma is 0.006968641114982549
- KS metric between Gender and Lymphoma is 0.006968641114982577
- NMI between Gender and Lymphoma is 5.825007056533386e-05
- NCMI: Subgroup was not provided.
- BR for Gender and Lymphoma is 1.08130081300813
- BD for Gender and Lymphoma is 0.006968641114982577
- CBD: Missing conditions for binary conditional difference.
- CORR between Gender and Lymphoma is 0.005433980154336431
- LR coefficients for Gender predicting Lymphoma are [[0.06472355]]
  Intercept is [-2.34733534]
```

## Contribution
- Contributions are welcome.
- Please fork the repository and submit pull requests with your enhancements.
- Ensure that new features are accompanied by appropriate tests and documentation.

## License
This project is licensed under the MIT License.<br />

## Additional notes
The script is designed to be modular and easy to extend for additional types of analyses.<br />
Future enhancements could include visualizations of the distributions and biases.<br />
