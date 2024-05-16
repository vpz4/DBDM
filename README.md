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

Insufficient clusters (1) for DB score calculation at 2 clusters.
Insufficient clusters (1) for DB score calculation at 3 clusters.
DB score for 4 clusters: 0.947099647714959
DB score for 5 clusters: 0.9854707833517176
DB score for 6 clusters: 0.9586702107017073
DB score for 7 clusters: 1.0515492504881534
DB score for 8 clusters: 1.0451974470258398
DB score for 9 clusters: 0.9901106832225159
DB score for 10 clusters: 1.0460494036553947
DB score for 11 clusters: 1.0555180431428168
DB score for 12 clusters: 1.060176025851847
DB score for 13 clusters: 0.9817792741187839
DB score for 14 clusters: 1.0579531824017483
DB score for 15 clusters: 0.9367636313432004
DB score for 16 clusters: 0.9920200141083888
DB score for 17 clusters: 1.008404782131915
DB score for 18 clusters: 0.9877050559780625
DB score for 19 clusters: 1.0131617428890853
DB score for 20 clusters: 0.9908491442005372
DB score for 21 clusters: 1.0817318254422816
DB score for 22 clusters: 0.9889997087817398
DB score for 23 clusters: 0.9985742412151416
DB score for 24 clusters: 1.0190581536740844
DB score for 25 clusters: 1.0562099787500703
DB score for 26 clusters: 1.1868725106727038
DB score for 27 clusters: 1.0882489726377005
DB score for 28 clusters: 1.125281977834633
DB score for 29 clusters: 1.0959063397904472
DB score for 30 clusters: 1.1513001276359944
The optimal number of clusters is 15
Applying the Minisom

Starting analysis for cluster 2 of 15 with 132 samples.

Analyzing cluster 2 / 15
Unique outcomes 2
Unique facets 2
Calculating pre-training data bias metrics...
- CI for Gender is 0.8787878787878789
- DPL for Gender given the outcome Lymphoma = 1 is -0.11290322580645162
>> Warning: Significant bias detected based on DPL metric!
- Average DD for Gender given the outcome Lymphoma is: -1.3877787807814457e-17
- Average CDD: Subgroup was not provided.
- Jensen-Shannon Divergence between Gender and Lymphoma is 0.010330835824001284
- L2 norm between Gender and Lymphoma is 0.1596692731711559
>> Warning: Significant bias detected based on L2 norm metric!
- TVD for Gender given Lymphoma is 0.11290322580645162
>> Warning: Significant bias detected based on TVD metric!
- KS metric between Gender and Lymphoma is 0.11290322580645162
>> Warning: Significant bias detected based on KS metric!
- NMI between Gender and Lymphoma is 0.007908386512665984
- NCMI: Subgroup was not provided.
- BR for Gender and Lymphoma is 0.5483870967741935
- BD for Gender and Lymphoma is -0.11290322580645162
- CBD: Missing conditions for binary conditional difference.
- CORR between Gender and Lymphoma is -0.07674430622296582
- LR coefficients for Gender predicting Lymphoma are [[-0.41205176]]
  Intercept is [-1.40006934]
Starting analysis for cluster 3 of 15 with 179 samples.

Analyzing cluster 3 / 15
Unique outcomes 2
Unique facets 2
Calculating pre-training data bias metrics...
- CI for Gender is 0.8659217877094971
- DPL for Gender given the outcome Lymphoma = 1 is -0.1307385229540918
>> Warning: Significant bias detected based on DPL metric!
- Average DD for Gender given the outcome Lymphoma is: -6.938893903907228e-18
- Average CDD: Subgroup was not provided.
- Jensen-Shannon Divergence between Gender and Lymphoma is 0.025253602790681316
- L2 norm between Gender and Lymphoma is 0.18489219228630283
>> Warning: Significant bias detected based on L2 norm metric!
- TVD for Gender given Lymphoma is 0.13073852295409183
>> Warning: Significant bias detected based on TVD metric!
- KS metric between Gender and Lymphoma is 0.13073852295409183
>> Warning: Significant bias detected based on KS metric!
- NMI between Gender and Lymphoma is 0.03721022415202665
- NCMI: Subgroup was not provided.
- BR for Gender and Lymphoma is 0.2155688622754491
- BD for Gender and Lymphoma is -0.1307385229540918
- CBD: Missing conditions for binary conditional difference.
- CORR between Gender and Lymphoma is -0.1582374894909201
- LR coefficients for Gender predicting Lymphoma are [[-0.87544112]]
  Intercept is [-2.27293816]
Starting analysis for cluster 4 of 15 with 163 samples.

Analyzing cluster 4 / 15
Unique outcomes 2
Unique facets 2
Calculating pre-training data bias metrics...
- CI for Gender is 0.852760736196319
- DPL for Gender given the outcome Lymphoma = 1 is 0.016004415011037526
- Average DD for Gender given the outcome Lymphoma is: -6.938893903907228e-18
- Average CDD: Subgroup was not provided.
- Jensen-Shannon Divergence between Gender and Lymphoma is 0.00038623614688024395
- L2 norm between Gender and Lymphoma is 0.02263366076645676
- TVD for Gender given Lymphoma is 0.016004415011037484
- KS metric between Gender and Lymphoma is 0.016004415011037526
- NMI between Gender and Lymphoma is 0.0003530855727628696
- NCMI: Subgroup was not provided.
- BR for Gender and Lymphoma is 1.1920529801324504
- BD for Gender and Lymphoma is 0.016004415011037526
- CBD: Missing conditions for binary conditional difference.
- CORR between Gender and Lymphoma is 0.0140475388716452
- LR coefficients for Gender predicting Lymphoma are [[0.09287414]]
  Intercept is [-2.30435945]
Starting analysis for cluster 5 of 15 with 147 samples.

Analyzing cluster 5 / 15
Unique outcomes 2
Unique facets 2
Calculating pre-training data bias metrics...
- CI for Gender is 0.9183673469387754
>> Warning: Significant bias detected based on CI metric!
Failed to calculate metrics: 0
Starting analysis for cluster 6 of 15 with 114 samples.

Analyzing cluster 6 / 15
Unique outcomes 2
Unique facets 2
Calculating pre-training data bias metrics...
- CI for Gender is 0.9473684210526316
>> Warning: Significant bias detected based on CI metric!
Failed to calculate metrics: 0
Starting analysis for cluster 7 of 15 with 184 samples.

Analyzing cluster 7 / 15
Unique outcomes 2
Unique facets 2
Calculating pre-training data bias metrics...
- CI for Gender is 0.8913043478260869
Failed to calculate metrics: 0
Starting analysis for cluster 8 of 15 with 100 samples.

Analyzing cluster 8 / 15
Unique outcomes 2
Unique facets 2
Calculating pre-training data bias metrics...
- CI for Gender is 0.94
>> Warning: Significant bias detected based on CI metric!
Failed to calculate metrics: 0
Starting analysis for cluster 9 of 15 with 79 samples.

Analyzing cluster 9 / 15
Unique outcomes 2
Unique facets 2
Calculating pre-training data bias metrics...
- CI for Gender is 0.9746835443037976
>> Warning: Significant bias detected based on CI metric!
Failed to calculate metrics: 0
Starting analysis for cluster 10 of 15 with 77 samples.

Analyzing cluster 10 / 15
Unique outcomes 2
Unique facets 2
Calculating pre-training data bias metrics...
- CI for Gender is 0.8701298701298701
Failed to calculate metrics: 0
Done
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
