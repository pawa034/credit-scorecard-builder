# Scorecard Builder Documentation

## Overview

The Scorecard Builder is a comprehensive Python package for building credit risk scorecards. It provides an end-to-end solution for data analysis, feature selection, model development, and scorecard creation.

## Table of Contents

1. [Installation](#installation)
2. [Package Structure](#package-structure)
3. [Module Details](#module-details)
4. [Usage](#usage)
5. [Sample Output](#sample-output)
6. [Interpretation](#interpretation)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd scorecard-builder

# Install required packages
pip install -r requirements.txt
```

Required dependencies:
- pandas >= 1.2.0
- numpy >= 1.19.0
- scikit-learn >= 0.24.0
- statsmodels >= 0.12.0
- optbinning >= 0.8.0
- lightgbm >= 3.2.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- tqdm >= 4.50.0
- probatus >= 1.4.0
- shap >= 0.39.0
- plotly >= 4.14.0

## Package Structure

The package has a modular structure with specialized modules for each step of the scorecard building process:

```
scorecard/
├── __init__.py           # Package initialization
├── scorecard_risk.py     # Main class definition
├── scorecard_builder.py  # Core builder implementation
├── data_analysis.py      # Data quality & analysis functions
├── feature_engineering.py # Feature transformation & selection
├── modeling.py           # Model training & evaluation
├── tuning.py             # Model tuning & optimization
└── utils.py              # Utility functions
```

## Module Details

### scorecard_risk.py

Contains the main `ScoreCardRisk` class, which serves as the entry point to the scorecard building process. This class integrates all other modules and provides a unified interface for the entire process.

**Key functions:**
- `ScoreCardBuilder`: Main function to build the scorecard
- `DataSelector`: Identifies and extracts numerical and categorical variables
- Various wrapper functions for accessing functionality from other modules

### scorecard_builder.py

Implements the end-to-end scorecard building process, orchestrating all steps from data preprocessing to final model selection.

**Key steps:**
1. Data preprocessing and initial filtering
2. Information Value (IV) calculation and filtering
3. Feature importance calculation
4. Correlation filtering
5. SHAP-based feature elimination
6. Model development (ML and Logistic)
7. Model tuning and selection
8. Performance evaluation

### data_analysis.py

Provides functions for data quality assessment and preliminary analysis.

**Key functions:**
- `data_report`: Generates statistics for numerical variables
- `report_builder`: Creates a report based on data quality checks
- `selector_sanity`: Performs variable selection based on quality checks
- `correlation_filter`: Filters variables based on correlation
- `chisq_feature_selection`: Selects categorical features using chi-square test

### feature_engineering.py

Contains functions for feature transformation and selection.

**Key functions:**
- `iv_maker`: Calculates Information Value for variables
- `fit_numeric`/`fit_categorical`: Fits numeric/categorical variables using OptimalBinning
- `transformer_numeric`/`transformer_categorical`: Transforms variables based on binning
- `rf_pimp_feature_importance`: Calculates feature importance using Random Forest
- `shap_rfecv`: Performs SHAP-based recursive feature elimination
- `ml_data_imputer`: Prepares data for machine learning

### modeling.py

Implements model training and evaluation functionality.

**Key functions:**
- `get_models`: Generates a library of base learners
- `train_predict`: Trains multiple classifiers and makes predictions
- `score_models`: Scores models based on predictions
- `ml_summary`: Generates ML model summary
- `gen_ks_calculator`: Generates KS table for scored population
- `modeling` / `modeling2`: Perform logistic regression

### tuning.py

Contains model tuning and optimization functions.

**Key functions:**
- `re_tune_models`: Performs iterative model tuning
- `best_model_selector`: Selects best model based on performance metrics
- `score_analyzer`: Analyzes score distribution across bins

### utils.py

Provides utility functions used across the package.

**Key functions:**
- `convert_decimal_to_double`: Converts decimal columns to double type
- `create_directory`: Creates a directory for storing results
- `split_numerical_categorical`: Splits data into numerical and categorical
- `filter_by_variance`: Filters out low-variance features
- `plot_roc_curve`: Plots ROC curves for models

## Usage

### Basic Usage

```python
from scorecard import ScoreCardRisk
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize the ScoreCardRisk class
sc = ScoreCardRisk()

# Build the scorecard
result, path, cols_to_use, model_vars, data = sc.ScoreCardBuilder(data)
```

### Example Usage with demo.csv

The package includes a demo script (`demo_scorecard.py`) that demonstrates how to use the package with the provided sample dataset:

```python
import pandas as pd
from scorecard import ScoreCardRisk

# Load the demo dataset
pdf = pd.read_csv("demo.csv")

# Create an instance of the ScoreCardRisk class
sc = ScoreCardRisk()

# Define columns to remove (if any)
columns_to_remove = []

# Build the scorecard model
result, path, cols_to_use, model_vars, data = sc.ScoreCardBuilder(pdf, columns_to_remove)
```

To run the demo:

```bash
python demo_scorecard.py
```

## Sample Output

When running the demo script with demo.csv, the following process and outputs were observed:

### Data Summary

```
Data Size  :  138204
Response Variable : EVER_01P_06M
Unique Id : CUSTOMER_ID
BadRate is: 6.4455%
```

### Feature Selection Process

1. **Initial Features**:
   ```
   Variables Included : 91
   ```

2. **After Data Quality Checks**:
   ```
   Variables Selected : 82
   ```

3. **After IV Threshold (1.0%)**:
   ```
   Variables Selected after IV cutoff : 45
   ```

4. **After Feature Importance**:
   ```
   Variables Selected after rf_pimp : 14
   ```

5. **After Correlation Analysis**:
   ```
   Variables Selected after Correlation : 13
   ```

6. **Final Model Features**:
   ```
   Variables Selected after SHAP : 13
   ```

### Machine Learning Model Results

```
----------------------------ROC SCORE Displayed Below for Each Model----------------------------
-------------------------------------------Fitting models-------------------------------------------
knn...
naive bayes...
mlp-nn...
random forest...
gbm...
logistic...

-------------------------------------------Scoring models-------------------------------------------
Model_Name                  logLoss                   
knn                       : 1.909
naive bayes               : 0.520
mlp-nn                    : 0.236
random forest             : 1.287
gbm                       : 0.232
logistic                  : 0.235
COMPLETED...

Best Model fitted on Raw Data is : gbm
```

### Feature Importance (GBM Model)

```
                                    importance
TTL_ENQ                               0.180935
BUR_NO_LOANS_CLOSED_SUC_WWF           0.139632
NT_BOUNCES_EVER_PB                    0.129765
ENQ_L12M                              0.115006
AGE                                   0.078689
SCORE_DROP_60                         0.075366
BUR_NO_UNSEC_LOANS_CLOSED_DELQ        0.063517
AVG_CC_UTILIZATION_BAND_NO_CC_card    0.059900
TOTAL_LIVE_LOANS                      0.036578
RATIO_BNC_3M_EVER                     0.035754
BUR_NO_CC_LOANS_CLOSED_SUC_WWF        0.035693
AVG_CC_UTILIZATION_BAND_Medium        0.020329
AVG_CC_UTILIZATION_BAND_Low           0.010491
EMPLOYMENT_TYPE_SEMP                  0.010028
```

### KS Tables (Machine Learning Model)

**Development KS**:
```
        min_prob  max_prob  events   count event_rate cum_eventrate    KS
Decile                                                                   
1       0.110300  0.637651  1657.0  9675.0     17.13%        26.91%  18.1
2       0.085644  0.110296   980.0  9674.0     10.13%        42.82%  24.4
3       0.072716  0.085643   859.0  9674.0      8.88%        56.77%  28.6
4       0.062403  0.072715   639.0  9674.0      6.61%        67.15%  29.0
5       0.054780  0.062402   527.0  9616.0      5.48%        75.71%  27.5
6       0.047302  0.054779   419.0  9732.0      4.31%        82.51%  24.0
7       0.041062  0.047299   374.0  9674.0      3.87%        88.58%  19.8
8       0.034692  0.041062   319.0  9673.0      3.30%        93.76%  14.7
9       0.027756  0.034692   232.0  9675.0      2.40%        97.53%   8.0
10      0.008951  0.027755   152.0  9675.0      1.57%       100.00%  -0.0
```

**Out-of-Time KS**:
```
        min_prob  max_prob  events   count event_rate cum_eventrate    KS
Decile                                                                   
1       0.110046  0.562917   669.0  4147.0     16.13%        24.33%  15.3
2       0.085422  0.110037   419.0  4146.0     10.11%        39.56%  21.0
3       0.072667  0.085421   347.0  4138.0      8.39%        52.18%  23.8
4       0.062307  0.072662   310.0  4154.0      7.46%        63.45%  25.1
5       0.054704  0.062306   259.0  4146.0      6.25%        72.87%  24.5
6       0.047464  0.054703   205.0  4146.0      4.94%        80.33%  21.8
7       0.041212  0.047463   180.0  4143.0      4.34%        86.87%  18.1
8       0.034708  0.041211   166.0  4149.0      4.00%        92.91%  13.8
9       0.027691  0.034706   121.0  4146.0      2.92%        97.31%   7.8
10      0.008650  0.027687    74.0  4147.0      1.78%       100.00%   0.0
```

### Logistic Regression Model Results

After 8 iterations of model tuning, the final logistic regression model included 7 significant variables:

```
                           Logit Regression Results                           
==============================================================================
Dep. Variable:           EVER_01P_06M   No. Observations:                96742
Model:                          Logit   Df Residuals:                    96735
Method:                           MLE   Df Model:                            6
Date:                Sun, 04 May 2025   Pseudo R-squ.:                 0.03207
Time:                        13:31:43   Log-Likelihood:                -22482.
converged:                       True   LL-Null:                       -23227.
Covariance Type:            nonrobust   LLR p-value:                     0.000
================================================================================
                                         coef    std err          z      P>|z|
    [0.025      0.975]
--------------------------------------------------------------------------------
const                                 -2.6676      0.013   -197.941      0.000
    -2.694      -2.641
woe_BUR_NO_LOANS_CLOSED_SUC_WWF       -1.3686      0.070    -19.436      0.000
    -1.507      -1.231
woe_NT_BOUNCES_EVER_PB                -0.8832      0.049    -17.977      0.000
    -0.979      -0.787
woe_BUR_NO_UNSEC_LOANS_CLOSED_DELQ    -1.5375      0.093    -16.451      0.000
    -1.721      -1.354
woe_AVG_CC_UTILIZATION_BAND           -0.7155      0.056    -12.867      0.000
    -0.825      -0.607
woe_SCORE_DROP                        -1.0338      0.093    -11.098      0.000
    -1.216      -0.851
woe_ENQ_L12M                          -0.8094      0.074    -10.869      0.000
    -0.955      -0.663
================================================================================
```

### KS Tables (Logistic Regression Model)

**Development KS**:
```
        min_prob  max_prob  events    count event_rate cum_eventrate    KS
Decile                                                                    
0       0.105661  0.272716  1299.0   9468.0     13.72%        20.71%  11.7
1       0.085411  0.105536   938.0   9864.0      9.51%        35.66%  16.8
2       0.075906  0.085371   772.0   9674.0      7.98%        47.97%  19.2
3       0.068279  0.075813   688.0   9682.0      7.11%        58.94%  20.3
4       0.059718  0.068264   606.0   9659.0      6.27%        68.60%  19.9
5       0.052123  0.059656   518.0   9586.0      5.40%        76.85%  18.1
6       0.046053  0.052032   468.0   9780.0      4.79%        84.31%  15.3
7       0.038666  0.046016   395.0   9523.0      4.15%        90.61%  11.5
8       0.030044  0.038642   319.0   9460.0      3.37%        95.70%   6.5
9       0.011697  0.030044   270.0  10046.0      2.69%       100.00%  -0.0
```

**Out-of-Time KS**:
```
        min_prob  max_prob  events   count event_rate cum_eventrate    KS
Decile                                                                   
0       0.105661  0.272716   586.0  4129.0     14.19%        22.24%  13.1
1       0.086502  0.105536   407.0  4164.0      9.77%        37.69%  18.9
2       0.076361  0.086469   279.0  3556.0      7.85%        48.27%  21.0
3       0.068786  0.076231   331.0  4703.0      7.04%        60.83%  22.3
4       0.059744  0.068640   228.0  4168.0      5.47%        69.49%  20.8
5       0.052123  0.059718   206.0  4152.0      4.96%        77.31%  18.5
6       0.045867  0.052032   192.0  4134.0      4.64%        84.59%  15.6
7       0.038595  0.045830   162.0  4158.0      3.90%        90.74%  11.5
8       0.029466  0.038472   136.0  4138.0      3.29%        95.90%   6.3
9       0.011697  0.029140   108.0  4160.0      2.60%       100.00%   0.0
```

## Interpretation

### Key Performance Metrics

1. **Information Value (IV)**:
   - IV < 0.02: Not predictive
   - 0.02 <= IV < 0.1: Weak predictor
   - 0.1 <= IV < 0.3: Medium predictor
   - 0.3 <= IV: Strong predictor

2. **KS Statistic**:
   - KS < 20: Poor discrimination
   - 20 <= KS < 30: Fair discrimination
   - 30 <= KS < 40: Good discrimination
   - 40 <= KS < 50: Very good discrimination
   - 50 <= KS: Excellent discrimination

3. **Log Loss**:
   - Lower values indicate better model performance
   - Used to compare different models

### Understanding Results

The scorecard building process results in both a Machine Learning model (GBM in this case) and a Logistic Regression model for interpretability.

1. **Feature Selection**: Started with 91 variables and ended with 13, removing redundant and non-predictive variables.

2. **Model Performance**: The GBM model achieved a KS of 29.0 on the development set and 25.1 on the test set, indicating fair to good discrimination.

3. **Logistic Regression**: The final model retained 7 significant variables with a KS of 20.3 on the development set and 22.3 on the test set.

4. **Important Variables**: Both models identified similar important variables, including:
   - BUR_NO_LOANS_CLOSED_SUC_WWF (loans closed successfully)
   - NT_BOUNCES_EVER_PB (number of bounces)
   - BUR_NO_UNSEC_LOANS_CLOSED_DELQ (unsecured loans closed delinquent)
   - ENQ_L12M (enquiries in last 12 months)
   - SCORE_DROP (credit score drop)
   - AVG_CC_UTILIZATION_BAND (credit card utilization)

5. **Output Files**: All model files, performance metrics, and variable reports are saved to the output directory for further analysis and implementation. 