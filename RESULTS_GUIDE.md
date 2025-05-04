# Interpreting Scorecard Results

This guide explains how to interpret and use the results from the Scorecard Builder package.

## Understanding Output Files

The Scorecard Builder generates several output files in the results directory:

| File | Description |
|------|-------------|
| `IVALL.csv` | Information Value for all variables |
| `FeatureImportance.csv` | Feature importance measures |
| `DEVKSML.csv` | KS table for ML model on development data |
| `OOTKSML.csv` | KS table for ML model on out-of-time data |
| `VarReport.csv` | Variable selection report |
| `output.csv` | Summary of all model iterations |
| `result_summary_all_iterations.txt` | Detailed model summaries |
| `*.sav` | Saved model files |

## Key Performance Metrics

### Information Value (IV)

Information Value measures the predictive power of a variable:

| IV Range | Interpretation |
|----------|---------------|
| < 0.02 | Not predictive |
| 0.02 - 0.1 | Weak predictor |
| 0.1 - 0.3 | Medium predictor |
| > 0.3 | Strong predictor |

Example from output:
```
Variable: TOTAL_LIVE_LOANS, IV: 0.157 (Medium predictor)
```

### Kolmogorov-Smirnov (KS) Statistic

KS measures the maximum separation between cumulative distributions of good and bad customers:

| KS Range | Discrimination Power |
|----------|---------------------|
| < 20 | Poor |
| 20 - 30 | Fair |
| 30 - 40 | Good |
| 40 - 50 | Very good |
| > 50 | Excellent |

From our example output:
```
Max KS for ML model: 29.0 (Fair/Good discrimination)
Max KS for Logistic model: 22.3 (Fair discrimination)
```

### Log Loss

Log Loss measures prediction accuracy - lower values indicate better performance:

```
gbm model log loss: 0.232 (Best performer)
logistic model log loss: 0.235
```

## Reading KS Tables

The KS table shows the discriminatory power at different score ranges:

```
        min_prob  max_prob  events   count event_rate cum_eventrate    KS
Decile                                                                   
1       0.110300  0.637651  1657.0  9675.0     17.13%        26.91%  18.1
2       0.085644  0.110296   980.0  9674.0     10.13%        42.82%  24.4
3       0.072716  0.085643   859.0  9674.0      8.88%        56.77%  28.6
4       0.062403  0.072715   639.0  9674.0      6.61%        67.15%  29.0  <- Max KS
...
```

Key columns:
- **min_prob/max_prob**: Probability range for this decile
- **events**: Number of bad customers in this decile
- **count**: Total customers in this decile
- **event_rate**: Percentage of bad customers in this decile
- **cum_eventrate**: Cumulative percentage of bad customers up to this decile
- **KS**: KS statistic at this decile

The highest KS value indicates the score cutoff with maximum discrimination power.

## Interpreting Logistic Regression Results

The logistic regression summary provides coefficient estimates and significance:

```
                           Logit Regression Results                           
==============================================================================
Dep. Variable:           EVER_01P_06M   No. Observations:                96742
Model:                          Logit   Df Residuals:                    96735
Method:                           MLE   Df Model:                            6
...
                                         coef    std err          z      P>|z|
--------------------------------------------------------------------------------
const                                 -2.6676      0.013   -197.941      0.000
woe_BUR_NO_LOANS_CLOSED_SUC_WWF       -1.3686      0.070    -19.436      0.000
woe_NT_BOUNCES_EVER_PB                -0.8832      0.049    -17.977      0.000
...
```

Key elements:
- **coef**: The coefficient value (direction and magnitude of impact)
- **std err**: Standard error of the coefficient estimate
- **z**: Test statistic (larger absolute value = more significant)
- **P>|z|**: P-value (smaller values indicate higher significance)
- **[0.025 0.975]**: 95% confidence interval for the coefficient

Variables with P-values < 0.05 are statistically significant predictors.

## Feature Importance

Feature importance shows the relative contribution of each feature to the model:

```
                                    importance
TTL_ENQ                               0.180935
BUR_NO_LOANS_CLOSED_SUC_WWF           0.139632
NT_BOUNCES_EVER_PB                    0.129765
ENQ_L12M                              0.115006
...
```

Higher values indicate more important variables in the model.

## Using the Model for Prediction

To use the saved model for scoring new customers:

```python
import pickle
import pandas as pd

# Load the saved model
with open('path/to/model.sav', 'rb') as file:
    model = pickle.load(file)

# Load new data for scoring
new_data = pd.read_csv('new_customers.csv')

# Prepare variables - ensure they match the training data format
# For logistic regression model:
X_new = new_data[model_variables]  # Use the cols_to_use from ScoreCardBuilder output

# For machine learning model:
# You may need to preprocess features the same way as during training

# Generate predictions
predictions = model.predict_proba(X_new)[:, 1]  # Probability of default

# Optionally convert to scores
scores = 600 - (50 * np.log(predictions / (1 - predictions)))  # Example transformation

# Add predictions to the data
new_data['default_probability'] = predictions
new_data['score'] = scores

# Save results
new_data.to_csv('scored_customers.csv', index=False)
```

## Setting Score Cutoffs

Use the KS table to identify appropriate cutoff points:

1. **Risk-based cutoffs**: Use the probability values at maximum KS
   - From our example: 0.062403 - 0.072715 (highest KS decile)

2. **Population-based cutoffs**: Set cutoffs to achieve target approval rates
   - Example: To approve 70% of applicants, set cutoff at the 7th decile boundary

3. **Expected loss-based cutoffs**: Balance approval rates with expected losses
   - Calculate: Expected Loss = Probability of Default × Exposure × Loss Given Default

## Monitoring Model Performance

After implementing your model, monitor these key metrics:

1. **Population Stability Index (PSI)**: Measures how much the population has changed
   - PSI < 0.1: No significant change
   - 0.1 ≤ PSI < 0.2: Moderate change
   - PSI ≥ 0.2: Significant change

2. **Actual vs. Expected Bad Rate**: Compare actual default rates to predicted rates

3. **KS over time**: Track if the model's discriminatory power decreases

## Business Implementation

When implementing the scorecard in business processes:

1. **Translate WoE values to scores**: Create a scorecard with points for each variable
2. **Set decision policies**: Define approval, rejection, and review thresholds
3. **Document override guidelines**: Rules for manual review of borderline cases
4. **Create monitoring dashboards**: Track performance metrics over time
5. **Define model refresh criteria**: When to rebuild or recalibrate the model 