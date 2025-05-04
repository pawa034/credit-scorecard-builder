# Scorecard Builder Usage Guide

This guide provides instructions on how to use the Scorecard Builder package for credit risk modeling.

## Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from scorecard import ScoreCardRisk
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize ScoreCardRisk
sc = ScoreCardRisk()

# Build scorecard
result, path, cols_to_use, model_vars, data = sc.ScoreCardBuilder(data)
```

## Detailed Usage

### Data Requirements

Your dataset should include:

1. A binary target variable (0/1) indicating credit risk
2. A unique identifier column
3. Predictor variables (both numerical and categorical)

Example columns in a typical credit risk dataset:
- Customer ID (unique identifier)
- Default flag (target variable)
- Age, income, loan amount (numerical predictors)
- Employment status, loan purpose (categorical predictors)

### Running the ScoreCardBuilder

When you run `sc.ScoreCardBuilder(data)`, you'll be prompted to provide:

1. **Project Name**: Name for your project (used for output directory)
2. **Volume Name**: Storage location identifier
3. **Response Variable**: Name of your target column (e.g., "default_flag")
4. **Unique ID**: Name of your ID column (e.g., "customer_id")
5. **IV Cutoff**: Information Value threshold (default: 0.01)
6. **Correlation Cutoff**: Threshold for correlation filtering (default: 0.65)

Example:

```python
# Run with specific columns to remove
result, path, cols_to_use, model_vars, data = sc.ScoreCardBuilder(
    data, 
    columns_to_remove=["redundant_col1", "redundant_col2"]
)
```

### Customizing Parameters

You can customize the ScoreCardRisk parameters before calling ScoreCardBuilder:

```python
from scorecard import ScoreCardRisk
import pandas as pd

# Load data
data = pd.read_csv("your_data.csv")

# Initialize with custom parameters
sc = ScoreCardRisk()
sc.iv_cutoff = 0.02  # Set higher IV threshold
sc.corr_cutoff = 0.7  # Set higher correlation threshold
sc.split = 0.25  # Set test split ratio (default: 0.3)

# Run scorecard builder
result, path, cols_to_use, model_vars, data = sc.ScoreCardBuilder(data)
```

### Understanding Outputs

The ScoreCardBuilder returns:

1. **result**: The best selected model
2. **path**: Path to the output directory with saved results
3. **cols_to_use**: List of columns used in the final model
4. **model_vars**: List of variables in the original data used
5. **data**: Transformed data with predictions

All model files, charts, and reports are saved to the output directory.

### Example with Demo Data

```python
import pandas as pd
from scorecard import ScoreCardRisk

# Load demo data
data = pd.read_csv("demo.csv")

# Initialize ScoreCardRisk
sc = ScoreCardRisk()

# Run without UI interaction (programmatically set parameters)
sc.Target = "ever_01p_06m"  # Set target column
sc.Id = "customer_id"      # Set ID column
sc.iv_cutoff = 0.01        # Set IV threshold
sc.corr_cutoff = 0.65      # Set correlation threshold

# Build scorecard
result, path, cols_to_use, model_vars, data = sc.ScoreCardBuilder(data)

# Print results
print(f"Model saved to: {path}")
print(f"Model variables: {cols_to_use}")
print(f"Model summary:")
print(result.summary())
```

## Output Files

The ScoreCardBuilder creates the following outputs in the results directory:

- **DEVKSML.csv**: KS table for ML model on development data
- **OOTKSML.csv**: KS table for ML model on out-of-time data
- **FeatureImportance.csv**: Feature importance measures
- **output.csv**: Summary of all model iterations
- **VarReport.csv**: Variable selection report
- **IVALL.csv**: Information Value for all variables
- **\*.sav**: Saved model files

## Advanced Usage

For advanced usage and more customization options, refer to the full [Documentation](DOCUMENTATION.md). 