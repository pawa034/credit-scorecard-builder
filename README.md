# Credit Scorecard Builder

A comprehensive Python package for building credit risk scorecards. Developed by Pawan Mishra.

## Overview

This package provides an end-to-end solution for credit risk scorecard development, including:

- Data quality analysis and preprocessing
- Feature selection using Information Value, correlation, and other techniques
- Optimal binning of variables using Weight of Evidence (WoE)
- Model development with multiple algorithms (machine learning and logistic regression)
- Model performance evaluation with KS statistics and other metrics
- Comprehensive documentation of the modeling process

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install credit-scorecard-builder
```

### Option 2: Install from source

```bash
# Clone the repository
git clone https://github.com/pawanmishra/credit-scorecard-builder.git
cd credit-scorecard-builder

# Install the package
pip install -e .
```

## Quick Start

### Python API

```python
from scorecard import ScoreCardRisk
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize the ScoreCardRisk class
sc = ScoreCardRisk()

# You can set parameters programmatically (optional)
sc.Target = "default_flag"  # Target variable name
sc.Id = "customer_id"       # ID column name
sc.iv_cutoff = 0.01         # Information Value cutoff
sc.corr_cutoff = 0.65       # Correlation cutoff

# Build the scorecard model
result, path, cols_to_use, model_vars, data = sc.ScoreCardBuilder(data)

# Use the model to make predictions on new data
# ...
```

### Command Line Interface

The package also provides a command-line interface:

```bash
# Basic usage
scorecard-builder your_data.csv

# Specifying parameters
scorecard-builder your_data.csv --target default_flag --id customer_id --iv-cutoff 0.02 --corr-cutoff 0.7
```

## Documentation

For detailed documentation, please refer to:

- [User Guide](USAGE_GUIDE.md): Comprehensive guide on how to use the package
- [Technical Documentation](DOCUMENTATION.md): Detailed explanation of the package architecture
- [Results Interpretation](RESULTS_GUIDE.md): How to interpret and use the scorecard results
- [Examples](examples/): Sample scripts demonstrating various use cases

## Features

- **Data Quality Analysis**: Identify and handle missing values, outliers, and other data quality issues
- **Feature Selection**: Select the most predictive variables using Information Value and other techniques
- **Optimal Binning**: Automatically bin continuous and categorical variables using Weight of Evidence
- **Model Development**: Train and evaluate multiple model types
- **Performance Metrics**: Calculate KS statistics, ROC curves, and other performance metrics
- **Comprehensive Documentation**: Generate detailed reports of the modeling process

## Requirements

- Python 3.6+
- pandas >= 1.2.0
- numpy >= 1.19.0
- scikit-learn >= 0.24.0
- statsmodels >= 0.12.0
- optbinning >= 0.8.0
- lightgbm >= 3.2.0
- And other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License with an additional restriction - see the [LICENSE](LICENSE) file for details.

**IMPORTANT:** Any commercial use of this software must be informed to and approved by Pawan Mishra (pawanbit034@gmail.com) prior to such use.

## Author

Developed by Pawan Mishra (pawanbit034@gmail.com).

## Acknowledgments

Thanks to everyone who has contributed to the development and testing of this package. 