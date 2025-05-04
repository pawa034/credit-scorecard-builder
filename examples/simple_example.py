#!/usr/bin/env python
"""
Simple example of using the credit-scorecard-builder package
"""
import pandas as pd
from scorecard import ScoreCardRisk

def main():
    # Load your data
    print("Loading data...")
    try:
        data = pd.read_csv("demo.csv")
        print(f"Data loaded successfully! Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Please ensure you have demo.csv in the current directory")
        return

    # Initialize the ScoreCardRisk class
    sc = ScoreCardRisk()
    
    # You can set parameters programmatically instead of via prompts
    sc.Target = "ever_01p_06m"  # Target variable name
    sc.Id = "customer_id"       # ID column name
    sc.iv_cutoff = 0.01         # Information Value cutoff
    sc.corr_cutoff = 0.65       # Correlation cutoff
    
    # Define columns to remove (if any)
    columns_to_remove = []
    
    # Build the scorecard model
    print("\nBuilding scorecard model...")
    result, path, cols_to_use, model_vars, data = sc.ScoreCardBuilder(data, columns_to_remove)
    
    # Display results
    print("\nScorecard model built successfully!")
    print(f"Results saved to: {path}")
    print(f"Number of features used in the model: {len(cols_to_use)}")
    print(f"Features: {cols_to_use}")
    
    # Display model summary if available
    if result is not None:
        try:
            print("\nModel Summary:")
            print(result.summary())
        except:
            print("\nDetailed model summary not available")

if __name__ == "__main__":
    main() 