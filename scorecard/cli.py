#!/usr/bin/env python
"""
Command-line interface for the credit scorecard builder.
"""
import argparse
import os
import pandas as pd
from scorecard import ScoreCardRisk


def main():
    """Main CLI entry point for the scorecard builder."""
    
    parser = argparse.ArgumentParser(
        description="Credit Scorecard Builder - A tool for building credit risk scorecards"
    )
    
    parser.add_argument(
        "input_file", 
        help="Path to the input CSV file containing the dataset"
    )
    
    parser.add_argument(
        "--target", 
        help="Name of the target/response variable column"
    )
    
    parser.add_argument(
        "--id", 
        help="Name of the unique identifier column"
    )
    
    parser.add_argument(
        "--iv-cutoff", 
        type=float, 
        default=0.01,
        help="Information Value cutoff threshold (default: 0.01)"
    )
    
    parser.add_argument(
        "--corr-cutoff", 
        type=float, 
        default=0.65,
        help="Correlation cutoff threshold (default: 0.65)"
    )
    
    parser.add_argument(
        "--split", 
        type=float, 
        default=0.3,
        help="Test/train split ratio (default: 0.3)"
    )
    
    parser.add_argument(
        "--remove-cols", 
        nargs="+",
        help="Space-separated list of columns to remove from analysis"
    )
    
    parser.add_argument(
        "--output-dir", 
        help="Output directory (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    try:
        data = pd.read_csv(args.input_file)
        print(f"Data loaded successfully. Shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return 1
    
    # Initialize ScoreCardRisk
    sc = ScoreCardRisk()
    
    # Set parameters if provided via command line
    if args.target:
        sc.Target = args.target
    if args.id:
        sc.Id = args.id
    
    sc.iv_cutoff = args.iv_cutoff
    sc.corr_cutoff = args.corr_cutoff
    sc.split = args.split
    
    # Run the scorecard builder
    print("Building scorecard model...")
    result, path, cols_to_use, model_vars, scored_data = sc.ScoreCardBuilder(
        data, 
        columns_to_remove=args.remove_cols
    )
    
    print("\nScorecard model built successfully!")
    print(f"Results saved to: {path}")
    print(f"Number of features used in the model: {len(cols_to_use)}")
    print(f"Total variables selected: {len(model_vars)}")
    
    # Display model summary if available
    if result is not None:
        try:
            print("\nModel Summary:")
            print(result.summary())
        except:
            print("\nModel summary not available")
    
    return 0


if __name__ == "__main__":
    exit(main()) 