"""
TANAP Boundaries Enhancement Script

This script enhances TANAP boundary annotations in CSV files by filling in missing values
between START and END markers.

USAGE:
    python scripts/enhance_TANAP_bounds.py input.csv output.csv

DESCRIPTION:
    - Reads a CSV file with a "TANAP Boundaries" column
    - Fills NaN values with "MIDDLE" if between START/END markers
    - Fills NaN values with "NONE" if outside document boundaries
    - Outputs enhanced CSV file with complete boundary annotations

EXAMPLE:
    python scripts/enhance_TANAP_bounds.py data/input.csv data/enhanced_output.csv
"""

import argparse
import pandas as pd
import numpy as np

def enhance_tanap_bounds(input_file, output_file):
    """
    Enhance TANAP boundaries in a CSV file by filling in MIDDLE and NONE values.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    # Read the dataset
    df = pd.read_csv(input_file, sep=",")

    # Flag to track whether we are within a document
    in_document = False

    # Iterate through each row
    for i, row in df.iterrows():
        current_value = row["TANAP Boundaries"]
        if current_value == "START":
            in_document = True
        elif current_value == "END":
            in_document = False
        elif pd.isna(current_value):
            # Mark "MIDDLE" if within a document, otherwise "NONE"
            df.at[i, "TANAP Boundaries"] = "MIDDLE" if in_document else "NONE"

    # Save the modified dataset to the output file
    df.to_csv(output_file, index=False)
    print(f"Processing complete. Enhanced file saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Enhance TANAP boundaries in a CSV file")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output CSV file")
    
    args = parser.parse_args()
    
    enhance_tanap_bounds(args.input_file, args.output_file)

if __name__ == "__main__":
    main()