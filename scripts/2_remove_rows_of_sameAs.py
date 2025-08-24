#!/usr/bin/env python3
"""
remove_rows_of_sameAs.py

Filter CSV file by removing rows where 'TANAP Boundaries' column 
starts with 'SAME AS'.

Usage:
    python remove_rows_of_sameAs.py input.csv output.csv
    python remove_rows_of_sameAs.py input.csv output.csv --column "Custom Column Name"
"""

import argparse
import sys
import pandas as pd


def remove_same_as_rows(input_csv, output_csv, column_name="TANAP Boundaries"):
    """
    Removes rows from CSV where the specified column starts with 'SAME AS'.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file  
        column_name: Name of column to check (default: "TANAP Boundaries")
    """
    print(f"Processing file: {input_csv}")
    
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[ERROR] Could not read {input_csv}: {e}")
        sys.exit(1)
    
    # Check if the specified column exists
    if column_name not in df.columns:
        print(f"[ERROR] Column '{column_name}' not found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Report original row count
    original_count = len(df)
    print(f"Original row count: {original_count}")
    
    # Convert to string (in case there are non-string values)
    # and filter out rows where the column starts with "SAME AS"
    df[column_name] = df[column_name].astype(str)
    
    # Keep rows where column does NOT start with 'SAME AS'
    df_filtered = df[~df[column_name].str.startswith("SAME AS")]
    
    removed_count = original_count - len(df_filtered)
    if removed_count > 0:
        print(f"Removed {removed_count} rows that started with 'SAME AS'.")
    else:
        print("No rows started with 'SAME AS'.")
    
    # Save filtered CSV
    try:
        df_filtered.to_csv(output_csv, index=False)
        print(f"Filtered CSV saved to: {output_csv}")
        print(f"Final row count: {len(df_filtered)}")
    except Exception as e:
        print(f"[ERROR] Could not save to {output_csv}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Remove rows from CSV where specified column starts with 'SAME AS'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_rows_of_sameAs.py input.csv filtered_output.csv
  python remove_rows_of_sameAs.py input.csv output.csv --column "Custom Column"
        """
    )
    
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument(
        "--column", 
        default="TANAP Boundaries",
        help="Name of column to check for 'SAME AS' (default: 'TANAP Boundaries')"
    )
    
    args = parser.parse_args()
    
    # Process the CSV
    remove_same_as_rows(args.input_csv, args.output_csv, args.column)


if __name__ == "__main__":
    main()