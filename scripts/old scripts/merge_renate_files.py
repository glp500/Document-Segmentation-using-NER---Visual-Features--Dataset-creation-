import os
import pandas as pd

def merge_csvs_with_blank(
    input_dir: str, 
    blank_csv_path: str, 
    output_file: str
) -> None:
    """
    Merges all CSV files from 'input_dir' into one CSV file, 
    inserting the rows from 'blank_csv_path' in between each file.
    
    Parameters:
    -----------
    input_dir : str
        Path to directory containing multiple CSV files to be merged.
    blank_csv_path : str
        Path to the small blank CSV file to be inserted between each CSV from the input directory.
    output_file : str
        Path (including filename) for the resulting merged CSV file.
    """
    # Read the small "blank" CSV once
    blank_df = pd.read_csv(blank_csv_path)
    
    # Collect all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    # Sort them so they’re processed in a predictable order (alphabetical by default)
    csv_files.sort()
    
    # Accumulate all DataFrames in this list
    dataframes = []
    
    for file_name in csv_files:
        full_path = os.path.join(input_dir, file_name)
        
        # Read each CSV file
        df = pd.read_csv(full_path)
        
        # Append the main CSV file DataFrame
        dataframes.append(df)
        
        # Append the blank CSV DataFrame
        dataframes.append(blank_df)
    
    # Concatenate all the data into one DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Export to the specified output CSV (single header row, no index column)
    merged_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Example usage:
    input_directory = r"/Users/gavinl/Desktop/TANAP Segmentation/Data/(3) Merged with Enhanced Annotations"
    blank_csv = r"/Users/gavinl/Desktop/TANAP Segmentation/Data/blank_page.csv"
    output_csv = r"/Users/gavinl/Desktop/TANAP Segmentation/Data/Testing/temp/merged_renate_xml.csv"
    
    merge_csvs_with_blank(input_directory, blank_csv, output_csv)
    print(f"Merged CSV created at: {output_csv}")