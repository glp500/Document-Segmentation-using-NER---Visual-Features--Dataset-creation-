import os
import re
import pandas as pd

def parse_inventory_number(csv_filename):
    """
    Extract the inventory identifier from the CSV filename.
    Assumes a pattern like: "Analysis Renate 123.csv"
    
    Example:
       "Analysis Renate 1120.csv" --> "1120"
    """
    base = csv_filename.replace(".csv", "")
    match = re.search(r"(\d+)$", base)
    if match:
        return match.group(1)
    else:
        return base

def load_xml_files_for_inventory(xml_parent_dir):
    """
    Load all XML files from 'page' subdirectory in xml_parent_dir.
    Return a dict: { exact_filename: file_contents }.
    """
    xml_dict = {}
    pages_dir = os.path.join(xml_parent_dir, "page")
    
    if not os.path.isdir(pages_dir):
        print(f"[WARNING] 'page' folder not found in: {xml_parent_dir}")
        return xml_dict
    
    for dirpath, _, filenames in os.walk(pages_dir):
        for filename in filenames:
            if filename.endswith(".xml"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        xml_content = f.read()
                    xml_filename = filename.strip()  # keep the exact filename
                    xml_dict[xml_filename] = xml_content
                except Exception as e:
                    print(f"[ERROR] Could not read {file_path}: {e}")
    return xml_dict

# ------------------ MAIN SCRIPT ------------------ #

csv_root_directory = "/Users/gavinl/Desktop/TANAP Segmentation/Data/Renate Annotations"
xml_root_directory = "/Users/gavinl/Desktop/TANAP Segmentation/Data/Inventories"

for file_name in os.listdir(csv_root_directory):
    if file_name.lower().startswith("analysis renate") and file_name.lower().endswith(".csv"):
        csv_path = os.path.join(csv_root_directory, file_name)

        # 1. Parse the inventory number
        inv_number = parse_inventory_number(file_name)
        print(f"\n=== Processing CSV '{file_name}' (inventory: {inv_number}) ===")

        # 2. Identify the XML directory
        possible_dir_name = f"inventory {inv_number}"
        xml_parent_dir = os.path.join(xml_root_directory, possible_dir_name)
        
        if not os.path.isdir(xml_parent_dir):
            xml_parent_dir = os.path.join(xml_root_directory, inv_number)
        if not os.path.isdir(xml_parent_dir):
            print(f"[ERROR] No matching XML folder for inventory '{inv_number}'. Skipping.")
            continue

        # 3. Load XML files
        xml_data_dict = load_xml_files_for_inventory(xml_parent_dir)
        print(f"   Found {len(xml_data_dict)} XML files in '{xml_parent_dir}/page'.")
        print(f"   Sample XML filenames: {list(xml_data_dict.keys())[:5]} ...")

        # 4. Read the CSV with sep=';' so it splits into correct columns
        try:
            df = pd.read_csv(csv_path, sep=';')
        except Exception as e:
            print(f"[ERROR] Could not read CSV at {csv_path}: {e}")
            continue
        
        print("   Columns in CSV:", df.columns.tolist())

        # If your CSV column is "Scan File_Name" (with a space), rename it
        if "Scan File_Name" in df.columns:
            df.rename(columns={"Scan File_Name": "Scan_File_Name"}, inplace=True)
        elif "Scan_File_Name" not in df.columns:
            # If it might appear differently, adjust accordingly
            print("[ERROR] Neither 'Scan File_Name' nor 'Scan_File_Name' found. Skipping this CSV.")
            continue

        # 5. Normalize file names to ensure .xml
        df["Scan_File_Name"] = df["Scan_File_Name"].astype(str).str.strip()
        df["Scan_File_Name"] = df["Scan_File_Name"].apply(
            lambda x: x if x.endswith(".xml") else f"{x}.xml"
        )

        # 6. Map to the XML content by exact filename
        df["xml_data"] = df["Scan_File_Name"].map(xml_data_dict)

        # 7. Fill in missing matches
        df["xml_data"] = df["xml_data"].fillna("XML file not found")

        unmatched = df[df["xml_data"] == "XML file not found"]
        if not unmatched.empty:
            print(f"   Unmatched rows: {len(unmatched)}")
            print("   Example unmatched filenames:", unmatched["Scan_File_Name"].unique()[:5])
        else:
            print("   All rows matched a corresponding XML file!")

        output_path = '/Users/gavinl/Desktop/TANAP Segmentation/Data/ (1) Merged Inventories & Annotations'
        
        # 8. Save to a new CSV
        output_csv_name = file_name.replace(".csv", "_with_xml.csv")
        output_csv_path = os.path.join(output_path, output_csv_name)
        df.to_csv(output_csv_path, index=False)
        print(f"   Saved updated CSV to: {output_csv_path}")