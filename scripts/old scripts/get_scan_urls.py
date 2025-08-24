import pandas as pd

# 1) Read your input file
df = pd.read_csv("/Volumes/Extreme SSD/Python_Projects/TANAP Segmentation (Random Forest)/Demo/2001_output.csv")  # replace with your actual file name/path

# 2) Drop the ".xml" suffix to get the base file name
df["base_name"] = df["xml_file_name"].str.replace(r"\.xml$", "", regex=True)

# 3) Extract the archive code (e.g. "1.04.02") and the inventory number (e.g. "2001")
parts = df["base_name"].str.split("_", expand=True)
df["archive_code"]   = parts[1]
df["inventory_number"] = parts[2]

# 4) Build the scan_url
df["scan_url"] = (
    "https://www.nationaalarchief.nl/onderzoeken/archief/"
    + df["archive_code"]
    + "/invnr/"
    + df["inventory_number"]
    + "/file/"
    + df["base_name"]
)

# 5) (Optional) drop the helper columns
df = df.drop(columns=["base_name", "archive_code", "inventory_number"])

# 6) Write out the new CSV
df.to_csv("output_with_scan_urls.csv", index=False)
print("Wrote", len(df), "rows to output_with_scan_urls.csv")