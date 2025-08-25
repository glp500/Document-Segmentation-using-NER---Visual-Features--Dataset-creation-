# Document Segmentation Dataset Generation

This project generates feature datasets for document boundary detection from historical Dutch archival documents using Named Entity Recognition (NER) and visual layout features. The pipeline processes XMI annotations and XML page data to create comprehensive feature vectors suitable for machine learning applications.

## 🎯 Project Purpose

The codebase processes historical Dutch documents from the TANAP (Towards a New Age of Partnership) archives to create labeled datasets for document segmentation research. It combines:

- **Named Entity Recognition (NER)** data from XMI files containing annotations for persons, locations, organizations, ships, and dates
- **Visual/Layout features** from PAGE XML files containing text regions, coordinates, and structural information
- **Document boundary labels** indicating START, END, MIDDLE, or NONE for each page

## 📁 Project Structure

```
├── CLAUDE.md                    # Project instructions and development commands
├── README.MD                    # This file - comprehensive usage guide
├── requirements.txt             # Python dependencies
├── data/                        # Generated datasets
│   ├── Training/               # Training datasets in various processing stages
│   └── Unseen/                # Unseen test data
├── notebooks/                  # Jupyter notebooks for data exploration
│   ├── data_viewer.ipynb      # Interactive data examination
│   └── data_editor.ipynb      # Manual data editing tools
├── scripts/                    # Core data processing pipeline
│   ├── 1_merge_all.py         # Step 1: Merge CSV, XML, and XMI data
│   ├── 2_remove_rows_of_sameAs.py   # Step 2: Clean duplicate labels
│   ├── 3_enhance_TANAP_bounds.py    # Step 3: Enhance boundary annotations
│   ├── process_xmi_xml.py     # Extract entities and regions from XMI/XML
│   ├── extract_document_features.py  # Generate comprehensive feature vectors
│   └── old scripts/           # Legacy processing scripts
└── Renate Inventories/        # Source historical data
    ├── Renate Annotations/    # Manual annotations and analysis
    ├── json/                  # Processed JSON data by inventory
    ├── xmi/                   # XMI files with NER annotations
    └── xml/                   # PAGE XML files with layout information
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the training set
```bash
# Step 1: Merge all data sources
python scripts/1_merge_all.py

# Step 2: Remove duplicate/ambiguous labels (sameAs rows)
python scripts/2_remove_rows_of_sameAs.py

# Step 3: Enhance TANAP boundary annotations
python scripts/3_enhance_TANAP_bounds.py
```

### 3. Explore the Data
```bash
# Launch Jupyter notebooks for interactive exploration
jupyter notebook notebooks/data_viewer.ipynb
jupyter notebook notebooks/data_editor.ipynb
```


### Path Configuration
Update hard-coded paths in `scripts/1_merge_all.py`:
```python
CSV_DIR = "path/to/renate/csvs"      # Renate inventory CSV files
XML_DIR = "path/to/xml/files"        # PAGE XML files  
XMI_DIR = "path/to/xmi/files"        # XMI annotation files
OUTPUT_PATH = "path/to/output.csv"   # Merged output location
```


### Validation Tools
```bash
# Inspect data format and quality
python scripts/extract_document_features.py --inspect-rows 10

# Check parsing success rates in logs
tail -f processing.log | grep "Successfully parsed"
```

## 🧪 Data Exploration

### Jupyter Notebooks
- **`data_viewer.ipynb`**: Interactive exploration of processed datasets
  - Feature distribution analysis
  - Boundary label analysis  
  - Visual inspection of page layouts
  - Statistical summaries

- **`data_editor.ipynb`**: Manual data correction tools
  - Label correction interface
  - Outlier identification
  - Quality assessment metrics

### Dependencies
```txt
pandas>=1.5.0          # Data manipulation
numpy>=1.21.0           # Numerical computations  
shapely>=1.8.0          # Polygon simplification
lxml>=4.6.0             # XML processing (optional, faster than ElementTree)
tqdm>=4.64.0            # Progress bars
jupyter>=1.0.0          # Notebooks (optional)
```

### Large Dataset Requirements
For processing 32M+ rows:
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ free space
- **Processing time**: 2-6 hours depending on CPU


