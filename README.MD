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

### 2. Run the Complete Pipeline
```bash
# Step 1: Merge all data sources
python scripts/1_merge_all.py

# Step 2: Remove duplicate/ambiguous labels (sameAs rows)
python scripts/2_remove_rows_of_sameAs.py

# Step 3: Enhance TANAP boundary annotations
python scripts/3_enhance_TANAP_bounds.py

# Step 4: Process XMI/XML data to extract entities and regions  
python scripts/4_process_xmi_xml.py merged_data.csv processed_data.csv --top-k 5 --xmi-column "xmi_data" --xml-column "xml_data"

# Step 5: Extract comprehensive features for machine learning
python scripts/5_extract_document_features.py processed_data.csv final_dataset.csv --json-column "region_data"
```

### 3. Explore the Data
```bash
# Launch Jupyter notebooks for interactive exploration
jupyter notebook notebooks/data_viewer.ipynb
jupyter notebook notebooks/data_editor.ipynb
```

## 📊 Data Processing Pipeline

### Stage 1: Data Merging (`1_merge_all.py`)
- **Input**: Separate CSV files (Renate inventories), XML files (page layouts), XMI files (NER annotations)
- **Process**: 
  - Automatic delimiter detection for CSV files
  - UTF-8 BOM handling for proper encoding
  - Public scan URL generation for document viewing
  - Consolidation into single dataset with embedded XML/XMI content
- **Output**: Unified CSV with columns: `[inventory_data, xmi_data, xml_data, public_scan_url]`

### Stage 2: Entity & Region Extraction (`process_xmi_xml.py`)
- **Input**: Merged CSV with XMI and XML data
- **Process**:
  - **XMI Processing**: Extract named entities (PER_NAME, LOC_NAME, ORG, SHIP, DATE) with begin/end offsets
  - **XML Processing**: Extract text regions (headers, paragraphs, marginalia, catch-words) with polygon coordinates
  - **Polygon Simplification**: Use Ramer-Douglas-Peucker algorithm with tolerance=200.0 to reduce coordinate complexity
  - **Top-K Selection**: Keep only the most frequent entity types per page (default: top 5)
- **Output**: CSV with additional columns: `[top_entities, region_data]` containing structured JSON

### Stage 3: Feature Engineering (`extract_document_features.py`)
- **Input**: CSV with extracted entities and regions in JSON format
- **Process**: Generate 60+ features across multiple categories (detailed below)
- **Output**: Feature-rich dataset ready for machine learning

### Stage 4: Data Cleaning (Optional)
- **Remove Duplicates**: `2_remove_rows_of_sameAs.py` removes rows with ambiguous labels
- **Enhance Boundaries**: `3_enhance_TANAP_bounds.py` improves boundary detection using heuristics

## 🔬 Comprehensive Feature Documentation

The feature extraction script generates **60+ features** across seven main categories:

### 1. Basic Region Statistics (4 features)
- `total_regions`: Total number of text regions on the page
- `unique_region_types`: Number of distinct region types (header, paragraph, etc.)
- `dominant_region_type`: Most frequent region type on the page
- `dominant_type_percentage`: Percentage of regions that are the dominant type

### 2. Region Type Indicators (8 features)
**Binary indicators** (0/1) for presence of specific region types:
- `has_title_region`: Page contains title regions
- `has_header_region`: Page contains header regions  
- `has_heading_region`: Page contains heading regions

**Count features** for specific region types:
- `header_count`: Number of header regions
- `paragraph_count`: Number of paragraph regions
- `marginalia_count`: Number of marginalia (margin notes)
- `catch-word_count`: Number of catch-words
- `*_count`: Additional region type counts as found in data

### 3. Text Content Analysis (15 features)
**Page-level text statistics**:
- `page_char_count`: Total characters on the page
- `page_word_count`: Total words on the page
- `page_avg_word_length`: Average word length across all text
- `page_has_date`: Binary indicator for date presence (regex: `\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b`)
- `page_number_count`: Count of numeric sequences
- `page_capital_word_count`: Count of all-caps words (potential proper nouns)
- `page_punctuation_density`: Ratio of punctuation to total characters
- `page_line_count`: Number of text lines (newline count + 1)

**Per-region text statistics**:
- `avg_words_per_region`: Average words per text region
- `std_words_per_region`: Standard deviation of words per region
- `max_words_in_region`: Maximum words found in any single region

### 4. Spatial Layout Features (17 features)
**Page dimensions and coverage**:
- `page_width`: Page width in coordinate units
- `page_height`: Page height in coordinate units  
- `total_text_area`: Sum of all region areas
- `avg_region_area`: Average area per region
- `std_region_area`: Standard deviation of region areas
- `page_coverage`: Fraction of page covered by text regions

**Spatial distribution**:
- `vertical_spread`: Standard deviation of region Y-coordinates (layout consistency)
- `horizontal_spread`: Standard deviation of region X-coordinates (margin consistency)
- `horizontal_alignment_score`: Measure of left-margin alignment (1/(1+std(x_positions)))

**Zone-based features** (page divided into top/middle/bottom thirds):
- `regions_in_top`: Count of regions in top third
- `regions_in_middle`: Count of regions in middle third  
- `regions_in_bottom`: Count of regions in bottom third
- `regions_in_top_pct`: Percentage of regions in top third
- `regions_in_middle_pct`: Percentage of regions in middle third
- `regions_in_bottom_pct`: Percentage of regions in bottom third

### 5. Geometric Features (Per-region polygon analysis)
For each text region's simplified polygon:
- `area`: Region area using shoelace formula
- `width`: Bounding box width (x_max - x_min)
- `height`: Bounding box height (y_max - y_min)
- `center_x`: Centroid X-coordinate
- `center_y`: Centroid Y-coordinate
- `x_min, x_max, y_min, y_max`: Bounding box coordinates

### 6. Layout Complexity (1 feature)
- `layout_complexity`: Combined measure (`unique_region_types × vertical_spread`) indicating page structure complexity

### 7. Sequential Features (Variable count)
Generated for each numeric feature to capture page-to-page transitions:

**Difference features** (`*_diff`):
- First-order differences from previous page for all numeric features
- Example: `total_regions_diff` = change in region count from previous page

**Moving averages** (`*_ma{window_size}`):  
- Rolling averages with configurable window (default: 3 pages)
- Example: `page_coverage_ma3` = 3-page moving average of page coverage

**Position features**:
- `page_position`: Absolute position in document sequence
- `relative_position`: Relative position (0.0 to 1.0) within document

**Change detection** (`*_significant_change`):
- Binary indicators for significant changes (> 2 standard deviations)
- Applied to key features: `total_regions`, `unique_region_types`, `page_coverage`

## 🎛️ Configuration Options

### Polygon Simplification
```python
# In process_xmi_xml.py
SIMPLIFICATION_TOLERANCE = 200.0  # Adjust based on coordinate precision needs
```

### Entity Extraction
```bash
# Extract top 5 most frequent entity types per page
python scripts/process_xmi_xml.py input.csv output.csv --top-k 5

# Custom column names
python scripts/process_xmi_xml.py input.csv output.csv --xmi-column "custom_xmi" --xml-column "custom_xml"
```

### Feature Engineering
```bash
# Sequential features with custom window size
python scripts/extract_document_features.py input.csv output.csv --json-column "region_data" --window-size 5

# Inspect data format before processing
python scripts/extract_document_features.py input.csv output.csv --json-column "region_data" --inspect-rows 10
```

### Path Configuration
Update hard-coded paths in `scripts/1_merge_all.py`:
```python
CSV_DIR = "path/to/renate/csvs"      # Renate inventory CSV files
XML_DIR = "path/to/xml/files"        # PAGE XML files  
XMI_DIR = "path/to/xmi/files"        # XMI annotation files
OUTPUT_PATH = "path/to/output.csv"   # Merged output location
```

## 📈 Data Quality and Validation

### Robust JSON Parsing
The feature extraction handles various JSON formats:
- Standard JSON with double quotes
- Python literals with single quotes  
- Escaped quotes and malformed JSON
- Empty and null values

### Error Handling
- **XMI Parse Errors**: Logged with warnings, empty entity lists returned
- **XML Parse Errors**: Graceful degradation with default values
- **Polygon Errors**: Invalid polygons return zero-valued geometric features
- **Missing Data**: Consistent empty feature dictionaries ensure uniform structure

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

### Feature Statistics
The processing pipeline automatically logs feature statistics:
```
INFO:__main__:Feature Statistics:
total_regions: mean=8.45, std=4.23
page_coverage: mean=0.34, std=0.18
layout_complexity: mean=12.67, std=8.91
```

## 🎯 Expected Output

### Final Dataset Structure
The complete pipeline produces a CSV with:
- **Original columns**: All data from Renate inventories
- **Processed columns**: `top_entities`, `region_data`, `public_scan_url`
- **60+ feature columns**: Comprehensive feature vectors
- **Sequential columns**: Difference, moving average, and change detection features

### Typical Dataset Size
- **Input**: ~19 inventory files with thousands of pages each
- **Processing time**: 5-15 minutes depending on dataset size
- **Output size**: 32M+ rows with 80+ columns (3+ GB)
- **Feature extraction success rate**: Typically 95%+ for well-formed data

## 🔧 Advanced Usage

### Custom Feature Development
Extend the `FeatureExtractor` class in `extract_document_features.py`:

```python
def extract_custom_features(self, regions):
    """Add your custom features here"""
    features = {}
    
    # Example: Count regions with specific text patterns
    features['formal_language_regions'] = sum(
        1 for r in regions 
        if re.search(r'\b(hereby|whereas|therefore)\b', r.get('text', ''), re.I)
    )
    
    return features
```

### Performance Optimization
For large datasets:
- Process in chunks using pandas `chunksize` parameter
- Use parallel processing for feature extraction
- Consider sampling for development and testing

### Integration with ML Pipelines
The generated features are ready for:
- **Classification tasks**: Document boundary detection (START/END/MIDDLE/NONE)
- **Regression tasks**: Boundary confidence scoring
- **Sequence modeling**: LSTM/Transformer models using sequential features
- **Clustering**: Unsupervised document structure discovery

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB (for small datasets)
- **Storage**: 10GB free space
- **CPU**: 2+ cores recommended

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

## 🚨 Common Issues and Solutions

### Memory Issues
```bash
# Process in smaller chunks
python scripts/process_xmi_xml.py input.csv output.csv --chunk-size 10000
```

### Encoding Problems
```python
# The pipeline handles UTF-8 BOM automatically
# For manual fixes, strip BOM:
df = pd.read_csv('file.csv', encoding='utf-8-sig')
```

### JSON Parsing Failures
- Use `--inspect-rows N` to examine data format
- Check for mixed JSON formats in the same column
- Verify column names match your data

### Performance Issues
- Use `--top-k` parameter to limit entity extraction
- Consider processing subsets for development
- Monitor memory usage during large batch processing

## 🎉 Getting Started

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd "Document Segmentation using NER & Visual Features (Dataset creation)"
   pip install -r requirements.txt
   ```

2. **Configure paths** in `scripts/1_merge_all.py`

3. **Run pipeline**:
   ```bash
   python scripts/1_merge_all.py
   python scripts/process_xmi_xml.py merged_data.csv processed_data.csv
   python scripts/extract_document_features.py processed_data.csv final_dataset.csv --json-column "region_data"
   ```

4. **Explore results**:
   ```bash
   jupyter notebook notebooks/data_viewer.ipynb
   ```

The pipeline is designed to be robust, scalable, and suitable for historical document analysis research. All processing steps include comprehensive logging and error handling to ensure reliable dataset generation from complex archival materials.