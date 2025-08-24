#!/usr/bin/env python3
"""
Feature Extraction Script for Document Boundary Detection

This script processes CSV files containing JSON data from page scans to extract
features for training a document boundary classifier. Each row represents a page
scan with text regions, their types, content, and polygon coordinates.

Features extracted include:
- Page-level statistics (region counts, text stats, spatial layout)
- Region type distribution and characteristics
- Text content patterns and linguistic features
- Spatial layout and geometric properties
- Sequential/contextual features for page sequences

Usage:
    python extract_document_features.py input.csv output.csv --json-column "region_data"

    # For debugging/inspection:
    python extract_document_features.py input.csv output.csv --json-column "region_data" --inspect-rows 5
"""

import argparse
import json
import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration constants
ZONE_BOUNDARIES = {"top": (0, 0.33), "middle": (0.33, 0.67), "bottom": (0.67, 1.0)}

# Common document boundary indicators
TITLE_INDICATORS = ["title", "header", "heading"]
DATE_PATTERN = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
CAPITAL_PATTERN = re.compile(r"\b[A-Z][A-Z]+\b")


class JSONParser:
    """Handle various JSON formats in CSV files."""

    @staticmethod
    def parse_json_value(value: Any) -> Optional[List[Dict]]:
        """Try multiple methods to parse JSON data from a CSV cell."""
        if pd.isna(value) or value is None:
            return None

        # If it's already a list, return it
        if isinstance(value, list):
            return value

        # If it's a string, try to parse it
        if isinstance(value, str):
            # Remove leading/trailing whitespace
            value = value.strip()

            # Empty string
            if not value:
                return None

            # Try standard JSON parsing
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
                else:
                    return None
            except json.JSONDecodeError:
                pass

            # Try ast.literal_eval for Python literals
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
                else:
                    return None
            except (ValueError, SyntaxError):
                pass

            # Try fixing common JSON issues
            try:
                # Replace single quotes with double quotes
                fixed = value.replace("'", '"')
                # Handle escaped quotes
                fixed = fixed.replace('\\"', '"')
                parsed = json.loads(fixed)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None


class FeatureExtractor:
    """Extract features from page scan JSON data for document boundary detection."""

    def __init__(self):
        self.feature_names = []
        self.json_parser = JSONParser()

    def extract_polygon_features(self, polygon: List[List[float]]) -> Dict[str, float]:
        """Extract geometric features from a polygon."""
        if not polygon or len(polygon) < 3:
            return {
                "area": 0,
                "width": 0,
                "height": 0,
                "x_min": 0,
                "x_max": 0,
                "y_min": 0,
                "y_max": 0,
                "center_x": 0,
                "center_y": 0,
            }

        try:
            points = np.array(polygon)
            x_coords = points[:, 0]
            y_coords = points[:, 1]

            # Calculate area using shoelace formula
            area = 0.5 * abs(
                sum(
                    x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]
                    for i in range(-1, len(x_coords) - 1)
                )
            )

            return {
                "area": area,
                "width": x_coords.max() - x_coords.min(),
                "height": y_coords.max() - y_coords.min(),
                "x_min": x_coords.min(),
                "x_max": x_coords.max(),
                "y_min": y_coords.min(),
                "y_max": y_coords.max(),
                "center_x": x_coords.mean(),
                "center_y": y_coords.mean(),
            }
        except Exception as e:
            logger.debug(f"Error processing polygon: {e}")
            return {
                "area": 0,
                "width": 0,
                "height": 0,
                "x_min": 0,
                "x_max": 0,
                "y_min": 0,
                "y_max": 0,
                "center_x": 0,
                "center_y": 0,
            }

    def get_page_zone(self, y_center: float, page_height: float) -> str:
        """Determine which zone of the page a region is in."""
        if page_height == 0:
            return "unknown"

        relative_y = y_center / page_height
        for zone, (min_y, max_y) in ZONE_BOUNDARIES.items():
            if min_y <= relative_y < max_y:
                return zone
        return "unknown"

    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic and content features from text."""
        if not text:
            return {
                "char_count": 0,
                "word_count": 0,
                "avg_word_length": 0,
                "has_date": 0,
                "number_count": 0,
                "capital_word_count": 0,
                "punctuation_density": 0,
                "line_count": 0,
            }

        words = text.split()
        char_count = len(text)
        word_count = len(words)

        return {
            "char_count": char_count,
            "word_count": word_count,
            "avg_word_length": (
                sum(len(w) for w in words) / word_count if word_count > 0 else 0
            ),
            "has_date": 1 if DATE_PATTERN.search(text) else 0,
            "number_count": len(NUMBER_PATTERN.findall(text)),
            "capital_word_count": len(CAPITAL_PATTERN.findall(text)),
            "punctuation_density": (
                sum(1 for c in text if c in ".,!?;:") / char_count
                if char_count > 0
                else 0
            ),
            "line_count": text.count("\n") + 1,
        }

    def extract_page_features(self, json_data: Any) -> Dict[str, Any]:
        """Extract all features from a single page's JSON data."""
        features = {}

        # Parse JSON data using the robust parser
        regions = self.json_parser.parse_json_value(json_data)

        # Handle empty or invalid data
        if not regions:
            logger.debug("No valid regions found in JSON data")
            return self._get_empty_features()

        # 1. Basic region statistics
        features["total_regions"] = len(regions)

        # 2. Region type analysis
        region_types = [
            r.get("type", "unknown") for r in regions if isinstance(r, dict)
        ]
        type_counts = Counter(region_types)

        features["unique_region_types"] = len(type_counts)
        features["dominant_region_type"] = (
            type_counts.most_common(1)[0][0] if type_counts else "none"
        )
        features["dominant_type_percentage"] = (
            type_counts.most_common(1)[0][1] / len(regions) if regions else 0
        )

        # Binary indicators for specific region types
        for indicator in TITLE_INDICATORS:
            features[f"has_{indicator}_region"] = (
                1 if any(indicator in t.lower() for t in region_types) else 0
            )

        # Count specific region types
        for region_type in ["header", "paragraph", "marginalia", "catch-word"]:
            features[f"{region_type}_count"] = type_counts.get(region_type, 0)

        # 3. Text content analysis
        all_text = " ".join(r.get("text", "") for r in regions if isinstance(r, dict))
        text_features = self.extract_text_features(all_text)
        features.update({f"page_{k}": v for k, v in text_features.items()})

        # Per-region text statistics
        region_texts = [r.get("text", "") for r in regions if isinstance(r, dict)]
        region_word_counts = [len(text.split()) for text in region_texts if text]

        features["avg_words_per_region"] = (
            np.mean(region_word_counts) if region_word_counts else 0
        )
        features["std_words_per_region"] = (
            np.std(region_word_counts) if region_word_counts else 0
        )
        features["max_words_in_region"] = (
            max(region_word_counts) if region_word_counts else 0
        )

        # 4. Spatial layout features
        polygon_features_list = []
        page_bounds = {
            "x_min": float("inf"),
            "x_max": 0,
            "y_min": float("inf"),
            "y_max": 0,
        }

        for region in regions:
            if isinstance(region, dict) and "simplified_polygon" in region:
                poly_features = self.extract_polygon_features(
                    region["simplified_polygon"]
                )
                if poly_features["area"] > 0:  # Valid polygon
                    polygon_features_list.append(poly_features)

                    # Update page bounds
                    page_bounds["x_min"] = min(
                        page_bounds["x_min"], poly_features["x_min"]
                    )
                    page_bounds["x_max"] = max(
                        page_bounds["x_max"], poly_features["x_max"]
                    )
                    page_bounds["y_min"] = min(
                        page_bounds["y_min"], poly_features["y_min"]
                    )
                    page_bounds["y_max"] = max(
                        page_bounds["y_max"], poly_features["y_max"]
                    )

        if polygon_features_list:
            # Page dimensions
            page_width = page_bounds["x_max"] - page_bounds["x_min"]
            page_height = page_bounds["y_max"] - page_bounds["y_min"]
            features["page_width"] = page_width
            features["page_height"] = page_height

            # Region area statistics
            areas = [pf["area"] for pf in polygon_features_list]
            features["total_text_area"] = sum(areas)
            features["avg_region_area"] = np.mean(areas)
            features["std_region_area"] = np.std(areas)
            features["page_coverage"] = (
                sum(areas) / (page_width * page_height)
                if page_width * page_height > 0
                else 0
            )

            # Spatial distribution
            y_centers = [pf["center_y"] for pf in polygon_features_list]
            x_centers = [pf["center_x"] for pf in polygon_features_list]

            features["vertical_spread"] = np.std(y_centers) if y_centers else 0
            features["horizontal_spread"] = np.std(x_centers) if x_centers else 0

            # Zone distribution
            zone_counts = Counter()
            for pf in polygon_features_list:
                zone = self.get_page_zone(
                    pf["center_y"] - page_bounds["y_min"], page_height
                )
                zone_counts[zone] += 1

            for zone in ["top", "middle", "bottom"]:
                features[f"regions_in_{zone}"] = zone_counts.get(zone, 0)
                features[f"regions_in_{zone}_pct"] = (
                    zone_counts.get(zone, 0) / len(regions) if regions else 0
                )

            # Alignment features
            x_positions = sorted(x_centers)
            if len(x_positions) > 1:
                x_diffs = [
                    x_positions[i + 1] - x_positions[i]
                    for i in range(len(x_positions) - 1)
                ]
                features["horizontal_alignment_score"] = (
                    1 / (1 + np.std(x_diffs)) if x_diffs else 0
                )
            else:
                features["horizontal_alignment_score"] = 1
        else:
            # Default spatial features
            features.update(
                {
                    "page_width": 0,
                    "page_height": 0,
                    "total_text_area": 0,
                    "avg_region_area": 0,
                    "std_region_area": 0,
                    "page_coverage": 0,
                    "vertical_spread": 0,
                    "horizontal_spread": 0,
                    "regions_in_top": 0,
                    "regions_in_middle": 0,
                    "regions_in_bottom": 0,
                    "regions_in_top_pct": 0,
                    "regions_in_middle_pct": 0,
                    "regions_in_bottom_pct": 0,
                    "horizontal_alignment_score": 0,
                }
            )

        # 5. Layout complexity
        features["layout_complexity"] = (
            features["unique_region_types"] * features["vertical_spread"]
        )

        return features

    def extract_sequential_features(
        self, df: pd.DataFrame, window_size: int = 3
    ) -> pd.DataFrame:
        """Extract features that consider page sequences and context."""
        logger.info(f"Extracting sequential features with window size {window_size}")

        # Create a copy to avoid fragmentation warnings
        df = df.copy()

        # Collect new columns to add
        new_columns = {}

        # Calculate differences from previous page
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not col.endswith("_diff") and not col.endswith("_ma"):
                # Difference from previous page
                new_columns[f"{col}_diff"] = df[col].diff()
                # Moving average
                new_columns[f"{col}_ma{window_size}"] = (
                    df[col].rolling(window=window_size, min_periods=1).mean()
                )

        # Page position features
        new_columns["page_position"] = np.arange(len(df))
        new_columns["relative_position"] = new_columns["page_position"] / len(df)

        # Add all new columns at once
        df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

        # Detect significant changes (potential boundaries)
        threshold_cols = ["total_regions", "unique_region_types", "page_coverage"]
        for col in threshold_cols:
            if col in df.columns and f"{col}_diff" in df.columns:
                col_std = df[col].std()
                df[f"{col}_significant_change"] = (
                    df[f"{col}_diff"].abs() > 2 * col_std
                ).astype(int)

        return df

    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty feature dictionary with all expected keys."""
        # This ensures consistent feature structure even for empty pages
        return {
            "total_regions": 0,
            "unique_region_types": 0,
            "dominant_region_type": "none",
            "dominant_type_percentage": 0,
            "has_title_region": 0,
            "has_header_region": 0,
            "has_heading_region": 0,
            "header_count": 0,
            "paragraph_count": 0,
            "marginalia_count": 0,
            "catch-word_count": 0,
            "page_char_count": 0,
            "page_word_count": 0,
            "page_avg_word_length": 0,
            "page_has_date": 0,
            "page_number_count": 0,
            "page_capital_word_count": 0,
            "page_punctuation_density": 0,
            "page_line_count": 0,
            "avg_words_per_region": 0,
            "std_words_per_region": 0,
            "max_words_in_region": 0,
            "page_width": 0,
            "page_height": 0,
            "total_text_area": 0,
            "avg_region_area": 0,
            "std_region_area": 0,
            "page_coverage": 0,
            "vertical_spread": 0,
            "horizontal_spread": 0,
            "regions_in_top": 0,
            "regions_in_middle": 0,
            "regions_in_bottom": 0,
            "regions_in_top_pct": 0,
            "regions_in_middle_pct": 0,
            "regions_in_bottom_pct": 0,
            "horizontal_alignment_score": 0,
            "layout_complexity": 0,
        }


def inspect_data(df: pd.DataFrame, json_column: str, num_rows: int = 5):
    """Inspect the data to understand JSON format and content."""
    logger.info(f"\n{'='*80}")
    logger.info("DATA INSPECTION REPORT")
    logger.info(f"{'='*80}")

    logger.info(f"\nDataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    if json_column not in df.columns:
        logger.error(f"Column '{json_column}' not found!")
        return

    logger.info(f"\nInspecting column '{json_column}':")
    logger.info(f"Data type: {df[json_column].dtype}")

    # Check for null values
    null_count = df[json_column].isna().sum()
    logger.info(f"Null values: {null_count} ({null_count/len(df)*100:.1f}%)")

    # Sample non-null values
    non_null_samples = df[df[json_column].notna()][json_column].head(num_rows)

    parser = JSONParser()
    successful_parses = 0

    for idx, (row_idx, value) in enumerate(non_null_samples.items()):
        logger.info(f"\n--- Row {row_idx} ---")
        logger.info(f"Raw value type: {type(value)}")
        logger.info(f"Raw value (first 200 chars): {str(value)[:200]}...")

        # Try to parse
        parsed = parser.parse_json_value(value)
        if parsed:
            successful_parses += 1
            logger.info(f"✓ Successfully parsed! Found {len(parsed)} regions")
            if parsed:
                first_region = parsed[0]
                logger.info(f"  First region keys: {list(first_region.keys())}")
                logger.info(f"  First region type: {first_region.get('type', 'N/A')}")
                logger.info(f"  Text preview: {first_region.get('text', '')[:50]}...")
        else:
            logger.info("✗ Failed to parse JSON")

    logger.info(f"\n{'='*80}")
    logger.info(
        f"Successfully parsed {successful_parses}/{len(non_null_samples)} samples"
    )
    logger.info(f"{'='*80}\n")


def process_csv(
    input_file: str,
    output_file: str,
    json_column: str,
    window_size: int = 3,
    inspect_rows: Optional[int] = None,
) -> None:
    """Process CSV file and extract features from JSON data."""
    logger.info(f"Processing {input_file}")

    # Load CSV
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return

    # Check if JSON column exists
    if json_column not in df.columns:
        logger.error(
            f"Column '{json_column}' not found in CSV. Available columns: {list(df.columns)}"
        )
        return

    # Inspect data if requested
    if inspect_rows:
        inspect_data(df, json_column, inspect_rows)
        if input("Continue with processing? (y/n): ").lower() != "y":
            return

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Extract page-level features
    logger.info("Extracting page-level features...")
    feature_dicts = []

    successful_extractions = 0
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing row {idx}/{len(df)}")

        json_data = row[json_column]
        features = extractor.extract_page_features(json_data)

        # Check if extraction was successful (non-zero features)
        if features["total_regions"] > 0:
            successful_extractions += 1

        feature_dicts.append(features)

    logger.info(
        f"Successfully extracted features from {successful_extractions}/{len(df)} rows"
    )

    # Create feature dataframe
    feature_df = pd.DataFrame(feature_dicts)

    # Combine with original data
    result_df = pd.concat([df, feature_df], axis=1)

    # Extract sequential features
    result_df = extractor.extract_sequential_features(result_df, window_size)

    # Save results
    result_df.to_csv(output_file, index=False)
    logger.info(
        f"Saved {len(result_df)} rows with {len(feature_df.columns)} new features to {output_file}"
    )

    # Log feature statistics (only for features with non-zero values)
    logger.info("\nFeature Statistics:")
    for col in feature_df.columns:
        if feature_df[col].dtype in [np.float64, np.int64]:
            col_mean = feature_df[col].mean()
            col_std = feature_df[col].std()
            if col_mean > 0 or col_std > 0:
                logger.info(f"{col}: mean={col_mean:.2f}, std={col_std:.2f}")


def generate_feature_documentation(output_file: str = "feature_documentation.txt"):
    """Generate documentation for all extracted features."""
    doc = """
Document Boundary Detection - Feature Documentation
================================================

Page-Level Features
------------------
- total_regions: Total number of text regions on the page
- unique_region_types: Number of distinct region types
- dominant_region_type: Most common region type
- dominant_type_percentage: Percentage of regions that are the dominant type

Region Type Features
-------------------
- has_title_region: Binary indicator for presence of title regions
- has_header_region: Binary indicator for presence of header regions
- has_heading_region: Binary indicator for presence of heading regions
- [type]_count: Count of specific region types (header, paragraph, marginalia, catch-word)

Text Content Features
--------------------
- page_char_count: Total character count across all regions
- page_word_count: Total word count across all regions
- page_avg_word_length: Average word length on the page
- page_has_date: Binary indicator for presence of date patterns
- page_number_count: Count of numeric patterns
- page_capital_word_count: Count of all-caps words
- page_punctuation_density: Ratio of punctuation to total characters
- page_line_count: Total number of lines
- avg_words_per_region: Average words per text region
- std_words_per_region: Standard deviation of words per region
- max_words_in_region: Maximum words in any single region

Spatial Layout Features
----------------------
- page_width/height: Dimensions of the page bounding box
- total_text_area: Sum of all region areas
- avg_region_area: Average area of text regions
- std_region_area: Standard deviation of region areas
- page_coverage: Ratio of text area to total page area
- vertical_spread: Standard deviation of region vertical positions
- horizontal_spread: Standard deviation of region horizontal positions
- regions_in_[zone]: Count of regions in top/middle/bottom zones
- regions_in_[zone]_pct: Percentage of regions in each zone
- horizontal_alignment_score: Measure of horizontal alignment consistency
- layout_complexity: Combined measure of type diversity and vertical spread

Sequential Features
------------------
- [feature]_diff: Change from previous page
- [feature]_ma[N]: Moving average over N pages
- page_position: Absolute position in sequence
- relative_position: Relative position (0-1)
- [feature]_significant_change: Binary indicator for large changes

Feature Importance Hints for ML
------------------------------
High importance for boundary detection:
1. has_title_region - New documents often start with titles
2. unique_region_types - Document starts may have distinct layouts
3. page_coverage_diff - Large layout changes indicate boundaries
4. regions_in_top_pct - Headers often appear at document starts
5. [feature]_significant_change - Abrupt changes suggest boundaries

Medium importance:
- Text density features (word counts, coverage)
- Spatial distribution features
- Sequential moving averages

Lower importance:
- Absolute position features
- Minor text statistics

Data Quality Troubleshooting
---------------------------
If all features are 0:
1. Check that the JSON column name is correct
2. Verify JSON format matches expected structure
3. Use --inspect-rows flag to examine raw data
4. Ensure JSON contains 'type', 'text', and 'simplified_polygon' fields
"""

    with open(output_file, "w") as f:
        f.write(doc)
    logger.info(f"Feature documentation saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from page scan JSON data for document boundary detection"
    )
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument(
        "--json-column",
        default="region_data",
        help="Name of column containing JSON data (default: region_data)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Window size for sequential features (default: 3)",
    )
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate feature documentation file",
    )
    parser.add_argument(
        "--inspect-rows",
        type=int,
        metavar="N",
        help="Inspect N rows of data before processing",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Process CSV
    process_csv(
        args.input_csv,
        args.output_csv,
        args.json_column,
        args.window_size,
        args.inspect_rows,
    )

    # Generate documentation if requested
    if args.generate_docs:
        generate_feature_documentation()


if __name__ == "__main__":
    main()
