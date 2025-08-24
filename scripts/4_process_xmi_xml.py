#!/usr/bin/env python3
"""
process_xmi_xml.py

Process a CSV file where each row represents a page scan containing XMI and XML data.
Extracts named entities from XMI data and text regions from XML data, adding the
results as new columns while preserving all original data.

Usage
-----
    python process_xmi_xml.py input.csv output.csv --top-k 5 --xmi-column "xmi_data" --xml-column "xml_data"

Requirements
------------
    pandas, shapely, lxml (or xml.etree.ElementTree), tqdm
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Tolerance for polygon simplification (adjust based on your needs)
SIMPLIFICATION_TOLERANCE = 200.0

# Default column names if not specified
DEFAULT_XMI_COLUMN = "xmi_data"
DEFAULT_XML_COLUMN = "xml_data"

# ──────────────────────────────────────────────────────────────────────────────
# XMI Processing Functions (adapted from xmi_viewer.py)
# ──────────────────────────────────────────────────────────────────────────────


def strip_namespace(tag: str) -> str:
    """Return the local part of an XML tag – e.g. '{uri}Foo' → 'Foo'."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def get_attr(attrs: Dict[str, str], local_name: str) -> Optional[str]:
    """Namespace-insensitive attribute lookup."""
    for k, v in attrs.items():
        if k.split("}", 1)[-1] == local_name:
            return v
    return None


def extract_entities_from_xmi(xmi_content: str) -> List[Dict[str, Any]]:
    """
    Extract all NamedEntity annotations from XMI content.

    Args:
        xmi_content: String containing XMI XML data

    Returns:
        List of dictionaries with entity information
    """
    entities = []

    if not xmi_content or pd.isna(xmi_content):
        return entities

    try:
        root = ET.fromstring(xmi_content)
    except ET.ParseError as e:
        print(f"[WARN] XMI parse error: {e}")
        return entities

    # 1) Get the sofaString (document text)
    sofa_text = None
    for elem in root.iter():
        if strip_namespace(elem.tag) == "Sofa":
            sofa_text = get_attr(elem.attrib, "sofaString")
            if sofa_text is not None:
                break

    # 2) Extract NamedEntity / EntityMention elements
    for elem in root.iter():
        tag_clean = strip_namespace(elem.tag)
        if tag_clean not in {"NamedEntity", "EntityMention"}:
            continue

        begin = get_attr(elem.attrib, "begin")
        end = get_attr(elem.attrib, "end")
        value = get_attr(elem.attrib, "value")

        # Convert offsets to integers
        try:
            begin_int = int(begin) if begin is not None else None
            end_int = int(end) if end is not None else None
        except ValueError:
            begin_int = end_int = None

        # Extract text span
        text_span = None
        if sofa_text and begin_int is not None and end_int is not None:
            text_span = sofa_text[begin_int:end_int]

        if value and text_span:  # Only add if we have both type and text
            entities.append(
                {
                    "entity_type": value,
                    "text": text_span,
                    "begin": begin_int,
                    "end": end_int,
                }
            )

    return entities


def get_top_k_entities(
    entities: List[Dict[str, Any]], k: int
) -> List[Tuple[str, str, int]]:
    """
    Get the top K most frequent entities from a list.

    Args:
        entities: List of entity dictionaries
        k: Number of top entities to return

    Returns:
        List of tuples (entity_type, text, count) for the top K entities
    """
    # Count occurrences of each (entity_type, text) pair
    entity_counter = Counter()
    for entity in entities:
        key = (entity["entity_type"], entity["text"])
        entity_counter[key] = entity_counter.get(key, 0) + 1

    # Get top K most common
    top_k = entity_counter.most_common(k)

    # Convert to desired format
    result = []
    for (entity_type, text), count in top_k:
        result.append((entity_type, text, count))

    # Pad with empty values if fewer than k entities
    while len(result) < k:
        result.append(("", "", 0))

    return result


# ──────────────────────────────────────────────────────────────────────────────
# XML Processing Functions (adapted from simplify_regions.py)
# ──────────────────────────────────────────────────────────────────────────────


def parse_points_string(points_str: str) -> List[Tuple[float, float]]:
    """
    Parse a string of space-separated 'x,y' coordinate pairs.

    Example: "10,20 30,40 50,60" → [(10, 20), (30, 40), (50, 60)]
    """
    coordinates = []
    if not points_str:
        return coordinates

    pairs = points_str.split(" ")
    for pair in pairs:
        try:
            x_str, y_str = pair.split(",")
            coordinates.append((float(x_str), float(y_str)))
        except ValueError:
            continue

    return coordinates


def simplify_coordinates(
    coords_list: List[Tuple[float, float]], tolerance: float
) -> List[List[float]]:
    """
    Simplify a list of [x,y] coordinates using the Ramer-Douglas-Peucker algorithm.
    """
    if not coords_list or len(coords_list) < 3:
        return [[x, y] for x, y in coords_list]

    # Ensure polygon is closed
    closed_coords = list(coords_list)
    if closed_coords[0] != closed_coords[-1]:
        closed_coords.append(closed_coords[0])

    try:
        line = LineString(closed_coords)
        simplified_line = line.simplify(tolerance, preserve_topology=True)

        if simplified_line.is_empty:
            return []

        simplified_coords = list(simplified_line.coords)

        # Ensure closed
        if simplified_coords and simplified_coords[0] != simplified_coords[-1]:
            simplified_coords.append(simplified_coords[0])

        return [[round(pt[0], 2), round(pt[1], 2)] for pt in simplified_coords]

    except Exception as e:
        print(f"[WARN] Simplification error: {e}")
        return [[round(pt[0], 2), round(pt[1], 2)] for pt in closed_coords]


def extract_regions_from_xml(xml_content: str) -> List[Dict[str, Any]]:
    """
    Extract text regions from PAGE XML content.

    Args:
        xml_content: String containing PAGE XML data

    Returns:
        List of dictionaries with region information
    """
    regions = []

    if not xml_content or pd.isna(xml_content):
        return regions

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"[WARN] XML parse error: {e}")
        return regions

    # Determine namespace
    ns_uri = root.tag.split("}")[0][1:] if "}" in root.tag else ""
    ns = {"page": ns_uri} if ns_uri else {}

    # Find Page element
    page_element = None
    for page_tag in [
        "Page",
        "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Page",
    ]:
        page_element = root.find(page_tag)
        if page_element is not None:
            break

    if page_element is None and root.tag.endswith("Page"):
        page_element = root

    if page_element is None:
        # Try to find under namespace
        if ns:
            page_element = root.find("page:Page", ns)

    if page_element is None:
        return regions

    # Process TextRegion elements
    find_query = "page:TextRegion" if ns else "TextRegion"
    for region_element in page_element.findall(find_query, ns):
        region_data = {}

        # Extract type from custom attribute
        custom_attr = region_element.get("custom", "")
        match = re.search(r"type:\s*([^;}]+)", custom_attr)
        if not match:
            continue

        region_type = match.group(1).strip()
        region_data["type"] = region_type

        # Extract text content
        region_text_parts = []
        text_lines = (
            region_element.findall(".//page:TextLine", ns)
            if ns
            else region_element.findall(".//TextLine")
        )

        if not text_lines:
            # Check for text directly under TextRegion
            text_path = "page:TextEquiv/page:Unicode" if ns else "TextEquiv/Unicode"
            text_equiv_direct = (
                region_element.find(text_path, ns)
                if ns
                else region_element.find(text_path)
            )
            if text_equiv_direct is not None and text_equiv_direct.text:
                region_text_parts.append(text_equiv_direct.text.strip())
        else:
            for text_line in text_lines:
                # Try line-level text first
                line_text_path = (
                    "page:TextEquiv/page:Unicode" if ns else "TextEquiv/Unicode"
                )
                line_text_equiv = (
                    text_line.find(line_text_path, ns)
                    if ns
                    else text_line.find(line_text_path)
                )
                if line_text_equiv is not None and line_text_equiv.text:
                    line_text = line_text_equiv.text.strip()
                    if line_text:
                        region_text_parts.append(line_text)
                else:
                    # Fallback to word-level text
                    word_texts = []
                    word_query = "page:Word" if ns else "Word"
                    for word in text_line.findall(word_query, ns):
                        word_text_path = (
                            "page:TextEquiv/page:Unicode" if ns else "TextEquiv/Unicode"
                        )
                        word_text_equiv = (
                            word.find(word_text_path, ns)
                            if ns
                            else word.find(word_text_path)
                        )
                        if word_text_equiv is not None and word_text_equiv.text:
                            word_text = word_text_equiv.text.strip()
                            if word_text:
                                word_texts.append(word_text)
                    if word_texts:
                        region_text_parts.append(" ".join(word_texts))

        region_data["text"] = " ".join(region_text_parts).strip()

        # Skip if no text extracted
        if not region_text_parts:
            continue

        # Extract and simplify coordinates
        coords_path = "page:Coords" if ns else "Coords"
        coords_element = (
            region_element.find(coords_path, ns)
            if ns
            else region_element.find(coords_path)
        )
        if coords_element is not None and coords_element.get("points"):
            points_str = coords_element.get("points")
            original_coords = parse_points_string(points_str)
            if original_coords:
                simplified_coords = simplify_coordinates(
                    original_coords, SIMPLIFICATION_TOLERANCE
                )
                region_data["simplified_polygon"] = simplified_coords
            else:
                region_data["simplified_polygon"] = []
        else:
            region_data["simplified_polygon"] = []

        # Only add if we have type and text
        if region_data.get("type") and region_data["text"]:
            regions.append(region_data)

    return regions


# ──────────────────────────────────────────────────────────────────────────────
# Main Processing Function
# ──────────────────────────────────────────────────────────────────────────────


def process_csv(
    input_file: str, output_file: str, top_k: int, xmi_column: str, xml_column: str
) -> None:
    """
    Process the CSV file, extracting XMI and XML data.

    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
        top_k: Number of top entities to extract per page
        xmi_column: Name of column containing XMI data
        xml_column: Name of column containing XML data
    """
    print(f"Reading input CSV: {input_file}")
    try:
        df = pd.read_csv(input_file, encoding="utf-8")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to read input CSV: {e}")

    print(f"Processing {len(df)} rows...")

    # Check if specified columns exist
    if xmi_column not in df.columns:
        print(
            f"[WARN] XMI column '{xmi_column}' not found in CSV. Available columns: {list(df.columns)}"
        )
        xmi_column = None

    if xml_column not in df.columns:
        print(
            f"[WARN] XML column '{xml_column}' not found in CSV. Available columns: {list(df.columns)}"
        )
        xml_column = None

    # Initialize new columns
    for i in range(1, top_k + 1):
        df[f"entity_{i}_type"] = ""
        df[f"entity_{i}_text"] = ""
        df[f"entity_{i}_count"] = 0

    df["region_data"] = ""

    # Process each row with progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # Process XMI data
        if xmi_column:
            try:
                xmi_content = row[xmi_column]
                entities = extract_entities_from_xmi(xmi_content)
                top_entities = get_top_k_entities(entities, top_k)

                # Add top entities to columns
                for i, (entity_type, text, count) in enumerate(top_entities, 1):
                    df.at[idx, f"entity_{i}_type"] = entity_type
                    df.at[idx, f"entity_{i}_text"] = text
                    df.at[idx, f"entity_{i}_count"] = count
            except Exception as e:
                print(f"[WARN] Error processing XMI data at row {idx}: {e}")

        # Process XML data
        if xml_column:
            try:
                xml_content = row[xml_column]
                regions = extract_regions_from_xml(xml_content)
                # Store as JSON string
                df.at[idx, "region_data"] = json.dumps(regions, ensure_ascii=False)
            except Exception as e:
                print(f"[WARN] Error processing XML data at row {idx}: {e}")

    # Save output
    print(f"Writing output CSV: {output_file}")
    try:
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"[OK] Successfully processed {len(df)} rows")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to write output CSV: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    """Main entry point with command-line argument parsing."""
    global SIMPLIFICATION_TOLERANCE
    
    parser = argparse.ArgumentParser(
        description="Process CSV containing XMI and XML data per page scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_xmi_xml.py input.csv output.csv --top-k 5
  python process_xmi_xml.py data.csv results.csv --xmi-column "xmi_content" --xml-column "page_xml"
        """,
    )

    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top entities to extract per page (default: 5)",
    )
    parser.add_argument(
        "--xmi-column",
        default=DEFAULT_XMI_COLUMN,
        help=f"Name of column containing XMI data (default: {DEFAULT_XMI_COLUMN})",
    )
    parser.add_argument(
        "--xml-column",
        default=DEFAULT_XML_COLUMN,
        help=f"Name of column containing XML data (default: {DEFAULT_XML_COLUMN})",
    )
    parser.add_argument(
        "--simplification-tolerance",
        type=float,
        default=SIMPLIFICATION_TOLERANCE,
        help=f"Tolerance for polygon simplification (default: {SIMPLIFICATION_TOLERANCE})",
    )

    args = parser.parse_args()

    # Update global tolerance if specified
    SIMPLIFICATION_TOLERANCE = args.simplification_tolerance

    # Validate arguments
    if args.top_k < 1:
        sys.exit("[ERROR] --top-k must be at least 1")

    # Process the CSV
    process_csv(
        input_file=args.input_csv,
        output_file=args.output_csv,
        top_k=args.top_k,
        xmi_column=args.xmi_column,
        xml_column=args.xml_column,
    )


if __name__ == "__main__":
    main()
