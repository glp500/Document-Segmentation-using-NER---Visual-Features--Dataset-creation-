import os
import xml.etree.ElementTree as ET
import json
import re
from shapely.geometry import LineString, Polygon # Added for simplification

# --- Configuration ---
# Adjust this tolerance value to control the level of simplification.
# Higher values mean more simplification (fewer points, less detail).
# Lower values mean less simplification (more points, more detail).
# You'll likely need to experiment with this value based on your specific documents.
SIMPLIFICATION_TOLERANCE = 200.0 # Example value, adjust as needed

# Base directories for batch processing
BASE_INPUT_DIRECTORY = '/Users/gavinl/Desktop/Renate Inventories/xml'
BASE_OUTPUT_DIRECTORY = '/Users/gavinl/Desktop/Renate Inventories/json'

def parse_points_string(points_str):
    """
    Parses a string of space-separated 'x,y' coordinate pairs into a list of [x, y] tuples.
    Example input: "10,20 30,40 50,60"
    Example output: [(10, 20), (30, 40), (50, 60)]
    """
    coordinates = []
    if not points_str:
        return coordinates
    pairs = points_str.split(' ')
    for pair in pairs:
        try:
            x_str, y_str = pair.split(',')
            coordinates.append((float(x_str), float(y_str)))
        except ValueError:
            # Handle potential errors in coordinate string format, e.g., empty strings if there are double spaces
            # print(f"Warning: Could not parse point pair '{pair}' in '{points_str}'. Skipping.")
            continue
    return coordinates

def simplify_coordinates(coords_list, tolerance):
    """
    Simplifies a list of [x,y] coordinates using the Ramer-Douglas-Peucker algorithm.
    Ensures the polygon is closed before simplification.
    """
    if not coords_list or len(coords_list) < 3: # Need at least 3 points for a polygon
        return coords_list

    # Ensure the polygon is closed (first and last points are the same)
    # This is important for shapely.geometry.Polygon and for sensible simplification of a closed region
    closed_coords_list = list(coords_list) # Make a copy
    if closed_coords_list[0] != closed_coords_list[-1]:
        closed_coords_list.append(closed_coords_list[0])
    
    if len(closed_coords_list) < 3: # Still not enough points after potential closure
        return closed_coords_list


    try:
        # For simplification, we can treat the boundary as a LineString.
        # If the original shape is a valid Polygon, simplify its exterior.
        # Using LineString is generally more robust for potentially "messy" input coordinates from tracing.
        line = LineString(closed_coords_list)
        simplified_line = line.simplify(tolerance, preserve_topology=True)
        
        # Get coordinates from the simplified geometry
        # The result might be a LineString or MultiLineString (if simplification breaks it)
        # For simplicity, we'll assume it remains a single LineString representing the polygon boundary
        if simplified_line.is_empty:
            return []
        
        simplified_coords = list(simplified_line.coords)

        # Ensure the simplified polygon is also explicitly closed in the output list
        if simplified_coords and simplified_coords[0] != simplified_coords[-1]:
            simplified_coords.append(simplified_coords[0])
            
        return [[round(pt[0], 2), round(pt[1], 2)] for pt in simplified_coords] # Round for cleaner JSON

    except Exception as e:
        print(f"Error during simplification: {e}. Returning original (closed) coordinates.")
        # Return the closed (but not simplified) coordinates in case of an error
        return [[round(pt[0], 2), round(pt[1], 2)] for pt in closed_coords_list]


def extract_data_from_xml(xml_file_path):
    """
    Parses a PAGE XML file, extracts text regions with their type, text,
    and simplified polygon coordinates.

    Args:
        xml_file_path (str): The path to the input XML file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              text region with its 'type', 'text', and 'simplified_polygon'.
              Returns an empty list if no processable regions are found.
    """
    json_output = []
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"Error parsing XML file: {xml_file_path}. Skipping.")
        return []

    # Define the namespace (assuming PAGE XML as per your samples)
    # The namespace URI might vary slightly in different versions of PAGE XML.
    # Check your XML files if you encounter issues.
    ns_uri = root.tag.split('}')[0][1:] if '}' in root.tag else ''
    ns = {'page': ns_uri} if ns_uri else {}
    
    # Find the Page element
    page_element_name = 'Page'
    if ns:
        page_element_name = f"{{{ns['page']}}}{page_element_name}"
    else: # If no namespace is detected at root, try finding Page without it (less common for PAGE XML)
        page_element_name = 'Page'

    page_element = root.find(page_element_name, ns)

    if page_element is None:
         # If root is PcGts, Page might be a direct child without prefix if ns wasn't properly caught or used by find
        if root.tag.endswith("PcGts"): # Common root tag for PAGE XML
            page_element = root.find('page:Page', {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}) # Try with common PAGE ns
            if page_element is None:
                 page_element = root.find('Page') # Try without explicit namespace mapping in find

        if page_element is None:
            print(f"Could not find Page element in {xml_file_path}. Skipping.")
            return []


    # Only process TextRegion (per instructions).
    region_types_to_process = ['TextRegion']

    for region_tag_name in region_types_to_process:
        
        find_query = f'page:{region_tag_name}' if ns else region_tag_name
        
        for region_element in page_element.findall(find_query, ns):
            region_data = {}
            
            custom_attr = region_element.get('custom', '')
            # Extract region type with the same regex used in xml_to_json.py
            match = re.search(r'type:\s*([^;}]+)', custom_attr)
            if not match:
                continue  # Skip regions that lack a type label
            region_type = match.group(1).strip()
            region_data['type'] = region_type

            # Extract text content using the same logic as xml_to_json.py
            region_text_parts = []
            text_lines = region_element.findall('.//page:TextLine', ns) if ns else region_element.findall('.//TextLine')

            if not text_lines:
                # Check for text directly under TextRegion
                text_equiv_direct = region_element.find('page:TextEquiv/page:Unicode', ns) if ns else region_element.find('TextEquiv/Unicode')
                if text_equiv_direct is not None and text_equiv_direct.text:
                    region_text_parts.append(text_equiv_direct.text.strip())
            else:
                for text_line in text_lines:
                    # Prefer full TextEquiv for the line
                    line_text_equiv = text_line.find('page:TextEquiv/page:Unicode', ns) if ns else text_line.find('TextEquiv/Unicode')
                    if line_text_equiv is not None and line_text_equiv.text:
                        line_text = line_text_equiv.text.strip()
                        if line_text:
                            region_text_parts.append(line_text)
                    else:
                        # Fallback to concatenating word-level text
                        word_texts = []
                        for word in text_line.findall('page:Word', ns) if ns else text_line.findall('Word'):
                            word_text_equiv = word.find('page:TextEquiv/page:Unicode', ns) if ns else word.find('TextEquiv/Unicode')
                            if word_text_equiv is not None and word_text_equiv.text:
                                word_text = word_text_equiv.text.strip()
                                if word_text:
                                    word_texts.append(word_text)
                        if word_texts:
                            region_text_parts.append(" ".join(word_texts))

            region_data['text'] = " ".join(region_text_parts).strip()

            # Skip this region entirely if no text was extracted
            if not region_text_parts:
                continue

            # Extract and simplify coordinates
            coords_element = region_element.find('page:Coords', ns) if ns else region_element.find('Coords')
            if coords_element is not None and coords_element.get('points'):
                points_str = coords_element.get('points')
                original_coords = parse_points_string(points_str)
                if original_coords:
                    simplified_poly_coords = simplify_coordinates(original_coords, SIMPLIFICATION_TOLERANCE)
                    region_data['simplified_polygon'] = simplified_poly_coords
                else:
                    region_data['simplified_polygon'] = [] # No valid points found
            else:
                region_data['simplified_polygon'] = [] # No Coords element or points attribute

            # Only add region if it has a type **and** non‑empty text
            if region_data.get('type') and region_data['text']:
                 json_output.append(region_data)
                 
    return json_output

def main():
    """
    Main function to process all XML files in the input directory and
    save the extracted data as JSON files in the output directory.
    """
    # Batch‑process every inventory folder that contains a "page" sub‑directory
    processed_inventories = 0
    for inventory_name in sorted(os.listdir(BASE_INPUT_DIRECTORY)):
        inventory_dir = os.path.join(BASE_INPUT_DIRECTORY, inventory_name)
        page_dir = os.path.join(inventory_dir, "page")

        # Skip if this isn't a valid inventory directory
        if not os.path.isdir(page_dir):
            continue

        output_dir = os.path.join(BASE_OUTPUT_DIRECTORY, inventory_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nProcessing inventory '{inventory_name}'")
        print(f"  XML source : {page_dir}")
        print(f"  JSON output: {output_dir}")
        print(f"  Simplification Tolerance: {SIMPLIFICATION_TOLERANCE}")
        print("-" * 30)

        processed_files = 0
        for filename in os.listdir(page_dir):
            if not filename.endswith(".xml"):
                continue

            xml_file_path = os.path.join(page_dir, filename)
            extracted_data = extract_data_from_xml(xml_file_path)

            base_filename = os.path.splitext(filename)[0]
            json_file_path = os.path.join(output_dir, f"{base_filename}.json")

            try:
                with open(json_file_path, "w", encoding="utf-8") as json_file:
                    json.dump(extracted_data or [], json_file, indent=4, ensure_ascii=False)
                processed_files += 1
            except IOError as e:
                print(f"Error writing JSON file {json_file_path}: {e}")

        print(f"Finished inventory '{inventory_name}'. Processed {processed_files} XML file(s).")
        processed_inventories += 1

    print(f"\nBatch complete. Processed {processed_inventories} inventories.")

if __name__ == "__main__":
    # Before running:
    # 1. Make sure you have the 'shapely' library installed.
    #    If not, run: pip install shapely
    # 2. Create a folder named "input_xml_files" in the same directory as this script 
    #    OR update INPUT_DIRECTORY to your actual path.
    # 3. Place your XML files (e.g., NL-HaNA_1.04.02_7923_0171.xml) into the input folder.
    # 4. An output folder (as specified in OUTPUT_DIRECTORY) will be created by the script if it doesn't exist.
    main()