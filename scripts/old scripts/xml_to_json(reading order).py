#!/usr/bin/env python3
"""Inventory Text Extractor

Batch-converts PAGE XML files stored in nested *inventory*/page/ folders into
simple JSON files that contain only the reading-order layout *type* and the
associated *text*.

Key differences from *simplify_regions.py*:
• Polygon coordinates are **not** stored.
• Regions whose type is "marginalia" are always appended to the **end** of each
  JSON list so that any margin notes appear after the main text when consumed
  later.

Folder structure expected
────────────────────────
<BASE_INPUT_DIRECTORY>/
    1120/
        page/   # PAGE XML files live here
    1547/
        page/
    ...

For every inventory folder (1120, 1547, …) the script writes JSON files with the
same base-name as the XML into a matching folder inside
<BASE_OUTPUT_DIRECTORY>/.

Run the script as a standalone program:
$ python inventory_text_extractor.py
"""
from __future__ import annotations

import os
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

# ─── Configuration ────────────────────────────────────────────────────────────
# Adjust these two paths before running.
BASE_INPUT_DIRECTORY = "/Users/gavinl/Desktop/Renate Inventories/xml"
BASE_OUTPUT_DIRECTORY = "/Users/gavinl/Desktop/Test"
# ─── Core XML parsing ─────────────────────────────────────────────────────────

def extract_data_from_xml(xml_path: str | Path) -> List[Dict[str, Any]]:
    """Parses a PAGE XML file and returns a list of dicts with 'type' & 'text'.

    All *marginalia* regions are placed at the end of the returned list so that
    downstream consumers read the main body first.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        print(f"[WARN] Could not parse XML: {xml_path}. Skipping.")
        return []

    root = tree.getroot()
    ns_uri = root.tag.split("}")[0][1:] if "}" in root.tag else ""
    ns = {"page": ns_uri} if ns_uri else {}

    page_elem_name = f"{{{ns_uri}}}Page" if ns_uri else "Page"
    page_elem = root.find(page_elem_name)
    if page_elem is None:
        # Fallbacks for slightly malformed namespace usage
        page_elem = root.find("page:Page", ns) or root.find("Page")
        if page_elem is None:
            print(f"[WARN] No <Page> element in {xml_path}. Skipping.")
            return []

    main_regions: List[Dict[str, Any]] = []
    margin_regions: List[Dict[str, Any]] = []

    for region in page_elem.findall("page:TextRegion", ns) if ns else page_elem.findall("TextRegion"):
        custom_attr = region.get("custom", "")
        match = re.search(r"type:\s*([^;}]+)", custom_attr)
        if not match:
            continue  # Skip regions without a type
        region_type = match.group(1).strip()

        # Gather full text for the region
        text_parts: List[str] = []
        for line in region.findall(".//page:TextLine", ns) if ns else region.findall(".//TextLine"):
            line_unicode = line.find("page:TextEquiv/page:Unicode", ns) if ns else line.find("TextEquiv/Unicode")
            if line_unicode is not None and line_unicode.text:
                text_parts.append(line_unicode.text.strip())
            else:
                # Fallback to word-level concatenation
                words = [
                    w.find("page:TextEquiv/page:Unicode", ns).text.strip()
                    for w in (line.findall("page:Word", ns) if ns else line.findall("Word"))
                    if w.find("page:TextEquiv/page:Unicode", ns) is not None
                    and w.find("page:TextEquiv/page:Unicode", ns).text
                ]
                if words:
                    text_parts.append(" ".join(words))
        # If region has no <TextLine>, check for text directly on region
        if not text_parts:
            direct = region.find("page:TextEquiv/page:Unicode", ns) if ns else region.find("TextEquiv/Unicode")
            if direct is not None and direct.text:
                text_parts.append(direct.text.strip())

        if not text_parts:
            continue  # skip empty text regions

        entry = {
            "type": region_type,
            "text": " ".join(text_parts),
        }
        (margin_regions if region_type.lower() == "marginalia" else main_regions).append(entry)

    # Non-marginalia first, marginalia afterwards
    return main_regions + margin_regions

# ─── Main driver ──────────────────────────────────────────────────────────────

def process_inventory(inventory_dir: Path, output_root: Path) -> None:
    """Convert all XML files in *inventory_dir/page* to JSON in output_root."""
    page_dir = inventory_dir / "page"
    if not page_dir.is_dir():
        return

    out_dir = output_root / inventory_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    files_processed = 0
    for xml_name in sorted(page_dir.iterdir()):
        if xml_name.suffix.lower() != ".xml":
            continue
        data = extract_data_from_xml(xml_name)
        json_path = out_dir / f"{xml_name.stem}.json"
        try:
            with json_path.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, indent=4, ensure_ascii=False)
            files_processed += 1
        except IOError as exc:
            print(f"[ERROR] Writing {json_path}: {exc}")

    print(
        f"Finished '{inventory_dir.name}': {files_processed} XML → JSON files written to {out_dir}"
    )

# ─── Entry-point ──────────────────────────────────────────────────────────────

def main() -> None:
    input_root = Path(BASE_INPUT_DIRECTORY)
    output_root = Path(BASE_OUTPUT_DIRECTORY)
    if not input_root.is_dir():
        raise SystemExit(f"Input root directory '{input_root}' does not exist.")
    output_root.mkdir(parents=True, exist_ok=True)

    inventories = [d for d in sorted(input_root.iterdir()) if d.is_dir()]
    if not inventories:
        print("No inventory folders found – nothing to do.")
        return

    total = 0
    for inv_dir in inventories:
        process_inventory(inv_dir, output_root)
        total += 1

    print(f"\nBatch complete. Processed {total} inventories.")

if __name__ == "__main__":
    main()
