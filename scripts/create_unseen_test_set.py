#!/usr/bin/env python3
"""
create_unseen_test_set.py  –  Generate test sets from individual unseen inventories
=====================================================================================

This script creates test datasets from individual unseen inventories by scanning 
XML and XMI directories within each inventory folder. Each test set contains data 
from only one inventory, making it suitable for per-inventory evaluation.

**Key features:**
* Processes inventories individually (one test set per inventory)
* Scans inventory-specific XML/XMI subdirectories
* Generates minimal dataset structure with filenames and content  
* Adds public scan URLs for document viewing
* No dependency on pre-existing CSV annotation files
* Works with new directory structure: /inventories/{inventory_id}/{xmi,xml}/

Update `BASE_DIR` and `OUTPUT_DIR` below as needed, then run:

```bash
python create_unseen_test_set.py [inventory_id]
```

If no inventory_id is provided, processes all inventories found.
Requires Python ≥ 3.9 and `pandas`.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Set, Tuple, List
from collections import defaultdict

import pandas as pd

###############################################################################
# Hard-coded configuration – edit these!                                      #
###############################################################################

BASE_DIR: Path = Path("/Volumes/T7/Python_Projects/Document Segmentation using NER & Visual Features (Dataset creation)/data/Unseen/Unseen Inventories")
OUTPUT_DIR: Path = Path("/Volumes/T7/Python_Projects/Document Segmentation using NER & Visual Features (Dataset creation)/data/Unseen")

FORCE_OVERWRITE = False  # set True to overwrite existing output files

###############################################################################
# Helper utilities (adapted from 1_merge_all.py)                             #
###############################################################################

_split = re.compile(r"[_-]")  # handles `_` or `-` as separators

def _parse_base_name(filename: str) -> Tuple[str, str, str, str]:
    """Parse a page filename into (archive_code, inventory_id, page_num, base).
    
    *Tolerant version* – accepts filenames where the page number may carry a 
    letter suffix (e.g. "0009r" or "0123b") and where the inventory id is the 
    numeric segment immediately before that page segment.  The rest (everything
    to the left) is treated as the archive code, which can itself contain „-"
    or „_" as in "NL-HaNA_1.04.02".
    """
    # Strip a known extension **only** if it truly looks like one; otherwise
    # keep the full string so dots inside archive codes ("1.04.02") are not lost.
    if re.search(r"\.(xml|xmi|tif|tiff|jpg|jpeg|png)$", filename, flags=re.I):
        base = filename.rsplit(".", 1)[0]
    else:
        base = filename
    base = base.strip()
    parts = _split.split(base)

    # Walk from the right‑hand side to find the first segment that *starts*
    # with digits – this is the page identifier (with optional letter suffix).

    page_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if re.match(r"^\d+[A-Za-z]*$", parts[i]):
            page_idx = i
            break
    if page_idx is None or page_idx == 0:
        raise ValueError(f"Filename '{filename}' does not contain a page number segment.")

    page_seg = parts[page_idx]
    inv_seg = parts[page_idx - 1]

    # Extract pure digits for page number; keep inventory id as‑is (digits expected)
    page_num = re.match(r"^\d+", page_seg).group(0)
    if not inv_seg.isdigit():
        raise ValueError(f"Filename '{filename}' – inventory id segment '{inv_seg}' is not all digits.")

    archive_code = "_".join(parts[: page_idx - 1])  # may be empty

    return archive_code, inv_seg, page_num, base

def build_scan_url(archive_code: str, inventory_id: str, base_name: str) -> str:
    """Return the public viewer URL for a page scan."""
    return (
        "https://www.nationaalarchief.nl/onderzoeken/archief/"
        + archive_code
        + "/invnr/"
        + inventory_id
        + "/file/"
        + base_name
    )

def _first_existing(*candidates: Path) -> Path | None:
    """Return the first path that exists from the candidate list, else None."""
    for p in candidates:
        if p.exists():
            return p
    return None

###############################################################################
# Core processing                                                             #
###############################################################################

def discover_pages_for_inventory(inventory_dir: Path, inventory_id: str) -> Set[Tuple[str, str, str, str]]:
    """Discover all available pages for a specific inventory.
    
    Args:
        inventory_dir: Path to the inventory directory containing xml/ and xmi/ subdirs
        inventory_id: The inventory ID for validation
    
    Returns:
        Set of tuples: (archive_code, inventory_id, page_num, base_name)
    """
    pages = set()
    xml_dir = inventory_dir / "xml"
    xmi_dir = inventory_dir / "xmi"
    
    print(f"  Discovering pages from XML directory: {xml_dir}")
    if xml_dir.exists():
        for xml_file in xml_dir.glob("*.xml"):
            try:
                base_name = xml_file.stem
                archive_code, inv_id, page_num, base = _parse_base_name(base_name)
                # Validate that this file belongs to the expected inventory
                if inv_id == inventory_id:
                    pages.add((archive_code, inv_id, page_num, base))
                else:
                    print(f"[WARN] XML file {xml_file.name} inventory ID {inv_id} doesn't match expected {inventory_id}")
            except ValueError as e:
                print(f"[WARN] Could not parse XML file {xml_file.name}: {e}", file=sys.stderr)
                continue
    else:
        print(f"[WARN] XML directory not found: {xml_dir}")
    
    print(f"  Discovering pages from XMI directory: {xmi_dir}")  
    if xmi_dir.exists():
        for xmi_file in xmi_dir.glob("*.xmi"):
            try:
                base_name = xmi_file.stem
                archive_code, inv_id, page_num, base = _parse_base_name(base_name)
                # Validate that this file belongs to the expected inventory
                if inv_id == inventory_id:
                    pages.add((archive_code, inv_id, page_num, base))
                else:
                    print(f"[WARN] XMI file {xmi_file.name} inventory ID {inv_id} doesn't match expected {inventory_id}")
            except ValueError as e:
                print(f"[WARN] Could not parse XMI file {xmi_file.name}: {e}", file=sys.stderr)
                continue
    else:
        print(f"[WARN] XMI directory not found: {xmi_dir}")
    
    return pages


def create_test_dataset_for_inventory(
    pages: Set[Tuple[str, str, str, str]], 
    inventory_dir: Path,
    output_path: Path,
    inventory_id: str
) -> None:
    """Create test dataset from discovered pages for a single inventory."""
    
    # Sort pages for consistent output
    sorted_pages = sorted(pages, key=lambda x: int(x[2]))  # sort by page_num
    
    print(f"  Creating dataset with {len(sorted_pages)} pages for inventory {inventory_id}...")
    
    # Initialize dataset structure
    data = []
    counters = dict(xml_found=0, xml_missing=0, xmi_found=0, xmi_missing=0, total_pages=0)
    
    xml_dir = inventory_dir / "xml"
    xmi_dir = inventory_dir / "xmi"
    
    for archive_code, inv_id, page_num, base_name in sorted_pages:
        counters["total_pages"] += 1
        
        if counters["total_pages"] % 500 == 0:
            print(f"    Processed {counters['total_pages']} pages...")
        
        # Find XML and XMI files directly in inventory subdirectories
        xml_path = xml_dir / f"{base_name}.xml" if xml_dir.exists() else None
        xmi_path = xmi_dir / f"{base_name}.xmi" if xmi_dir.exists() else None
        
        # Read file contents
        xml_data = ""
        xmi_data = ""
        
        if xml_path and xml_path.exists():
            try:
                xml_data = xml_path.read_text(encoding="utf-8", errors="ignore")
                counters["xml_found"] += 1
            except Exception as exc:
                print(f"[ERROR] Reading XML {xml_path}: {exc}", file=sys.stderr)
        else:
            counters["xml_missing"] += 1
        
        if xmi_path and xmi_path.exists():
            try:
                xmi_data = xmi_path.read_text(encoding="utf-8", errors="ignore")
                counters["xmi_found"] += 1
            except Exception as exc:
                print(f"[ERROR] Reading XMI {xmi_path}: {exc}", file=sys.stderr)
        else:
            counters["xmi_missing"] += 1
        
        # Create scan URL
        scan_url = build_scan_url(archive_code, inv_id, base_name)
        
        # Add to dataset
        row = {
            "Scan File_Name": f"{base_name}.xml",  # Use standard column name
            "archive_code": archive_code,
            "inventory_id": inv_id, 
            "page_num": page_num,
            "base_name": base_name,
            "xml_data": xml_data,
            "xmi_data": xmi_data,
            "scan_url": scan_url,
        }
        data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"  Test dataset created for inventory {inventory_id} ✔")
    print(f"     Total pages:          {counters['total_pages']}")
    print(f"     XML found:            {counters['xml_found']}")
    print(f"     XML missing:          {counters['xml_missing']}")
    print(f"     XMI found:            {counters['xmi_found']}")
    print(f"     XMI missing:          {counters['xmi_missing']}")
    print(f"     Output → {output_path}")

def process_single_inventory(base_dir: Path, output_dir: Path, inventory_id: str, force: bool) -> None:
    """Process a single inventory to create its test dataset."""
    
    inventory_dir = base_dir / inventory_id
    if not inventory_dir.exists():
        raise RuntimeError(f"Inventory directory does not exist: {inventory_dir}")
    
    output_path = output_dir / f"unseen_test_set_{inventory_id}.csv"
    
    if output_path.exists() and not force:
        print(f"[SKIP] '{output_path}' already exists – enable FORCE_OVERWRITE to replace.")
        return
    
    print(f"Processing inventory {inventory_id}...")
    
    # Discover pages for this inventory
    pages = discover_pages_for_inventory(inventory_dir, inventory_id)
    
    if not pages:
        print(f"[WARN] No valid pages found for inventory {inventory_id}")
        return
    
    print(f"  Found {len(pages)} pages for inventory {inventory_id}")
    
    # Create the dataset for this inventory
    create_test_dataset_for_inventory(pages, inventory_dir, output_path, inventory_id)

def process_all_inventories(base_dir: Path, output_dir: Path, force: bool) -> None:
    """Process all inventories found in the base directory."""
    
    if not base_dir.exists():
        raise RuntimeError(f"Base directory does not exist: {base_dir}")
    
    # Find all inventory directories (directories with numeric names)
    inventory_dirs = [d for d in base_dir.iterdir() 
                     if d.is_dir() and d.name.isdigit()]
    
    if not inventory_dirs:
        raise RuntimeError(f"No inventory directories found in {base_dir}")
    
    inventory_dirs.sort(key=lambda x: int(x.name))  # Sort numerically
    
    print(f"Found {len(inventory_dirs)} inventories to process:")
    for inv_dir in inventory_dirs:
        print(f"  - {inv_dir.name}")
    print()
    
    # Process each inventory
    for inv_dir in inventory_dirs:
        try:
            process_single_inventory(base_dir, output_dir, inv_dir.name, force)
        except Exception as e:
            print(f"[ERROR] Failed to process inventory {inv_dir.name}: {e}")
            continue
    
    print(f"\nProcessing complete! Output files saved to: {output_dir}")

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate test sets from individual unseen inventories",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inventory_id", 
        nargs="?", 
        help="Specific inventory ID to process (if not provided, processes all)"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help=f"Base directory containing inventory folders (default: {BASE_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path, 
        default=OUTPUT_DIR,
        help=f"Output directory for test sets (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Override force setting if specified via argument
    force = args.force or FORCE_OVERWRITE
    
    if args.inventory_id:
        # Process specific inventory
        process_single_inventory(args.base_dir, args.output_dir, args.inventory_id, force)
    else:
        # Process all inventories
        process_all_inventories(args.base_dir, args.output_dir, force)

###############################################################################
# Entry-point                                                                 #
###############################################################################

if __name__ == "__main__":
    main()