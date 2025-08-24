

"""
xmi_viewer.py

Walk through a directory of UIMA‑style XMI files, collect every Named‑Entity
annotation, resolve its character offsets to the original document text, and
store the results in a CSV.

Usage
-----
    python xmi_viewer.py /path/to/xmi_dir  [--output named_entities.csv]

If no directory is passed the script defaults to the current working dir.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
# ⚠️  Change this to the folder that holds your .xmi files.
HARDCODED_DIR = '/Users/gavinl/Desktop/NER Inventories/Renate/1120'

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────


def strip_namespace(tag: str) -> str:
    """Return the local part of an XML tag – e.g. '{uri}Foo' ➜ 'Foo'."""
    return tag.split("}", 1)[-1]


def get_attr(attrs: Dict[str, str], local_name: str) -> Optional[str]:
    """Namespace‑insensitive attribute lookup."""
    for k, v in attrs.items():
        if k.split("}", 1)[-1] == local_name:
            return v
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Core parsing logic
# ──────────────────────────────────────────────────────────────────────────────


def parse_xmi(path: Path) -> List[Dict[str, str | int]]:
    """
    Extract all NamedEntity annotations from *one* XMI file.

    Returns
    -------
    rows : list of dictionaries ready for the Pandas DataFrame constructor
    """
    rows: List[Dict[str, str | int]] = []

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        sys.stderr.write(f"[WARN] Parse error in {path.name}: {exc}\n")
        return rows

    root = tree.getroot()

    # 1) Grab the sofaString (document text)
    sofa_text: Optional[str] = None
    for elem in root.iter():
        if strip_namespace(elem.tag) == "Sofa":
            sofa_text = get_attr(elem.attrib, "sofaString")
            # Use the *first* Sofa element found
            if sofa_text is not None:
                break

    # 2) Iterate through NamedEntity / EntityMention elements
    for elem in root.iter():
        tag_clean = strip_namespace(elem.tag)
        if tag_clean not in {"NamedEntity", "EntityMention"}:
            continue

        begin = get_attr(elem.attrib, "begin")
        end = get_attr(elem.attrib, "end")
        value = get_attr(elem.attrib, "value")

        # Convert begin/end to integers when possible
        try:
            begin_int = int(begin) if begin is not None else None
            end_int = int(end) if end is not None else None
        except ValueError:
            begin_int = end_int = None

        text_span = (
            sofa_text[begin_int:end_int] if sofa_text and begin_int is not None and end_int is not None else None
        )

        rows.append(
            {
                "file": path.name,
                "entity_type": value,
                "begin": begin_int,
                "end": end_int,
                "text": text_span,
            }
        )

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# CLI wrapper
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract named entities from XMI files.")
    parser.add_argument(
        "--output",
        "-o",
        default="named_entities.csv",
        help="CSV file to write results to (default: named_entities.csv)",
    )
    args = parser.parse_args()

    xmi_dir = Path(HARDCODED_DIR).expanduser().resolve()
    if not xmi_dir.is_dir():
        sys.exit(f"[ERROR] '{xmi_dir}' is not a directory.")

    all_rows: List[Dict[str, str | int]] = []
    for file in sorted(xmi_dir.iterdir()):
        if file.suffix.lower() == ".xmi":
            all_rows.extend(parse_xmi(file))

    if not all_rows:
        sys.exit("[INFO] No NamedEntity annotations found.")

    df = pd.DataFrame(all_rows)

    # Print a quick preview and save to CSV
    print(df.head())
    df.to_csv(args.output, index=False)
    print(f"[OK] Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()