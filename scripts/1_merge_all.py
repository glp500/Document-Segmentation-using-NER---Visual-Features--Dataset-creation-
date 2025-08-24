#!/usr/bin/env python3
"""
build_renate_dataset.py  –  *hard-coded paths*
================================================

This standalone script merges every Renate inventory CSV, adds the matching
XML/XMI paths plus a public `scan_url`, and writes a single master file.

**Key changes (2025-06-30):**
* Detects CSV delimiter automatically (`;`, `,`, `\t`, etc.), so inventories with
  semicolon-separated columns load correctly.
* Strips a potential UTF-8 BOM (\ufeff) from header names.
* Expanded filename-column detection to handle the most common variants.

Update `CSV_DIR`, `XML_DIR`, `XMI_DIR`, and `OUTPUT_PATH` below as needed, then
run:

```bash
python build_renate_dataset.py
```

Requires Python ≥ 3.9 and `pandas`.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

###############################################################################
# Hard-coded configuration – edit these!                                      #
###############################################################################

CSV_DIR: Path = Path("/Users/gavinl/Desktop/Renate Inventories/Renate Annotations")
XML_DIR: Path = Path("/Users/gavinl/Desktop/Renate Inventories/xml")
XMI_DIR: Path = Path("/Users/gavinl/Desktop/Renate Inventories/xmi")
OUTPUT_PATH: Path = Path("/Users/gavinl/Desktop/Renate Inventories/renate_merged_pages.csv")

CHUNKSIZE = 50_000      # rows streamed at once
FORCE_OVERWRITE = False # set True to overwrite an existing OUTPUT_PATH

###############################################################################
# Helper utilities                                                            #
###############################################################################

def _detect_filename_column(columns: Iterable[str]) -> str | None:
    """Return the column holding the original page filename, if any.

    Known variants: *Scan File_Name*, *Scan FileName*, *xml_file_name*, etc.
    The function lowers case, replaces spaces with underscores, strips BOMs and
    compares against a curated list.
    """
    targets = {
        "scan_file_name",
        "scanfilename",
        "xml_file_name",
        "file_name",
        "scanfile",
    }
    for original in columns:
        norm = original.lower().replace(" ", "_").lstrip("\ufeff")
        if norm in targets:
            return original
    return None

_split = re.compile(r"[_-]")  # handles `_` or `-` as separators

def _parse_base_name(filename: str) -> Tuple[str, str, str, str]:
    """Parse a page filename into (archive_code, inventory_id, page_num, base).

    *Tolerant version* – accepts filenames where the page number may carry a 
    letter suffix (e.g. “0009r” or “0123b”) and where the inventory id is the 
    numeric segment immediately before that page segment.  The rest (everything
    to the left) is treated as the archive code, which can itself contain „-”
    or „_” as in “NL-HaNA_1.04.02”.
    """
    # Strip a known extension **only** if it truly looks like one; otherwise
    # keep the full string so dots inside archive codes (“1.04.02”) are not lost.
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

def _reader(csv_path: Path, chunksize: int):
    """Yield chunks from *csv_path* with automatic delimiter sniffing."""
    return pd.read_csv(
        csv_path,
        sep=None,            # let pandas sniff ',' ';' '\t' ...
        engine="python",     # required when sep=None or unusual delims
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    )


def process_csv(
    csv_path: Path,
    xml_dir: Path,
    xmi_dir: Path,
    chunksize: int,
    output_path: Path,
    mode: str,
    counters: dict[str, int],
) -> None:
    """Stream *csv_path* in *chunksize* blocks and append results to *output_path*."""

    filename_col: str | None = None

    for chunk in _reader(csv_path, chunksize):
        # Detect the filename column once per CSV
        if filename_col is None:
            filename_col = _detect_filename_column(chunk.columns)
            if filename_col is None:
                raise RuntimeError(f"{csv_path.name}: no recognised filename column found.")

        # Ensure new columns exist
        chunk["xml_data"] = ""
        chunk["xmi_data"] = ""
        chunk["scan_url"] = ""

        for idx, row in chunk.iterrows():
            counters["rows"] += 1
            xml_file = row[filename_col]

            try:
                archive_code, inv_id, page_num, base_name = _parse_base_name(xml_file)
            except ValueError as err:
                print(f"[WARN] {err}", file=sys.stderr)
                counters["filename_parse_errors"] += 1
                continue

            # All plausible XML/XMI locations (support 'inventory 1120' folder etc.)
            xml_candidates = (
                xml_dir / inv_id / "page" / f"{base_name}.xml",
                xml_dir / f"inventory {inv_id}" / "page" / f"{base_name}.xml",
                xml_dir / f"inventory_{inv_id}" / "page" / f"{base_name}.xml",
                xml_dir / inv_id / f"{base_name}.xml",
                xml_dir / f"inventory {inv_id}" / f"{base_name}.xml",
            )
            xmi_candidates = (
                xmi_dir / inv_id / f"{base_name}.xmi",
                xmi_dir / f"inventory {inv_id}" / f"{base_name}.xmi",
                xmi_dir / f"inventory_{inv_id}" / f"{base_name}.xmi",
            )

            xml_path = _first_existing(*xml_candidates)
            xmi_path = _first_existing(*xmi_candidates)

            if xml_path is not None:
                try:
                    chunk.at[idx, "xml_data"] = xml_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as exc:
                    print(f"[ERROR] Reading XML {xml_path}: {exc}", file=sys.stderr)
                counters["xml_found"] += 1
            else:
                print(f"[ERROR] Missing XML for base '{base_name}' (inventory {inv_id})", file=sys.stderr)
                counters["xml_missing"] += 1

            if xmi_path is not None:
                try:
                    chunk.at[idx, "xmi_data"] = xmi_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as exc:
                    print(f"[ERROR] Reading XMI {xmi_path}: {exc}", file=sys.stderr)
                counters["xmi_found"] += 1
            else:
                counters["xmi_missing"] += 1

            # Public URL
            chunk.at[idx, "scan_url"] = build_scan_url(archive_code, inv_id, base_name)

        # Append to output
        chunk.to_csv(output_path, mode=mode, index=False, header=(mode == "w"))
        mode = "a"  # subsequent writes append


def merge_inventories(
    csv_dir: Path,
    xml_dir: Path,
    xmi_dir: Path,
    output_path: Path,
    chunksize: int,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"[ABORT] '{output_path}' exists – enable FORCE_OVERWRITE to replace.")
        sys.exit(1)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {csv_dir}")

    counters = dict(rows=0, xml_found=0, xml_missing=0, xmi_found=0, xmi_missing=0, filename_parse_errors=0)
    mode = "w"
    for csv_path in csv_files:
        print(f"→ {csv_path.name}")
        process_csv(csv_path, xml_dir, xmi_dir, chunksize, output_path, mode, counters)
        mode = "a"

    # Summary
    print("\nDone ✔")
    print(f"   Rows processed:       {counters['rows']}")
    print(f"   XML found:            {counters['xml_found']}")
    print(f"   XML missing:          {counters['xml_missing']}")
    print(f"   XMI found:            {counters['xmi_found']}")
    print(f"   XMI missing:          {counters['xmi_missing']}")
    if counters['filename_parse_errors']:
        print(f"   Filename parse errors: {counters['filename_parse_errors']}")
    print(f"   Output → {output_path}")

###############################################################################
# Entry-point                                                                 #
###############################################################################

if __name__ == "__main__":
    merge_inventories(
        csv_dir=CSV_DIR,
        xml_dir=XML_DIR,
        xmi_dir=XMI_DIR,
        output_path=OUTPUT_PATH,
        chunksize=CHUNKSIZE,
        force=FORCE_OVERWRITE,
    )
