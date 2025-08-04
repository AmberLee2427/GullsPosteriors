#!/usr/bin/env python
"""Convert a space-delimited master file to a comma-separated CSV.

Usage
-----
python scripts/space_to_csv.py  path/to/input.out  [--output path/to/output.csv]

The script uses pandas with ``delim_whitespace=True`` so any amount of
spaces / tabs between columns is accepted.  The first non-comment line is
assumed to be the header.  If *output* is omitted the CSV is written next
to the input file with the same base-name but ``.csv`` extension.
"""
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description="Convert space-delimited file to CSV for spreadsheet inspection.")
parser.add_argument("input_file", help="Path to the .out text file (space-delimited)")
parser.add_argument("--output", "-o", help="Destination CSV path (optional)")
args = parser.parse_args()

in_path = Path(args.input_file)
if not in_path.exists():
    parser.error(f"Input file '{in_path}' does not exist.")

out_path = Path(args.output) if args.output else in_path.with_suffix(".csv")

print(f"Reading '{in_path}' …")

df = pd.read_csv(in_path, delim_whitespace=True, comment="#", engine="python")

print(f"Detected {df.shape[1]} columns and {df.shape[0]} rows.")
print(f"Saving to '{out_path}' …")

df.to_csv(out_path, index=False)

print("Done.") 