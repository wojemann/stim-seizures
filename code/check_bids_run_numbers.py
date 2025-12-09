#!/usr/bin/env python3
"""
Diagnostic script to find BIDS files that still have single-digit run numbers.
This helps identify which files need to be fixed.
"""

import os
import re
from pathlib import Path
from config import Config

# Loading CONFIG to get datapath
datapath = Config.deal(['datapath'])
bids_root = Path(datapath) / 'BIDS'
print(f"Data path: {datapath}")
print(f"Searching in: {bids_root}\n")

# Track files that need fixing
files_with_single_digit_run = []
files_with_zero_padded_run = []

# Walk through the BIDS directory
for root, dirs, files in os.walk(bids_root):
    for filename in files:
        file_path = Path(root) / filename
        
        # Check for single-digit run number (needs fixing)
        single_digit_match = re.search(r'_run-(\d)([_.])', filename)
        if single_digit_match:
            run_digit = single_digit_match.group(1)
            # Check if it's actually single digit (0-9) and not already zero-padded
            if len(run_digit) == 1:
                files_with_single_digit_run.append(file_path.relative_to(bids_root))
        
        # Also track zero-padded for comparison
        zero_padded_match = re.search(r'_run-(\d{2})([_.])', filename)
        if zero_padded_match:
            files_with_zero_padded_run.append(file_path.relative_to(bids_root))

print("="*60)
print("FILES WITH SINGLE-DIGIT RUN NUMBERS (need fixing):")
print("="*60)
if files_with_single_digit_run:
    for file_path in sorted(files_with_single_digit_run):
        print(f"  {file_path}")
    print(f"\nTotal: {len(files_with_single_digit_run)} files need fixing")
else:
    print("  ✓ No files with single-digit run numbers found!")
    print("  All files appear to be correctly formatted.")

print("\n" + "="*60)
print("FILES WITH ZERO-PADDED RUN NUMBERS (already correct):")
print("="*60)
print(f"  Total: {len(files_with_zero_padded_run)} files")

# Check for specific problematic file mentioned in error
print("\n" + "="*60)
print("Checking for specific file mentioned in error:")
print("="*60)
problem_file = "ieeg/sub-HUP097_ses-clinical01_task-ictal349667_run-00_ieeg.edf"
problem_file_path = bids_root / problem_file
if problem_file_path.exists():
    print(f"  ✓ Found: {problem_file_path}")
else:
    print(f"  ✗ Not found: {problem_file_path}")
    # Check for run-0 version
    alt_file = problem_file.replace("run-00", "run-0")
    alt_file_path = bids_root / alt_file
    if alt_file_path.exists():
        print(f"  ⚠ Found with run-0: {alt_file_path}")
        print(f"     This file needs to be renamed!")

print("\n" + "="*60)

