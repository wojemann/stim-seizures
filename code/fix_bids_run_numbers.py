#!/usr/bin/env python3
"""
Script to fix BIDS run numbers from single-digit to zero-padded format.
Changes '_run-0_' to '_run-00_', '_run-1_' to '_run-01_', etc. in filenames.
If the target file already exists, removes the single-digit version.
"""

import os
import re
from pathlib import Path
from config import Config

# ============================================
# SET TO False TO ACTUALLY PERFORM OPERATIONS
DRY_RUN = False
# ============================================

# Loading CONFIG to get datapath
datapath = Config.deal(['datapath'])
bids_root = Path(datapath) / 'BIDS'
print(datapath)
print(f"Searching in: {bids_root}")
if DRY_RUN:
    print("\n*** DRY RUN MODE - No files will be modified ***\n")
else:
    print("\n*** LIVE MODE - Files will be modified ***\n")

# Track statistics
renamed_count = 0
deleted_count = 0
skipped_count = 0

# Collect examples for dry run
rename_examples = []
delete_examples = []

# Walk through the BIDS directory
for root, dirs, files in os.walk(bids_root):
    for filename in files:
        # Check if this file has a single-digit run number pattern
        # Match pattern like '_run-0_', '_run-1_', etc. but not '_run-00_', '_run-01_'
        match = re.search(r'_run-(\d)_', filename)
        if match:
            old_path = Path(root) / filename
            # Create the new filename with zero-padded run number
            run_digit = match.group(1)
            new_filename = filename.replace(f'_run-{run_digit}_', f'_run-0{run_digit}_')
            new_path = Path(root) / new_filename
            
            # Check if the target file already exists
            if new_path.exists():
                deleted_count += 1
                if DRY_RUN:
                    if len(delete_examples) < 5:
                        delete_examples.append((old_path.relative_to(bids_root), new_path.relative_to(bids_root)))
                else:
                    print(f"DELETE (target exists): {old_path.relative_to(bids_root)}")
                    try:
                        old_path.unlink()
                    except Exception as e:
                        print(f"  ERROR deleting: {e}")
                        skipped_count += 1
            else:
                renamed_count += 1
                if DRY_RUN:
                    if len(rename_examples) < 5:
                        rename_examples.append((old_path.relative_to(bids_root), new_filename))
                else:
                    print(f"RENAME: {old_path.relative_to(bids_root)}")
                    print(f"    --> {new_filename}")
                    try:
                        old_path.rename(new_path)
                    except Exception as e:
                        print(f"  ERROR renaming: {e}")
                        skipped_count += 1

print("\n" + "="*60)
print("Summary:")
print(f"  Files to rename: {renamed_count}")
print(f"  Files to delete: {deleted_count}")
print(f"  Files skipped (errors): {skipped_count}")
print("="*60)

if DRY_RUN:
    print("\nDRY RUN EXAMPLES:")
    print("-" * 60)
    
    if rename_examples:
        print("\nFILES TO RENAME (showing up to 3 examples):")
        for old_file, new_filename in rename_examples:
            print(f"  {old_file}")
            print(f"    --> {new_filename}")
    else:
        print("\nNo files need to be renamed.")
    
    if delete_examples:
        print("\nFILES TO DELETE (duplicate, showing up to 3 examples):")
        for old_file, existing_file in delete_examples:
            print(f"  DELETE: {old_file}")
            print(f"  (already exists as: {existing_file})")
    else:
        print("\nNo duplicate files to delete.")
    
    print("\n" + "="*60)
    print("To perform these operations, set DRY_RUN = False in the script")
    print("="*60)

