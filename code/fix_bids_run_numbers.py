#!/usr/bin/env python3
"""
Script to fix BIDS run numbers from single-digit to zero-padded format.
Changes single-digit run numbers to two-digit zero-padded format:
- '_run-0_' -> '_run-00_'
- '_run-1_' -> '_run-01_'
- '_run-2_' -> '_run-02_'
- '_run-3_' -> '_run-03_'
- ... up to '_run-9_' -> '_run-09_'

If the target file already exists, removes the single-digit version.

This script handles ALL BIDS file types:
- .edf (main data files)
- .edf.json (sidecar JSON metadata)
- .channels.tsv (channel information)
- .events.tsv (events file)
- Any other files with the run number pattern

This script is idempotent - it can be run multiple times safely.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from config import Config

# Try to import mne_bids to check what it sees
try:
    from mne_bids import get_entity_vals
    MNE_BIDS_AVAILABLE = True
except ImportError:
    MNE_BIDS_AVAILABLE = False
    print("Note: mne_bids not available, skipping mne_bids checks")

# ============================================
# SET TO False TO ACTUALLY PERFORM OPERATIONS
DRY_RUN = True
# ============================================

# Loading CONFIG to get datapath
datapath = Config.deal(['datapath'])
bids_root = Path(datapath) / 'BIDS'

# Also construct the path the same way run_NDD_seizure_annotations_v2.py does
from os.path import join as ospj
bids_root_string = ospj(datapath, "BIDS")
bids_root_from_string = Path(bids_root_string)

print(f"Data path: {datapath}")
print(f"BIDS root (Path): {bids_root}")
print(f"BIDS root (ospj): {bids_root_string}")
print(f"Paths match: {str(bids_root) == str(bids_root_from_string)}")
print(f"BIDS directory exists: {bids_root.exists()}")

# Debug: Check if directory exists and list some files
if not bids_root.exists():
    print(f"⚠ WARNING: BIDS directory does not exist at {bids_root}")
    print("Please check your config.py datapath setting.")
else:
    # Count total files and show a few examples
    try:
        all_files = list(bids_root.rglob('*'))
        files_only = [f for f in all_files if f.is_file()]
        print(f"Total files found in BIDS directory: {len(files_only)}")
        if files_only:
            print(f"Sample files (first 5):")
            for f in files_only[:5]:
                print(f"  {f.relative_to(bids_root)}")
            # Check for files with single-digit run pattern (using same regex as processing)
            run_single_digit_files = [f for f in files_only if re.search(r'_run-(\d)([_.])', f.name)]
            print(f"\nFiles matching '_run-(\\d)([_.])' pattern (single digit): {len(run_single_digit_files)}")
            if run_single_digit_files:
                print("Sample matches:")
                for f in run_single_digit_files[:10]:  # Show more examples
                    print(f"  {f.relative_to(bids_root)}")
            # Also check for zero-padded files for comparison
            run_zero_padded_files = [f for f in files_only if re.search(r'_run-(\d{2})([_.])', f.name)]
            print(f"\nFiles matching '_run-(\\d{{2}})([_.])' pattern (zero-padded): {len(run_zero_padded_files)}")
            
            # Check what mne_bids sees for run numbers
            if MNE_BIDS_AVAILABLE:
                try:
                    print(f"\nChecking what mne_bids sees:")
                    runs_seen = get_entity_vals(str(bids_root), 'run')
                    print(f"  Run values found by mne_bids: {sorted(runs_seen)[:20]}")  # Show first 20
                    # Check for single-digit runs
                    single_digit_runs = [r for r in runs_seen if len(r) == 1]
                    if single_digit_runs:
                        print(f"  ⚠ Single-digit run values found: {single_digit_runs}")
                        print(f"     These need to be zero-padded!")
                    else:
                        print(f"  ✓ All run values are zero-padded")
                    
                    # Test BIDSPath construction for one of the problematic files
                    try:
                        from mne_bids import BIDSPath
                        test_path = BIDSPath(
                            root=str(bids_root),
                            subject='HUP097',
                            session='clinical01',
                            task='ictal349667',
                            run='00',
                            datatype='ieeg',
                            suffix='ieeg',
                            extension='.edf'
                        )
                        print(f"\n  Testing BIDSPath construction:")
                        print(f"    Constructed path: {test_path.fpath}")
                        print(f"    Path exists: {test_path.fpath.exists()}")
                        # Check what files actually exist in that directory
                        if test_path.fpath.parent.exists():
                            actual_files = list(test_path.fpath.parent.glob('*ictal349667*run-*'))
                            print(f"    Files in directory matching pattern:")
                            for f in actual_files[:5]:
                                print(f"      {f.name}")
                    except Exception as e:
                        print(f"  Error testing BIDSPath: {e}")
                except Exception as e:
                    print(f"  Error querying mne_bids: {e}")
            
            # Also check specifically for files in ieeg subdirectories
            ieeg_files = [f for f in files_only if 'ieeg' in str(f.relative_to(bids_root)) and f.suffix == '.edf']
            print(f"\nTotal .edf files in 'ieeg' subdirectories: {len(ieeg_files)}")
            if ieeg_files:
                # Check how many have single-digit run numbers
                ieeg_single_digit = [f for f in ieeg_files if re.search(r'_run-(\d)([_.])', f.name)]
                print(f"  Files with single-digit run numbers: {len(ieeg_single_digit)}")
                if ieeg_single_digit:
                    print("  Sample files with single-digit run:")
                    for f in ieeg_single_digit[:5]:
                        print(f"    {f.relative_to(bids_root)}")
            
            # Check for specific files mentioned in error messages
            print(f"\nChecking for specific files from error messages:")
            error_file_patterns = [
                'sub-HUP098_ses-clinical01_task-ictal107365_run-0_ieeg.edf',
                'sub-HUP097_ses-clinical01_task-ictal349667_run-0_ieeg.edf',
                'sub-HUP098_ses-clinical01_task-ictal200024_run-0_ieeg.edf',
            ]
            for pattern in error_file_patterns:
                # Check for run-0 version
                matching_files = [f for f in files_only if pattern in f.name]
                if matching_files:
                    print(f"  ✓ Found run-0 version: {pattern}")
                    for f in matching_files:
                        print(f"      Path: {f.relative_to(bids_root)}")
                else:
                    print(f"  ✗ NOT found run-0: {pattern}")
                    # Check for run-00 version
                    pattern_00 = pattern.replace('_run-0_', '_run-00_')
                    matching_files_00 = [f for f in files_only if pattern_00 in f.name]
                    if matching_files_00:
                        print(f"  ✓ Found run-00 version: {pattern_00}")
                        for f in matching_files_00:
                            print(f"      Path: {f.relative_to(bids_root)}")
                    # Try to find ALL related files (both run-0 and run-00)
                    base_pattern = pattern.split('_run-')[0]
                    similar = [f for f in files_only if base_pattern in f.name and 'run-' in f.name]
                    if similar:
                        print(f"      All related files found ({len(similar)} total):")
                        # Group by run number
                        run_0_files = [f for f in similar if re.search(r'_run-0([^0]|$)', f.name)]
                        run_00_files = [f for f in similar if re.search(r'_run-00', f.name)]
                        if run_0_files:
                            print(f"        Files with run-0 (single digit): {len(run_0_files)}")
                            for f in run_0_files[:3]:
                                print(f"          {f.relative_to(bids_root)}")
                        if run_00_files:
                            print(f"        Files with run-00 (zero-padded): {len(run_00_files)}")
                            for f in run_00_files[:3]:
                                print(f"          {f.relative_to(bids_root)}")
    except Exception as e:
        print(f"Error scanning directory: {e}")

# Test regex pattern with example filenames from error messages
test_filenames = [
    'sub-HUP098_ses-clinical01_task-ictal107365_run-0_ieeg.edf',
    'sub-HUP097_ses-clinical01_task-ictal349667_run-0_ieeg.edf',
    'sub-HUP098_ses-clinical01_task-ictal200024_run-0_ieeg.edf',
]
print("\nTesting regex pattern on example filenames:")
for test_name in test_filenames:
    match = re.search(r'_run-(\d)([_.])', test_name)
    if match:
        new_name = re.sub(r'_run-(\d)([_.])', rf'_run-0\1\2', test_name, count=1)
        print(f"  ✓ '{test_name}' -> '{new_name}'")
    else:
        print(f"  ✗ '{test_name}' - NO MATCH (this is a problem!)")

if DRY_RUN:
    print("\n*** DRY RUN MODE - No files will be modified ***\n")
else:
    print("\n*** LIVE MODE - Files will be modified ***\n")

# Track statistics
renamed_count = 0
deleted_count = 0
skipped_count = 0
already_correct_count = 0

# Group files by base name for better reporting
files_by_base = defaultdict(list)

# Collect examples for dry run
rename_examples = []
delete_examples = []
error_examples = []

# First pass: collect all files that need fixing
files_to_process = []
total_files_scanned = 0
# Also track files that might match but need verification
potential_matches = []

for root, dirs, files in os.walk(bids_root):
    for filename in files:
        total_files_scanned += 1
        # Check if this file has a single-digit run number pattern
        # Match pattern like '_run-0_', '_run-1_', '_run-2_', ..., '_run-9_'
        # but NOT '_run-00_', '_run-01_', etc. (already zero-padded)
        # Match single digit (0-9) followed by underscore or dot (for file extensions)
        match = re.search(r'_run-(\d)([_.])', filename)
        if match:
            # Verify it's actually a single digit (not part of a two-digit number)
            run_digit = match.group(1)
            # Check that the character before 'run-' is not a digit (to avoid matching run-10, run-20, etc.)
            # and that we're not matching run-00, run-01, etc.
            prev_char_idx = filename.find(f'_run-{run_digit}')
            if prev_char_idx > 0:
                prev_char = filename[prev_char_idx - 1]
                # If previous char is a digit, this might be part of a larger number
                if prev_char.isdigit():
                    continue
            
            # Double-check: make sure we're not matching run-00, run-01, etc.
            # by checking that the character after the digit is not another digit
            match_end = match.end()
            if match_end < len(filename):
                next_char = filename[match_end]
                if next_char.isdigit():
                    # This is part of a two-digit number like run-00, skip it
                    continue
            old_path = Path(root) / filename
            
            # Additional safety check: make sure this is really a single-digit run number
            # by checking the filename doesn't contain _run-XX where XX is two digits
            if '_run-00' in filename or '_run-01' in filename or '_run-02' in filename or \
               '_run-03' in filename or '_run-04' in filename or '_run-05' in filename or \
               '_run-06' in filename or '_run-07' in filename or '_run-08' in filename or \
               '_run-09' in filename:
                # This file already has zero-padded run number, skip it
                continue
            
            run_digit = match.group(1)
            separator = match.group(2)  # underscore or dot
            
            # Create the new filename with zero-padded run number
            # Replace '_run-X_' with '_run-0X_' where X is any single digit (0-9)
            # Use re.sub to replace only the first occurrence to be safe
            # Examples: '_run-0_' -> '_run-00_', '_run-1_' -> '_run-01_', '_run-9_' -> '_run-09_'
            new_filename = re.sub(r'_run-(\d)([_.])', rf'_run-0\1\2', filename, count=1)
            new_path = Path(root) / new_filename
            
            # Skip if already correct (shouldn't happen with our regex, but be safe)
            if old_path.name == new_filename:
                already_correct_count += 1
                continue
            
            # Extract base name (without extension) for grouping
            # Remove the run number part to group related files
            base_match = re.search(r'(.+)_run-\d([_.].+)', filename)
            if base_match:
                base_name = base_match.group(1) + base_match.group(2)
                files_by_base[base_name].append((old_path, new_path, new_filename))
            
            files_to_process.append((old_path, new_path, new_filename))

# Process files
print(f"\nScanned {total_files_scanned} total files")
print(f"Found {len(files_to_process)} files with single-digit run numbers to process\n")

for old_path, new_path, new_filename in files_to_process:
    # Check if the target file already exists
    if new_path.exists():
        deleted_count += 1
        if DRY_RUN:
            if len(delete_examples) < 5:
                delete_examples.append((old_path.relative_to(bids_root), new_path.relative_to(bids_root)))
            print(f"Would DELETE (target exists): {old_path.relative_to(bids_root)}")
        else:
            print(f"DELETE (target exists): {old_path.relative_to(bids_root)}")
            try:
                # Verify files are the same before deleting (optional safety check)
                if old_path.stat().st_size == new_path.stat().st_size:
                    old_path.unlink()
                    print(f"  ✓ Deleted successfully")
                else:
                    print(f"  ⚠ WARNING: File sizes differ! Skipping deletion.")
                    skipped_count += 1
                    if len(error_examples) < 3:
                        error_examples.append((old_path.relative_to(bids_root), "size mismatch"))
            except Exception as e:
                print(f"  ✗ ERROR deleting: {e}")
                skipped_count += 1
                if len(error_examples) < 3:
                    error_examples.append((old_path.relative_to(bids_root), str(e)))
    else:
        renamed_count += 1
        if DRY_RUN:
            if len(rename_examples) < 5:
                rename_examples.append((old_path.relative_to(bids_root), new_filename))
            print(f"Would RENAME: {old_path.relative_to(bids_root)}")
            print(f"         --> {new_filename}")
        else:
            print(f"RENAME: {old_path.relative_to(bids_root)}")
            print(f"    --> {new_filename}")
            try:
                old_path.rename(new_path)
                print(f"  ✓ Renamed successfully")
            except Exception as e:
                print(f"  ✗ ERROR renaming: {e}")
                skipped_count += 1
                if len(error_examples) < 3:
                    error_examples.append((old_path.relative_to(bids_root), str(e)))

print("\n" + "="*60)
print("Summary:")
print(f"  Files renamed: {renamed_count}")
print(f"  Files deleted (duplicates): {deleted_count}")
print(f"  Files skipped (errors): {skipped_count}")
print(f"  Files already correct: {already_correct_count}")
print("="*60)

# Show grouped files for better visibility
if files_by_base and (DRY_RUN or renamed_count > 0):
    print("\n" + "="*60)
    print("FILES GROUPED BY RECORDING (showing first 3 groups):")
    print("="*60)
    group_count = 0
    for base_name, file_list in list(files_by_base.items())[:3]:
        if group_count >= 3:
            break
        print(f"\nRecording group: {base_name.split('_run-')[0] if '_run-' in base_name else base_name}")
        for old_path, new_path, new_filename in file_list:
            status = "✓" if new_path.exists() else "→"
            print(f"  {status} {old_path.name} -> {new_filename}")
        group_count += 1
    if len(files_by_base) > 3:
        print(f"\n  ... and {len(files_by_base) - 3} more recording groups")

if DRY_RUN:
    print("\n" + "="*60)
    print("DRY RUN EXAMPLES:")
    print("="*60)
    
    if rename_examples:
        print("\nFILES TO RENAME (showing up to 5 examples):")
        for old_file, new_filename in rename_examples[:5]:
            print(f"  {old_file}")
            print(f"    --> {new_filename}")
    else:
        print("\nNo files need to be renamed.")
    
    if delete_examples:
        print("\nFILES TO DELETE (duplicate, showing up to 5 examples):")
        for old_file, existing_file in delete_examples[:5]:
            print(f"  DELETE: {old_file}")
            print(f"  (already exists as: {existing_file})")
    else:
        print("\nNo duplicate files to delete.")
    
    if error_examples:
        print("\nERRORS ENCOUNTERED (showing up to 3 examples):")
        for file_path, error_msg in error_examples:
            print(f"  {file_path}: {error_msg}")
    
    print("\n" + "="*60)
    print("To perform these operations, set DRY_RUN = False in the script")
    print("="*60)
else:
    if error_examples:
        print("\n" + "="*60)
        print("ERRORS ENCOUNTERED:")
        print("="*60)
        for file_path, error_msg in error_examples:
            print(f"  {file_path}: {error_msg}")

