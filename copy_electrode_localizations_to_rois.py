#!/usr/bin/env python3
"""
Script to copy electrode localization files from BIDS structure to rois folder.

For each subject in /Users/wojemann/local_data/stim_dataset_data/RAW_DATA/DATA/sub-xxxxxxx,
copies the electrodes.tsv file from sub-xxxxxxx/derivatives/electrodes.tsv
to /Users/wojemann/Documents/CNT/stim_seizures_data/PROCESSED_DATA/rois/xxxxxxx/electrodes.tsv
(removing the "sub-" prefix from the subject name).
"""

import os
import shutil
from pathlib import Path
import argparse


def find_subject_folders(source_dir):
    """Find all sub-* folders in the source directory."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    subject_folders = [d for d in source_path.iterdir() 
                      if d.is_dir() and d.name.startswith('sub-')]
    
    return sorted(subject_folders)


def copy_electrode_file(subject_folder, dest_base_dir):
    """
    Copy electrodes.tsv from subject folder to destination.
    
    Args:
        subject_folder: Path to sub-xxxxxxx folder
        dest_base_dir: Base destination directory (rois folder)
    
    Returns:
        tuple: (success: bool, subject_id: str, message: str)
    """
    subject_id = subject_folder.name.replace('sub-', '')
    source_file = subject_folder / 'derivatives' / 'electrodes.tsv'
    
    if not source_file.exists():
        return False, subject_id, f"Source file not found: {source_file}"
    
    # Create destination directory
    dest_dir = Path(dest_base_dir) / subject_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy file
    dest_file = dest_dir / 'electrodes.tsv'
    try:
        shutil.copy2(source_file, dest_file)
        return True, subject_id, f"Copied to {dest_file}"
    except Exception as e:
        return False, subject_id, f"Error copying file: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Copy electrode localization files from BIDS structure to rois folder"
    )
    parser.add_argument(
        "--source-dir",
        default="/Users/wojemann/local_data/stim_dataset_data/RAW_DATA/DATA",
        help="Source directory containing sub-* folders (default: /Users/wojemann/local_data/stim_dataset_data/RAW_DATA/DATA)"
    )
    parser.add_argument(
        "--dest-dir",
        default="/Users/wojemann/Documents/CNT/stim_seizures_data/PROCESSED_DATA/rois",
        help="Destination base directory (default: /Users/wojemann/Documents/CNT/stim_seizures_data/PROCESSED_DATA/rois)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying"
    )
    
    args = parser.parse_args()
    
    print(f"Source directory: {args.source_dir}")
    print(f"Destination directory: {args.dest_dir}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
    print()
    
    # Find all subject folders
    try:
        subject_folders = find_subject_folders(args.source_dir)
        print(f"Found {len(subject_folders)} subject folders")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    if not subject_folders:
        print("No subject folders found!")
        return 1
    
    # Process each subject
    successful = []
    failed = []
    
    for subject_folder in subject_folders:
        subject_id = subject_folder.name.replace('sub-', '')
        source_file = subject_folder / 'derivatives' / 'electrodes.tsv'
        dest_file = Path(args.dest_dir) / subject_id / 'electrodes.tsv'
        
        if args.dry_run:
            if source_file.exists():
                print(f"  Would copy: {subject_id}")
                print(f"    From: {source_file}")
                print(f"    To:   {dest_file}")
            else:
                print(f"  Would skip: {subject_id} (file not found: {source_file})")
        else:
            success, subj_id, message = copy_electrode_file(subject_folder, args.dest_dir)
            if success:
                successful.append(subj_id)
                print(f"  ✓ {subj_id}: {message}")
            else:
                failed.append((subj_id, message))
                print(f"  ✗ {subj_id}: {message}")
    
    # Summary
    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Total subjects found: {len(subject_folders)}")
    if not args.dry_run:
        print(f"  Successfully copied: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        
        if successful:
            print(f"\n  Successfully processed subjects:")
            for subj_id in successful:
                print(f"    - {subj_id}")
        
        if failed:
            print(f"\n  Failed subjects:")
            for subj_id, message in failed:
                print(f"    - {subj_id}: {message}")
    else:
        existing_count = sum(1 for sf in subject_folders 
                           if (sf / 'derivatives' / 'electrodes.tsv').exists())
        print(f"  Subjects with electrodes.tsv: {existing_count}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main())

