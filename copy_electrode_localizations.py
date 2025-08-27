#!/usr/bin/env python3

import os
import sys
import re
import subprocess
from pathlib import Path
import argparse

def extract_patients_from_config(config_path):
    """Extract patient IDs from config.py file."""
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Extract patient IDs using regex
        pattern = r'"ptID":\s*"([^"]+)"'
        patients = re.findall(pattern, content)
        return patients
    except FileNotFoundError:
        print(f"Error: config.py not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error reading config.py: {e}")
        return None

def create_local_structure(local_repo, patients):
    """Create local directory structure for all patients."""
    print("Creating local directory structure...")
    
    # Create main directories
    directories = [
        "PROCESSED_DATA",
        "code", 
        "METADATA",
        "figures"
    ]
    
    for dir_name in directories:
        Path(local_repo, dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create patient directories
    for patient in patients:
        patient_dir = Path(local_repo, "PROCESSED_DATA", patient)
        patient_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created directory for {patient}")
    
    print(f"Directory structure created in {local_repo}")

def copy_config_file(server, local_repo):
    """Copy config.py from server if it doesn't exist locally."""
    local_config = Path(local_repo, "code", "config.py")
    
    if not local_config.exists():
        print("Copying config.py from server...")
        server_config = "/mnt/sauce/littlab/users/wojemann/stim-seizures/code/config.py"
        
        try:
            subprocess.run([
                "scp", f"{server}:{server_config}", str(local_config)
            ], check=True)
            print("  config.py copied successfully")
        except subprocess.CalledProcessError as e:
            print(f"  Error copying config.py: {e}")
            return False
    else:
        print("  config.py already exists locally")
    
    return True

def copy_electrode_files(server, local_repo, patients):
    """Copy electrode localization files for all patients."""
    print(f"Copying electrode localization files for {len(patients)} patients...")
    
    server_processed = "/mnt/sauce/littlab/users/wojemann/stim-seizures/PROCESSED_DATA"
    files_copied = 0
    
    for patient in patients:
        print(f"Processing {patient}...")
        
        local_patient_dir = Path(local_repo, "PROCESSED_DATA", patient)
        server_patient_dir = f"{server_processed}/{patient}"
        
        # File types to copy
        file_types = ["electrode_localizations_CHOPR.pkl", "electrode_localizations_dkt.pkl"]
        
        for file_type in file_types:
            server_file = f"{server}:{server_patient_dir}/{file_type}"
            local_file = local_patient_dir / file_type
            
            try:
                subprocess.run([
                    "scp", server_file, str(local_file)
                ], check=True, capture_output=True)
                print(f"  ✓ {file_type}")
                files_copied += 1
            except subprocess.CalledProcessError:
                print(f"  ✗ {file_type} (not found)")
    
    return files_copied

def main():
    parser = argparse.ArgumentParser(description="Copy electrode localization files from server")
    parser.add_argument("server", help="Server connection string (e.g., username@server.edu)")
    parser.add_argument("--local-repo", "-l", default="./local-stim-seizures", 
                       help="Local repository path (default: ./local-stim-seizures)")
    parser.add_argument("--config-path", "-c", 
                       help="Path to config.py (default: auto-detect)")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only create directory structure, don't copy files")
    
    args = parser.parse_args()
    
    local_repo = Path(args.local_repo)
    print(f"Local repository: {local_repo.absolute()}")
    
    # Determine config path
    if args.config_path:
        config_path = Path(args.config_path)
    else:
        # Try to find config.py locally first, then use server copy
        config_path = local_repo / "code" / "config.py"
        if not config_path.exists():
            # Copy from server first
            if not args.setup_only:
                copy_config_file(args.server, local_repo)
            else:
                print("Error: No config.py found and --setup-only specified")
                print("Please provide config.py path with --config-path")
                sys.exit(1)
    
    # Extract patient list
    patients = extract_patients_from_config(config_path)
    if not patients:
        sys.exit(1)
    
    print(f"Found {len(patients)} patients: {', '.join(patients[:5])}{'...' if len(patients) > 5 else ''}")
    
    # Create directory structure
    create_local_structure(local_repo, patients)
    
    if args.setup_only:
        print("Setup complete! (--setup-only specified)")
        return
    
    # Copy electrode localization files
    files_copied = copy_electrode_files(args.server, local_repo, patients)
    
    # Summary
    print(f"\nCopy complete!")
    print(f"Files copied: {files_copied}")
    print(f"Local repository: {local_repo.absolute()}")
    
    # List copied files
    pkl_files = list(Path(local_repo, "PROCESSED_DATA").glob("*/electrode_localizations_*.pkl"))
    if pkl_files:
        print(f"\nCopied files:")
        for file in sorted(pkl_files):
            print(f"  {file.relative_to(local_repo)}")

if __name__ == "__main__":
    main() 