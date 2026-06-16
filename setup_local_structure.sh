#!/bin/bash

# Script to create local directory structure based on patient list in config.py
# Usage: ./setup_local_structure.sh [local_repo_path] [config_path]

LOCAL_REPO=${1:-"./local-stim-seizures"}
CONFIG_PATH=${2:-"./code/config.py"}

echo "Setting up local repository structure..."
echo "Local repo: $LOCAL_REPO"
echo "Config file: $CONFIG_PATH"

# Check if config.py exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config.py not found at $CONFIG_PATH"
    echo "Please ensure config.py is available locally or provide the correct path"
    exit 1
fi

# Extract patient IDs from config.py
echo "Extracting patient list from config.py..."
PATIENTS=$(grep -o '"ptID": "[^"]*"' "$CONFIG_PATH" | sed 's/"ptID": "\([^"]*\)"/\1/')

if [ -z "$PATIENTS" ]; then
    echo "Error: Could not extract patient list from config.py"
    exit 1
fi

echo "Found patients: $(echo $PATIENTS | tr '\n' ' ')"

# Create local directory structure
echo "Creating directory structure..."
mkdir -p "$LOCAL_REPO/PROCESSED_DATA"
mkdir -p "$LOCAL_REPO/code"
mkdir -p "$LOCAL_REPO/METADATA"
mkdir -p "$LOCAL_REPO/figures"

# Create patient directories
for PATIENT in $PATIENTS; do
    echo "Creating directory for $PATIENT..."
    mkdir -p "$LOCAL_REPO/PROCESSED_DATA/$PATIENT"
done

echo "Directory structure created successfully!"
echo "Local repository structure:"
tree "$LOCAL_REPO" -d -L 3 2>/dev/null || find "$LOCAL_REPO" -type d | head -20 