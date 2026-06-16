#!/bin/bash

# Script to copy electrode localization files from server to local repository
# Usage: ./copy_electrode_localizations.sh username@server local_repo_path

SERVER=$1
LOCAL_REPO=${2:-"./local-stim-seizures"}

if [ -z "$SERVER" ]; then
    echo "Usage: $0 username@server [local_repo_path]"
    echo "Example: $0 user@server.edu ./local-stim-seizures"
    echo ""
    echo "Note: Run setup_local_structure.sh first to create the directory structure"
    exit 1
fi

# Server paths
SERVER_BASE="/mnt/sauce/littlab/users/wojemann/stim-seizures"
SERVER_CONFIG="$SERVER_BASE/code/config.py"
SERVER_PROCESSED="$SERVER_BASE/PROCESSED_DATA"

# Check if local directory structure exists
if [ ! -d "$LOCAL_REPO/PROCESSED_DATA" ]; then
    echo "Error: Local directory structure not found at $LOCAL_REPO"
    echo "Please run setup_local_structure.sh first to create the directory structure"
    exit 1
fi

echo "Copying files from server: $SERVER"
echo "Local repository: $LOCAL_REPO"

# Copy config.py if it doesn't exist locally
if [ ! -f "$LOCAL_REPO/code/config.py" ]; then
    echo "Copying config.py..."
    scp "$SERVER:$SERVER_CONFIG" "$LOCAL_REPO/code/"
fi

# Extract patient list from local config
echo "Reading patient list from local config.py..."
PATIENTS=$(grep -o '"ptID": "[^"]*"' "$LOCAL_REPO/code/config.py" | sed 's/"ptID": "\([^"]*\)"/\1/')

if [ -z "$PATIENTS" ]; then
    echo "Error: Could not extract patient list from local config.py"
    exit 1
fi

echo "Copying electrode localization files for $(echo $PATIENTS | wc -w) patients..."

# Use rsync for more efficient copying
for PATIENT in $PATIENTS; do
    echo "Processing $PATIENT..."
    
    # Use rsync to copy both electrode localization files at once
    rsync -av --progress \
        --include="electrode_localizations_CHOPR.pkl" \
        --include="electrode_localizations_dkt.pkl" \
        --exclude="*" \
        "$SERVER:$SERVER_PROCESSED/$PATIENT/" "$LOCAL_REPO/PROCESSED_DATA/$PATIENT/" || echo "  - No electrode localization files found for $PATIENT"
done

echo ""
echo "Copy complete! Summary:"
echo "Files copied:"
find "$LOCAL_REPO/PROCESSED_DATA" -name "electrode_localizations_*.pkl" | wc -l
echo ""
echo "Files by patient:"
find "$LOCAL_REPO/PROCESSED_DATA" -name "electrode_localizations_*.pkl" | sort 