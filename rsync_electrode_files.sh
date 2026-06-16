#!/bin/bash

# Simple rsync script to copy electrode localization files from server
# Usage: ./rsync_electrode_files.sh username@server [local_repo_path]

SERVER=$1
LOCAL_REPO=${2:-"./local-stim-seizures"}

if [ -z "$SERVER" ]; then
    echo "Usage: $0 username@server [local_repo_path]"
    echo "Example: $0 user@server.edu ./local-stim-seizures"
    exit 1
fi

# Server path
SERVER_PROCESSED="/mnt/sauce/littlab/users/wojemann/stim-seizures/PROCESSED_DATA"

echo "Copying electrode localization files from server..."
echo "Server: $SERVER"
echo "Local repo: $LOCAL_REPO"

# Create local PROCESSED_DATA directory
mkdir -p "$LOCAL_REPO/PROCESSED_DATA"

# Use rsync to copy only the electrode localization files while preserving directory structure
rsync -av --progress \
    --include="*/" \
    --include="electrode_localizations_CHOPR.pkl" \
    --include="electrode_localizations_dkt.pkl" \
    --exclude="*" \
    "$SERVER:$SERVER_PROCESSED/" "$LOCAL_REPO/PROCESSED_DATA/"

echo ""
echo "Copy complete!"
echo "Files copied:"
find "$LOCAL_REPO/PROCESSED_DATA" -name "electrode_localizations_*.pkl" | wc -l
echo ""
echo "Directory structure:"
find "$LOCAL_REPO/PROCESSED_DATA" -name "electrode_localizations_*.pkl" | sort 