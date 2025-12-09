#!/usr/bin/env python3
"""
Script to clear mne_bids-related caches and force fresh reads of BIDS data.

mne_bids doesn't maintain a persistent cache, but Python caches imports
and get_entity_vals() results may be cached in memory during a session.
"""

import sys
import importlib
from pathlib import Path
from config import Config

def clear_python_cache():
    """Clear Python's import cache for mne_bids related modules."""
    modules_to_reload = [
        'mne_bids',
        'mne_bids.read',
        'mne_bids.path',
        'mne_bids.utils',
    ]
    
    print("Clearing Python import cache...")
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            print(f"  Removing {module_name} from cache")
            del sys.modules[module_name]
    
    print("✓ Python import cache cleared")
    print("  Note: You'll need to re-import mne_bids in your scripts")

def clear_bids_pycache(bids_root):
    """Remove __pycache__ directories from BIDS root (if any)."""
    bids_path = Path(bids_root)
    if not bids_path.exists():
        print(f"⚠ BIDS directory not found: {bids_root}")
        return
    
    pycache_dirs = list(bids_path.rglob('__pycache__'))
    if pycache_dirs:
        print(f"\nFound {len(pycache_dirs)} __pycache__ directories:")
        for cache_dir in pycache_dirs[:10]:  # Show first 10
            print(f"  {cache_dir.relative_to(bids_path)}")
        if len(pycache_dirs) > 10:
            print(f"  ... and {len(pycache_dirs) - 10} more")
        
        # Ask before deleting (safety)
        print("\n⚠ These are Python bytecode caches, not mne_bids caches.")
        print("  They're safe to delete but will be regenerated automatically.")
        print("  To delete them, run: find <bids_root> -type d -name __pycache__ -exec rm -r {} +")
    else:
        print("\n✓ No __pycache__ directories found in BIDS root")

def main():
    """Main function to clear caches."""
    print("="*60)
    print("Clearing mne_bids-related caches")
    print("="*60)
    
    # Get BIDS root from config
    datapath = Config.deal(['datapath'])
    bids_root = Path(datapath) / 'BIDS'
    
    print(f"\nBIDS root: {bids_root}")
    
    # Clear Python import cache
    clear_python_cache()
    
    # Check for __pycache__ directories
    clear_bids_pycache(bids_root)
    
    print("\n" + "="*60)
    print("Cache clearing complete!")
    print("="*60)
    print("\nTo ensure fresh reads:")
    print("1. Restart your Python kernel/process")
    print("2. Re-import mne_bids: from mne_bids import BIDSPath, get_entity_vals")
    print("3. Re-run your code")
    print("="*60)

if __name__ == "__main__":
    main()

