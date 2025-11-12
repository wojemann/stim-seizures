#!/usr/bin/env python
"""
Diagnostic script to test imports and identify what's hanging
Run this directly in your terminal with:
    python code/test_imports.py
"""
import sys
import time
import os

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print("-" * 60)

def test_import(module_name):
    """Test importing a module and time it"""
    print(f"Testing {module_name}...", end="", flush=True)
    start = time.time()
    try:
        __import__(module_name)
        elapsed = time.time() - start
        print(f" ✓ {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f" ✗ {elapsed:.2f}s - {e}")
        return False

# Set environment variables to fix known issues
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.makedirs('/tmp/matplotlib-cache', exist_ok=True)

# Test imports in order
modules = [
    'numpy',
    'pandas',
    'matplotlib',
    'torch',
    'tensorflow',
]

print("\nTesting imports:\n")
for module in modules:
    if not test_import(module):
        print(f"\n❌ Failed at {module}")
        sys.exit(1)

print("\n✅ All imports successful!")

