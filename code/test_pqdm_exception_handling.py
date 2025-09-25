#!/usr/bin/env python3
"""
Test script to verify how pqdm handles exceptions in worker functions
"""
from pqdm.processes import pqdm
import pandas as pd

def test_function(x):
    """Test function that fails for certain inputs"""
    if x == 2:
        raise ValueError(f"Intentional failure for x={x}")
    return x * 2

def main():
    # Test data - item 2 will cause an exception
    test_data = [1, 2, 3, 4, 5]
    
    print("Testing pqdm exception handling...")
    print(f"Input data: {test_data}")
    
    try:
        # Run pqdm and see what happens
        results = pqdm(test_data, test_function, n_jobs=2)
        
        print(f"Raw results: {results}")
        print(f"Result types: {[type(r) for r in results]}")
        
        # Test the filtering approach
        filtered_results = [r for r in results if not isinstance(r, Exception)]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        print(f"Filtered results: {filtered_results}")
        print(f"Exceptions found: {exceptions}")
        
        if exceptions:
            print("✓ Exception filtering approach will work!")
        else:
            print("✗ Exception filtering approach may not work as expected")
            
    except Exception as e:
        print(f"pqdm raised an exception: {type(e).__name__}: {e}")
        print("✗ Exception filtering won't help - pqdm re-raises exceptions")

if __name__ == "__main__":
    main() 