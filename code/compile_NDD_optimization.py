'''
Script to compile all NDD optimization checkpoint results into a single CSV file.
Loads all .pkl files from the ndd_checkpoints directory and combines them.
'''

import os
from os.path import join as ospj
import pandas as pd
import pickle
import glob
from tqdm import tqdm

# Get the project root and add DynaSD to path
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

import sys
if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

from config import Config
datapath, prodatapath = Config.deal(['datapath', 'prodatapath'])

def compile_ndd_results():
    """Load all NDD checkpoint files and combine into a single DataFrame."""
    
    checkpoint_dir = ospj(prodatapath, "ndd_checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Find all pickle files in the checkpoint directory
    pkl_files = glob.glob(ospj(checkpoint_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"No pickle files found in {checkpoint_dir}")
        return None
    
    print(f"Found {len(pkl_files)} checkpoint files")
    
    # Load and combine all results
    all_results = []
    
    for pkl_file in tqdm(pkl_files, desc="Loading checkpoint files"):
        try:
            with open(pkl_file, 'rb') as f:
                df = pickle.load(f)
                
            # If it's a DataFrame, add it to our list
            if isinstance(df, pd.DataFrame):
                all_results.append(df)
            else:
                print(f"Warning: {pkl_file} does not contain a DataFrame")
                
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    
    if not all_results:
        print("No valid DataFrames found in checkpoint files")
        return None
    
    # Combine all DataFrames
    combined_df = pd.concat(all_results, ignore_index=True)
    
    print(f"Combined {len(all_results)} checkpoint files into DataFrame with {len(combined_df)} rows")
    
    return combined_df

def main():
    """Main function to compile results and save as CSV."""
    
    # Compile all results
    results_df = compile_ndd_results()
    
    if results_df is None:
        print("Failed to compile results")
        return
    
    # Save combined results as CSV
    output_file = ospj(prodatapath, 'compiled_ndd_optimization_results.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"Saved combined results to: {output_file}")
    print(f"Total rows: {len(results_df)}")
    print(f"Columns: {list(results_df.columns)}")
    
    # Print summary statistics
    print("\nSummary:")
    if 'patient' in results_df.columns:
        print(f"Patients: {results_df['patient'].unique()}")
    if 'model' in results_df.columns:
        print(f"Models: {results_df['model'].unique()}")
    if 'sequence_length' in results_df.columns:
        print(f"Sequence lengths: {sorted(results_df['sequence_length'].unique())}")
    if 'duration' in results_df.columns:
        print(f"Durations: {sorted(results_df['duration'].unique())}")

if __name__ == "__main__":
    main() 