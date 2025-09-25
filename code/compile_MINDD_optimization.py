'''
Script to compile all MINDD optimization checkpoint results into a single CSV file.
Loads all .pkl files from the mindd_checkpoints directory and combines them.
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

def compile_mindd_results():
    """Load all MINDD checkpoint files and combine into a single DataFrame."""
    
    checkpoint_dir = ospj(prodatapath, "ndd_checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Find all MINDD optimization pickle files in the checkpoint directory
    pkl_files = glob.glob(ospj(checkpoint_dir, "*MINDD*.pkl"))
    
    # Filter to only include MINDD optimization files (not NDD files with MINDD models)
    # MINDD optimization files have structure: patient_duration_MINDD_sequence_layers_dropout_weights_stacks.pkl
    # NDD optimization files have structure: patient_duration_MINDD_sequence_layers_param_scale_hidden_size_stacks.pkl
    mindd_files = []
    for f in pkl_files:
        basename = os.path.basename(f)
        parts = basename.replace('.pkl', '').split('_')
        # Check if this has the MINDD optimization parameter structure
        # MINDD opt files have dropout and weights parameters, not param_scale and hidden_size
        if len(parts) >= 8 and 'MINDD' in basename:
            # Check if this looks like MINDD optimization format (has non-numeric weights section)
            try:
                # In MINDD opt: parts[-3] is weights_str (could be "None" or numbers separated by dots)
                # In NDD opt: parts[-3] is hidden_size (numeric)
                weights_part = parts[-3]
                hidden_size_part = parts[-2] if len(parts) > 7 else parts[-3]
                
                # If weights_part is "None" or contains dots (weight values), it's MINDD opt
                # If it's purely numeric, it's likely NDD opt with hidden_size
                if weights_part == "None" or "." in weights_part or not weights_part.isdigit():
                    mindd_files.append(f)
                elif hidden_size_part.isdigit() and weights_part.isdigit():
                    # This looks like NDD opt format (param_scale_hidden_size)
                    continue
                else:
                    # Default to including if uncertain
                    mindd_files.append(f)
            except (IndexError, ValueError):
                # If parsing fails, include the file
                mindd_files.append(f)
        elif 'MINDD' in basename:
            # Include any other MINDD files
            mindd_files.append(f)
    
    if not mindd_files:
        print(f"No MINDD optimization pickle files found in {checkpoint_dir}")
        return None
    
    print(f"Found {len(mindd_files)} MINDD checkpoint files")
    
    # Load and combine all results
    all_results = []
    
    for pkl_file in tqdm(mindd_files, desc="Loading MINDD checkpoint files"):
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
    results_df = compile_mindd_results()
    
    if results_df is None:
        print("Failed to compile results")
        return
    
    # Save combined results as CSV
    output_file = ospj(prodatapath, 'compiled_mindd_optimization_results.csv')
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