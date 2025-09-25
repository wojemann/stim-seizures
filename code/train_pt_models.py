import os
import sys
from os.path import join as ospj
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Get the project root and add DynaSD to path
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')
if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

from utils import get_data_from_bids, preprocess_for_detection, remove_scalp_electrodes, clean_labels
from config import Config
from DynaSD import MINDD

# Load config
datapath, prodatapath = Config.deal(['datapath', 'prodatapath'])

def train_patient_model(patient, train_kwargs, save_dir=None):
    """
    Train MINDD model for a single patient
    
    Args:
        patient: Patient ID (e.g., 'HUP065')
        train_kwargs: Dictionary with training parameters
        save_dir: Directory to save models and results
    
    Returns:
        dict: Training results including validation R2 and model path
    """
    print(f"Training model for patient {patient}")
    
    # Set default save directory
    if save_dir is None:
        save_dir = ospj(prodatapath, "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Load interictal data
        X, fs_raw = get_data_from_bids(ospj(datapath, "BIDS"), patient, 'interictal')
        X.columns = clean_labels(X.columns, patient)
        neural_channels = remove_scalp_electrodes(X.columns)
        
        # Preprocess data
        X, fs_raw, _ = preprocess_for_detection(X.loc[:, neural_channels], fs_raw)
        print(f"Data shape for {patient}: {X.shape}")
        
        # Create MINDD model with provided parameters
        model = MINDD(
            fs=256,
            use_cuda=True,
            verbose=train_kwargs.get('verbose', False),
            **{k: v for k, v in train_kwargs.items() if k != 'verbose'}
        )
        
        # Train model and measure time
        train_start = time.perf_counter()
        model.fit(X)
        train_time = time.perf_counter() - train_start
        
        # Calculate validation R2 (sequential split)
        val_split = train_kwargs.get('val_split', 0.1)
        val_idx = int(X.shape[0] * (1 - val_split))
        
        # Get predictions for validation set
        x_pred = model.predict(X)
        sequence_length = train_kwargs.get('sequence_length', 32)
        
        # Calculate validation R2 on the sequential validation split
        val_data = X.iloc[val_idx:, :]
        val_pred = x_pred[val_idx - sequence_length:, :]
        
        # Align predictions with validation data
        min_len = min(len(val_data), len(val_pred))
        if min_len > 0:
            val_r2 = r2_score(val_data.iloc[:min_len, :], val_pred[:min_len, :])
        else:
            val_r2 = np.nan
            print(f"Warning: Could not calculate validation R2 for {patient}")
        
        # Save trained model
        model_filename = f"{patient}_mindd_model.pkl"
        model_path = ospj(save_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Prepare results
        results = {
            'patient': patient,
            'val_r2': val_r2,
            'train_time': train_time,
            'model_path': model_path,
            'data_shape': X.shape,
            'num_epochs': getattr(model, 'early_stop_epoch', train_kwargs.get('num_epochs', 'unknown')),
            'batch_size': getattr(model, 'batch_size', train_kwargs.get('batch_size', 'unknown')),
            **train_kwargs
        }
        
        print(f"✓ {patient}: Validation R2 = {val_r2:.4f}, Training time = {train_time:.1f}s")
        return results
        
    except Exception as e:
        print(f"✗ Error training {patient}: {str(e)}")
        return {
            'patient': patient,
            'val_r2': np.nan,
            'error': str(e),
            **train_kwargs
        }

def main():
    """Main training function"""
    
    # Training parameters - modify these as needed
    train_kwargs = {
        'sequence_length': 32,
        'forecast_length': 1,
        'hidden_sizes': [64, 32],  # List of hidden layer sizes
        'num_epochs': 200,
        'batch_size': 2048,
        'patience': 3,
        'lr': 0.0005,
        'val_split': 0.1,
        'early_stopping': True,
        'verbose': False
    }
    
    # Patient list
    patients = ['HUP065', 'HUP078', 'HUP126', 'HUP221', 'HUP276']
    
    # Create results directory
    save_dir = ospj(prodatapath, "trained_models")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Training MINDD models for {len(patients)} patients")
    print(f"Training parameters: {train_kwargs}")
    print("-" * 60)
    
    # Train models for each patient
    all_results = []
    for patient in patients:
        result = train_patient_model(patient, train_kwargs, save_dir)
        all_results.append(result)
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_path = ospj(save_dir, "training_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Save results as pickle too
    pickle_path = ospj(save_dir, "training_results.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    print("-" * 60)
    print("Training completed!")
    print(f"Results saved to: {results_path}")
    print(f"Models saved to: {save_dir}")
    
    # Print summary
    successful = results_df[~results_df['val_r2'].isna()]
    if len(successful) > 0:
        print(f"\nSummary ({len(successful)}/{len(patients)} successful):")
        print(f"Mean validation R2: {successful['val_r2'].mean():.4f} ± {successful['val_r2'].std():.4f}")
        print(f"Mean training time: {successful['train_time'].mean():.1f}s ± {successful['train_time'].std():.1f}s")
    
    return results_df

if __name__ == "__main__":
    results = main()
