# File system imports
import sys
import os
import glob
import pickle
from os.path import join as ospj

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARN messages
os.environ['TF_TRT_DISABLED'] = '1'       # Silence TF-TRT warnings if TensorRT not installed

# Scientific imports
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

# Plotting imports
import matplotlib.pyplot as plt

# Utility imports
from utils import preprocess_for_detection, get_data_from_bids, index_of_union_threshold, clean_labels

# Parallel execution
from pqdm.threads import pqdm

# Get the project root (parent directory of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

from DynaSD import ABSSLP, IMPRINT, WVNT, HFER, LiNDDA, GIN, MINDD
from config import Config

# Get paths from config 
datapath,prodatapath,figpath,metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])


# Set default colormap for visualizations
plt.rcParams['image.cmap'] = 'magma'

# Global configuration
OVERWRITE = True  # Whether to overwrite existing probability matrix files

def find_optimal_f1_threshold(y_true, y_scores):
    """Find threshold that maximizes F1 score."""
    thresholds = np.unique(y_scores)
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        y_pred = y_scores > threshold
        if len(np.unique(y_pred)) > 1:  # Ensure both classes are predicted
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    return best_threshold

def compute_all_metrics(y_true, y_scores, threshold):
    """Compute all binary classification metrics for a given threshold."""
    # Convert inputs to numpy arrays for consistent handling
    y_true = np.array(y_true, dtype=int)
    y_scores = np.array(y_scores)
    y_pred = (y_scores > threshold).astype(int)
    
    # Handle edge cases where all predictions are same class
    if len(np.unique(y_pred)) == 1:
        return {
            'f1': np.nan,
            'phi': np.nan,
            'sensitivity': np.nan,
            'specificity': np.nan,
            'precision': np.nan,
            'recall': np.nan
        }
    
    f1 = f1_score(y_true, y_pred)
    phi = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Calculate sensitivity and specificity manually
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    # Note: sensitivity = recall for binary classification
    return {
        'f1': f1,
        'phi': phi,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall
    }

def get_metrics(onset_mask, onset_prob):
    optimal_threshold, opt_se, opt_sp, auc = index_of_union_threshold(
                    onset_mask, onset_prob
                )
    onset_pred = onset_prob > optimal_threshold
    f1 = f1_score(onset_mask, onset_pred)
    phi = matthews_corrcoef(onset_mask, onset_pred)
    return optimal_threshold, opt_se, opt_sp, auc, f1, phi

def run_model_task(params: tuple) -> list:
    """
    Parallelizable worker: loads seizure, preprocesses, then evaluates loaded model for that seizure.

    Parameters
    ----------
    params : tuple
        (
            patient: str,
            onset_run: str,
            onset_labels: list,
            montage: str,
            onset_time_sec: float,
            model_path: str
        )

    Returns
    -------
    list[dict]
        One dict per model with summary metrics for this seizure.
    """
    (
        patient,
        onset_run,
        onset_labels,
        montage,
        onset_time_sec,
        model_path,
    ) = params
    try:
        # Load pre-trained model and associated data
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        model = model_dict['model']
        
        # Move model to desired device (GPU if available)
        if isinstance(model,LiNDDA):
            model.device = 'cpu'
        else:
            model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if hasattr(model, 'model') and model.model is not None:
                model.model = model.model.to(model.device)
        
        saved_mask = model_dict['mask']
        neural_channels = model_dict['neural_channels']
        fs = model_dict['fs']
        
        # Extract model parameters
        sequence_length = model.sequence_length
        forecast_length = model.forecast_length
        model_name = str(model) + '_pt'

        # Load seizure recording
        seizure, fs_raw, _, _, task, run = get_data_from_bids(
            ospj(datapath, "BIDS"), patient, onset_run, return_path=True, verbose=0
        )

        # Clean labels
        seizure.columns = clean_labels(seizure.columns, patient)

        # Create output directory
        out_dir = ospj(prodatapath, 'sz_prob', patient)
        os.makedirs(out_dir, exist_ok=True)

        # Preprocess seizure using saved mask
        seizure_pre, fs = preprocess_for_detection(
            seizure.loc[:, neural_channels],
            fs_raw,
            montage,
            target=fs_raw,
            wavenet=False,
            pre_mask=saved_mask,
        )

        art_channel_mask = seizure_pre.loc[180*fs:,:].abs().max() <= (np.median(seizure_pre.loc[180*fs:,:].abs().max())*50)
        # seizure_nart = seizure_pre.loc[:,art_channel_mask]
        seizure_nart = seizure_pre
        if len(seizure_nart.columns) == 0:
            print(f"No channels kept found for {patient} {onset_run}")
            return []
        onset_mask = [ch.split('-')[0] in onset_labels for ch in seizure_nart.columns[art_channel_mask]]
        
        # Apply model (no fitting)
        out_path = ospj(out_dir, f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_sz_prob_forecast-{forecast_length}.pkl")
        
        sz_prob = model(seizure_nart).loc[:,art_channel_mask]
        mse_prob = model.mse_df.loc[:,art_channel_mask]
        mse_z_prob = model.mse_z_df.abs().loc[:,art_channel_mask]
        sz_prob_times = model.get_win_times(len(seizure_nart))
        sz_prob_df = pd.concat((sz_prob,pd.Series(sz_prob_times,name='time')),axis=1)
        mse_prob_df = pd.concat((mse_prob,pd.Series(sz_prob_times,name='time')),axis=1)
        mse_z_prob_df = pd.concat((mse_z_prob,pd.Series(sz_prob_times,name='time')),axis=1)
        sz_prob_df.to_pickle(out_path)
        mse_prob_df.to_pickle(out_path.replace('sz_prob_','mse_prob_'))
        mse_z_prob_df.to_pickle(out_path.replace('sz_prob_','mse_z_prob_'))
        onset_idx = int(np.argmin(np.abs(sz_prob_times - onset_time_sec)))
        onset_odx = int(np.argmin(np.abs(sz_prob_times - (onset_time_sec + 3))))
        onset_prob = sz_prob.iloc[onset_idx:onset_odx, :].mean()
        onset_mse = mse_prob.iloc[onset_idx:onset_odx,:].mean()
        onset_mse_z = mse_z_prob.iloc[onset_idx:onset_odx,:].mean()
        
        results = []
        for df,metric in zip([onset_prob,onset_mse,onset_mse_z],['prob','mse','mse_z']):
            if len(np.unique(onset_mask)) == 2:
                # IoU-optimized threshold metrics
                iou_threshold, iou_sensitivity, iou_specificity, auc, iou_f1, iou_phi = get_metrics(onset_mask, df)
                # Calculate additional IoU metrics (precision, recall)
                iou_pred = df > iou_threshold
                iou_precision = precision_score(onset_mask, iou_pred)
                iou_recall = recall_score(onset_mask, iou_pred)
                
                # F1-optimized threshold metrics
                f1_threshold = find_optimal_f1_threshold(onset_mask, df)
                f1_metrics = compute_all_metrics(onset_mask, df, f1_threshold)

                results.append(
                    dict(
                        patient=patient,
                        onset=int(onset_run),
                        model=model_name+'_'+metric,
                        sequence = sequence_length,
                        forecast=forecast_length,
                        auc=auc,
                        # IoU-optimized metrics
                        iou_threshold=iou_threshold,
                        iou_f1=iou_f1,
                        iou_phi=iou_phi,
                        iou_sensitivity=iou_sensitivity,
                        iou_specificity=iou_specificity,
                        iou_precision=iou_precision,
                        iou_recall=iou_recall,
                        # F1-optimized metrics
                        f1_threshold=f1_threshold,
                        f1_f1=f1_metrics['f1'],
                        f1_phi=f1_metrics['phi'],
                        f1_sensitivity=f1_metrics['sensitivity'],
                        f1_specificity=f1_metrics['specificity'],
                        f1_precision=f1_metrics['precision'],
                        f1_recall=f1_metrics['recall'],
                    )
                )
            else:
                results.append(
                    dict(
                        patient=patient,
                        onset=int(onset_run),
                        model=model_name+'_'+metric,
                        sequence = sequence_length,
                        forecast=forecast_length,
                        auc=np.nan,
                        # IoU-optimized metrics
                        iou_threshold=np.nan,
                        iou_f1=np.nan,
                        iou_phi=np.nan,
                        iou_sensitivity=np.nan,
                        iou_specificity=np.nan,
                        iou_precision=np.nan,
                        iou_recall=np.nan,
                        # F1-optimized metrics
                        f1_threshold=np.nan,
                        f1_f1=np.nan,
                        f1_phi=np.nan,
                        f1_sensitivity=np.nan,
                        f1_specificity=np.nan,
                        f1_precision=np.nan,
                        f1_recall=np.nan,
                    )
                )
        return results
    except Exception as e:
        print(f"Task failed with error: {e}")
        return []

def main():
    """
    Main seizure detection pipeline using pre-trained models.
    
    Workflow:
    1. Load configuration and seizure metadata from BIDS format
    2. For each patient:
       - Load pre-trained MINDD model(s)
       - For each seizure recording:
         - Apply model with saved masks
         - Generate predictions across full seizure recording
         - Save probability matrices and compute metrics
    """
    
    # Load seizure metadata from BIDS processing
    seizures_df = pd.read_csv(ospj(metapath,"metadata_v6_BIDS.csv"))
    seizures_df = seizures_df[seizures_df.split == 1] # Filter for only seizures that have soft onset labels
    
    # Detection parameters
    onset_time = 180          # Seizure onset time in recording (seconds)
    montage = 'bipolar'       # Electrode montage for preprocessing
    
    # Path to trained models
    model_dir = ospj(prodatapath, "trained_models")

    # Build all tasks grouped by patient
    tasks = []
    
    # Group seizures by patient
    for patient, patient_seizures in seizures_df.groupby('Patient'):
        # Find all trained models for this patient
        patient_model_dir = ospj(model_dir, patient)
        if not os.path.exists(patient_model_dir):
            print(f"No trained models found for {patient}, skipping")
            continue
        
        model_files = glob.glob(ospj(patient_model_dir, f"{patient}_mindd_seq64*.pkl"))
        model_files_small = glob.glob(ospj(patient_model_dir, f"{patient}_mindd_seq32*.pkl"))
        model_files_linear = glob.glob(ospj(patient_model_dir, f"{patient}_lindda_seq64*.pkl"))
        model_files.extend(model_files_small)
        model_files.extend(model_files_linear)
        if not model_files:
            print(f"No model files found for {patient}, skipping")
            continue
        
        # For each model trained for this patient
        for model_path in model_files:
            # For each seizure for this patient
            for _, row in patient_seizures.iterrows():
                onset_run = str(int(row.onset))
                onset_labels = clean_labels([l.strip() for l in row.SOZ.split(',')], patient)
                tasks.append((patient, onset_run, onset_labels, montage, onset_time, model_path))

    print(f"Processing {len(tasks)} tasks across {seizures_df.Patient.nunique()} patients")

    # Execute all tasks
    results_nested = []
    for task in tqdm(tasks[:300], total=300):
        results_nested.append(run_model_task(task))
    # results_nested = pqdm(tasks, run_model_task, n_jobs=16)

    # Filter out exceptions and flatten list of lists into a single list of dicts
    flat_results = []
    for result in results_nested:
        if isinstance(result, Exception):
            print(f"Task failed with error: {result}")
            continue
        for mdl in result:
            flat_results.append(mdl)

    result_df = pd.DataFrame(flat_results)
    result_df.to_csv(ospj(prodatapath,f"mindd_model_validation_results_100.csv"),index=False)

    
if __name__ == "__main__":
    main()
