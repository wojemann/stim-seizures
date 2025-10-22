# File system imports
import sys
import os
import glob
from os.path import join as ospj

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARN messages
os.environ['TF_TRT_DISABLED'] = '1'       # Silence TF-TRT warnings if TensorRT not installed

# Scientific imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

# Plotting imports
import matplotlib.pyplot as plt

# Deep learning imports
from tensorflow.config.experimental import set_memory_growth, list_physical_devices
import tensorflow as tf
from absl import logging as absl_logging

# Suppress TensorFlow logging
absl_logging.set_verbosity(absl_logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Configure GPU memory growth to prevent allocation issues
try:
    for _gpu in list_physical_devices('GPU'):
        set_memory_growth(_gpu, True)
except Exception:
    pass

# Utility imports
from utils import preprocess_for_detection, get_data_from_bids, index_of_union_threshold, clean_labels

# Parallel execution
from pqdm.threads import pqdm

# Get the project root (parent directory of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

from DynaSD import ABSSLP, IMPRINT, WVNT, HFER
from config import Config

# Get paths from config 
datapath,prodatapath,figpath,metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])


# Set default colormap for visualizations
plt.rcParams['image.cmap'] = 'magma'

# Global configuration
OVERWRITE = False  # Whether to overwrite existing probability matrix files

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

def find_optimal_phi_threshold(y_true, y_scores):
    """Find threshold that maximizes phi/MCC score."""
    thresholds = np.unique(y_scores)
    best_phi = -1
    best_threshold = 0
    
    for threshold in thresholds:
        y_pred = y_scores > threshold
        if len(np.unique(y_pred)) > 1:  # Ensure both classes are predicted
            phi = matthews_corrcoef(y_true, y_pred)
            if phi > best_phi:
                best_phi = phi
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
    Parallelizable worker: loads seizure, preprocesses, then evaluates all models for that seizure.

    Parameters
    ----------
    task : tuple
        (
            patient: str,
            onset_run: str,
            montage: str,
            onset_time_sec: float,
            model_classes: list[type]
        )

    Returns
    -------
    list[dict]
        One dict per model with summary metrics for this seizure.
    """
    (
        patient,
        onset_run,
        split,
        stim,
        onset_labels,
        montage,
        onset_time_sec,
        model_classes,
    ) = params

    # Check if all models already have probability files saved (without loading EEG)
    all_models_exist = True
    model_paths = {}
    #TODO add forecast to the output path
    for model_class in model_classes:
        # Get model name
        model_name = getattr(model_class, '__name__','UNKNOWN')
        
        # Create output directory and path using onset_run as task name
        out_dir = ospj(prodatapath, 'sz_prob', patient)
        out_path = ospj(out_dir, f"{patient}_task-ictal{onset_run}_run-*_mdl-{model_name}_sz_prob.pkl")
        model_paths[model_class] = out_path
        
        if not glob.glob(out_path) or OVERWRITE:
            # print(f"All models do not exist for {patient} {onset_run} {model_class}")
            all_models_exist = False
            break
    
    # If all models exist and we're not overwriting, process existing files without loading EEG
    if all_models_exist:
        # print(f"All models exist for {patient} {onset_run}")
        results: list[dict] = []
        for model_class in model_classes:
            model_name = getattr(model_class, '__name__','UNKNOWN')
                
            out_path = glob.glob(model_paths[model_class])[0]
            sz_prob = pd.read_pickle(out_path)
            sz_prob_times = sz_prob.pop('time')
            onset_mask = [ch.split('-')[0] in onset_labels for ch in sz_prob.columns]
            
            if len(np.unique(onset_mask)) == 2:
                onset_idx = int(np.argmin(np.abs(sz_prob_times - onset_time_sec)))
                onset_odx = int(np.argmin(np.abs(sz_prob_times - (onset_time_sec + 3))))
                onset_prob = sz_prob.iloc[onset_idx:onset_odx, :].mean()
                
                # IoU-optimized threshold metrics
                iou_threshold, iou_sensitivity, iou_specificity, auc, iou_f1, iou_phi = get_metrics(onset_mask, onset_prob)
                # Calculate additional IoU metrics (precision, recall)
                iou_pred = onset_prob > iou_threshold
                iou_precision = precision_score(onset_mask, iou_pred)
                iou_recall = recall_score(onset_mask, iou_pred)
                
                # F1-optimized threshold metrics
                f1_threshold = find_optimal_f1_threshold(onset_mask, onset_prob)
                f1_metrics = compute_all_metrics(onset_mask, onset_prob, f1_threshold)
                
                # Phi-optimized threshold metrics
                phi_threshold = find_optimal_phi_threshold(onset_mask, onset_prob)
                phi_metrics = compute_all_metrics(onset_mask, onset_prob, phi_threshold)

                results.append(
                    dict(
                        patient=patient,
                        onset=int(onset_run),
                        split=split,
                        stim=stim,
                        model=model_name,
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
                        # Phi-optimized metrics
                        phi_threshold=phi_threshold,
                        phi_f1=phi_metrics['f1'],
                        phi_phi=phi_metrics['phi'],
                        phi_sensitivity=phi_metrics['sensitivity'],
                        phi_specificity=phi_metrics['specificity'],
                        phi_precision=phi_metrics['precision'],
                        phi_recall=phi_metrics['recall'],
                    )
                )
            else:
                results.append(
                    dict(
                        patient=patient,    
                        onset=int(onset_run),
                        split=split,
                        stim=stim,
                        model=model_name,
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
                        # Phi-optimized metrics
                        phi_threshold=np.nan,
                        phi_f1=np.nan,
                        phi_phi=np.nan,
                        phi_sensitivity=np.nan,
                        phi_specificity=np.nan,
                        phi_precision=np.nan,
                        phi_recall=np.nan,
                    )
                )
        
        return results

    # Load seizure recording (only if not all models exist)
    seizure, fs_raw, _, _, task, run = get_data_from_bids(
        ospj(datapath, "BIDS"), patient, onset_run, return_path=True, verbose=0
    )

    # Clean labels
    seizure.columns = clean_labels(seizure.columns, patient)

    # Initial mask from first 60s
    _, _, channel_mask = preprocess_for_detection(
        seizure.iloc[: 120 * fs_raw, :],
        fs_raw,
        montage,
        target=fs_raw,
        wavenet=False,
        pre_mask=None,
    )

    results: list[dict] = []
    for model_class in model_classes:

        # Get model name
        model_name = getattr(model_class, '__name__','UNKNOWN')

        # Create output directory
        out_dir = ospj(prodatapath, 'sz_prob', patient)
        os.makedirs(out_dir, exist_ok=True)

        # Check if output file already exists
        all_paths_exist = True
        out_path = ospj(out_dir, f"{patient}_task-ictal{onset_run}_run-{run}_mdl-{model_name}_sz_prob.pkl")
        if glob.glob(out_path) and not OVERWRITE:  
            sz_prob = pd.read_pickle(out_path)
            sz_prob_times = sz_prob.pop('time')
            onset_mask = [ch.split('-')[0] in onset_labels for ch in sz_prob.columns]
        else:
            all_paths_exist = False
        if not all_paths_exist:
            # print(f"All paths do not exist for {patient} {onset_run} {model_name}")
            # Preprocess seizure
            wavecheck = model_class == WVNT
            seizure_pre, fs = preprocess_for_detection(
                seizure,
                fs_raw,
                montage,
                target=fs_raw,
                wavenet=wavecheck,
                pre_mask=channel_mask,
            )

            art_channel_mask = seizure_pre.loc[180*fs:,:].abs().max() <= (np.median(seizure_pre.loc[180*fs:,:].abs().max())*50)
            seizure_nart = seizure_pre.loc[:,art_channel_mask]

            onset_mask = [ch.split('-')[0] in onset_labels for ch in seizure_nart.columns]

            if wavecheck:
                model = model_class(fs=fs, w_size=1, w_stride = 0.5, 
                model_path = ospj(prodatapath,'CHECKPOINTS/WaveNet/v111.hdf5'),
                verbose = False,
                batch_size = 512)
            else:
                model = model_class(fs=fs, w_size=1, w_stride=0.5)

            model.fit(seizure_nart.iloc[: fs * 120])

            sz_prob = model(seizure_nart)
            sz_prob_times = model.get_win_times(len(seizure_nart))
            sz_prob_df = pd.concat((sz_prob,pd.Series(sz_prob_times,name='time')),axis=1)
            out_path = ospj(out_dir, f"{patient}_task-ictal{onset_run}_run-{run}_mdl-{model_name}_sz_prob.pkl")
            sz_prob_df.to_pickle(out_path)

        if len(np.unique(onset_mask)) == 2:
            onset_idx = int(np.argmin(np.abs(sz_prob_times - onset_time_sec)))
            onset_odx = int(np.argmin(np.abs(sz_prob_times - (onset_time_sec + 3))))
            onset_prob = sz_prob.iloc[onset_idx:onset_odx, :].mean()
            
            # IoU-optimized threshold metrics
            iou_threshold, iou_sensitivity, iou_specificity, auc, iou_f1, iou_phi = get_metrics(onset_mask, onset_prob)
            # Calculate additional IoU metrics (precision, recall)
            iou_pred = onset_prob > iou_threshold
            iou_precision = precision_score(onset_mask, iou_pred)
            iou_recall = recall_score(onset_mask, iou_pred)
            
            # F1-optimized threshold metrics
            f1_threshold = find_optimal_f1_threshold(onset_mask, onset_prob)
            f1_metrics = compute_all_metrics(onset_mask, onset_prob, f1_threshold)
            
            # Phi-optimized threshold metrics
            phi_threshold = find_optimal_phi_threshold(onset_mask, onset_prob)
            phi_metrics = compute_all_metrics(onset_mask, onset_prob, phi_threshold)

            results.append(
                dict(
                    patient=patient,
                    onset=int(onset_run),
                    split=split,
                    stim=stim,
                    model=model_name,
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
                    # Phi-optimized metrics
                    phi_threshold=phi_threshold,
                    phi_f1=phi_metrics['f1'],
                    phi_phi=phi_metrics['phi'],
                    phi_sensitivity=phi_metrics['sensitivity'],
                    phi_specificity=phi_metrics['specificity'],
                    phi_precision=phi_metrics['precision'],
                    phi_recall=phi_metrics['recall'],
                )
            )
        else:
            results.append(
                dict(
                    patient=patient,
                    onset=int(onset_run),
                    split=split,
                    stim=stim,
                    model=model_name,
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
                    # Phi-optimized metrics
                    phi_threshold=np.nan,
                    phi_f1=np.nan,
                    phi_phi=np.nan,
                    phi_sensitivity=np.nan,
                    phi_specificity=np.nan,
                    phi_precision=np.nan,
                    phi_recall=np.nan,
                )
            )

    return results


def main():
    """
    Main seizure detection pipeline using pre-trained models.
    
    Workflow:
    1. Load configuration and seizure metadata from BIDS format
    2. Configure GPU settings for optimal performance  
    3. For each patient with available data:
       - Load interictal training data from BIDS
       - Clean electrode labels and localize neural channels
       - For each seizure recording:
         - Train model on interictal/early seizure data
         - Generate predictions across full seizure recording
         - Save probability matrices and visualizations
    4. Support three detection algorithms: LSTM, AbsSlope, WaveNet
    
    Models are trained patient-specifically on interictal data and applied
    to detect seizure onset patterns in ictal recordings.
    """
    # Configure GPU memory growth for TensorFlow/PyTorch compatibility
    gpus = list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Load seizure metadata from BIDS processing
    seizures_df = pd.read_csv(ospj(metapath,"metadata_v7_BIDS.csv"))
    seizures_df['stim'] = seizures_df['stim'].fillna(0)
    seizures_df = seizures_df[(seizures_df.split == 2) & (seizures_df.stim == 1)] # Filter for only seizures that have soft onset labels
    
    # Detection parameters
    onset_time = 180          # Seizure onset time in recording (seconds)
    montage = 'bipolar'       # Electrode montage for preprocessing
    all_models = [ABSSLP,IMPRINT,WVNT,HFER]  # Models to run

    # Build all tasks across all patients and seizures (models are handled inside)
    tasks = []
    
    # Iterating through each patient that we have annotations for
    pbar = tqdm(seizures_df.iterrows(),total=len(seizures_df))
    for _,row in pbar:
        pt = row.Patient
        sz = row.onset
        split = row.split
        stim = row.stim
        pbar.set_description(desc=f"Patient: {pt} | Seizure: {sz}",refresh=True)
        onset_run = str(int(row.onset))
        if not isinstance(row.SOZ, str):
            onset_labels = []
        else:
            onset_labels = clean_labels([l.strip() for l in row.SOZ.split(',')], pt)
        tasks.append((pt, onset_run, split, stim,onset_labels, montage, onset_time, all_models))

    # Execute all tasks in parallel (each task loads seizure and runs all models)
    # y = []
    # for task in tasks[-5:]:
    #     y.append(run_model_task(task))
    # results_nested = pqdm(tasks, run_model_task, n_jobs=12)
    results_nested = []
    for task in tasks:
        results_nested.append(run_model_task(task))
    # Filter out exceptions and flatten list of lists into a single list of dicts
    flat_results = []
    for result in results_nested:
        if isinstance(result, Exception):
            # flat_results.append(dict(patient=None,onset=None,model=None,auc=np.nan,sens=np.nan,spec=np.nan,threshold=np.nan))
            print(f"Task failed with error: {result}")
            continue
        for mdl in result:
            flat_results.append(mdl)

    result_df = pd.DataFrame(flat_results)
    # print(result_df)
    # result_df.to_csv(ospj(prodatapath,f"benchmark_model_validation_results.csv"),index=False)
    
if __name__ == "__main__":
    main()
