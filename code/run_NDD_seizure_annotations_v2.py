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

# Utility imports
from utils import preprocess_for_detection, get_data_from_bids, index_of_union_threshold, clean_labels

# Parallel execution
from pqdm.threads import pqdm

# Get the project root (parent directory of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

from DynaSD import ABSSLP, IMPRINT, WVNT, HFER, LiNDDA, GIN
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
        onset_labels,
        montage,
        onset_time_sec,
        model_dicts,
    ) = params
    try:
        # Check if all models already have probability files saved (without loading EEG)
        all_models_exist = True
        model_paths = {}
        
        for model_dict in model_dicts:
            model_class = model_dict['model']
            sequence_length = model_dict['sequence_length']
            forecast_length = model_dict['forecast_length']
            model_name = getattr(model_class, '__name__','UNKNOWN')
            
            # Check all three output file types
            out_dir = ospj(prodatapath, 'sz_prob', patient)
            base_path = ospj(out_dir, f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_sz_prob_forecast-{forecast_length}.pkl")
            
            model_paths[f"{model_name}_{sequence_length}_{forecast_length}"] = {
                'mse_prob': base_path.replace('sz_prob_', 'mse_prob_'),
                'mse_z_prob': base_path.replace('sz_prob_', 'mse_z_prob_'),
                'mse_zs_prob': base_path.replace('sz_prob_', 'mse_zs_prob_')
            }
            
            # Check if any file is missing
            for file_path in model_paths[f"{model_name}_{sequence_length}_{forecast_length}"].values():
                if not os.path.exists(file_path) or OVERWRITE:
                    all_models_exist = False
                    break
            
            if not all_models_exist:
                break
        
        # If all models exist and we're not overwriting, process existing files without loading EEG
        if all_models_exist:
            results: list[dict] = []
            
            for model_dict in model_dicts:
                model_class = model_dict['model']
                sequence_length = model_dict['sequence_length']
                forecast_length = model_dict['forecast_length']
                model_name = getattr(model_class, '__name__','UNKNOWN')
                
                paths = model_paths[f"{model_name}_{sequence_length}_{forecast_length}"]
                
                for metric, file_path in zip(['mse', 'mse_z', 'mse_zs'], [paths['mse_prob'], paths['mse_z_prob'], paths['mse_zs_prob']]):
                    df = pd.read_pickle(file_path)
                    sz_prob_times = df.pop('time')
                    onset_mask = [ch.split('-')[0] in onset_labels for ch in df.columns]
                    
                    if len(np.unique(onset_mask)) == 2:
                        onset_idx = int(np.argmin(np.abs(sz_prob_times - onset_time_sec)))
                        onset_odx = int(np.argmin(np.abs(sz_prob_times - (onset_time_sec + 3))))
                        onset_prob = df.iloc[onset_idx:onset_odx, :].mean()
                        
                        # IoU-optimized threshold metrics
                        iou_threshold, iou_sensitivity, iou_specificity, auc, iou_f1, iou_phi = get_metrics(onset_mask, onset_prob)
                        iou_pred = onset_prob > iou_threshold
                        iou_precision = precision_score(onset_mask, iou_pred)
                        iou_recall = recall_score(onset_mask, iou_pred)
                        
                        # F1-optimized threshold metrics
                        f1_threshold = find_optimal_f1_threshold(onset_mask, onset_prob)
                        f1_metrics = compute_all_metrics(onset_mask, onset_prob, f1_threshold)

                        results.append(
                            dict(
                                patient=patient,
                                onset=int(onset_run),
                                split=split,
                                model=model_name+'_'+metric,
                                sequence=sequence_length,
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
                                split=split,
                                model=model_name+'_'+metric,
                                sequence=sequence_length,
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
        for model_dict in model_dicts:
            model_class = model_dict['model']
            sequence_length = model_dict['sequence_length']
            forecast_length = model_dict['forecast_length']
            # Get model name
            model_name = getattr(model_class, '__name__','UNKNOWN')

            # Create output directory
            out_dir = ospj(prodatapath, 'sz_prob', patient)
            os.makedirs(out_dir, exist_ok=True)

            # Preprocess seizure
            seizure_pre, fs = preprocess_for_detection(
                seizure,
                fs_raw,
                montage,
                target=fs_raw,
                wavenet=False,
                pre_mask=channel_mask,
            )

            art_channel_mask = seizure_pre.loc[180*fs:,:].abs().max() <= (np.median(seizure_pre.loc[180*fs:,:].abs().max())*50)
            seizure_nart = seizure_pre.loc[:,art_channel_mask]
            if len(seizure_nart.columns) == 0:
                print(f"No channels kept found for {patient} {onset_run}")
                return []
            onset_mask = [ch.split('-')[0] in onset_labels for ch in seizure_nart.columns]
            
            batch_size = 2048
            val_split = 0.1
            early_stopping = True
            patience = 1
            verbose = False

            if model_class == LiNDDA:
                model = model_class(
                    fs = fs,
                    w_size = 1,
                    w_stride = 0.5,
                    sequence_length = sequence_length,
                    forecast_length = forecast_length,
                    closeform = True,
                    batch_size = batch_size,
                    verbose = verbose,
                )
            elif model_class == GIN:
                model = model_class(
                fs=fs,
                w_size=1,
                w_stride=0.5,
                sequence_length = sequence_length,
                forecast_length = forecast_length,
                batch_size = batch_size,
                val_split = val_split,
                patience = patience,
                lr = 0.01,
                hidden_size=10 if sequence_length == 12 else seizure_nart.shape[1],
                num_layers=1,
                num_stacks=1,
                num_epochs = 10 if sequence_length == 12 else 100,
                verbose=verbose,
                use_cuda=True,
                early_stopping = False if sequence_length == 12 else early_stopping
                )
            else:
                raise ValueError(f"Model {model_class} not supported")

            model.fit(seizure_nart.iloc[:120*fs,:])
            out_path = ospj(out_dir, f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_sz_prob_forecast-{forecast_length}.pkl")
            
            mse_zs_prob = model(seizure_nart)
            mse_prob = model.mse_df
            mse_z_prob = model.mse_z_df.abs()
            sz_prob_times = model.get_win_times(len(seizure_nart))
            # sz_prob_df = pd.concat((sz_prob,pd.Series(sz_prob_times,name='time')),axis=1)
            mse_prob_df = pd.concat((mse_prob,pd.Series(sz_prob_times,name='time')),axis=1)
            mse_z_prob_df = pd.concat((mse_z_prob,pd.Series(sz_prob_times,name='time')),axis=1)
            mse_zs_prob_df = pd.concat((mse_zs_prob,pd.Series(sz_prob_times,name='time')),axis=1)
            # sz_prob_df.to_pickle(out_path)
            mse_prob_df.to_pickle(out_path.replace('sz_prob_','mse_prob_'))
            mse_z_prob_df.to_pickle(out_path.replace('sz_prob_','mse_z_prob_'))
            mse_zs_prob_df.to_pickle(out_path.replace('sz_prob_','mse_zs_prob_'))

            onset_idx = int(np.argmin(np.abs(sz_prob_times - onset_time_sec)))
            onset_odx = int(np.argmin(np.abs(sz_prob_times - (onset_time_sec + 3))))
            # onset_prob = sz_prob.iloc[onset_idx:onset_odx, :].mean()
            onset_mse = mse_prob.iloc[onset_idx:onset_odx,:].mean()
            onset_mse_z = mse_z_prob.iloc[onset_idx:onset_odx,:].mean()
            onset_mse_zs = mse_zs_prob.iloc[onset_idx:onset_odx,:].mean()
            for df,metric in zip([onset_mse,onset_mse_z,onset_mse_zs],['mse','mse_z','mse_zs']):
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
                            split=split,
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
                            split=split,
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
        return e

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
    
    # Load seizure metadata from BIDS processing
    seizures_df = pd.read_csv(ospj(metapath,"metadata_v6_BIDS.csv"))
    seizures_df = seizures_df[seizures]
    # seizures_df = seizures_df[seizures_df.split == 1] # Filter for only seizures that have soft onset labels
    
    # Detection parameters
    onset_time = 180          # Seizure onset time in recording (seconds)
    montage = 'bipolar'       # Electrode montage for preprocessing
    # all_models = [{'model': LiNDDA, 'sequence_length': 1},{'model':GIN,'sequence_length':12}]
    # all_models = [{'model': LiNDDA, 'sequence_length': 1}]
    # all_models = [{'model': LiNDDA, 'sequence_length': 32}]
    all_models = [
        {'model': LiNDDA, 'sequence_length': 1, 'forecast_length': 1},
        {'model': LiNDDA, 'sequence_length': 32, 'forecast_length': 1},
        {'model': GIN, 'sequence_length': 12, 'forecast_length': 1},
    ]
    # all_models = [
    #     {'model': LiNDDA, 'sequence_length': 2, 'forecast_length': 1},
    #     {'model': LiNDDA, 'sequence_length': 4, 'forecast_length': 1},
    #     {'model': LiNDDA, 'sequence_length': 8, 'forecast_length': 1},
    # ]


    # Build all tasks across all patients and seizures (models are handled inside)
    tasks = []
    
    # Iterating through each patient that we have annotations for
    pbar = tqdm(seizures_df.iterrows(),total=len(seizures_df))
    for _,row in pbar:
        pt = row.Patient
        sz = row.onset
        split = row.split
        pbar.set_description(desc=f"Patient: {pt} | Seizure: {sz}",refresh=True)
        onset_run = str(int(row.onset))
        if not isinstance(row.SOZ, str):
            onset_labels = []
        else:
            onset_labels = clean_labels([l.strip() for l in row.SOZ.split(',')], pt)
        tasks.append((pt, onset_run, split, onset_labels, montage, onset_time, all_models))

    # Execute all tasks in parallel (each task loads seizure and runs all models)
    results_nested = []
    for task in tqdm(tasks,total=len(tasks)):
        results_nested.append(run_model_task(task))
    # results_nested = pqdm(tasks[100:], run_model_task, n_jobs=16)

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
    result_df.to_csv(ospj(prodatapath,f"ndd_model_validation_results_v2.csv"),index=False)

if __name__ == "__main__":
    main()
