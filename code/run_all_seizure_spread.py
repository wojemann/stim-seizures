# File system imports
import sys
import os
import glob
from os.path import join as ospj
import pickle

# Scientific imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy as sc

# Utility imports
from utils import clean_labels

# Sklearn imports
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc as sklearn_auc

# Get the project root (parent directory of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

# Import all models
from DynaSD import HFER, ABSSLP, IMPRINT, WVNT, LiNDDA, GIN, MINDD, NDD

from config import Config

# Get paths from config 
datapath, prodatapath, figpath, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])

FILE_KEYS = ['f1', 'iou', 'phi', 'f1_plateau', 'phi_plateau']  # Different threshold optimization metrics


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

def get_all_models():
    """
    Define all models to process with their configuration.
    Returns a list of model config dictionaries.
    """
    models = []
    
    # Benchmark models
    for model_name, model_class in [('ABSSLP', ABSSLP), ('IMPRINT', IMPRINT), 
                                     ('WVNT', WVNT), ('HFER', HFER)]:
        models.append({
            'key': model_name,
            'name': model_name,
            'type': 'benchmark',
            'class': model_class,
        })
    
    # NDD models
    ndd_configs = [
        ('LiNDDA', LiNDDA, 1, 1), ('LiNDDA', LiNDDA, 2, 1), ('LiNDDA', LiNDDA, 3, 2),
        ('LiNDDA', LiNDDA, 4, 3), ('LiNDDA', LiNDDA, 5, 4), ('LiNDDA', LiNDDA, 6, 5),
        ('LiNDDA', LiNDDA, 7, 6),
        ('GIN', GIN, 4, 1), ('GIN', GIN, 8, 1), ('GIN', GIN, 12, 1),
        ('MINDD', MINDD, 3, 2), ('NDD', NDD, 12, 1),
    ]
    
    for name, cls, seq_len, forecast in ndd_configs:
        key = f"{name}_mse_sl{seq_len}_fl{forecast}"
        models.append({
            'key': key,
            'name': name,
            'type': 'ndd',
            'class': cls,
            'sequence_length': seq_len,
            'forecast_length': forecast,
            'metric': 'mse',
            'suffix': '',
        })
        if name == 'GIN':
            models[-1]['suffix'] = 'nopass_nolayernorm'
    
    return models


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================

def load_prob_file(patient, onset_run, model_config):
    """
    Load probability file for a given model configuration.
    
    Args:
        patient: Patient ID
        onset_run: Seizure onset run number
        model_config: Model configuration dictionary
        
    Returns:
        DataFrame with probability data or None if file not found
    """
    prob_dir = ospj(prodatapath, 'sz_prob', patient)
    
    if model_config['type'] == 'benchmark':
        pattern = f"{patient}_task-ictal{onset_run}_run-*_mdl-{model_config['name']}_sz_prob.pkl"
    else:  # ndd
        name = model_config['name']
        seq_len = model_config['sequence_length']
        forecast = model_config['forecast_length']
        metric = model_config['metric']
        suffix = model_config.get('suffix', '')
        
        pattern = f"{patient}_task-ictal{onset_run}_mdl-{name}_seq-{seq_len}"
        pattern += f"_{metric}_prob_forecast-{forecast}{suffix}.pkl"
    
    prob_paths = glob.glob(ospj(prob_dir, pattern))
    if prob_paths:
        return pd.read_pickle(prob_paths[0])
    return None


def create_model(model_config, num_channels=None):
    """
    Create model instance based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        num_channels: Number of channels (needed for some NDD models)
        
    Returns:
        Model instance
    """
    model_class = model_config['class']
    
    if model_config['type'] == 'benchmark':
        if model_config['name'] == 'WVNT':
            return model_class(
                fs=128, 
                w_size=1, 
                w_stride=0.5,
                model_path='',
                verbose=False,
                batch_size=512
            )
        else:
            return model_class(w_size=1, w_stride=0.5, fs=256)
    
    else:  # ndd
        seq_len = model_config['sequence_length']
        forecast = model_config['forecast_length']
        
        if model_class == LiNDDA:
            return model_class(
                fs = 256,
                w_size = 1,
                w_stride = 0.5,
                sequence_length = seq_len,
                forecast_length = forecast,
                closeform = False,
                verbose = False
            )
        elif model_class == GIN:
            return model_class(
                fs=256,
                w_size=1,
                w_stride=0.5,
                sequence_length=seq_len,
                forecast_length=forecast,
                verbose=False,
            )
        elif model_class == MINDD:
            return model_class(
                fs=256,
                w_size=1,
                w_stride=0.5,
                sequence_length=seq_len,
                forecast_length=forecast,
                verbose=False,
            )
        elif model_class == NDD:
            return model_class(
                fs=256,
                w_size=1,
                w_stride=0.5,
                sequence_length=seq_len,
                forecast_length=forecast,
                verbose=False,
            )
        else:
            raise ValueError(f"Unsupported NDD model class: {model_class}")


def preprocess_prob_data(prob_data, prob_times, model_config, patient, onset_run):
    """
    Preprocess probability data (extract time, drop NaNs, filter, trim).
    
    Args:
        prob_data: Raw probability DataFrame
        prob_times: Time array
        model_config: Model configuration
        patient: Patient ID
        onset_run: Seizure run
        
    Returns:
        (processed_prob_data, prob_times) or (None, None) if invalid
    """
    # Check window stride for benchmark models
    if model_config['type'] == 'benchmark' and np.diff(prob_times).mean() < 0.5:
        print(f"Warning: Incorrect window stride for {patient} {onset_run} {model_config['key']}")
        return None, None
    
    # Drop NaN columns and apply smoothing
    prob_data = prob_data.dropna(axis=1)
    prob_data = pd.DataFrame(
        sc.ndimage.uniform_filter1d(prob_data, size=20, mode='nearest', axis=0, origin=0),
        columns=prob_data.columns
    )
    prob_times[np.isnan(prob_times)] = max(prob_times) + 0.5
    return prob_data, prob_times


def analyze_spread(model, prob_data, prob_times, threshold, onset_labels, model_config):
    """
    Run spread analysis for a model.
    
    Args:
        model: Model instance
        prob_data: Preprocessed probability data
        prob_times: Time array
        threshold: Detection threshold
        onset_labels: SOZ labels
        model_config: Model configuration
        
    Returns:
        Dictionary with spread metrics or None if analysis failed
    """
    # Define time windows
    onset_idx = int(np.argmin(np.abs(prob_times)))
    onset_odx = int(np.argmin(np.abs(prob_times - 3)))
    first_onset_idx = int(np.argmin(np.abs(prob_times - 180)))
    offset_idx = int(np.argmin(np.abs(prob_times - (prob_times.max() - 120))))
    
    # Extract seizure probability window
    sz_prob = prob_data.iloc[first_onset_idx:offset_idx, :]
    
    # Create onset mask
    onset_mask = np.array([ch.split('-')[0] in onset_labels for ch in sz_prob.columns])
    
    # Calculate AUC and AUPRC
    if sum(onset_mask) > 0:
        prob_scores = sz_prob.iloc[onset_idx:onset_odx, :].mean()
        auc_score = roc_auc_score(onset_mask, prob_scores)
        
        # Calculate AUPRC
        precision_vals, recall_vals, _ = precision_recall_curve(onset_mask, prob_scores)
        auprc_raw = sklearn_auc(recall_vals, precision_vals)
        
        # Calculate normalized AUPRC
        baseline = np.sum(onset_mask) / len(onset_mask)  # Fraction of positive samples
        auprc_normalized = (auprc_raw - baseline) / (1 - baseline) if baseline < 1 else np.nan
    else:
        auc_score = np.nan
        auprc_raw = np.nan
        auprc_normalized = np.nan
    
    # Run spread detection
    spread_df, sz_clf = model.get_onset_and_spread(
        sz_prob, 
        threshold=threshold, 
        ret_smooth_mat=True, 
        filter_w=10, 
        rwin_size=5, 
        rwin_req=4
    )

    if spread_df is None or spread_df.empty:
        return None
    
    spread_df.fillna(offset_idx - first_onset_idx, inplace=True)
    
    # Calculate onset metrics
    onset_idx_adj = onset_idx + int(spread_df.iloc[0].values[0])
    onset_odx_adj = onset_odx + int(spread_df.iloc[0].values[0])
    onset_pred = sz_clf.iloc[onset_idx_adj:onset_odx_adj, :].sum() > 0
    
    sensitivity = np.sum(onset_mask & onset_pred) / np.sum(onset_mask)
    specificity = np.sum(~onset_mask & ~onset_pred) / np.sum(~onset_mask)
    precision = precision_score(onset_mask, onset_pred, zero_division=0)
    recall = recall_score(onset_mask, onset_pred, zero_division=0)
    f1 = f1_score(onset_mask, onset_pred)
    phi = matthews_corrcoef(onset_mask, onset_pred)
    
    # Calculate recruitment times
    recruitment_indices = spread_df.iloc[0].values
    recruitment_times = prob_times[recruitment_indices.astype(int)]
    recruitment_times -= min(recruitment_times)
    channel_times = dict(zip(spread_df.columns, recruitment_times))
    
    # Filter for SOZ channels
    soz_channels = [ch for ch in spread_df.columns if ch.split('-')[0] in onset_labels]
    
    if len(soz_channels) == 0:
        return None
    
    # Calculate spread ranks
    sorted_channels = sorted(channel_times.items(), key=lambda x: x[1])
    channel_ranks = {ch: rank+1 for rank, (ch, _) in enumerate(sorted_channels)}
    total_channels = len(sorted_channels)
    
    soz_ranks = [channel_ranks[ch] for ch in soz_channels]
    soz_rank_percentages = [(rank / total_channels) * 100 for rank in soz_ranks]
    
    avg_soz_rank_pct = np.mean(soz_rank_percentages)
    median_soz_rank_pct = np.median(soz_rank_percentages)
    
    soz_channel_pct_adjustment = (len(soz_channels) / 2 / total_channels) * 100
    adjusted_soz_rank_pct = avg_soz_rank_pct - soz_channel_pct_adjustment
    adjusted_med_soz_rank_pct = median_soz_rank_pct - soz_channel_pct_adjustment
    
    # Calculate recruitment latencies
    soz_times = [channel_times[ch] for ch in soz_channels]
    avg_recruitment_latency = np.mean(soz_times)
    median_recruitment_latency = np.median(soz_times)
    max_recruitment_latency = np.max(soz_times)
    
    # Prepare result dictionary
    result = {
        'num_soz_channels': len(soz_channels),
        'total_channels': total_channels,
        'avg_soz_spread_rank_pct': avg_soz_rank_pct,
        'med_soz_spread_rank_pct': median_soz_rank_pct,
        'adj_avg_soz_spread_rank_pct': adjusted_soz_rank_pct,
        'adj_med_soz_spread_rank_pct': adjusted_med_soz_rank_pct,
        'avg_soz_recruitment_latency': avg_recruitment_latency,
        'med_soz_recruitment_latency': median_recruitment_latency,
        'max_soz_recruitment_latency': max_recruitment_latency,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'phi': phi,
        'auc': auc_score,
        'auprc_raw': auprc_raw,
        'auprc_normalized': auprc_normalized,
    }
    
    # Add NDD-specific metrics
    if model_config['type'] == 'ndd':
        soz = sz_prob.iloc[onset_idx:onset_odx, onset_mask].mean().mean()
        nsoz = sz_prob.iloc[onset_idx:onset_odx, ~onset_mask].mean().mean()
        result['soz'] = soz
        result['nsoz'] = nsoz
    
    return result


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """
    Main unified seizure spread analysis pipeline.
    
    Loads pre-computed probability matrices for all models and applies 
    model-specific thresholds to analyze seizure onset and spread patterns.
    """
    
    # Load seizure metadata
    seizures_df = pd.read_csv(ospj(metapath, "metadata_v7_BIDS.csv"))
    seizures_df = seizures_df[seizures_df.split == 1]
    seizures_df[['notes','source']] = seizures_df[['notes','source']].fillna('')
    seizures_df = seizures_df[seizures_df.notes.apply(lambda x: 'nina' not in x.lower())]
    seizures_df = seizures_df[seizures_df.source.apply(lambda x: 'nina' not in x.lower())]
    
    # Get all model configurations
    all_models = get_all_models()
    
    # Load all thresholds once
    thresholds_df = pd.read_csv('/Users/wojemann/local_data/dynasd_data/PROCESSED_DATA/all_thresholds_v6.csv')
    
    # Results storage for all threshold metrics
    all_spread_results = []
    
    # Process each seizure
    pbar = tqdm(seizures_df.iterrows(), total=len(seizures_df))
    for _, row in pbar:
        patient = row.Patient
        onset_run = str(int(row.onset))
        onset_labels = clean_labels([l.strip() for l in row.SOZ.split(',')], patient)
        
        pbar.set_description(f"Patient: {patient} | Seizure: {onset_run}")
        
        # Process each model
        for model_config in all_models:
            key = model_config['key']
            
            # Load probability file once per model/seizure
            prob_data = load_prob_file(patient, onset_run, model_config)
            if prob_data is None:
                print(f"Warning: No probability file found for {patient} {onset_run} {key}")
                continue
            
            # Extract time array
            if 'time' not in prob_data.columns:
                print(f"Warning: No time column found for {patient} {onset_run} {key}")
                continue
            
            prob_times = prob_data.pop('time').values
            
            # Preprocess data once per model/seizure
            prob_data, prob_times = preprocess_prob_data(
                prob_data, prob_times, model_config, patient, onset_run
            )
            if prob_data is None:
                continue
            
            # Process each threshold metric
            for FILE_KEY in FILE_KEYS:
                # Get thresholds for this metric
                thresholds_mean = thresholds_df[(thresholds_df.metric == FILE_KEY) & (thresholds_df.aggregation == 'mean')]
                thresholds_median = thresholds_df[(thresholds_df.metric == FILE_KEY) & (thresholds_df.aggregation == 'median')]
                threshold_dict_mean = dict(zip(thresholds_mean.model, thresholds_mean.threshold))
                threshold_dict_median = dict(zip(thresholds_median.model, thresholds_median.threshold))
                
                # Process with both mean and median aggregations
                for aggregation, threshold_dict in zip(['mean', 'median'], 
                                                       [threshold_dict_mean, threshold_dict_median]):
                    if key not in threshold_dict:
                        continue
                    
                    threshold = threshold_dict[key]
                    
                    # Create model instance
                    model = create_model(model_config, num_channels=len(prob_data.columns))
                    
                    # Run spread analysis
                    spread_metrics = analyze_spread(
                        model, prob_data.copy(), prob_times, threshold, 
                        onset_labels, model_config
                    )
                    
                    if spread_metrics is None:
                        continue
                    
                    # Compile result dictionary
                    result_dict = {
                        'patient': patient,
                        'onset': int(onset_run),
                        'model': key,
                        'model_name': model_config['name'],
                        'model_type': model_config['type'],
                        'aggregation': aggregation,
                        'threshold_metric': FILE_KEY,
                        'threshold': threshold,
                    }
                    
                    # Add type-specific fields
                    if model_config['type'] == 'ndd':
                        result_dict.update({
                            'metric': model_config['metric'],
                            'sequence_length': model_config['sequence_length'],
                            'forecast_length': model_config['forecast_length'],
                        })
                    else:
                        result_dict.update({
                            'sequence_length': np.nan,
                            'forecast_length': np.nan,
                        })
                    
                    # Add spread metrics
                    result_dict.update(spread_metrics)
                    
                    all_spread_results.append(result_dict)

            if model_config['type'] == 'ndd':
                model = create_model(model_config, num_channels=len(prob_data.columns))
                first_onset_idx = int(np.argmin(np.abs(prob_times - 180)))
                threshold = model.get_threshold(prob_data.iloc[first_onset_idx:,:], method='automedian')
                spread_metrics = analyze_spread(
                    model, prob_data.copy(), prob_times, threshold, 
                    onset_labels, model_config
                )
                result_dict = {
                    'patient': patient,
                    'onset': int(onset_run),
                    'model': key,
                    'model_name': model_config['name'],
                    'model_type': model_config['type'],
                    'aggregation': 'median',
                    'threshold_metric': 'tau',
                    'threshold': threshold,
                    'metric': model_config['metric'],
                    'sequence_length': model_config['sequence_length'],
                    'forecast_length': model_config['forecast_length'],
                }
                result_dict.update(spread_metrics)
                all_spread_results.append(result_dict)

    # Save all results to single file
    if all_spread_results:
        results_df = pd.DataFrame(all_spread_results)
        output_path = ospj(prodatapath, "all_models_spread_analysis_results_v8.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"All results saved to {output_path}")
        print(f"Total rows: {len(results_df)}")
        print(f"Threshold metrics: {results_df['threshold_metric'].unique()}")
        print(f"{'='*80}")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
