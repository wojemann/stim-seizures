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
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

# Get the project root (parent directory of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

# Import all models
from DynaSD import HFER, ABSSLP, IMPRINT, WVNT, LiNDDA, GIN, MINDD

from config import Config

# Get paths from config 
datapath, prodatapath, figpath, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])

FILE_KEYS = ['f1', 'iou', 'phi']  # Different threshold optimization metrics


def load_probability_files(patient, onset_run, benchmark_models, ndd_model_dicts):
    """Load probability files for all models (both benchmark and NDD) for a given patient/seizure"""
    prob_files = {}
    
    # Load benchmark model probability files
    for model_name in benchmark_models:
        prob_dir = ospj(prodatapath, 'sz_prob', patient)
        pattern = f"{patient}_task-ictal{onset_run}_run-*_mdl-{model_name}_sz_prob.pkl"
        prob_path = glob.glob(ospj(prob_dir, pattern))
        
        if prob_path:
            prob_files[model_name] = {
                'data': pd.read_pickle(prob_path[0]),
                'model_type': 'benchmark',
                'model_name': model_name,
            }
        else:
            print(f"Warning: No probability file found for {patient} {onset_run} {model_name}")
    
    # Load NDD model probability files
    for model_dict in ndd_model_dicts:
        model_class = model_dict['model']
        model_name = model_dict['model_name']
        sequence_length = model_dict['sequence_length']
        forecast_length = model_dict['forecast_length']
        suffix = model_dict.get('suffix', '')
        
        for metric in ['mse']:
            # Create the key for this combination
            key = f"{model_name}_{metric}_sl{sequence_length}_fl{forecast_length}{suffix}"
                
            # Find probability files
            prob_dir = ospj(prodatapath, 'sz_prob', patient)
            pattern = f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}"
            if metric == 'prob':
                pattern += f"_sz_prob_forecast-{forecast_length}"
            else:
                pattern += f"_{metric}_prob_forecast-{forecast_length}"
            
            # Apply suffix from model configuration
            pattern += f"{suffix}.pkl"
            
            prob_path = glob.glob(ospj(prob_dir, pattern))
            if prob_path:
                prob_files[key] = {
                    'data': pd.read_pickle(prob_path[0]),
                    'model_type': 'ndd',
                    'model_class': model_class,
                    'model_name': model_name,
                    'suffix': suffix,
                    'sequence_length': sequence_length,
                    'forecast_length': forecast_length,
                    'metric': metric
                }
            else:
                print(f"Warning: No probability file found for {patient} {onset_run} {key}")
    
    return prob_files


def main():
    """
    Main unified seizure spread analysis pipeline.
    
    Loads pre-computed probability matrices for both benchmark and NDD models
    and applies model-specific thresholds to analyze seizure onset and spread patterns.
    """
    
    # Load seizure metadata
    seizures_df = pd.read_csv(ospj(metapath, "metadata_v7_BIDS.csv"))
    seizures_df = seizures_df[seizures_df.split == 1]
    seizures_df[['notes','source']] = seizures_df[['notes','source']].fillna('')
    seizures_df = seizures_df[seizures_df.notes.apply(lambda x: 'nina' not in x.lower())]
    seizures_df = seizures_df[seizures_df.source.apply(lambda x: 'nina' not in x.lower())]
    
    # Benchmark models from run_seizure_spread.py
    benchmark_model_names = ['ABSSLP', 'IMPRINT', 'WVNT', 'HFER']
    benchmark_model_dict = {
        'ABSSLP': ABSSLP,
        'IMPRINT': IMPRINT,
        'WVNT': WVNT,
        'HFER': HFER
    }
    
    # NDD models from run_NDD_seizure_spread.py
    ndd_model_dicts = [
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 1, 'forecast_length': 1, 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 2, 'forecast_length': 1, 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 3, 'forecast_length': 2, 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 4, 'forecast_length': 3, 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 5, 'forecast_length': 4, 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 6, 'forecast_length': 5, 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 7, 'forecast_length': 6, 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 8, 'forecast_length': 7, 'suffix': ''},
        {'model': GIN, 'model_name': 'GIN', 'sequence_length': 4, 'forecast_length': 1, 'suffix': ''},
        {'model': GIN, 'model_name': 'GIN', 'sequence_length': 8, 'forecast_length': 1, 'suffix': ''},
        {'model': GIN, 'model_name': 'GIN', 'sequence_length': 12, 'forecast_length': 1, 'suffix': ''},
        {'model': MINDD, 'model_name': 'MINDD', 'sequence_length': 3, 'forecast_length': 2, 'suffix': ''},
        {'model': MINDD, 'model_name': 'NDD', 'sequence_length': 12, 'forecast_length': 1, 'suffix': ''},
    ]
    
    # Iterate through each threshold metric (FILE_KEY)
    for FILE_KEY in FILE_KEYS:
        print(f"\n{'='*80}")
        print(f"Processing with threshold metric: {FILE_KEY}")
        print(f"{'='*80}\n")
        
        # Load all_thresholds_v5.csv
        thresholds_df = pd.read_csv('/Users/wojemann/local_data/dynasd_data/PROCESSED_DATA/all_thresholds_v5.csv')
        # Filter for current metric and create dictionaries by aggregation
        thresholds_mean = thresholds_df[(thresholds_df.metric == FILE_KEY) & (thresholds_df.agg == 'mean')]
        thresholds_median = thresholds_df[(thresholds_df.metric == FILE_KEY) & (thresholds_df.agg == 'median')]
        threshold_dict_mean = dict(zip(thresholds_mean.model, thresholds_mean.threshold))
        threshold_dict_median = dict(zip(thresholds_median.model, thresholds_median.threshold))
        
        # Results storage
        spread_results = []
        
        # Process each seizure
        pbar = tqdm(seizures_df.iterrows(), total=len(seizures_df))
        for _, row in pbar:
            patient = row.Patient
            onset_run = str(int(row.onset))
            onset_labels = clean_labels([l.strip() for l in row.SOZ.split(',')], patient)
            
            pbar.set_description(f"Patient: {patient} | Seizure: {onset_run}")
            
            # Load probability files for this seizure (both benchmark and NDD)
            prob_files = load_probability_files(patient, onset_run, benchmark_model_names, ndd_model_dicts)
            
            # Process each model that has probability data
            for key, prob_info in prob_files.items():
                model_type = prob_info['model_type']
                prob_data = prob_info['data']
                
                # Extract time array
                if 'time' in prob_data.columns:
                    prob_times = prob_data.pop('time').values
                    # Check window stride for benchmark models
                    if model_type == 'benchmark' and np.diff(prob_times).mean() < 0.5:
                        print(f'Warning: Incorrect window stride for {patient} {onset_run} {key}')
                        continue
                else:
                    print(f"Warning: No time column found for {patient} {onset_run} {key}")
                    continue
                
                # Drop NaN columns for benchmark models
                if model_type == 'benchmark':
                    prob_data = prob_data.dropna(axis=1)
                
                # Process with both mean and median aggregations
                for aggregation, threshold_dict in zip(['mean', 'median'], [threshold_dict_mean, threshold_dict_median]):
                    if key in threshold_dict:
                        threshold = threshold_dict[key]
                        prob_data = prob_info['data'].copy()  # Re-copy for each aggregation
                        if 'time' in prob_data.columns:
                            prob_data.pop('time')
                        if model_type == 'benchmark':
                            prob_data = prob_data.dropna(axis=1)
                        
                        # Create model instance based on type
                        if model_type == 'benchmark':
                            model_name = prob_info['model_name']
                            if model_name == 'WVNT':
                                model = benchmark_model_dict[model_name](
                                    fs=128, 
                                    w_size=1, 
                                    w_stride=0.5,
                                    model_path='',
                                    verbose=False,
                                    batch_size=512
                                )
                            else:
                                model = benchmark_model_dict[model_name](w_size=1, w_stride=0.5, fs=256)
                            
                            # Benchmark model spread analysis
                            onset_mask = np.array([ch.split('-')[0] in onset_labels for ch in prob_data.columns])
                            onset_idx = int(np.argmin(np.abs(prob_times)))
                            onset_odx = int(np.argmin(np.abs(prob_times - 3)))
                            
                            first_onset_idx = int(np.argmin(np.abs(prob_times - 180)))
                            offset_idx = int(np.argmin(np.abs(prob_times - (prob_times.max() - 120))))
                            sz_prob = prob_data.iloc[first_onset_idx:offset_idx, :]
                            sz_prob = pd.DataFrame(
                                sc.ndimage.uniform_filter1d(sz_prob, size=20, mode='nearest', axis=0, origin=0),
                                columns=sz_prob.columns
                            )
                            
                            spread_df, sz_clf = model.get_onset_and_spread(
                                sz_prob, 
                                threshold=threshold, 
                                ret_smooth_mat=True, 
                                filter_w=10, 
                                rwin_size=5, 
                                rwin_req=4
                            )
                            spread_df.fillna(offset_idx - first_onset_idx, inplace=True)
                            
                            auc = roc_auc_score(onset_mask, sz_prob.iloc[onset_idx:onset_odx, :].mean()) if sum(onset_mask) > 0 else np.nan
                            
                            if spread_df is not None and not spread_df.empty:
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
                                
                                if len(soz_channels) > 0:
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
                                    
                                    soz_times = [channel_times[ch] for ch in soz_channels]
                                    avg_recruitment_latency = np.mean(soz_times)
                                    median_recruitment_latency = np.median(soz_times)
                                    max_recruitment_latency = np.max(soz_times)
                                    
                                    result_dict = {
                                        'patient': patient,
                                        'onset': int(onset_run),
                                        'model': model_name,
                                        'model_type': 'benchmark',
                                        'aggregation': aggregation,
                                        'threshold_metric': FILE_KEY,
                                        'threshold': threshold,
                                        'sequence_length': np.nan,
                                        'forecast_length': np.nan,
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
                                        'auc': auc,
                                    }
                                    spread_results.append(result_dict)
                                else:
                                    print(f"Warning: No SOZ channels found in spread data for {patient} {onset_run} {key}")
                            else:
                                print(f"Warning: No spread data returned for {patient} {onset_run} {key}")
                        
                        elif model_type == 'ndd':
                            # NDD model spread analysis
                            model_class = prob_info['model_class']
                            model_name = prob_info['model_name']
                            sequence_length = prob_info['sequence_length']
                            forecast_length = prob_info['forecast_length']
                            metric = prob_info['metric']
                            
                            # Create model instance
                            if model_class == LiNDDA:
                                model = model_class(
                                    fs=256,
                                    w_size=1,
                                    w_stride=0.5,
                                    sequence_length=sequence_length,
                                    forecast_length=forecast_length,
                                    closeform=True,
                                    batch_size=2048,
                                    verbose=False,
                                )
                            elif model_class == GIN:
                                model = model_class(
                                    fs=256,
                                    w_size=1,
                                    w_stride=0.5,
                                    sequence_length=sequence_length,
                                    forecast_length=forecast_length,
                                    batch_size=2048,
                                    val_split=0.1,
                                    patience=1,
                                    lr=0.01,
                                    hidden_size=10 if sequence_length == 12 else len(prob_data.columns),
                                    num_layers=1,
                                    num_stacks=1,
                                    num_epochs=10 if sequence_length == 12 else 100,
                                    verbose=False,
                                    use_cuda=False,
                                    early_stopping=False if sequence_length == 12 else True
                                )
                            elif model_class == MINDD:
                                model = model_class(
                                    fs=256,
                                    w_size=1,
                                    w_stride=0.5,
                                    sequence_length=sequence_length,
                                    forecast_length=forecast_length,
                                )
                            else:
                                print(f"Warning: Unsupported model class {model_class}")
                                continue
                            
                            first_onset_idx = int(np.argmin(np.abs(prob_times - 180)))
                            offset_idx = int(np.argmin(np.abs(prob_times - (prob_times.max() - 120))))
                            sz_prob = prob_data.iloc[first_onset_idx:offset_idx, :]
                            sz_prob = pd.DataFrame(
                                sc.ndimage.uniform_filter1d(sz_prob, size=20, mode='nearest', axis=0, origin=0),
                                columns=sz_prob.columns
                            )
                            
                            # Calculate AUC before spread detection
                            onset_idx_auc = int(np.argmin(np.abs(prob_times)))
                            onset_odx_auc = int(np.argmin(np.abs(prob_times - 3)))
                            onset_mask = np.array([ch.split('-')[0] in onset_labels for ch in sz_prob.columns])
                            auc = roc_auc_score(onset_mask, sz_prob.iloc[onset_idx_auc:onset_odx_auc, :].mean()) if sum(onset_mask) > 0 else np.nan
                            
                            spread_df, sz_clf = model.get_onset_and_spread(sz_prob, threshold=threshold, ret_smooth_mat=True)
                            spread_df.fillna(offset_idx - first_onset_idx, inplace=True)
                            
                            if spread_df is not None and not spread_df.empty:
                                onset_idx = int(np.argmin(np.abs(prob_times)))
                                onset_odx = int(np.argmin(np.abs(prob_times - 3)))
                                
                                # Calculate onset metrics
                                onset_idx += int(spread_df.iloc[0].values[0])
                                onset_odx += int(spread_df.iloc[0].values[0])
                                onset_pred = sz_clf.iloc[onset_idx:onset_odx, :].sum() > 0
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
                                
                                if len(soz_channels) > 0:
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
                                    
                                    soz_times = [channel_times[ch] for ch in soz_channels]
                                    avg_recruitment_latency = np.mean(soz_times)
                                    median_recruitment_latency = np.median(soz_times)
                                    max_recruitment_latency = np.max(soz_times)
                                    
                                    # Calculate raw NDD values
                                    soz = sz_prob.iloc[onset_idx_auc:onset_odx_auc, onset_mask].mean().mean()
                                    nsoz = sz_prob.iloc[onset_idx_auc:onset_odx_auc, ~onset_mask].mean().mean()
                                    
                                    result_dict = {
                                        'patient': patient,
                                        'onset': int(onset_run),
                                        'model': key,
                                        'model_type': 'ndd',
                                        'model_name': model_name,
                                        'aggregation': aggregation,
                                        'threshold_metric': FILE_KEY,
                                        'threshold': threshold,
                                        'metric': metric,
                                        'sequence_length': sequence_length,
                                        'forecast_length': forecast_length,
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
                                        'soz': soz,
                                        'nsoz': nsoz,
                                        'auc': auc,
                                    }
                                    spread_results.append(result_dict)
                                else:
                                    print(f"Warning: No SOZ channels found in spread data for {patient} {onset_run} {key}")
                            else:
                                print(f"Warning: No spread data found for {patient} {onset_run} {key} with threshold {threshold}")
                    else:
                        print(f"Warning: No threshold found for model {key}")
        
        # Save results for this FILE_KEY
        if spread_results:
            results_df = pd.DataFrame(spread_results)
            output_path = ospj(prodatapath, f"all_models_spread_analysis_results_{FILE_KEY}_v5.csv")
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            print(f"No results to save for FILE_KEY: {FILE_KEY}")


if __name__ == "__main__":
    main()

