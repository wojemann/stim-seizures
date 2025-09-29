# File system imports
import sys
import os
import glob
from os.path import join as ospj

# Scientific imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Utility imports
from utils import get_data_from_bids, clean_labels

# Sklearn imports
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

# Get the project root (parent directory of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

from DynaSD import LiNDDA, GIN

from config import Config

# Get paths from config 
datapath, prodatapath, figpath, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])


def load_probability_files(patient, onset_run, model_dicts, all_forecasts):
    """Load probability files for all model-sequence-forecast combinations for a given patient/seizure"""
    prob_files = {}
    
    for model_dict in model_dicts:
        model_class = model_dict['model']
        sequence_length = model_dict['sequence_length']
        model_name = getattr(model_class, '__name__', 'UNKNOWN')
        
        for forecast in all_forecasts:
            for metric in ['prob', 'mse', 'mse_z']:
                # Create the key for this combination
                key = f"{model_name}_{metric}_{sequence_length}_{forecast}"
                
                # Find probability files
                prob_dir = ospj(prodatapath, 'sz_prob', patient)
                if metric == 'prob':
                    pattern = f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_sz_prob_forecast-{forecast}.pkl"
                else:
                    pattern = f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_{metric}_prob_forecast-{forecast}.pkl"
                
                prob_path = glob.glob(ospj(prob_dir, pattern))
                
                if prob_path:
                    prob_files[key] = {
                        'data': pd.read_pickle(prob_path[0]),
                        'model_class': model_class,
                        'sequence_length': sequence_length,
                        'forecast': forecast,
                        'metric': metric
                    }
                else:
                    print(f"Warning: No probability file found for {patient} {onset_run} {key}")
    
    return prob_files

def main():
    """
    Main seizure spread analysis pipeline.
    
    Loads pre-computed probability matrices and applies model-specific thresholds
    to analyze seizure onset and spread patterns.
    """
    
    # Load seizure metadata
    seizures_df = pd.read_csv(ospj(metapath, "metadata_v6_BIDS.csv"))
    seizures_df = seizures_df[seizures_df.split == 1]
    
    # Load model thresholds - expecting format: model_metric_sequence_forecast
    thresholds_df = pd.read_csv(ospj(prodatapath, "ndd_val_thresholds_f1.csv"))  # Update path as needed
    threshold_dict = dict(zip(thresholds_df.model, thresholds_df.f1_threshold))
    
    # Model configurations matching annotation script
    all_models = [
        {'model': LiNDDA, 'sequence_length': 1},
        {'model': LiNDDA, 'sequence_length': 8}, 
        {'model': LiNDDA, 'sequence_length': 32},
        {'model': GIN, 'sequence_length': 12},
        {'model': GIN, 'sequence_length': 32}
    ]
    all_forecasts = [1,16]  # Forecast lengths to process
    
    # Results storage
    spread_results = []
    
    # Process each seizure
    pbar = tqdm(seizures_df.iterrows(), total=len(seizures_df))
    for _, row in pbar:
        patient = row.Patient
        onset_run = str(int(row.onset))
        onset_labels = clean_labels([l.strip() for l in row.SOZ.split(',')], patient)
        
        pbar.set_description(f"Patient: {patient} | Seizure: {onset_run}")
        
        # Load probability files for this seizure
        prob_files = load_probability_files(patient, onset_run, all_models, all_forecasts)
        
        # Process each model-metric-sequence-forecast combination that has probability data
        for key, prob_info in prob_files.items():
            if key in threshold_dict:
                threshold = threshold_dict[key]
                prob_data = prob_info['data']
                model_class = prob_info['model_class']
                sequence_length = prob_info['sequence_length']
                forecast = prob_info['forecast']
                metric = prob_info['metric']

                # Extract time array
                if 'time' in prob_data.columns:
                    prob_times = prob_data.pop('time').values
                else:
                    print(f"Warning: No time column found for {patient} {onset_run} {key}")
                    continue
                
                # Create model instance for spread analysis
                if model_class == LiNDDA:
                    model = model_class(
                        fs=256,  # Assuming standard fs for spread analysis
                        w_size=1,
                        w_stride=0.5,
                        sequence_length=sequence_length,
                        forecast_length=forecast,
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
                        forecast_length=forecast,
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
                else:
                    print(f"Warning: Unsupported model class {model_class}")
                    continue
                
                # Call spread analysis function
                first_onset_idx = int(np.argmin(np.abs(prob_times - 120)))
                spread_df, sz_clf = model.get_onset_and_spread(prob_data.iloc[first_onset_idx:,:], threshold=threshold, ret_smooth_mat=True)

                if spread_df is not None and not spread_df.empty:
                    onset_idx = int(np.argmin(np.abs(prob_times))) + spread_df.iloc[0].values[0]
                    onset_odx = int(np.argmin(np.abs(prob_times - 1))) + spread_df.iloc[0].values[0]
                    
                    # Calculate onset metrics
                    onset_mask = np.array([ch.split('-')[0] in onset_labels for ch in prob_data.columns])
                    onset_pred = sz_clf.iloc[onset_idx:onset_odx,:].sum()>0
                    sensitivity = np.sum(onset_mask & onset_pred) / np.sum(onset_mask)
                    specificity = np.sum(~onset_mask & ~onset_pred) / np.sum(~onset_mask)
                    precision = precision_score(onset_mask, onset_pred, zero_division=0)
                    recall = recall_score(onset_mask, onset_pred, zero_division=0)
                    f1 = f1_score(onset_mask, onset_pred)
                    phi = matthews_corrcoef(onset_mask, onset_pred)

                    # Calculate recruitment times from indices
                    recruitment_indices = spread_df.iloc[0].values  # First row contains indices
                    recruitment_times = prob_times[recruitment_indices.astype(int)]
                    recruitment_times -= min(recruitment_times)
                    
                    # Create channel-time mapping
                    channel_times = dict(zip(spread_df.columns, recruitment_times))
                    
                    # Filter for SOZ channels only
                    soz_channels = [ch for ch in spread_df.columns if ch.split('-')[0] in onset_labels]
                    
                    if len(soz_channels) > 0:
                        # Calculate spread ranks
                        # Sort all channels by recruitment time to get ranks
                        sorted_channels = sorted(channel_times.items(), key=lambda x: x[1])
                        channel_ranks = {ch: rank+1 for rank, (ch, _) in enumerate(sorted_channels)}
                        total_channels = len(sorted_channels)
                        
                        # Get ranks for SOZ channels and convert to percentages
                        soz_ranks = [channel_ranks[ch] for ch in soz_channels]
                        soz_rank_percentages = [(rank / total_channels) * 100 for rank in soz_ranks]
                        
                        avg_soz_rank_pct = np.mean(soz_rank_percentages)
                        median_soz_rank_pct = np.median(soz_rank_percentages)
                        
                        # Calculate adjusted percentages (subtract percentage equivalent of num_SOZ_channels/2)
                        soz_channel_pct_adjustment = (len(soz_channels) / 2 / total_channels) * 100
                        adjusted_soz_rank_pct = avg_soz_rank_pct - soz_channel_pct_adjustment
                        adjusted_med_soz_rank_pct = median_soz_rank_pct - soz_channel_pct_adjustment
                        
                        # Calculate recruitment latencies for SOZ channels
                        soz_times = [channel_times[ch] for ch in soz_channels]
                        avg_recruitment_latency = np.mean(soz_times)
                        median_recruitment_latency = np.median(soz_times)
                        max_recruitment_latency = np.max(soz_times)
                        
                        # Store results
                        result_dict = {
                            'patient': patient,
                            'onset': int(onset_run),
                            'model': key,  # Full model_metric_sequence_forecast identifier
                            'model_name': getattr(model_class, '__name__', 'UNKNOWN'),
                            'metric': metric,
                            'sequence_length': sequence_length,
                            'forecast_length': forecast,
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
                        }
                    
                    else:
                        print(f"Warning: No SOZ channels found in spread data for {patient} {onset_run} {key}")
                        # Create minimal result dict for cases without SOZ channels in spread data
                        total_channels = len(channel_times) if channel_times else 0
                        result_dict = {
                            'patient': patient,
                            'onset': int(onset_run),
                            'model': key,
                            'model_name': getattr(model_class, '__name__', 'UNKNOWN'),
                            'metric': metric,
                            'sequence_length': sequence_length,
                            'forecast_length': forecast,
                            'num_soz_channels': 0,
                            'total_channels': total_channels,
                            'avg_soz_spread_rank_pct': np.nan,
                            'med_soz_spread_rank_pct': np.nan,
                            'adj_avg_soz_spread_rank_pct': np.nan,
                            'adj_med_soz_spread_rank_pct': np.nan,
                            'avg_soz_recruitment_latency': np.nan,
                            'med_soz_recruitment_latency': np.nan,
                            'max_soz_recruitment_latency': np.nan,
                            'sensitivity': np.nan,
                            'specificity': np.nan,
                            'precision': np.nan,
                            'recall': np.nan,
                            'f1': np.nan,
                            'phi': np.nan,
                        }
                    
                    spread_results.append(result_dict)
            else:
                print(f"Warning: No threshold found for model {key}")
    
    # Save results
    if spread_results:
        results_df = pd.DataFrame(spread_results)
        output_path = ospj(prodatapath, "ndd_spread_analysis_results_f1.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No results to save")

if __name__ == "__main__":
    main()
