#!/usr/bin/env python3
"""
Test validation analysis for NDD and benchmark models.
Calculates Phi at various thresholds and AUC without thresholding.
"""
# File system imports
import sys
import os
import glob
from os.path import join as ospj

# Scientific imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy as sc
import scipy.ndimage

# Utility imports
from utils import clean_labels

# Sklearn imports
from sklearn.metrics import matthews_corrcoef, roc_auc_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Get the project root for DynaSD
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

# Import models
from DynaSD.LiNDDA import LiNDDA
from DynaSD.GIN import GIN
from DynaSD.HFER import HFER
from DynaSD.ABSSLP import ABSSLP
from DynaSD.IMPRINT import IMPRINT
from DynaSD.WAVENET import WVNT

from config import Config

# Get paths from config 
datapath, prodatapath, figpath, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])


def load_ndd_probability_files(patient, onset_run, model_dicts):
    """Load probability files for NDD models"""
    prob_files = {}
    
    for model_dict in model_dicts:
        model_name = model_dict['model_name']
        sequence_length = model_dict['sequence_length']
        forecast_length = model_dict['forecast_length']
        metric = model_dict.get('metric', 'mse')
        
        key = f"{model_name}_seq{sequence_length}_fc{forecast_length}"
        
        prob_dir = ospj(prodatapath, 'sz_prob', patient)
        pattern = f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_{metric}_prob_forecast-{forecast_length}.pkl"
        prob_path = glob.glob(ospj(prob_dir, pattern))
        
        if prob_path:
            prob_files[key] = {
                'data': pd.read_pickle(prob_path[0]),
                'model_name': model_name,
                'sequence_length': sequence_length,
                'forecast_length': forecast_length,
                'metric': metric
            }
        else:
            print(f"Warning: No probability file found for {patient} {onset_run} {key}")
    
    return prob_files


def load_benchmark_probability_files(patient, onset_run, model_names):
    """Load probability files for benchmark models"""
    prob_files = {}
    
    for model_name in model_names:
        prob_dir = ospj(prodatapath, 'sz_prob', patient)
        pattern = f"{patient}_task-ictal{onset_run}_run-*_mdl-{model_name}_sz_prob.pkl"
        prob_path = glob.glob(ospj(prob_dir, pattern))
        
        if prob_path:
            prob_files[model_name] = {
                'data': pd.read_pickle(prob_path[0]),
                'model_name': model_name,
                'sequence_length': None,
                'forecast_length': None,
                'metric': 'prob'
            }
        else:
            print(f"Warning: No probability file found for {patient} {onset_run} {model_name}")
    
    return prob_files


def apply_smoothing(prob_array, window_size=20):
    """Apply temporal smoothing to probability matrix"""
    return sc.ndimage.uniform_filter1d(prob_array, size=window_size, mode='nearest', axis=1, origin=0)


def get_channel_predictions(sz_prob, prob_chs, threshold, onset_idx, spread_idx):
    """Get channel predictions at onset and spread timepoints"""
    sz_clf = sz_prob > threshold
    
    # UEO channels - seizing for 5 consecutive timepoints at onset
    ueo_idx = np.all(sz_clf[:, onset_idx:onset_idx+5], axis=1)
    ueo_ch_strict = np.array([s.split("-")[0] for s in prob_chs[ueo_idx]]) if np.any(ueo_idx) else np.array([])
    
    # SEC channels - seizing for 5 consecutive timepoints at spread
    sec_idx = np.all(sz_clf[:, spread_idx:spread_idx+5], axis=1)
    sec_ch_strict = np.array([s.split("-")[0] for s in prob_chs[sec_idx]]) if np.any(sec_idx) else np.array([])
    
    return ueo_ch_strict, sec_ch_strict


def wideform_preds(element, all_labels):
    """Convert element list to boolean array"""
    return np.array([label in element for label in all_labels])


def calculate_phi(pred_labels, true_labels, all_labels):
    """Calculate Matthews Correlation Coefficient (Phi)"""
    pred_bool = wideform_preds(pred_labels, all_labels)
    true_bool = wideform_preds(true_labels, all_labels)
    
    if len(pred_bool) == 0 or len(true_bool) == 0:
        return np.nan
    
    return matthews_corrcoef(true_bool, pred_bool)


def calculate_auc_and_probs(sz_prob, prob_chs, onset_labels, onset_idx):
    """Calculate AUC and average probabilities without thresholding"""
    # Create onset mask
    onset_mask = np.array([ch.split('-')[0] in onset_labels for ch in prob_chs])
    
    if not np.any(onset_mask):
        return np.nan, np.nan, np.nan, np.nan
    
    # Average probabilities at onset window (5 timepoints)
    avg_soz_prob = np.mean(sz_prob[onset_mask, onset_idx:onset_idx+5])
    avg_nsoz_prob = np.mean(sz_prob[~onset_mask, onset_idx:onset_idx+5])
    
    # Calculate AUC for onset and spread
    onset_pred = sz_prob[:, onset_idx:onset_idx+5].mean(axis=1)
    try:
        onset_auc = roc_auc_score(onset_mask, onset_pred)
    except:
        onset_auc = np.nan
    
    spread_pred = sz_prob[:, onset_idx+10:onset_idx+15].mean(axis=1) if sz_prob.shape[1] > onset_idx+15 else onset_pred
    try:
        spread_auc = roc_auc_score(onset_mask, spread_pred)
    except:
        spread_auc = np.nan
    
    return avg_soz_prob, avg_nsoz_prob, onset_auc, spread_auc


def find_optimal_phi_threshold(sz_prob, prob_chs, onset_idx, spread_idx, all_chs,
                               ueo_consensus, sec_consensus, ueo_annotators, sec_annotators):
    """Find optimal threshold for Phi by testing unique probability values"""
    
    # Get unique probability values as candidate thresholds
    unique_probs = np.unique(sz_prob[:, onset_idx:onset_idx+5])
    
    onset_phi_vals = []
    onset_phi_per_annotator = {i: [] for i in range(len(ueo_annotators))}
    spread_phi_per_annotator = {i: [] for i in range(len(sec_annotators))}
    
    for threshold in unique_probs:
        # Get predictions
        ueo_pred, sec_pred = get_channel_predictions(sz_prob, prob_chs, threshold, onset_idx, spread_idx)
        
        # Calculate onset phi with consensus
        onset_phi = calculate_phi(ueo_pred, ueo_consensus, all_chs)
        onset_phi_vals.append(onset_phi)
        
        # Calculate phi with each annotator
        for i, annotator_labels in enumerate(ueo_annotators):
            phi = calculate_phi(ueo_pred, annotator_labels, all_chs)
            onset_phi_per_annotator[i].append(phi)
        
        for i, annotator_labels in enumerate(sec_annotators):
            phi = calculate_phi(sec_pred, annotator_labels, all_chs)
            spread_phi_per_annotator[i].append(phi)
    
    onset_phi_vals = np.array(onset_phi_vals)
    
    # Find optimal threshold
    if np.any(~np.isnan(onset_phi_vals)):
        optimal_idx = np.nanargmax(onset_phi_vals)
        optimal_threshold = unique_probs[optimal_idx]
        max_phi = onset_phi_vals[optimal_idx]
        onset_phi_annotators = [onset_phi_per_annotator[i][optimal_idx] for i in range(len(ueo_annotators))]
        spread_phi_annotators = [spread_phi_per_annotator[i][optimal_idx] for i in range(len(sec_annotators))]
    else:
        optimal_threshold = np.nan
        max_phi = np.nan
        onset_phi_annotators = [np.nan] * len(ueo_annotators)
        spread_phi_annotators = [np.nan] * len(sec_annotators)
    
    return {
        'optimal_threshold': optimal_threshold,
        'max_phi': max_phi,
        'onset_phi_annotators': onset_phi_annotators,
        'spread_phi_annotators': spread_phi_annotators,
        'unique_probs': unique_probs,
        'phi_curve': onset_phi_vals
    }


def calculate_phi_at_threshold(sz_prob, prob_chs, onset_idx, spread_idx, all_chs,
                               ueo_consensus, sec_consensus, ueo_annotators, sec_annotators, threshold):
    """Calculate Phi at a given threshold"""
    
    if np.isnan(threshold):
        return {
            'onset_phi': np.nan,
            'spread_phi': np.nan,
            'onset_phi_annotators': [np.nan] * len(ueo_annotators),
            'spread_phi_annotators': [np.nan] * len(sec_annotators)
        }
    
    # Get predictions
    ueo_pred, sec_pred = get_channel_predictions(sz_prob, prob_chs, threshold, onset_idx, spread_idx)
    
    # Calculate consensus phi
    onset_phi = calculate_phi(ueo_pred, ueo_consensus, all_chs)
    spread_phi = calculate_phi(sec_pred, sec_consensus, all_chs)
    
    # Calculate per-annotator phi
    onset_phi_annotators = [calculate_phi(ueo_pred, ann, all_chs) for ann in ueo_annotators]
    spread_phi_annotators = [calculate_phi(sec_pred, ann, all_chs) for ann in sec_annotators]
    
    return {
        'onset_phi': onset_phi,
        'spread_phi': spread_phi,
        'onset_phi_annotators': onset_phi_annotators,
        'spread_phi_annotators': spread_phi_annotators
    }


def generate_example_figure(sz_prob, prob_chs, onset_idx, spread_idx, all_chs,
                           ueo_consensus, sec_consensus, figpath):
    """Generate threshold sweep figure for HUP238 example"""
    
    # Get unique probability values - use all for detailed figure
    unique_probs = np.unique(sz_prob[:, onset_idx:onset_idx+5])
    phi_vals = []
    
    for threshold in unique_probs:
        ueo_pred, _ = get_channel_predictions(sz_prob, prob_chs, threshold, onset_idx, spread_idx)
        phi = calculate_phi(ueo_pred, ueo_consensus, all_chs)
        phi_vals.append(phi)
    
    phi_vals = np.array(phi_vals)
    
    if np.any(~np.isnan(phi_vals)):
        optimal_idx = np.nanargmax(phi_vals)
        optimal_threshold = unique_probs[optimal_idx]
        max_phi = phi_vals[optimal_idx]
        
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(unique_probs, phi_vals, 
               color=sns.color_palette('spring_r', n_colors=3)[0],
               linewidth=4)
        
        # Mark optimal threshold
        ax.plot([optimal_threshold]*2, 
               [0, max_phi], 
               '--o', c='purple',
               markersize=15,
               linewidth=4,
               markevery=[1],
               fillstyle='none',
               markeredgewidth=3)
        
        ax.set_ylabel('Agreement ($\phi$)')
        ax.set_xlabel('Threshold')
        ax.set_ylim([-0.2, 1.1])
        sns.despine()
        
        fig_path = ospj(figpath, 'threshold_tune_example_test.pdf')
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
        plt.close()


def main():
    """Main validation analysis pipeline for test seizures"""
    
    print("Loading metadata and annotations...")
    
    # Load seizure metadata
    seizures_df = pd.read_csv(ospj(metapath, "metadata_v6_BIDS.csv"))
    seizures_df = seizures_df[(seizures_df.split == 2) & (seizures_df.stim == 0)]
    
    print(f"Found {len(seizures_df)} test seizures")
    
    # Load clinical annotations
    annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v2.pkl"))
    
    # Load learned thresholds
    ndd_thresholds = {}
    benchmark_thresholds = {}
    
    for metric in ['phi', 'f1', 'iou']:
        try:
            ndd_thresh_df = pd.read_csv(ospj(prodatapath, f"ndd_val_thresholds_{metric}_v2.csv"))
            ndd_thresholds[metric] = dict(zip(ndd_thresh_df.model, ndd_thresh_df[f'{metric}_threshold']))
        except:
            print(f"Warning: Could not load ndd_val_thresholds_{metric}_v2.csv")
            ndd_thresholds[metric] = {}
        
        try:
            bench_thresh_df = pd.read_csv(ospj(prodatapath, f"benchmark_val_thresholds_{metric}.csv"))
            benchmark_thresholds[metric] = dict(zip(bench_thresh_df.model, bench_thresh_df[f'{metric}_threshold']))
        except:
            print(f"Warning: Could not load benchmark_val_thresholds_{metric}.csv")
            benchmark_thresholds[metric] = {}
    
    # Model configurations
    ndd_models = [
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 1, 'forecast_length': 1, 'metric': 'mse'},
        {'model': GIN, 'model_name': 'GIN', 'sequence_length': 12, 'forecast_length': 1, 'metric': 'mse'},
    ]
    
    benchmark_models = ['ABSSLP', 'IMPRINT', 'WVNT', 'HFER']
    
    # Results storage
    results = []
    
    # Process each seizure
    pbar = tqdm(seizures_df.iterrows(), total=len(seizures_df))
    for _, row in pbar:
        patient = row.Patient
        onset_run = str(int(row.onset))
        approx_onset = row.onset
        
        pbar.set_description(f"Patient: {patient} | Seizure: {onset_run}")
        
        # Get clinical annotations
        annot_matches = annotations_df[
            (annotations_df['patient'] == patient) & 
            (np.abs(annotations_df['approximate_onset'].astype(float) - approx_onset) < 1)
        ]
        
        if len(annot_matches) == 0:
            print(f"No annotations found for {patient} {onset_run}")
            continue
        
        annot_row = annot_matches.iloc[0]
        consensus_time = annot_row['ueo_time_consensus']
        all_chs = annot_row['all_chs']
        ueo_consensus = annot_row['ueo_consensus']
        sec_consensus = annot_row['sec_consensus']
        ueo_annotators = annot_row['ueo']
        sec_annotators = annot_row['sec']
        
        # Get onset labels from consensus
        onset_labels = [ch for ch, is_onset in zip(all_chs, ueo_consensus) if is_onset]
        
        # Process NDD models
        ndd_prob_files = load_ndd_probability_files(patient, onset_run, ndd_models)
        
        for model_key, prob_info in ndd_prob_files.items():
            prob_data = prob_info['data'].copy()
            model_name = prob_info['model_name']
            
            if 'time' in prob_data.columns:
                prob_times = prob_data.pop('time').values
            else:
                continue
            
            prob_chs = prob_data.columns.to_numpy()
            sz_prob = prob_data.to_numpy().T
            
            # Apply smoothing
            sz_prob_smooth = apply_smoothing(sz_prob, window_size=20)
            
            # Calculate temporal alignment
            time_diff = consensus_time - approx_onset
            onset_idx = int(np.argmin(np.abs((prob_times - 120) + time_diff)))
            spread_idx = int(np.argmin(np.abs((prob_times - 130) + time_diff)))
            
            # Calculate AUC and average probabilities (no thresholding)
            avg_soz_prob, avg_nsoz_prob, onset_auc, spread_auc = calculate_auc_and_probs(
                sz_prob_smooth, prob_chs, onset_labels, onset_idx
            )
            
            # Find optimal threshold for Phi
            optimal_results = find_optimal_phi_threshold(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                all_chs, ueo_consensus, sec_consensus,
                ueo_annotators, sec_annotators
            )
            
            # Get full model key for learned thresholds
            full_model_key = f"{model_name}_mse_sl{prob_info['sequence_length']}_fl{prob_info['forecast_length']}"
            
            # Calculate Phi at learned thresholds
            phi_at_learned = {}
            for metric_name in ['phi', 'f1', 'iou']:
                learned_thresh = ndd_thresholds[metric_name].get(full_model_key, np.nan)
                phi_at_learned[metric_name] = calculate_phi_at_threshold(
                    sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                    all_chs, ueo_consensus, sec_consensus,
                    ueo_annotators, sec_annotators, learned_thresh
                )
            
            # Store results
            result_dict = {
                'patient': patient,
                'onset': int(onset_run),
                'model_name': model_name,
                'model_key': model_key,
                'full_model_key': full_model_key,
                'sequence_length': prob_info['sequence_length'],
                'forecast_length': prob_info['forecast_length'],
                'metric': prob_info['metric'],
                
                # Probability-based metrics (no thresholding)
                'avg_soz_prob': avg_soz_prob,
                'avg_nsoz_prob': avg_nsoz_prob,
                'onset_auc': onset_auc,
                'spread_auc': spread_auc,
                
                # Optimal threshold for Phi
                'optimal_phi_threshold': optimal_results['optimal_threshold'],
                'phi_at_optimal_threshold': optimal_results['max_phi'],
                'onset_phi_annotators_at_optimal': optimal_results['onset_phi_annotators'],
                'spread_phi_annotators_at_optimal': optimal_results['spread_phi_annotators'],
                
                # Phi at learned thresholds
                'learned_phi_threshold': ndd_thresholds['phi'].get(full_model_key, np.nan),
                'onset_phi_at_learned_phi': phi_at_learned['phi']['onset_phi'],
                'spread_phi_at_learned_phi': phi_at_learned['phi']['spread_phi'],
                'onset_phi_annotators_at_learned_phi': phi_at_learned['phi']['onset_phi_annotators'],
                
                'learned_f1_threshold': ndd_thresholds['f1'].get(full_model_key, np.nan),
                'onset_phi_at_learned_f1': phi_at_learned['f1']['onset_phi'],
                'spread_phi_at_learned_f1': phi_at_learned['f1']['spread_phi'],
                'onset_phi_annotators_at_learned_f1': phi_at_learned['f1']['onset_phi_annotators'],
                
                'learned_iou_threshold': ndd_thresholds['iou'].get(full_model_key, np.nan),
                'onset_phi_at_learned_iou': phi_at_learned['iou']['onset_phi'],
                'spread_phi_at_learned_iou': phi_at_learned['iou']['spread_phi'],
                'onset_phi_annotators_at_learned_iou': phi_at_learned['iou']['onset_phi_annotators'],
            }
            
            results.append(result_dict)
            
            # Generate example figure
            if patient == 'HUP238' and int(onset_run) == 290006 and model_name == 'LiNDDA':
                print(f"\nGenerating example figure for {patient} {onset_run} {model_name}...")
                generate_example_figure(sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                                      all_chs, ueo_consensus, sec_consensus, figpath)
        
        # Process benchmark models
        benchmark_prob_files = load_benchmark_probability_files(patient, onset_run, benchmark_models)
        
        for model_name, prob_info in benchmark_prob_files.items():
            prob_data = prob_info['data'].copy()
            
            if 'time' in prob_data.columns:
                prob_times = prob_data.pop('time').values
            else:
                continue
            
            prob_chs = prob_data.columns.to_numpy()
            sz_prob = prob_data.to_numpy().T
            
            # Apply smoothing
            sz_prob_smooth = apply_smoothing(sz_prob, window_size=20)
            
            # Calculate temporal alignment
            time_diff = consensus_time - approx_onset
            onset_idx = int(np.argmin(np.abs((prob_times - 120) + time_diff)))
            spread_idx = int(np.argmin(np.abs((prob_times - 130) + time_diff)))
            
            # Calculate AUC and average probabilities
            avg_soz_prob, avg_nsoz_prob, onset_auc, spread_auc = calculate_auc_and_probs(
                sz_prob_smooth, prob_chs, onset_labels, onset_idx
            )
            
            # Find optimal threshold for Phi
            optimal_results = find_optimal_phi_threshold(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                all_chs, ueo_consensus, sec_consensus,
                ueo_annotators, sec_annotators
            )
            
            # Calculate Phi at learned thresholds
            phi_at_learned = {}
            for metric_name in ['phi', 'f1', 'iou']:
                learned_thresh = benchmark_thresholds[metric_name].get(model_name, np.nan)
                phi_at_learned[metric_name] = calculate_phi_at_threshold(
                    sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                    all_chs, ueo_consensus, sec_consensus,
                    ueo_annotators, sec_annotators, learned_thresh
                )
            
            # Store results
            result_dict = {
                'patient': patient,
                'onset': int(onset_run),
                'model_name': model_name,
                'model_key': model_name,
                'full_model_key': model_name,
                'sequence_length': None,
                'forecast_length': None,
                'metric': 'prob',
                
                # Probability-based metrics
                'avg_soz_prob': avg_soz_prob,
                'avg_nsoz_prob': avg_nsoz_prob,
                'onset_auc': onset_auc,
                'spread_auc': spread_auc,
                
                # Optimal threshold for Phi
                'optimal_phi_threshold': optimal_results['optimal_threshold'],
                'phi_at_optimal_threshold': optimal_results['max_phi'],
                'onset_phi_annotators_at_optimal': optimal_results['onset_phi_annotators'],
                'spread_phi_annotators_at_optimal': optimal_results['spread_phi_annotators'],
                
                # Phi at learned thresholds
                'learned_phi_threshold': benchmark_thresholds['phi'].get(model_name, np.nan),
                'onset_phi_at_learned_phi': phi_at_learned['phi']['onset_phi'],
                'spread_phi_at_learned_phi': phi_at_learned['phi']['spread_phi'],
                'onset_phi_annotators_at_learned_phi': phi_at_learned['phi']['onset_phi_annotators'],
                
                'learned_f1_threshold': benchmark_thresholds['f1'].get(model_name, np.nan),
                'onset_phi_at_learned_f1': phi_at_learned['f1']['onset_phi'],
                'spread_phi_at_learned_f1': phi_at_learned['f1']['spread_phi'],
                'onset_phi_annotators_at_learned_f1': phi_at_learned['f1']['onset_phi_annotators'],
                
                'learned_iou_threshold': benchmark_thresholds['iou'].get(model_name, np.nan),
                'onset_phi_at_learned_iou': phi_at_learned['iou']['onset_phi'],
                'spread_phi_at_learned_iou': phi_at_learned['iou']['spread_phi'],
                'onset_phi_annotators_at_learned_iou': phi_at_learned['iou']['onset_phi_annotators'],
            }
            
            results.append(result_dict)
    
    # Save results
    print("\nSaving results...")
    if results:
        results_df = pd.DataFrame(results)
        output_path = ospj(prodatapath, "test_validation_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(f"Total results: {len(results_df)}")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()


