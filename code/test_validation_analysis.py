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
    
    # UEO channels - seizing for at least 4/5 timepoints at onset (80%)
    ueo_idx = np.sum(sz_clf[:, onset_idx:onset_idx+5], axis=1) >= 4
    ueo_ch_strict = np.array([s.split("-")[0] for s in prob_chs[ueo_idx]]) if np.any(ueo_idx) else np.array([])
    
    # SEC channels - seizing for at least 4/5 timepoints at spread (80%)
    sec_idx = np.sum(sz_clf[:, spread_idx:spread_idx+5], axis=1) >= 4
    sec_ch_strict = np.array([s.split("-")[0] for s in prob_chs[sec_idx]]) if np.any(sec_idx) else np.array([])
    
    return ueo_ch_strict, sec_ch_strict


def wideform_preds(element, all_labels):
    """Convert element list to boolean array"""
    return np.array([label in element for label in all_labels])


def calculate_phi(pred_labels, true_bool, all_labels):
    """Calculate Matthews Correlation Coefficient (Phi)"""
    pred_bool = wideform_preds(pred_labels, all_labels)
    
    if len(pred_bool) == 0 or len(true_bool) == 0:
        return np.nan
    
    return matthews_corrcoef(true_bool, pred_bool)


def calculate_inter_rater_reliability(annotators, all_labels):
    """Calculate average pairwise phi between all annotators"""
    n_annotators = len(annotators)
    
    if n_annotators < 2:
        return np.nan
    
    phi_values = []
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            phi = matthews_corrcoef(annotators[i], annotators[j])
            if not np.isnan(phi):
                phi_values.append(phi)
    
    if len(phi_values) == 0:
        return np.nan
    
    return np.mean(phi_values)


def calculate_auc_and_probs(sz_prob, prob_chs, onset_labels, spread_labels, onset_idx, spread_idx):
    """Calculate AUC and average probabilities without thresholding"""
    # Create onset mask
    onset_mask = np.array([ch.split('-')[0] in onset_labels for ch in prob_chs])
    
    # Create spread mask
    spread_mask = np.array([ch.split('-')[0] in spread_labels for ch in prob_chs])
    
    # Calculate onset metrics
    if np.any(onset_mask):
        # Average probabilities at onset window (5 timepoints)
        avg_onset_soz_prob = np.mean(sz_prob[onset_mask, onset_idx:onset_idx+5])
        avg_onset_nsoz_prob = np.mean(sz_prob[~onset_mask, onset_idx:onset_idx+5])
        
        # Calculate AUC for onset
        onset_pred = sz_prob[:, onset_idx:onset_idx+5].mean(axis=1)
        try:
            onset_auc = roc_auc_score(onset_mask, onset_pred)
        except:
            onset_auc = np.nan
    else:
        avg_onset_soz_prob = np.nan
        avg_onset_nsoz_prob = np.nan
        onset_auc = np.nan
    
    # Calculate spread metrics
    if np.any(spread_mask):
        # Average probabilities at spread window (5 timepoints)
        avg_spread_soz_prob = np.mean(sz_prob[spread_mask, spread_idx:spread_idx+5])
        avg_spread_nsoz_prob = np.mean(sz_prob[~spread_mask, spread_idx:spread_idx+5])
        
        # Calculate AUC for spread
        spread_pred = sz_prob[:, spread_idx:spread_idx+5].mean(axis=1)
        try:
            spread_auc = roc_auc_score(spread_mask, spread_pred)
        except:
            spread_auc = np.nan
    else:
        avg_spread_soz_prob = np.nan
        avg_spread_nsoz_prob = np.nan
        spread_auc = np.nan
    
    return avg_onset_soz_prob, avg_onset_nsoz_prob, onset_auc, avg_spread_soz_prob, avg_spread_nsoz_prob, spread_auc


def find_optimal_phi_threshold(sz_prob, prob_chs, window_idx, all_chs, consensus_labels, annotators):
    """
    Find optimal threshold for Phi by testing unique probability values.
    
    Parameters:
    -----------
    sz_prob : np.array
        Smoothed probability matrix (n_channels × n_timepoints)
    prob_chs : np.array
        Channel names
    window_idx : int
        Index of the window to evaluate (onset_idx or spread_idx)
    all_chs : list
        All channel labels
    consensus_labels : np.array
        Boolean array of consensus labels
    annotators : list
        List of boolean arrays, one per annotator
    
    Returns:
    --------
    dict with optimal_threshold, max_phi, phi_annotators, unique_probs, phi_curve
    """
    
    # Get unique probability values as candidate thresholds from the window
    unique_probs = np.unique(sz_prob[:, window_idx:window_idx+5])
    
    phi_vals = []
    phi_per_annotator = {i: [] for i in range(len(annotators))}
    
    for threshold in unique_probs:
        # Get predictions at this window (at least 4/5 timepoints above threshold = 80%)
        sz_clf = sz_prob > threshold
        window_idx_pred = np.sum(sz_clf[:, window_idx:window_idx+5], axis=1) >= 4
        predicted_chs = np.array([s.split("-")[0] for s in prob_chs[window_idx_pred]]) if np.any(window_idx_pred) else np.array([])
        
        # Calculate phi with consensus
        phi = calculate_phi(predicted_chs, consensus_labels, all_chs)
        phi_vals.append(phi)
        
        # Calculate phi with each annotator
        for i, annotator_labels in enumerate(annotators):
            phi_annot = calculate_phi(predicted_chs, annotator_labels, all_chs)
            phi_per_annotator[i].append(phi_annot)
    
    phi_vals = np.array(phi_vals)
    
    # Find optimal threshold
    if np.any(~np.isnan(phi_vals)):
        optimal_idx = np.nanargmax(phi_vals)
        optimal_threshold = unique_probs[optimal_idx]
        max_phi = phi_vals[optimal_idx]
        phi_annotators = [phi_per_annotator[i][optimal_idx] for i in range(len(annotators))]
    else:
        optimal_threshold = np.nan
        max_phi = np.nan
        phi_annotators = [np.nan] * len(annotators)
    
    return {
        'optimal_threshold': optimal_threshold,
        'max_phi': max_phi,
        'phi_annotators': phi_annotators,
        'unique_probs': unique_probs,
        'phi_curve': phi_vals
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


def generate_example_figure(sz_prob, prob_chs, onset_idx, all_chs,
                           ueo_consensus, ueo_annotators, figpath):
    """Generate threshold sweep figure for HUP238 example"""
    
    # Use the new function to get phi curve
    results = find_optimal_phi_threshold(
        sz_prob, prob_chs, onset_idx,
        all_chs, ueo_consensus, ueo_annotators
    )
    
    unique_probs = results['unique_probs']
    phi_vals = results['phi_curve']
    optimal_threshold = results['optimal_threshold']
    max_phi = results['max_phi']
    
    if np.any(~np.isnan(phi_vals)):
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
        ax.set_xlim([0, 1])
        sns.despine()
        
        fig_path = ospj(figpath, 'threshold_tune_example_test.pdf')
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
        plt.close()


def main():
    """Main validation analysis pipeline for test seizures"""
    
    print("Loading metadata and annotations...")
    
    # Load seizure metadata
    seizures_df = pd.read_csv(ospj(metapath, "metadata_v7_BIDS.csv"))
    seizures_df = seizures_df[(seizures_df.split == 2) & (seizures_df.stim == 0)]
    
    print(f"Found {len(seizures_df)} test seizures")
    
    # Load clinical annotations
    annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v2.pkl"))
    annotations_df = annotations_df[annotations_df.stim == 0]
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
            (np.abs(annotations_df['approximate_onset'].astype(float) - approx_onset) < 360)
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
        
        # Get onset and spread labels from consensus
        onset_labels = [ch for ch, is_onset in zip(all_chs, ueo_consensus) if is_onset]
        spread_labels = [ch for ch, is_spread in zip(all_chs, sec_consensus) if is_spread]
        
        # Calculate inter-rater reliability (independent of model)
        onset_inter_rater = calculate_inter_rater_reliability(ueo_annotators, all_chs)
        spread_inter_rater = calculate_inter_rater_reliability(sec_annotators, all_chs)
        
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
            onset_idx = int(np.argmin(np.abs((prob_times - 180) + time_diff)))
            spread_idx = int(np.argmin(np.abs((prob_times - 180 + 10) + time_diff)))
            
            # Calculate AUC and average probabilities (no thresholding)
            avg_onset_soz_prob, avg_onset_nsoz_prob, onset_auc, avg_spread_soz_prob, avg_spread_nsoz_prob, spread_auc = calculate_auc_and_probs(
                sz_prob_smooth, prob_chs, onset_labels, spread_labels, onset_idx, spread_idx
            )
            
            # Find optimal threshold for Phi (onset)
            optimal_onset_results = find_optimal_phi_threshold(
                sz_prob_smooth, prob_chs, onset_idx,
                all_chs, ueo_consensus, ueo_annotators
            )
            
            # Find optimal threshold for Phi (spread)
            optimal_spread_results = find_optimal_phi_threshold(
                sz_prob_smooth, prob_chs, spread_idx,
                all_chs, sec_consensus, sec_annotators
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
                
                # Inter-rater reliability (independent of model)
                'onset_inter_rater_reliability': onset_inter_rater,
                'spread_inter_rater_reliability': spread_inter_rater,
                
                # Probability-based metrics (no thresholding)
                'avg_onset_soz_prob': avg_onset_soz_prob,
                'avg_onset_nsoz_prob': avg_onset_nsoz_prob,
                'onset_auc': onset_auc,
                'avg_spread_soz_prob': avg_spread_soz_prob,
                'avg_spread_nsoz_prob': avg_spread_nsoz_prob,
                'spread_auc': spread_auc,
                
                # Optimal threshold for Phi (onset)
                'optimal_onset_threshold': optimal_onset_results['optimal_threshold'],
                'max_onset_phi': optimal_onset_results['max_phi'],
                'onset_phi_annotators_at_optimal': optimal_onset_results['phi_annotators'],

                # Optimal threshold for Phi (spread)
                'optimal_spread_threshold': optimal_spread_results['optimal_threshold'],
                'max_spread_phi': optimal_spread_results['max_phi'],
                'spread_phi_annotators_at_optimal': optimal_spread_results['phi_annotators'],
                
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
                generate_example_figure(sz_prob_smooth, prob_chs, onset_idx,
                                      all_chs, ueo_consensus, ueo_annotators, figpath)
        
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
            avg_onset_soz_prob, avg_onset_nsoz_prob, onset_auc, avg_spread_soz_prob, avg_spread_nsoz_prob, spread_auc = calculate_auc_and_probs(
                sz_prob_smooth, prob_chs, onset_labels, spread_labels, onset_idx, spread_idx
            )
            
            # Find optimal threshold for Phi (onset)
            optimal_onset_results = find_optimal_phi_threshold(
                sz_prob_smooth, prob_chs, onset_idx,
                all_chs, ueo_consensus, ueo_annotators
            )
            
            # Find optimal threshold for Phi (spread)
            optimal_spread_results = find_optimal_phi_threshold(
                sz_prob_smooth, prob_chs, spread_idx,
                all_chs, sec_consensus, sec_annotators
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
                
                # Inter-rater reliability (independent of model)
                'onset_inter_rater_reliability': onset_inter_rater,
                'spread_inter_rater_reliability': spread_inter_rater,
                
                # Probability-based metrics
                'avg_onset_soz_prob': avg_onset_soz_prob,
                'avg_onset_nsoz_prob': avg_onset_nsoz_prob,
                'onset_auc': onset_auc,
                'avg_spread_soz_prob': avg_spread_soz_prob,
                'avg_spread_nsoz_prob': avg_spread_nsoz_prob,
                'spread_auc': spread_auc,
                
                # Optimal threshold for Phi (onset)
                'optimal_onset_threshold': optimal_onset_results['optimal_threshold'],
                'max_onset_phi': optimal_onset_results['max_phi'],
                'onset_phi_annotators_at_optimal': optimal_onset_results['phi_annotators'],
                
                # Optimal threshold for Phi (spread)
                'optimal_spread_threshold': optimal_spread_results['optimal_threshold'],
                'max_spread_phi': optimal_spread_results['max_phi'],
                'spread_phi_annotators_at_optimal': optimal_spread_results['phi_annotators'],
                
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
        output_path = ospj(prodatapath, "test_validation_results_channels.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(f"Total results: {len(results_df)}")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()

