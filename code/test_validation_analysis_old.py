#!/usr/bin/env python3
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
import scipy.ndimage

# Utility imports
from utils import clean_labels

# Sklearn imports
from sklearn.metrics import (f1_score, matthews_corrcoef, precision_score, 
                            recall_score, roc_auc_score)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Get the project root for DynaSD
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

# Import models directly to avoid NDD dependency issues
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
        model_class = model_dict['model']
        model_name = model_dict['model_name']
        sequence_length = model_dict['sequence_length']
        forecast_length = model_dict['forecast_length']
        metric = model_dict.get('metric', 'mse')
        
        key = f"{model_name}_seq{sequence_length}_fc{forecast_length}"
        
        # Find probability files
        prob_dir = ospj(prodatapath, 'sz_prob', patient)
        pattern = f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_{metric}_prob_forecast-{forecast_length}.pkl"
        prob_path = glob.glob(ospj(prob_dir, pattern))
        
        if prob_path:
            prob_files[key] = {
                'data': pd.read_pickle(prob_path[0]),
                'model_class': model_class,
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
        # Find probability files
        prob_dir = ospj(prodatapath, 'sz_prob', patient)
        pattern = f"{patient}_task-ictal{onset_run}_run-*_mdl-{model_name}_sz_prob.pkl"
        prob_path = glob.glob(ospj(prob_dir, pattern))
        
        if prob_path:
            prob_files[model_name] = {
                'data': pd.read_pickle(prob_path[0]),
                'model_class': None,
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
    # Binary classification
    sz_clf = sz_prob > threshold
    
    # UEO channels - must be seizing for 5 consecutive timepoints at onset
    ueo_idx = np.all(sz_clf[:, onset_idx:onset_idx+5], axis=1)
    ueo_ch_bp = prob_chs[ueo_idx]
    ueo_ch_strict = np.array([s.split("-")[0] for s in ueo_ch_bp]) if len(ueo_ch_bp) > 0 else np.array([])
    ueo_ch_loose = np.unique(np.array([s.split("-") for s in ueo_ch_bp]).flatten()) if len(ueo_ch_bp) > 0 else np.array([])
    
    # SEC channels - must be seizing for 5 consecutive timepoints at spread
    sec_idx = np.all(sz_clf[:, spread_idx:spread_idx+5], axis=1)
    sec_ch_bp = prob_chs[sec_idx]
    sec_ch_strict = np.array([s.split("-")[0] for s in sec_ch_bp]) if len(sec_ch_bp) > 0 else np.array([])
    sec_ch_loose = np.unique(np.array([s.split("-") for s in sec_ch_bp]).flatten()) if len(sec_ch_bp) > 0 else np.array([])
    
    return {
        'ueo_strict': ueo_ch_strict,
        'ueo_loose': ueo_ch_loose,
        'sec_strict': sec_ch_strict,
        'sec_loose': sec_ch_loose,
        'sz_clf': sz_clf
    }


def wideform_preds(element, all_labels):
    """Convert element list to boolean array"""
    return np.array([label in element for label in all_labels])


def calculate_phi(pred_labels, true_labels, all_labels):
    """Calculate MCC/phi"""
    pred_bool = wideform_preds(pred_labels, all_labels)
    true_bool = wideform_preds(true_labels, all_labels)
    
    if len(pred_bool) == 0 or len(true_bool) == 0:
        return np.nan
    
    return matthews_corrcoef(pred_bool, true_bool)


def calculate_f1(pred_labels, true_labels, all_labels):
    """Calculate F1 score"""
    pred_bool = wideform_preds(pred_labels, all_labels)
    true_bool = wideform_preds(true_labels, all_labels)
    
    if len(pred_bool) == 0 or len(true_bool) == 0:
        return np.nan
    
    return f1_score(true_bool, pred_bool, zero_division=0)


def calculate_iou(pred_labels, true_labels, all_labels):
    """Calculate IOU (Jaccard index)"""
    pred_set = set(pred_labels)
    true_set = set(true_labels)
    
    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0
    
    intersection = len(pred_set & true_set)
    union = len(pred_set | true_set)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def load_region_mapping(patient, prodatapath):
    """Load electrode to region mapping"""
    region_file = ospj(prodatapath, 'electrode_localizations', f'{patient}.csv')
    
    if not os.path.exists(region_file):
        return None
    
    try:
        region_df = pd.read_csv(region_file)
        if 'label' not in region_df.columns:
            return None
        
        # Clean labels
        region_df['label_clean'] = region_df['label'].apply(lambda x: clean_labels([x], patient)[0] if pd.notna(x) else None)
        
        # Filter out excluded regions
        excluded_regions = ['white', 'ventricle', 'csf', 'outside']
        region_df = region_df[~region_df['label'].str.lower().str.contains('|'.join(excluded_regions), na=False)]
        
        return region_df
    except Exception as e:
        print(f"Error loading region mapping for {patient}: {e}")
        return None


def aggregate_to_regions(prob_df, prob_chs, region_df):
    """Aggregate channel probabilities to regions"""
    if region_df is None:
        return None, None
    
    # Create mapping from channel to region
    ch_to_region = {}
    for _, row in region_df.iterrows():
        if pd.notna(row.get('label_clean')):
            ch_to_region[row['label_clean']] = row['label']
    
    # Group channels by region
    region_probs = {}
    region_names = []
    
    for i, ch in enumerate(prob_chs):
        # Get contact name (first part of bipolar pair)
        contact = ch.split('-')[0]
        
        if contact in ch_to_region:
            region = ch_to_region[contact]
            if region not in region_probs:
                region_probs[region] = []
                region_names.append(region)
            region_probs[region].append(prob_df[i, :])
    
    if len(region_probs) == 0:
        return None, None
    
    # Average within each region
    region_prob_array = np.array([np.mean(region_probs[r], axis=0) for r in region_names])
    
    return region_prob_array, np.array(region_names)


def map_annotations_to_regions(annotations, region_df, patient):
    """Map channel annotations to regions"""
    if region_df is None or annotations is None:
        return []
    
    # Create mapping from channel to region
    ch_to_region = {}
    for _, row in region_df.iterrows():
        if pd.notna(row.get('label_clean')):
            ch_to_region[row['label_clean']] = row['label']
    
    # Map annotations
    regions = set()
    for ch in annotations:
        ch_clean = clean_labels([ch], patient)[0] if isinstance(ch, str) else ch
        if ch_clean in ch_to_region:
            regions.add(ch_to_region[ch_clean])
    
    return list(regions)


def find_optimal_thresholds(sz_prob, prob_chs, onset_idx, spread_idx, all_chs,
                            ueo_consensus, sec_consensus, n_thresholds=100):
    """Find optimal thresholds for phi, f1, and iou metrics"""
    
    # Sample thresholds
    thresholds = np.linspace(0, 4, n_thresholds)
    
    onset_phi_vals = []
    onset_f1_vals = []
    onset_iou_vals = []
    spread_phi_vals = []
    spread_f1_vals = []
    spread_iou_vals = []
    
    for threshold in thresholds:
        # Get predictions
        preds = get_channel_predictions(sz_prob, prob_chs, threshold, onset_idx, spread_idx)
        
        # Calculate onset metrics
        onset_phi = calculate_phi(preds['ueo_strict'], ueo_consensus, all_chs)
        onset_f1 = calculate_f1(preds['ueo_strict'], ueo_consensus, all_chs)
        onset_iou = calculate_iou(preds['ueo_strict'], ueo_consensus, all_chs)
        
        onset_phi_vals.append(onset_phi)
        onset_f1_vals.append(onset_f1)
        onset_iou_vals.append(onset_iou)
        
        # Calculate spread metrics
        spread_phi = calculate_phi(preds['sec_strict'], sec_consensus, all_chs)
        spread_f1 = calculate_f1(preds['sec_strict'], sec_consensus, all_chs)
        spread_iou = calculate_iou(preds['sec_strict'], sec_consensus, all_chs)
        
        spread_phi_vals.append(spread_phi)
        spread_f1_vals.append(spread_f1)
        spread_iou_vals.append(spread_iou)
    
    # Convert to arrays
    onset_phi_vals = np.array(onset_phi_vals)
    onset_f1_vals = np.array(onset_f1_vals)
    onset_iou_vals = np.array(onset_iou_vals)
    spread_phi_vals = np.array(spread_phi_vals)
    spread_f1_vals = np.array(spread_f1_vals)
    spread_iou_vals = np.array(spread_iou_vals)
    
    # Find optimal thresholds
    results = {}
    
    # Onset metrics
    if np.any(~np.isnan(onset_phi_vals)):
        idx = np.nanargmax(onset_phi_vals)
        results['onset_phi_threshold'] = thresholds[idx]
        results['onset_phi_max'] = onset_phi_vals[idx]
    else:
        results['onset_phi_threshold'] = np.nan
        results['onset_phi_max'] = np.nan
    
    if np.any(~np.isnan(onset_f1_vals)):
        idx = np.nanargmax(onset_f1_vals)
        results['onset_f1_threshold'] = thresholds[idx]
        results['onset_f1_max'] = onset_f1_vals[idx]
    else:
        results['onset_f1_threshold'] = np.nan
        results['onset_f1_max'] = np.nan
    
    if np.any(~np.isnan(onset_iou_vals)):
        idx = np.nanargmax(onset_iou_vals)
        results['onset_iou_threshold'] = thresholds[idx]
        results['onset_iou_max'] = onset_iou_vals[idx]
    else:
        results['onset_iou_threshold'] = np.nan
        results['onset_iou_max'] = np.nan
    
    # Spread metrics
    if np.any(~np.isnan(spread_phi_vals)):
        idx = np.nanargmax(spread_phi_vals)
        results['spread_phi_threshold'] = thresholds[idx]
        results['spread_phi_max'] = spread_phi_vals[idx]
    else:
        results['spread_phi_threshold'] = np.nan
        results['spread_phi_max'] = np.nan
    
    if np.any(~np.isnan(spread_f1_vals)):
        idx = np.nanargmax(spread_f1_vals)
        results['spread_f1_threshold'] = thresholds[idx]
        results['spread_f1_max'] = spread_f1_vals[idx]
    else:
        results['spread_f1_threshold'] = np.nan
        results['spread_f1_max'] = np.nan
    
    if np.any(~np.isnan(spread_iou_vals)):
        idx = np.nanargmax(spread_iou_vals)
        results['spread_iou_threshold'] = thresholds[idx]
        results['spread_iou_max'] = spread_iou_vals[idx]
    else:
        results['spread_iou_threshold'] = np.nan
        results['spread_iou_max'] = np.nan
    
    return results


def calculate_metrics_at_threshold(sz_prob, prob_chs, onset_idx, spread_idx,
                                   onset_labels, all_chs, ueo_consensus, sec_consensus,
                                   ueo_annotators, sec_annotators, threshold):
    """Calculate all metrics at a given threshold"""
    
    # Get predictions
    preds = get_channel_predictions(sz_prob, prob_chs, threshold, onset_idx, spread_idx)
    
    # Calculate consensus metrics
    onset_phi = calculate_phi(preds['ueo_strict'], ueo_consensus, all_chs)
    onset_f1 = calculate_f1(preds['ueo_strict'], ueo_consensus, all_chs)
    onset_iou = calculate_iou(preds['ueo_strict'], ueo_consensus, all_chs)
    
    spread_phi = calculate_phi(preds['sec_strict'], sec_consensus, all_chs)
    spread_f1 = calculate_f1(preds['sec_strict'], sec_consensus, all_chs)
    spread_iou = calculate_iou(preds['sec_strict'], sec_consensus, all_chs)
    
    # Calculate per-annotator phi
    onset_phi_annotators = [calculate_phi(preds['ueo_strict'], ann, all_chs) for ann in ueo_annotators]
    spread_phi_annotators = [calculate_phi(preds['sec_strict'], ann, all_chs) for ann in sec_annotators]
    
    # Calculate average probabilities and AUC
    onset_mask = np.array([ch.split('-')[0] in onset_labels for ch in prob_chs])
    
    if np.any(onset_mask):
        avg_soz_prob = np.mean(sz_prob[onset_mask, onset_idx:onset_idx+5])
        avg_nsoz_prob = np.mean(sz_prob[~onset_mask, onset_idx:onset_idx+5])
        
        # Calculate AUC for onset
        onset_pred = sz_prob[:, onset_idx:onset_idx+5].mean(axis=1)
        try:
            onset_auc = roc_auc_score(onset_mask, onset_pred)
        except:
            onset_auc = np.nan
        
        # Calculate AUC for spread
        spread_pred = sz_prob[:, spread_idx:spread_idx+5].mean(axis=1)
        try:
            spread_auc = roc_auc_score(onset_mask, spread_pred)
        except:
            spread_auc = np.nan
    else:
        avg_soz_prob = np.nan
        avg_nsoz_prob = np.nan
        onset_auc = np.nan
        spread_auc = np.nan
    
    return {
        'onset_phi': onset_phi,
        'onset_f1': onset_f1,
        'onset_iou': onset_iou,
        'spread_phi': spread_phi,
        'spread_f1': spread_f1,
        'spread_iou': spread_iou,
        'onset_phi_annotators': onset_phi_annotators,
        'spread_phi_annotators': spread_phi_annotators,
        'avg_soz_prob': avg_soz_prob,
        'avg_nsoz_prob': avg_nsoz_prob,
        'onset_auc': onset_auc,
        'spread_auc': spread_auc
    }


def generate_example_figure(sz_prob, prob_chs, onset_idx, spread_idx, all_chs,
                           ueo_consensus, sec_consensus, figpath):
    """Generate threshold sweep figure for HUP238 example"""
    
    thresholds = np.linspace(0, 4, 750)
    phi_vals = []
    
    for threshold in thresholds:
        preds = get_channel_predictions(sz_prob, prob_chs, threshold, onset_idx, spread_idx)
        phi = calculate_phi(preds['ueo_strict'], ueo_consensus, all_chs)
        phi_vals.append(phi)
    
    phi_vals = np.array(phi_vals)
    
    if np.any(~np.isnan(phi_vals)):
        optimal_idx = np.nanargmax(phi_vals)
        optimal_threshold = thresholds[optimal_idx]
        max_phi = phi_vals[optimal_idx]
        
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(thresholds, phi_vals, 
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
    """
    Main validation analysis pipeline for test seizures
    """
    
    print("Loading metadata and annotations...")
    
    # Load seizure metadata
    seizures_df = pd.read_csv(ospj(metapath, "metadata_v6_BIDS.csv"))
    # Filter for test seizures (split==2) and spontaneous (stim==0)
    seizures_df = seizures_df[(seizures_df.split == 2) & (seizures_df.stim == 0)]
    
    print(f"Found {len(seizures_df)} test seizures")
    
    # Load clinical annotations
    annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v2.pkl"))
    
    # Load learned thresholds for NDD and benchmark models
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
        
        # Get clinical annotations for this seizure
        # Match by approximate onset time and patient
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
        
        # Get onset labels from consensus annotations (list of channel labels)
        # Convert boolean array to list of channel labels
        onset_labels = [ch for ch, is_onset in zip(all_chs, ueo_consensus) if is_onset]
        
        # Load region mapping
        region_df = load_region_mapping(patient, prodatapath)
        
        # Process NDD models
        ndd_prob_files = load_ndd_probability_files(patient, onset_run, ndd_models)
        
        for model_key, prob_info in ndd_prob_files.items():
            prob_data = prob_info['data'].copy()
            model_name = prob_info['model_name']
            
            # Extract time array
            if 'time' in prob_data.columns:
                prob_times = prob_data.pop('time').values
            else:
                print(f"Warning: No time column for {patient} {onset_run} {model_key}")
                continue
            
            prob_chs = prob_data.columns.to_numpy()
            sz_prob = prob_data.to_numpy().T
            
            # Apply smoothing
            sz_prob_smooth = apply_smoothing(sz_prob, window_size=20)
            
            # Calculate temporal alignment
            time_diff = consensus_time - approx_onset
            onset_idx = int(np.argmin(np.abs((prob_times - 120) + time_diff)))
            spread_idx = int(np.argmin(np.abs((prob_times - 130) + time_diff)))
            
            # Find optimal thresholds for this seizure
            optimal_thresholds = find_optimal_thresholds(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                all_chs, ueo_consensus, sec_consensus
            )
            
            # Calculate metrics at optimal thresholds
            metrics_at_optimal_phi = calculate_metrics_at_threshold(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                onset_labels, all_chs, ueo_consensus, sec_consensus,
                ueo_annotators, sec_annotators,
                optimal_thresholds['onset_phi_threshold']
            )
            
            metrics_at_optimal_f1 = calculate_metrics_at_threshold(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                onset_labels, all_chs, ueo_consensus, sec_consensus,
                ueo_annotators, sec_annotators,
                optimal_thresholds['onset_f1_threshold']
            )
            
            metrics_at_optimal_iou = calculate_metrics_at_threshold(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                onset_labels, all_chs, ueo_consensus, sec_consensus,
                ueo_annotators, sec_annotators,
                optimal_thresholds['onset_iou_threshold']
            )
            
            # Get full model identifier for learned thresholds
            full_model_key = f"{model_name}_mse_sl{prob_info['sequence_length']}_fl{prob_info['forecast_length']}"
            
            # Calculate metrics at learned thresholds
            metrics_at_learned = {}
            for metric_name in ['phi', 'f1', 'iou']:
                if full_model_key in ndd_thresholds[metric_name]:
                    learned_thresh = ndd_thresholds[metric_name][full_model_key]
                    metrics_at_learned[metric_name] = calculate_metrics_at_threshold(
                        sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                        onset_labels, all_chs, ueo_consensus, sec_consensus,
                        ueo_annotators, sec_annotators, learned_thresh
                    )
                else:
                    metrics_at_learned[metric_name] = None
            
            # Region-level analysis
            region_prob_array, region_names = aggregate_to_regions(sz_prob_smooth, prob_chs, region_df)
            
            if region_prob_array is not None:
                # Map annotations to regions
                ueo_consensus_regions = map_annotations_to_regions(ueo_consensus, region_df, patient)
                sec_consensus_regions = map_annotations_to_regions(sec_consensus, region_df, patient)
                ueo_annotators_regions = [map_annotations_to_regions(ann, region_df, patient) for ann in ueo_annotators]
                sec_annotators_regions = [map_annotations_to_regions(ann, region_df, patient) for ann in sec_annotators]
                all_regions = list(region_names)
                onset_labels_regions = map_annotations_to_regions(onset_labels, region_df, patient)
                
                # Find optimal thresholds for regions
                region_optimal_thresholds = find_optimal_thresholds(
                    region_prob_array, region_names, onset_idx, spread_idx,
                    all_regions, ueo_consensus_regions, sec_consensus_regions
                )
                
                # Calculate region metrics at optimal thresholds
                region_metrics_at_optimal_phi = calculate_metrics_at_threshold(
                    region_prob_array, region_names, onset_idx, spread_idx,
                    onset_labels_regions, all_regions, ueo_consensus_regions, sec_consensus_regions,
                    ueo_annotators_regions, sec_annotators_regions,
                    region_optimal_thresholds['onset_phi_threshold']
                )
            else:
                region_optimal_thresholds = None
                region_metrics_at_optimal_phi = None
            
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
                
                # Optimal seizure-specific thresholds
                'optimal_phi_threshold': optimal_thresholds['onset_phi_threshold'],
                'optimal_f1_threshold': optimal_thresholds['onset_f1_threshold'],
                'optimal_iou_threshold': optimal_thresholds['onset_iou_threshold'],
                
                # Metrics at optimal phi threshold
                'phi_at_optimal_phi': metrics_at_optimal_phi['onset_phi'],
                'f1_at_optimal_phi': metrics_at_optimal_phi['onset_f1'],
                'iou_at_optimal_phi': metrics_at_optimal_phi['onset_iou'],
                'spread_phi_at_optimal_phi': metrics_at_optimal_phi['spread_phi'],
                'onset_phi_annotators_at_optimal_phi': metrics_at_optimal_phi['onset_phi_annotators'],
                
                # Metrics at optimal f1 threshold
                'phi_at_optimal_f1': metrics_at_optimal_f1['onset_phi'],
                'f1_at_optimal_f1': metrics_at_optimal_f1['onset_f1'],
                'iou_at_optimal_f1': metrics_at_optimal_f1['onset_iou'],
                
                # Metrics at optimal iou threshold
                'phi_at_optimal_iou': metrics_at_optimal_iou['onset_phi'],
                'f1_at_optimal_iou': metrics_at_optimal_iou['onset_f1'],
                'iou_at_optimal_iou': metrics_at_optimal_iou['onset_iou'],
                
                # Metrics at learned thresholds
                'phi_at_learned_phi': metrics_at_learned['phi']['onset_phi'] if metrics_at_learned['phi'] else np.nan,
                'f1_at_learned_phi': metrics_at_learned['phi']['onset_f1'] if metrics_at_learned['phi'] else np.nan,
                'iou_at_learned_phi': metrics_at_learned['phi']['onset_iou'] if metrics_at_learned['phi'] else np.nan,
                'learned_phi_threshold': ndd_thresholds['phi'].get(full_model_key, np.nan),
                
                'phi_at_learned_f1': metrics_at_learned['f1']['onset_phi'] if metrics_at_learned['f1'] else np.nan,
                'f1_at_learned_f1': metrics_at_learned['f1']['onset_f1'] if metrics_at_learned['f1'] else np.nan,
                'iou_at_learned_f1': metrics_at_learned['f1']['onset_iou'] if metrics_at_learned['f1'] else np.nan,
                'learned_f1_threshold': ndd_thresholds['f1'].get(full_model_key, np.nan),
                
                'phi_at_learned_iou': metrics_at_learned['iou']['onset_phi'] if metrics_at_learned['iou'] else np.nan,
                'f1_at_learned_iou': metrics_at_learned['iou']['onset_f1'] if metrics_at_learned['iou'] else np.nan,
                'iou_at_learned_iou': metrics_at_learned['iou']['onset_iou'] if metrics_at_learned['iou'] else np.nan,
                'learned_iou_threshold': ndd_thresholds['iou'].get(full_model_key, np.nan),
                
                # Probability and AUC metrics
                'avg_soz_prob': metrics_at_optimal_phi['avg_soz_prob'],
                'avg_nsoz_prob': metrics_at_optimal_phi['avg_nsoz_prob'],
                'onset_auc': metrics_at_optimal_phi['onset_auc'],
                'spread_auc': metrics_at_optimal_phi['spread_auc'],
                
                # Region-level metrics
                'optimal_phi_threshold_region': region_optimal_thresholds['onset_phi_threshold'] if region_optimal_thresholds else np.nan,
                'phi_at_optimal_phi_region': region_metrics_at_optimal_phi['onset_phi'] if region_metrics_at_optimal_phi else np.nan,
                'f1_at_optimal_phi_region': region_metrics_at_optimal_phi['onset_f1'] if region_metrics_at_optimal_phi else np.nan,
                'iou_at_optimal_phi_region': region_metrics_at_optimal_phi['onset_iou'] if region_metrics_at_optimal_phi else np.nan,
                'avg_soz_prob_region': region_metrics_at_optimal_phi['avg_soz_prob'] if region_metrics_at_optimal_phi else np.nan,
                'onset_auc_region': region_metrics_at_optimal_phi['onset_auc'] if region_metrics_at_optimal_phi else np.nan,
            }
            
            results.append(result_dict)
            
            # Generate example figure for HUP238, seizure 290006, LiNDDA
            if patient == 'HUP238' and int(onset_run) == 290006 and model_name == 'LiNDDA':
                print(f"\nGenerating example figure for {patient} {onset_run} {model_name}...")
                generate_example_figure(sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                                      all_chs, ueo_consensus, sec_consensus, figpath)
        
        # Process benchmark models
        benchmark_prob_files = load_benchmark_probability_files(patient, onset_run, benchmark_models)
        
        for model_name, prob_info in benchmark_prob_files.items():
            prob_data = prob_info['data'].copy()
            
            # Extract time array
            if 'time' in prob_data.columns:
                prob_times = prob_data.pop('time').values
            else:
                print(f"Warning: No time column for {patient} {onset_run} {model_name}")
                continue
            
            prob_chs = prob_data.columns.to_numpy()
            sz_prob = prob_data.to_numpy().T
            
            # Apply smoothing
            sz_prob_smooth = apply_smoothing(sz_prob, window_size=20)
            
            # Calculate temporal alignment
            time_diff = consensus_time - approx_onset
            onset_idx = int(np.argmin(np.abs((prob_times - 120) + time_diff)))
            spread_idx = int(np.argmin(np.abs((prob_times - 130) + time_diff)))
            
            # Find optimal thresholds for this seizure
            optimal_thresholds = find_optimal_thresholds(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                all_chs, ueo_consensus, sec_consensus
            )
            
            # Calculate metrics at optimal thresholds
            metrics_at_optimal_phi = calculate_metrics_at_threshold(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                onset_labels, all_chs, ueo_consensus, sec_consensus,
                ueo_annotators, sec_annotators,
                optimal_thresholds['onset_phi_threshold']
            )
            
            metrics_at_optimal_f1 = calculate_metrics_at_threshold(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                onset_labels, all_chs, ueo_consensus, sec_consensus,
                ueo_annotators, sec_annotators,
                optimal_thresholds['onset_f1_threshold']
            )
            
            metrics_at_optimal_iou = calculate_metrics_at_threshold(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                onset_labels, all_chs, ueo_consensus, sec_consensus,
                ueo_annotators, sec_annotators,
                optimal_thresholds['onset_iou_threshold']
            )
            
            # Calculate metrics at learned thresholds
            metrics_at_learned = {}
            for metric_name in ['phi', 'f1', 'iou']:
                if model_name in benchmark_thresholds[metric_name]:
                    learned_thresh = benchmark_thresholds[metric_name][model_name]
                    metrics_at_learned[metric_name] = calculate_metrics_at_threshold(
                        sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                        onset_labels, all_chs, ueo_consensus, sec_consensus,
                        ueo_annotators, sec_annotators, learned_thresh
                    )
                else:
                    metrics_at_learned[metric_name] = None
            
            # Region-level analysis
            region_prob_array, region_names = aggregate_to_regions(sz_prob_smooth, prob_chs, region_df)
            
            if region_prob_array is not None:
                ueo_consensus_regions = map_annotations_to_regions(ueo_consensus, region_df, patient)
                sec_consensus_regions = map_annotations_to_regions(sec_consensus, region_df, patient)
                ueo_annotators_regions = [map_annotations_to_regions(ann, region_df, patient) for ann in ueo_annotators]
                sec_annotators_regions = [map_annotations_to_regions(ann, region_df, patient) for ann in sec_annotators]
                all_regions = list(region_names)
                onset_labels_regions = map_annotations_to_regions(onset_labels, region_df, patient)
                
                region_optimal_thresholds = find_optimal_thresholds(
                    region_prob_array, region_names, onset_idx, spread_idx,
                    all_regions, ueo_consensus_regions, sec_consensus_regions
                )
                
                region_metrics_at_optimal_phi = calculate_metrics_at_threshold(
                    region_prob_array, region_names, onset_idx, spread_idx,
                    onset_labels_regions, all_regions, ueo_consensus_regions, sec_consensus_regions,
                    ueo_annotators_regions, sec_annotators_regions,
                    region_optimal_thresholds['onset_phi_threshold']
                )
            else:
                region_optimal_thresholds = None
                region_metrics_at_optimal_phi = None
            
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
                
                # Optimal seizure-specific thresholds
                'optimal_phi_threshold': optimal_thresholds['onset_phi_threshold'],
                'optimal_f1_threshold': optimal_thresholds['onset_f1_threshold'],
                'optimal_iou_threshold': optimal_thresholds['onset_iou_threshold'],
                
                # Metrics at optimal phi threshold
                'phi_at_optimal_phi': metrics_at_optimal_phi['onset_phi'],
                'f1_at_optimal_phi': metrics_at_optimal_phi['onset_f1'],
                'iou_at_optimal_phi': metrics_at_optimal_phi['onset_iou'],
                'spread_phi_at_optimal_phi': metrics_at_optimal_phi['spread_phi'],
                'onset_phi_annotators_at_optimal_phi': metrics_at_optimal_phi['onset_phi_annotators'],
                
                # Metrics at optimal f1 threshold
                'phi_at_optimal_f1': metrics_at_optimal_f1['onset_phi'],
                'f1_at_optimal_f1': metrics_at_optimal_f1['onset_f1'],
                'iou_at_optimal_f1': metrics_at_optimal_f1['onset_iou'],
                
                # Metrics at optimal iou threshold
                'phi_at_optimal_iou': metrics_at_optimal_iou['onset_phi'],
                'f1_at_optimal_iou': metrics_at_optimal_iou['onset_f1'],
                'iou_at_optimal_iou': metrics_at_optimal_iou['onset_iou'],
                
                # Metrics at learned thresholds
                'phi_at_learned_phi': metrics_at_learned['phi']['onset_phi'] if metrics_at_learned['phi'] else np.nan,
                'f1_at_learned_phi': metrics_at_learned['phi']['onset_f1'] if metrics_at_learned['phi'] else np.nan,
                'iou_at_learned_phi': metrics_at_learned['phi']['onset_iou'] if metrics_at_learned['phi'] else np.nan,
                'learned_phi_threshold': benchmark_thresholds['phi'].get(model_name, np.nan),
                
                'phi_at_learned_f1': metrics_at_learned['f1']['onset_phi'] if metrics_at_learned['f1'] else np.nan,
                'f1_at_learned_f1': metrics_at_learned['f1']['onset_f1'] if metrics_at_learned['f1'] else np.nan,
                'iou_at_learned_f1': metrics_at_learned['f1']['onset_iou'] if metrics_at_learned['f1'] else np.nan,
                'learned_f1_threshold': benchmark_thresholds['f1'].get(model_name, np.nan),
                
                'phi_at_learned_iou': metrics_at_learned['iou']['onset_phi'] if metrics_at_learned['iou'] else np.nan,
                'f1_at_learned_iou': metrics_at_learned['iou']['onset_f1'] if metrics_at_learned['iou'] else np.nan,
                'iou_at_learned_iou': metrics_at_learned['iou']['onset_iou'] if metrics_at_learned['iou'] else np.nan,
                'learned_iou_threshold': benchmark_thresholds['iou'].get(model_name, np.nan),
                
                # Probability and AUC metrics
                'avg_soz_prob': metrics_at_optimal_phi['avg_soz_prob'],
                'avg_nsoz_prob': metrics_at_optimal_phi['avg_nsoz_prob'],
                'onset_auc': metrics_at_optimal_phi['onset_auc'],
                'spread_auc': metrics_at_optimal_phi['spread_auc'],
                
                # Region-level metrics
                'optimal_phi_threshold_region': region_optimal_thresholds['onset_phi_threshold'] if region_optimal_thresholds else np.nan,
                'phi_at_optimal_phi_region': region_metrics_at_optimal_phi['onset_phi'] if region_metrics_at_optimal_phi else np.nan,
                'f1_at_optimal_phi_region': region_metrics_at_optimal_phi['onset_f1'] if region_metrics_at_optimal_phi else np.nan,
                'iou_at_optimal_phi_region': region_metrics_at_optimal_phi['onset_iou'] if region_metrics_at_optimal_phi else np.nan,
                'avg_soz_prob_region': region_metrics_at_optimal_phi['avg_soz_prob'] if region_metrics_at_optimal_phi else np.nan,
                'onset_auc_region': region_metrics_at_optimal_phi['onset_auc'] if region_metrics_at_optimal_phi else np.nan,
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
