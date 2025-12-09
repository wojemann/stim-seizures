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
from utils import clean_labels, load_electrode_localizations

# Sklearn imports
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as sklearn_auc

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
from DynaSD.NDD import NDD
from DynaSD.HFER import HFER
from DynaSD.ABSSLP import ABSSLP
from DynaSD.IMPRINT import IMPRINT
from DynaSD.WAVENET import WVNT
from DynaSD.ONCET import ONCET
from config import Config

# Get paths from config 
datapath, prodatapath, figpath, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])

MODEL_VERSION = 'nopass_nolayernorm'  # Version suffix for probability files (use '' for default, 'nopass_nolayernorm' for new hyperparameters, etc.)


def map_channels_to_regions(channel_list, ch_to_region):
    """Convert list of channel names (first contacts) to list of unique region names"""
    region_set = set()
    
    for ch in channel_list:
        # ch is already first contact from clinical annotations
        region = ch_to_region.get(ch)
        if region is not None:
            region_set.add(region)
    
    return list(region_set)


def aggregate_to_regions(sz_prob, prob_chs, ch_to_region):
    """
    Aggregate channel probabilities to region level by averaging.
    
    Parameters:
    -----------
    sz_prob : np.array
        Channel probability matrix (n_channels × n_timepoints)
    prob_chs : np.array
        Channel names
    ch_to_region : dict
        Mapping from first contact to region name
    
    Returns:
    --------
    region_prob_matrix : np.array
        Region probability matrix (n_regions × n_timepoints)
    region_names : list
        List of region names
    """
    # Group channels by region
    region_prob_dict = {}
    
    for i, ch in enumerate(prob_chs):
        # Extract first contact from bipolar channel
        first_contact = ch.split('-')[0]
        region = ch_to_region.get(first_contact)
        
        if region is None:
            continue
        
        if region not in region_prob_dict:
            region_prob_dict[region] = []
        
        region_prob_dict[region].append(sz_prob[i, :])
    
    if len(region_prob_dict) == 0:
        return None, None
    
    # Average probabilities within each region
    region_names = sorted(region_prob_dict.keys())
    region_prob_list = []
    
    for region in region_names:
        # Average across all channels in this region
        region_avg = np.mean(region_prob_dict[region], axis=0)
        region_prob_list.append(region_avg)
    
    region_prob_matrix = np.array(region_prob_list)
    
    return region_prob_matrix, np.array(region_names)


def load_ndd_probability_files(patient, onset_run, model_dicts):
    """Load probability files for NDD models"""
    prob_files = {}
    
    for model_dict in model_dicts:
        model_name = model_dict['model_name']
        model = model_dict['model']
        sequence_length = model_dict['sequence_length']
        forecast_length = model_dict['forecast_length']
        metric = model_dict.get('metric', 'mse')
        suffix = model_dict.get('suffix', '')
        
        key = f"{model_name}_seq{sequence_length}_fc{forecast_length}"
        
        prob_dir = ospj(prodatapath, 'sz_prob', patient)
        pattern = f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_{metric}_prob_forecast-{forecast_length}"
        
        # Add suffix if provided
        if suffix:
            pattern += f"{suffix}.pkl"
        else:
            pattern += ".pkl"
        
        prob_path = glob.glob(ospj(prob_dir, pattern))
        
        if prob_path:
            prob_files[key] = {
                'data': pd.read_pickle(prob_path[0]),
                'model': model(fs=256,w_size=1,w_stride=0.5,sequence_length=sequence_length,forecast_length=forecast_length),
                'model_name': model_name,
                'sequence_length': sequence_length,
                'forecast_length': forecast_length,
                'metric': metric
            }
        else:
            print(f"Warning: No probability file found for {patient} {onset_run} {key}")
    
    return prob_files


def load_benchmark_probability_files(patient, onset_run, model_dicts):
    """Load probability files for benchmark models"""
    prob_files = {}
    
    for model_dict in model_dicts:
        model_name = model_dict['model_name']
        model = model_dict['model']
        suffix = model_dict.get('suffix', '')
        
        prob_dir = ospj(prodatapath, 'sz_prob', patient)
        
        # Build pattern with optional suffix
        if suffix:
            pattern = f"{patient}_task-ictal{onset_run}_run-*_mdl-{model_name}{suffix}_sz_prob.pkl"
        else:
            pattern = f"{patient}_task-ictal{onset_run}_run-*_mdl-{model_name}_sz_prob.pkl"
        
        prob_path = glob.glob(ospj(prob_dir, pattern))
        
        if prob_path:
            prob_files[model_name] = {
                'data': pd.read_pickle(prob_path[0]),
                'model_name': model_name,
                'model': model(w_size=1,w_stride=0.5,fs=256),
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
    # return sc.ndimage.median_filter(prob_array, size=window_size, mode='nearest', axes=1, origin=0)

def get_channel_predictions(sz_prob, prob_chs, threshold, onset_idx, spread_idx):
    """Get channel predictions at onset and spread timepoints"""
    # sz_clf = sz_prob > threshold
    
    # UEO channels - seizing for at least 4/5 timepoints at onset (80%)
    # ueo_idx = np.sum(sz_clf[:, onset_idx:onset_idx+5], axis=1) >= 5
    ueo_idx = sz_prob[:, onset_idx:onset_idx+5].mean(axis=1) > threshold
    ueo_ch_strict = prob_chs[ueo_idx] if np.any(ueo_idx) else np.array([])
    
    # SEC channels - seizing for at least 4/5 timepoints at spread (80%)
    # sec_idx = np.sum(sz_clf[:, spread_idx:spread_idx+5], axis=1) >= 5
    sec_idx = sz_prob[:, spread_idx:spread_idx+5].mean(axis=1) > threshold
    sec_ch_strict = prob_chs[sec_idx] if np.any(sec_idx) else np.array([])
    
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


def calculate_auc_and_probs(sz_prob, onset_mask, spread_mask, onset_idx, spread_idx):
    """Calculate AUC, AUPRC, and average probabilities without thresholding"""
    # Calculate onset metrics
    if np.any(onset_mask):
        # Average probabilities at onset window (2 timepoints)
        avg_onset_soz_prob = np.mean(sz_prob[onset_mask, onset_idx:onset_idx+2])
        avg_onset_nsoz_prob = np.mean(sz_prob[~onset_mask, onset_idx:onset_idx+2])
        
        # Calculate AUC for onset
        onset_pred = sz_prob[:, onset_idx:onset_idx+2].mean(axis=1)
        try:
            onset_auc = roc_auc_score(onset_mask, onset_pred)
        except:
            onset_auc = np.nan
        
        # Calculate AUPRC for onset
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(onset_mask, onset_pred)
            onset_auprc_raw = sklearn_auc(recall_vals, precision_vals)
            
            # Calculate normalized AUPRC
            baseline = np.sum(onset_mask) / len(onset_mask)  # Fraction of positive samples
            onset_auprc_normalized = (onset_auprc_raw - baseline) / (1 - baseline) if baseline < 1 else np.nan
        except:
            onset_auprc_raw = np.nan
            onset_auprc_normalized = np.nan
    else:
        avg_onset_soz_prob = np.nan
        avg_onset_nsoz_prob = np.nan
        onset_auc = np.nan
        onset_auprc_raw = np.nan
        onset_auprc_normalized = np.nan
    
    # Calculate spread metrics
    if np.any(spread_mask):
        # Average probabilities at spread window (2 timepoints)
        avg_spread_soz_prob = np.mean(sz_prob[spread_mask, spread_idx:spread_idx+2])
        avg_spread_nsoz_prob = np.mean(sz_prob[~spread_mask, spread_idx:spread_idx+2])
        
        # Calculate AUC for spread
        spread_pred = sz_prob[:, spread_idx:spread_idx+2].mean(axis=1)
        try:
            spread_auc = roc_auc_score(spread_mask, spread_pred)
        except:
            spread_auc = np.nan
        
        # Calculate AUPRC for spread
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(spread_mask, spread_pred)
            spread_auprc_raw = sklearn_auc(recall_vals, precision_vals)
            
            # Calculate normalized AUPRC
            baseline = np.sum(spread_mask) / len(spread_mask)  # Fraction of positive samples
            spread_auprc_normalized = (spread_auprc_raw - baseline) / (1 - baseline) if baseline < 1 else np.nan
        except:
            spread_auprc_raw = np.nan
            spread_auprc_normalized = np.nan
    else:
        avg_spread_soz_prob = np.nan
        avg_spread_nsoz_prob = np.nan
        spread_auc = np.nan
        spread_auprc_raw = np.nan
        spread_auprc_normalized = np.nan
    
    return (avg_onset_soz_prob, avg_onset_nsoz_prob, onset_auc, onset_auprc_raw, onset_auprc_normalized,
            avg_spread_soz_prob, avg_spread_nsoz_prob, spread_auc, spread_auprc_raw, spread_auprc_normalized)


def find_optimal_phi_threshold(sz_prob, prob_chs, window_idx, all_chs, consensus_labels, annotators, tolerance=0.01):
    """
    Find optimal threshold for Phi using plateau methodology.
    
    This method finds thresholds within tolerance of the best score, groups them into
    continuous plateaus, and selects the midpoint of the longest plateau for better generalization.
    
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
    tolerance : float
        Score tolerance for plateau detection (default: 0.01)
    
    Returns:
    --------
    dict with optimal_threshold, max_phi, phi_annotators, unique_probs, phi_curve
    """
    
    # Get average probabilities over the window for threshold evaluation
    window_probs = sz_prob[:, window_idx:window_idx+5].mean(axis=1)
    
    # Use linear space of thresholds between 5th and 99th percentiles (like plateau method)
    min_prob = np.percentile(window_probs, 5)
    max_prob = np.percentile(window_probs, 99)
    thresholds = np.linspace(min_prob, max_prob, 301)
    
    phi_vals = []
    phi_per_annotator = {i: [] for i in range(len(annotators))}
    
    for threshold in thresholds:
        # Get predictions at this window (mean probability above threshold)
        window_idx_pred = window_probs > threshold
        predicted_chs = prob_chs[window_idx_pred] if np.any(window_idx_pred) else np.array([])
        
        # Calculate phi with consensus
        phi = calculate_phi(predicted_chs, consensus_labels, all_chs)
        phi_vals.append(phi)
        
        # Calculate phi with each annotator
        for i, annotator_labels in enumerate(annotators):
            phi_annot = calculate_phi(predicted_chs, annotator_labels, all_chs)
            phi_per_annotator[i].append(phi_annot)
    
    phi_vals = np.array(phi_vals)
    
    # Find optimal threshold using plateau methodology
    if np.any(~np.isnan(phi_vals)):
        best_score = np.nanmax(phi_vals)
        
        # Find all thresholds within tolerance of best score
        within_tolerance = phi_vals >= (best_score - tolerance)
        candidate_indices = np.where(within_tolerance)[0]
        
        if len(candidate_indices) == 0:
            # Fallback to best score if no candidates
            optimal_idx = np.nanargmax(phi_vals)
            optimal_threshold = thresholds[optimal_idx]
            max_phi = phi_vals[optimal_idx]
            phi_annotators = [phi_per_annotator[i][optimal_idx] for i in range(len(annotators))]
        else:
            # Group into continuous segments (plateaus)
            segments = []
            current_segment = [candidate_indices[0]]
            
            for i in range(1, len(candidate_indices)):
                if candidate_indices[i] == candidate_indices[i-1] + 1:
                    # Continuous
                    current_segment.append(candidate_indices[i])
                else:
                    # Gap found, start new segment
                    segments.append(current_segment)
                    current_segment = [candidate_indices[i]]
            segments.append(current_segment)  # Add last segment
            
            # Find longest continuous plateau
            longest_segment = max(segments, key=len)
            
            # Take midpoint of longest plateau
            mid_idx = longest_segment[len(longest_segment) // 2]
            optimal_threshold = thresholds[mid_idx]
            max_phi = phi_vals[mid_idx]
            phi_annotators = [phi_per_annotator[i][mid_idx] for i in range(len(annotators))]
    else:
        optimal_threshold = np.nan
        max_phi = np.nan
        phi_annotators = [np.nan] * len(annotators)
    
    return {
        'optimal_threshold': optimal_threshold,
        'max_phi': max_phi,
        'phi_annotators': phi_annotators,
        'unique_probs': thresholds,
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


def calculate_all_metrics(sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                         onset_mask, spread_mask, all_chs,
                         ueo_consensus, sec_consensus, ueo_annotators, sec_annotators,
                         onset_labels_regions, spread_labels_regions, 
                         ueo_annotators_regions, sec_annotators_regions,
                         ch_to_region, learned_thresholds_dict):
    """
    Calculate all metrics for a model including channel-level and region-level analysis.
    
    Parameters:
    -----------
    sz_prob_smooth : np.array
        Smoothed probability matrix
    prob_chs : np.array
        Channel names
    onset_idx, spread_idx : int
        Temporal indices for onset and spread
    onset_mask, spread_mask : np.array
        Boolean masks for ground truth labels
    all_chs : list
        All channel labels
    ueo_consensus, sec_consensus : np.array
        Consensus boolean arrays
    ueo_annotators, sec_annotators : list
        List of annotator boolean arrays
    onset_labels_regions, spread_labels_regions : list or None
        Region labels (or None if unavailable)
    ueo_annotators_regions, sec_annotators_regions : list or None
        Annotator region labels
    ch_to_region : dict or None
        Channel to region mapping
    learned_thresholds_dict : dict
        Dict with keys 'phi', 'f1', 'iou' containing learned thresholds
    
    Returns:
    --------
    dict with all calculated metrics
    """
    metrics = {}
    
    # Channel-level AUC and probabilities
    (avg_onset_soz_prob, avg_onset_nsoz_prob, onset_auc, onset_auprc_raw, onset_auprc_normalized,
     avg_spread_soz_prob, avg_spread_nsoz_prob, spread_auc, spread_auprc_raw, spread_auprc_normalized) = calculate_auc_and_probs(
        sz_prob_smooth, onset_mask, spread_mask, onset_idx, spread_idx
    )
    
    metrics.update({
        'avg_onset_soz_prob': avg_onset_soz_prob,
        'avg_onset_nsoz_prob': avg_onset_nsoz_prob,
        'onset_auc': onset_auc,
        'onset_auprc_raw': onset_auprc_raw,
        'onset_auprc_normalized': onset_auprc_normalized,
        'avg_spread_soz_prob': avg_spread_soz_prob,
        'avg_spread_nsoz_prob': avg_spread_nsoz_prob,
        'spread_auc': spread_auc,
        'spread_auprc_raw': spread_auprc_raw,
        'spread_auprc_normalized': spread_auprc_normalized,
    })
    
    # Optimal thresholds for Phi
    optimal_onset_results = find_optimal_phi_threshold(
        sz_prob_smooth, prob_chs, onset_idx,
        all_chs, ueo_consensus, ueo_annotators
    )
    
    optimal_spread_results = find_optimal_phi_threshold(
        sz_prob_smooth, prob_chs, spread_idx,
        all_chs, sec_consensus, sec_annotators
    )
    
    metrics.update({
        'optimal_onset_threshold': optimal_onset_results['optimal_threshold'],
        'max_onset_phi': optimal_onset_results['max_phi'],
        'onset_phi_annotators_at_optimal': optimal_onset_results['phi_annotators'],
        'optimal_spread_threshold': optimal_spread_results['optimal_threshold'],
        'max_spread_phi': optimal_spread_results['max_phi'],
        'spread_phi_annotators_at_optimal': optimal_spread_results['phi_annotators'],
    })
    
    # Phi at learned thresholds
    # Handle both old format (metric_name) and new format (metric_name_aggregation)
    for threshold_key, learned_thresh in learned_thresholds_dict.items():
        if np.isnan(learned_thresh):
            continue
            
        phi_at_learned = calculate_phi_at_threshold(
            sz_prob_smooth, prob_chs, onset_idx, spread_idx,
            all_chs, ueo_consensus, sec_consensus,
            ueo_annotators, sec_annotators, learned_thresh
        )
        
        # Use the threshold_key as the identifier (e.g., 'f1_mean', 'phi_median', 'tau')
        metrics[f'learned_{threshold_key}_threshold'] = learned_thresh
        metrics[f'onset_phi_at_learned_{threshold_key}'] = phi_at_learned['onset_phi']
        metrics[f'spread_phi_at_learned_{threshold_key}'] = phi_at_learned['spread_phi']
        metrics[f'onset_phi_annotators_at_learned_{threshold_key}'] = phi_at_learned['onset_phi_annotators']
    
    # Region-level analysis
    if ch_to_region is not None and onset_labels_regions is not None:
        region_prob_matrix, region_names = aggregate_to_regions(sz_prob_smooth, prob_chs, ch_to_region)
        
        if region_prob_matrix is not None:
            ueo_consensus_regions_bool = np.array([r in onset_labels_regions for r in region_names])
            sec_consensus_regions_bool = np.array([r in spread_labels_regions for r in region_names])
            
            ueo_annotators_regions_bool = [
                np.array([r in ann_regions for r in region_names])
                for ann_regions in ueo_annotators_regions
            ]
            sec_annotators_regions_bool = [
                np.array([r in ann_regions for r in region_names])
                for ann_regions in sec_annotators_regions
            ]
            
            onset_mask_regions = np.array([r in onset_labels_regions for r in region_names])
            spread_mask_regions = np.array([r in spread_labels_regions for r in region_names])
            
            # Region AUC and probabilities
            (region_avg_onset_soz_prob, region_avg_onset_nsoz_prob, region_onset_auc, 
             region_onset_auprc_raw, region_onset_auprc_normalized,
             region_avg_spread_soz_prob, region_avg_spread_nsoz_prob, region_spread_auc,
             region_spread_auprc_raw, region_spread_auprc_normalized) = calculate_auc_and_probs(
                region_prob_matrix, onset_mask_regions, spread_mask_regions, onset_idx, spread_idx
            )
            
            # Region inter-rater reliability
            region_onset_inter_rater = calculate_inter_rater_reliability(ueo_annotators_regions_bool, region_names)
            region_spread_inter_rater = calculate_inter_rater_reliability(sec_annotators_regions_bool, region_names)
            
            # Region optimal thresholds
            region_optimal_onset_results = find_optimal_phi_threshold(
                region_prob_matrix, region_names, onset_idx,
                region_names, ueo_consensus_regions_bool, ueo_annotators_regions_bool
            )
            
            region_optimal_spread_results = find_optimal_phi_threshold(
                region_prob_matrix, region_names, spread_idx,
                region_names, sec_consensus_regions_bool, sec_annotators_regions_bool
            )
            
            # Region Phi at learned thresholds
            region_phi_at_learned = {}
            for threshold_key, learned_thresh in learned_thresholds_dict.items():
                if np.isnan(learned_thresh):
                    continue
                region_phi_at_learned[threshold_key] = calculate_phi_at_threshold(
                    region_prob_matrix, region_names, onset_idx, spread_idx,
                    region_names, ueo_consensus_regions_bool, sec_consensus_regions_bool,
                    ueo_annotators_regions_bool, sec_annotators_regions_bool, learned_thresh
                )
            
            metrics.update({
                'region_onset_inter_rater_reliability': region_onset_inter_rater,
                'region_spread_inter_rater_reliability': region_spread_inter_rater,
                'region_avg_onset_soz_prob': region_avg_onset_soz_prob,
                'region_avg_onset_nsoz_prob': region_avg_onset_nsoz_prob,
                'region_onset_auc': region_onset_auc,
                'region_onset_auprc_raw': region_onset_auprc_raw,
                'region_onset_auprc_normalized': region_onset_auprc_normalized,
                'region_avg_spread_soz_prob': region_avg_spread_soz_prob,
                'region_avg_spread_nsoz_prob': region_avg_spread_nsoz_prob,
                'region_spread_auc': region_spread_auc,
                'region_spread_auprc_raw': region_spread_auprc_raw,
                'region_spread_auprc_normalized': region_spread_auprc_normalized,
                'region_optimal_onset_threshold': region_optimal_onset_results['optimal_threshold'],
                'region_max_onset_phi': region_optimal_onset_results['max_phi'],
                'region_onset_phi_annotators_at_optimal': region_optimal_onset_results['phi_annotators'],
                'region_optimal_spread_threshold': region_optimal_spread_results['optimal_threshold'],
                'region_max_spread_phi': region_optimal_spread_results['max_phi'],
                'region_spread_phi_annotators_at_optimal': region_optimal_spread_results['phi_annotators'],
            })
            
            for threshold_key in region_phi_at_learned.keys():
                metrics[f'region_onset_phi_at_learned_{threshold_key}'] = region_phi_at_learned[threshold_key]['onset_phi']
                metrics[f'region_spread_phi_at_learned_{threshold_key}'] = region_phi_at_learned[threshold_key]['spread_phi']
                metrics[f'region_onset_phi_annotators_at_learned_{threshold_key}'] = region_phi_at_learned[threshold_key]['onset_phi_annotators']
        else:
            # Empty region results
            metrics.update(_empty_region_metrics())
    else:
        # No region mapping
        metrics.update(_empty_region_metrics())
    
    return metrics

def _empty_region_metrics():
    """Return dict of NaN region metrics"""
    # Base metrics that are always present
    base_metrics = {
        'region_onset_inter_rater_reliability': np.nan,
        'region_spread_inter_rater_reliability': np.nan,
        'region_avg_onset_soz_prob': np.nan,
        'region_avg_onset_nsoz_prob': np.nan,
        'region_onset_auc': np.nan,
        'region_onset_auprc_raw': np.nan,
        'region_onset_auprc_normalized': np.nan,
        'region_avg_spread_soz_prob': np.nan,
        'region_avg_spread_nsoz_prob': np.nan,
        'region_spread_auc': np.nan,
        'region_spread_auprc_raw': np.nan,
        'region_spread_auprc_normalized': np.nan,
        'region_optimal_onset_threshold': np.nan,
        'region_max_onset_phi': np.nan,
        'region_onset_phi_annotators_at_optimal': [],
        'region_optimal_spread_threshold': np.nan,
        'region_max_spread_phi': np.nan,
        'region_spread_phi_annotators_at_optimal': [],
    }
    # Note: Learned threshold metrics are added dynamically based on what's in learned_thresholds_dict
    # so we don't need to pre-populate them here
    return base_metrics

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
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.plot(unique_probs, phi_vals, 
               color='#ff7f00',#sns.color_palette('spring_r', n_colors=3)[0],
               linewidth=4)
        
        # Mark optimal threshold
        ax.plot([optimal_threshold]*2, 
               [0, max_phi], 
               '--o', c='purple',
               markersize=12,
               linewidth=4,
               markevery=[1],
               fillstyle='none',
               markeredgewidth=3)
        
        ax.set_ylabel('Agreement ($\phi$)')
        ax.set_xlabel('Threshold')
        ax.set_ylim([-0.05, 1])
        ax.set_xlim([np.min(unique_probs), np.max(unique_probs)])
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
    # seizures_df = seizures_df[seizures_df.Patient == 'HUP238']
    # seizures_df = seizures_df[seizures_df.onset.astype(int) == 290006]
    print(f"Found {len(seizures_df)} test seizures")
    
    # Load clinical annotations
    annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v3.pkl"))
    annotations_df = annotations_df[annotations_df.stim == 0]
    
    # Load all thresholds from v6 file only
    thresholds_path = ospj(prodatapath, 'all_thresholds_v6.csv')
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"Could not find all_thresholds_v6.csv in {prodatapath}")
    thresholds_df = pd.read_csv(thresholds_path)
    print(f"Loaded thresholds from all_thresholds_v6.csv")
    
    # Pre-build threshold lookup dictionary for faster access
    # Structure: {(metric, aggregation, model): threshold}
    threshold_lookup = {}
    for _, row in thresholds_df.iterrows():
        key = (row['metric'], row['aggregation'], row['model'])
        threshold_lookup[key] = row['threshold']
    
    # Define threshold metrics to process
    THRESHOLD_METRICS = ['f1', 'iou', 'phi', 'f1_plateau', 'phi_plateau']
    # Process both mean and median aggregations
    AGGREGATIONS = ['mean', 'median']
    # Model configurations
    ndd_models = [
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 1, 'forecast_length': 1, 'metric': 'mse', 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 2, 'forecast_length': 1, 'metric': 'mse', 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 3, 'forecast_length': 2, 'metric': 'mse', 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 4, 'forecast_length': 3, 'metric': 'mse', 'suffix': ''},
        {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 5, 'forecast_length': 4, 'metric': 'mse', 'suffix': ''},
        # {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 9, 'forecast_length': 6, 'metric': 'mse', 'suffix': ''},
        {'model': GIN, 'model_name': 'GIN', 'sequence_length': 12, 'forecast_length': 1, 'metric': 'mse', 'suffix': MODEL_VERSION},
        {'model': GIN, 'model_name': 'GIN', 'sequence_length': 8, 'forecast_length': 1, 'metric': 'mse', 'suffix': MODEL_VERSION},
        # {'model': ONCET, 'model_name': 'ONCET', 'suffix': ''},
        {'model': NDD, 'model_name': 'NDD', 'sequence_length': 12, 'forecast_length': 1, 'metric': 'mse', 'suffix': ''},
    ]
    
    benchmark_models = [
        {'model': ABSSLP, 'model_name': 'ABSSLP', 'suffix': ''},
        {'model': IMPRINT, 'model_name': 'IMPRINT', 'suffix': ''}, 
        {'model': WVNT, 'model_name': 'WVNT', 'suffix': ''},
        {'model': HFER, 'model_name': 'HFER', 'suffix': ''},
        {'model': ONCET, 'model_name': 'ONCET', 'suffix': ''}
    ]

    # Results storage
    results = []
    
    # Process each seizure
    pbar = tqdm(seizures_df.iterrows(), total=len(seizures_df))
    for _, row in pbar:
        patient = row.Patient
        onset_run = str(int(row.onset))
        approx_onset = row.onset
        stim = row.stim
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
        
        # Load electrode localizations for region-level analysis
        ch_to_region = load_electrode_localizations(patient, prodatapath)
        
        # Map clinical annotations to regions if localizations available
        if ch_to_region is not None:
            onset_labels_regions = map_channels_to_regions(onset_labels, ch_to_region)
            spread_labels_regions = map_channels_to_regions(spread_labels, ch_to_region)
            
            # Map annotator labels to regions
            ueo_annotators_regions = []
            for annotator_bool in ueo_annotators:
                annotator_chs = [ch for ch, is_ueo in zip(all_chs, annotator_bool) if is_ueo]
                annotator_regions = map_channels_to_regions(annotator_chs, ch_to_region)
                # Convert back to boolean array for all regions (will be used later)
                ueo_annotators_regions.append(annotator_regions)
            
            sec_annotators_regions = []
            for annotator_bool in sec_annotators:
                annotator_chs = [ch for ch, is_sec in zip(all_chs, annotator_bool) if is_sec]
                annotator_regions = map_channels_to_regions(annotator_chs, ch_to_region)
                sec_annotators_regions.append(annotator_regions)
        else:
            onset_labels_regions = None
            spread_labels_regions = None
            ueo_annotators_regions = None
            sec_annotators_regions = None
        
        # Combine all models into single list
        all_models = []
        
        # Add NDD models
        ndd_prob_files = load_ndd_probability_files(patient, onset_run, ndd_models)
        for model_key, prob_info in ndd_prob_files.items():
            all_models.append({
                'type': 'ndd',
                'model_key': model_key,
                'prob_info': prob_info
            })
        
        # Add benchmark models
        benchmark_prob_files = load_benchmark_probability_files(patient, onset_run, benchmark_models)
        for model_name, prob_info in benchmark_prob_files.items():
            all_models.append({
                'type': 'benchmark',
                'model_key': model_name,
                'prob_info': prob_info
            })
        
        # Process all models
        for model_dict in all_models:
            model_type = model_dict['type']
            model_key = model_dict['model_key']
            prob_info = model_dict['prob_info']
            model = prob_info['model']
            prob_data = prob_info['data'].copy()
            model_name = prob_info['model_name']
            
            if 'time' not in prob_data.columns:
                continue
            
            prob_times = prob_data.pop('time').values
            prob_times[np.isnan(prob_times)] = np.max(prob_times[~np.isnan(prob_times)])+0.5
            prob_chs_raw = prob_data.columns.to_numpy()
            sz_prob = prob_data.to_numpy().T
            
            # Apply smoothing
            sz_prob_smooth = apply_smoothing(sz_prob, window_size=20)
            
            # Calculate temporal alignment
            time_diff = consensus_time - approx_onset
            onset_idx = int(np.argmin(np.abs((prob_times - 180) + time_diff)))
            spread_idx = int(np.argmin(np.abs((prob_times - 190) + time_diff)))
            offset_idx = int(np.argmin(np.abs((prob_times - (np.max(prob_times)-120)) + time_diff)))
            
            # Extract first contacts and create masks (same logic for both types)
            prob_chs = np.array([ch.split('-')[0] for ch in prob_chs_raw])
            onset_mask = np.array([ch in onset_labels for ch in prob_chs])
            spread_mask = np.array([ch in spread_labels for ch in prob_chs])
            
            if model_type == 'ndd':
                full_model_key = f"{model_name}_mse_sl{prob_info['sequence_length']}_fl{prob_info['forecast_length']}"
            else:  # benchmark
                full_model_key = model_name
            
            # Pre-compute all learned thresholds once to avoid repeated threshold optimization
            learned_thresholds_dict = {}
            
            # Collect all thresholds from pre-built lookup dictionary for this model
            for threshold_metric in THRESHOLD_METRICS:
                for aggregation in AGGREGATIONS:
                    key = (threshold_metric, aggregation, full_model_key)
                    if key in threshold_lookup:
                        learned_thresholds_dict[f'{threshold_metric}_{aggregation}'] = threshold_lookup[key]
            
            # Also add tau threshold for NDD models
            if model_type == 'ndd':
                offset_idx = int(np.argmin(np.abs((prob_times - (np.max(prob_times)-120)))))
                tau_threshold = model.get_threshold(
                    pd.DataFrame(sz_prob_smooth.T, columns=prob_chs_raw).iloc[:offset_idx,:],
                    method='automedian'
                )
                learned_thresholds_dict['tau'] = tau_threshold
            
            # Calculate all metrics once with all learned thresholds
            # This avoids expensive find_optimal_phi_threshold calls in the loop
            calculated_metrics_all = calculate_all_metrics(
                sz_prob_smooth, prob_chs, onset_idx, spread_idx,
                onset_mask, spread_mask, all_chs,
                ueo_consensus, sec_consensus, ueo_annotators, sec_annotators,
                onset_labels_regions, spread_labels_regions,
                ueo_annotators_regions, sec_annotators_regions,
                ch_to_region, learned_thresholds_dict
            )
            
            # Create dataframes with probabilities and labels for onset and spread (shared across all thresholds)
            # Average probabilities over the 5-timepoint window
            onset_probs = sz_prob_smooth[:, onset_idx:onset_idx+5].mean(axis=1)
            spread_probs = sz_prob_smooth[:, spread_idx:spread_idx+5].mean(axis=1)
            
            onset_prob_data = pd.DataFrame({
                ch: [prob, int(label)] 
                for ch, prob, label in zip(prob_chs, onset_probs, onset_mask)
            }, index=['probability', 'label'])
            
            spread_prob_data = pd.DataFrame({
                ch: [prob, int(label)] 
                for ch, prob, label in zip(prob_chs, spread_probs, spread_mask)
            }, index=['probability', 'label'])
            
            # Base result dict with shared fields
            base_result_dict = {
                'patient': patient,
                'onset': int(onset_run),
                'stim': stim,
                'model_name': model_name,
                'model_key': model_key,
                'full_model_key': full_model_key,
                'sequence_length': prob_info['sequence_length'],
                'forecast_length': prob_info['forecast_length'],
                'metric': prob_info['metric'],
                'onset_inter_rater_reliability': onset_inter_rater,
                'spread_inter_rater_reliability': spread_inter_rater,
                'onset_prob_data': onset_prob_data,
                'spread_prob_data': spread_prob_data,
            }
            
            # Extract metrics for each threshold metric/aggregation combination
            for threshold_metric in THRESHOLD_METRICS:
                for aggregation in AGGREGATIONS:
                    threshold_key = f'{threshold_metric}_{aggregation}'
                    
                    if threshold_key not in learned_thresholds_dict:
                        continue
                    
                    threshold = learned_thresholds_dict[threshold_key]
                    
                    # Extract metrics specific to this threshold
                    result_dict = base_result_dict.copy()
                    result_dict.update({
                        'threshold_metric': threshold_metric,
                        'aggregation': aggregation,
                        'threshold': threshold,
                    })
                    
                    # Add all shared metrics (optimal thresholds, AUC, etc.)
                    # These are the same for all threshold metrics since they're computed once
                    shared_metrics = {
                        'optimal_onset_threshold': calculated_metrics_all['optimal_onset_threshold'],
                        'max_onset_phi': calculated_metrics_all['max_onset_phi'],
                        'onset_phi_annotators_at_optimal': calculated_metrics_all['onset_phi_annotators_at_optimal'],
                        'optimal_spread_threshold': calculated_metrics_all['optimal_spread_threshold'],
                        'max_spread_phi': calculated_metrics_all['max_spread_phi'],
                        'spread_phi_annotators_at_optimal': calculated_metrics_all['spread_phi_annotators_at_optimal'],
                        'avg_onset_soz_prob': calculated_metrics_all['avg_onset_soz_prob'],
                        'avg_onset_nsoz_prob': calculated_metrics_all['avg_onset_nsoz_prob'],
                        'onset_auc': calculated_metrics_all['onset_auc'],
                        'onset_auprc_raw': calculated_metrics_all['onset_auprc_raw'],
                        'onset_auprc_normalized': calculated_metrics_all['onset_auprc_normalized'],
                        'avg_spread_soz_prob': calculated_metrics_all['avg_spread_soz_prob'],
                        'avg_spread_nsoz_prob': calculated_metrics_all['avg_spread_nsoz_prob'],
                        'spread_auc': calculated_metrics_all['spread_auc'],
                        'spread_auprc_raw': calculated_metrics_all['spread_auprc_raw'],
                        'spread_auprc_normalized': calculated_metrics_all['spread_auprc_normalized'],
                    }
                    
                    # Add region-level metrics if available
                    if 'region_onset_inter_rater_reliability' in calculated_metrics_all:
                        shared_metrics.update({
                            'region_onset_inter_rater_reliability': calculated_metrics_all['region_onset_inter_rater_reliability'],
                            'region_spread_inter_rater_reliability': calculated_metrics_all['region_spread_inter_rater_reliability'],
                            'region_avg_onset_soz_prob': calculated_metrics_all['region_avg_onset_soz_prob'],
                            'region_avg_onset_nsoz_prob': calculated_metrics_all['region_avg_onset_nsoz_prob'],
                            'region_onset_auc': calculated_metrics_all['region_onset_auc'],
                            'region_onset_auprc_raw': calculated_metrics_all['region_onset_auprc_raw'],
                            'region_onset_auprc_normalized': calculated_metrics_all['region_onset_auprc_normalized'],
                            'region_avg_spread_soz_prob': calculated_metrics_all['region_avg_spread_soz_prob'],
                            'region_avg_spread_nsoz_prob': calculated_metrics_all['region_avg_spread_nsoz_prob'],
                            'region_spread_auc': calculated_metrics_all['region_spread_auc'],
                            'region_spread_auprc_raw': calculated_metrics_all['region_spread_auprc_raw'],
                            'region_spread_auprc_normalized': calculated_metrics_all['region_spread_auprc_normalized'],
                            'region_optimal_onset_threshold': calculated_metrics_all['region_optimal_onset_threshold'],
                            'region_max_onset_phi': calculated_metrics_all['region_max_onset_phi'],
                            'region_onset_phi_annotators_at_optimal': calculated_metrics_all['region_onset_phi_annotators_at_optimal'],
                            'region_optimal_spread_threshold': calculated_metrics_all['region_optimal_spread_threshold'],
                            'region_max_spread_phi': calculated_metrics_all['region_max_spread_phi'],
                            'region_spread_phi_annotators_at_optimal': calculated_metrics_all['region_spread_phi_annotators_at_optimal'],
                        })
                    
                    result_dict.update(shared_metrics)
                    
                    # Add metrics specific to this learned threshold (only if they were calculated)
                    if f'learned_{threshold_key}_threshold' in calculated_metrics_all:
                        result_dict[f'learned_{threshold_key}_threshold'] = calculated_metrics_all[f'learned_{threshold_key}_threshold']
                        result_dict[f'onset_phi_at_learned_{threshold_key}'] = calculated_metrics_all[f'onset_phi_at_learned_{threshold_key}']
                        result_dict[f'spread_phi_at_learned_{threshold_key}'] = calculated_metrics_all[f'spread_phi_at_learned_{threshold_key}']
                        result_dict[f'onset_phi_annotators_at_learned_{threshold_key}'] = calculated_metrics_all[f'onset_phi_annotators_at_learned_{threshold_key}']
                    
                    # Add region-level metrics for this learned threshold if available
                    if f'region_onset_phi_at_learned_{threshold_key}' in calculated_metrics_all:
                        result_dict[f'region_onset_phi_at_learned_{threshold_key}'] = calculated_metrics_all[f'region_onset_phi_at_learned_{threshold_key}']
                        result_dict[f'region_spread_phi_at_learned_{threshold_key}'] = calculated_metrics_all[f'region_spread_phi_at_learned_{threshold_key}']
                        result_dict[f'region_onset_phi_annotators_at_learned_{threshold_key}'] = calculated_metrics_all[f'region_onset_phi_annotators_at_learned_{threshold_key}']
                    
                    results.append(result_dict)
            
            # Process tau threshold separately if it exists (for NDD models)
            if 'tau' in learned_thresholds_dict:
                tau_threshold = learned_thresholds_dict['tau']
                
                result_dict = base_result_dict.copy()
                result_dict.update({
                    'threshold_metric': 'tau',
                    'aggregation': 'automedian',
                    'threshold': tau_threshold,
                })
                
                # Add all shared metrics
                shared_metrics = {
                    'optimal_onset_threshold': calculated_metrics_all['optimal_onset_threshold'],
                    'max_onset_phi': calculated_metrics_all['max_onset_phi'],
                    'onset_phi_annotators_at_optimal': calculated_metrics_all['onset_phi_annotators_at_optimal'],
                    'optimal_spread_threshold': calculated_metrics_all['optimal_spread_threshold'],
                    'max_spread_phi': calculated_metrics_all['max_spread_phi'],
                    'spread_phi_annotators_at_optimal': calculated_metrics_all['spread_phi_annotators_at_optimal'],
                    'avg_onset_soz_prob': calculated_metrics_all['avg_onset_soz_prob'],
                    'avg_onset_nsoz_prob': calculated_metrics_all['avg_onset_nsoz_prob'],
                    'onset_auc': calculated_metrics_all['onset_auc'],
                    'onset_auprc_raw': calculated_metrics_all['onset_auprc_raw'],
                    'onset_auprc_normalized': calculated_metrics_all['onset_auprc_normalized'],
                    'avg_spread_soz_prob': calculated_metrics_all['avg_spread_soz_prob'],
                    'avg_spread_nsoz_prob': calculated_metrics_all['avg_spread_nsoz_prob'],
                    'spread_auc': calculated_metrics_all['spread_auc'],
                    'spread_auprc_raw': calculated_metrics_all['spread_auprc_raw'],
                    'spread_auprc_normalized': calculated_metrics_all['spread_auprc_normalized'],
                }
                
                # Add region-level metrics if available
                if 'region_onset_inter_rater_reliability' in calculated_metrics_all:
                    shared_metrics.update({
                        'region_onset_inter_rater_reliability': calculated_metrics_all['region_onset_inter_rater_reliability'],
                        'region_spread_inter_rater_reliability': calculated_metrics_all['region_spread_inter_rater_reliability'],
                        'region_avg_onset_soz_prob': calculated_metrics_all['region_avg_onset_soz_prob'],
                        'region_avg_onset_nsoz_prob': calculated_metrics_all['region_avg_onset_nsoz_prob'],
                        'region_onset_auc': calculated_metrics_all['region_onset_auc'],
                        'region_onset_auprc_raw': calculated_metrics_all['region_onset_auprc_raw'],
                        'region_onset_auprc_normalized': calculated_metrics_all['region_onset_auprc_normalized'],
                        'region_avg_spread_soz_prob': calculated_metrics_all['region_avg_spread_soz_prob'],
                        'region_avg_spread_nsoz_prob': calculated_metrics_all['region_avg_spread_nsoz_prob'],
                        'region_spread_auc': calculated_metrics_all['region_spread_auc'],
                        'region_spread_auprc_raw': calculated_metrics_all['region_spread_auprc_raw'],
                        'region_spread_auprc_normalized': calculated_metrics_all['region_spread_auprc_normalized'],
                        'region_optimal_onset_threshold': calculated_metrics_all['region_optimal_onset_threshold'],
                        'region_max_onset_phi': calculated_metrics_all['region_max_onset_phi'],
                        'region_onset_phi_annotators_at_optimal': calculated_metrics_all['region_onset_phi_annotators_at_optimal'],
                        'region_optimal_spread_threshold': calculated_metrics_all['region_optimal_spread_threshold'],
                        'region_max_spread_phi': calculated_metrics_all['region_max_spread_phi'],
                        'region_spread_phi_annotators_at_optimal': calculated_metrics_all['region_spread_phi_annotators_at_optimal'],
                    })
                
                result_dict.update(shared_metrics)
                
                # Add metrics specific to tau threshold
                result_dict['learned_tau_threshold'] = calculated_metrics_all['learned_tau_threshold']
                result_dict['onset_phi_at_learned_tau'] = calculated_metrics_all['onset_phi_at_learned_tau']
                result_dict['spread_phi_at_learned_tau'] = calculated_metrics_all['spread_phi_at_learned_tau']
                result_dict['onset_phi_annotators_at_learned_tau'] = calculated_metrics_all['onset_phi_annotators_at_learned_tau']
                
                # Add region-level metrics for tau if available
                if 'region_onset_phi_at_learned_tau' in calculated_metrics_all:
                    result_dict['region_onset_phi_at_learned_tau'] = calculated_metrics_all['region_onset_phi_at_learned_tau']
                    result_dict['region_spread_phi_at_learned_tau'] = calculated_metrics_all['region_spread_phi_at_learned_tau']
                    result_dict['region_onset_phi_annotators_at_learned_tau'] = calculated_metrics_all['region_onset_phi_annotators_at_learned_tau']
                
                results.append(result_dict)
            
            # Generate example figure (only once per model)
            if patient == 'HUP238' and int(onset_run) == 290006 and model_name == 'LiNDDA' and model_key == 'LiNDDA_mse_sl3_fl2':
                print(f"\nGenerating example figure for {patient} {onset_run} {model_name}...")
                generate_example_figure(sz_prob_smooth, prob_chs, onset_idx,
                                      all_chs, ueo_consensus, ueo_annotators, figpath)
    
    # Save results
    print("\nSaving results...")
    if results:
        # Save full results as pickle (includes probability dataframes)
        results_df_full = pd.DataFrame(results)
        # pickle_path = ospj(prodatapath, "test_validation_results_nopass_nolayernorm_v3_mean_thresholds.pkl")
        pickle_path = ospj(prodatapath, "test_validation_results_v6.pkl")
        results_df_full.to_pickle(pickle_path)
        print(f"Full results (with probability data) saved to {pickle_path}")
        
        # Create CSV version without complex dataframe fields
        results_csv = []
        for result in results:
            result_csv = result.copy()
            # Remove the dataframe fields that don't work well in CSV
            result_csv.pop('onset_prob_data', None)
            result_csv.pop('spread_prob_data', None)
            results_csv.append(result_csv)
        
        results_df_csv = pd.DataFrame(results_csv)
        # csv_path = ospj(prodatapath, "test_validation_results_v3_mean_thresholds.csv")
        csv_path = ospj(prodatapath, "test_validation_results_v6.csv")
        results_df_csv.to_csv(csv_path, index=False)
        print(f"CSV results (without probability data) saved to {csv_path}")
        
        print(f"Total results: {len(results_df_full)}")
        print(f"Channel-level and region-level metrics included")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()

