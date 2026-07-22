#!/usr/bin/env python3
"""
WAVENET validation analysis script.

Evaluates WAVENET model performance on split 2 seizures from HUP patients that are not stim-induced.
Generates ROC and PRC curves for onset and spread probability distributions, and calculates agreement metrics with consensus annotations.
"""
# File system imports
import sys
import os
import glob
import json
from os.path import join as ospj

# Scientific imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy as sc
import scipy.ndimage

# Sklearn imports
from sklearn.metrics import (
    matthews_corrcoef, 
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve,
    auc
)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical modeling
import statsmodels.formula.api as smf

# Get the project root for DynaSD
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

# WVNT is imported lazily inside load_or_generate_wavenet_probabilities so that the
# validation runs from precomputed probability files without requiring DynaSD to be installed.
from config import Config
from utils import (
    clean_labels, 
    preprocess_for_detection, 
    get_data_from_bids
)

# Get paths from config 
datapath, prodatapath, figpath, metapath, repopath = Config.deal(['datapath','prodatapath','figpath','metapath','repopath'])

# Hardcoded threshold for WAVENET predictions
WAVENET_THRESHOLD = 0.69  # Change this value as needed

# Overwrite flag - set to True to regenerate probability files
OVERWRITE = False

# Spread time offset (in seconds) from onset
SPREAD_TIME_OFFSET = 3.0  # 3 seconds after onset


def load_or_generate_wavenet_probabilities(patient, onset_run, onset_labels, montage='bipolar', overwrite=False):
    """
    Load existing WAVENET probability files or generate them if they don't exist.
    
    Parameters:
    -----------
    patient : str
        Patient ID
    onset_run : str
        Onset run identifier
    onset_labels : list
        List of onset channel labels
    montage : str
        Montage type ('bipolar' or 'car')
    overwrite : bool
        Whether to overwrite existing files
        
    Returns:
    --------
    sz_prob : pd.DataFrame
        Probability matrix with time column
    sz_prob_times : np.array
        Time array
    """
    # Check for existing probability file
    prob_dir = ospj(prodatapath, 'sz_prob', patient)
    pattern = f"{patient}_task-ictal{onset_run}_run-*_mdl-WVNT_sz_prob.pkl"
    prob_paths = glob.glob(ospj(prob_dir, pattern))
    
    if prob_paths and not overwrite:
        # Load existing file
        sz_prob = pd.read_pickle(prob_paths[0])
        sz_prob_times = sz_prob.pop('time').values
        return sz_prob, sz_prob_times
    
    # Generate probability file
    print(f"Generating WAVENET probabilities for {patient} {onset_run}...")

    # Lazy import: only needed when regenerating probabilities (requires DynaSD)
    from DynaSD.WAVENET import WVNT
    
    # Load seizure recording
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
        target=128,  # WAVENET uses 128 Hz
        wavenet=True,
        pre_mask=None,
    )
    
    # Preprocess seizure
    seizure_pre, fs = preprocess_for_detection(
        seizure,
        fs_raw,
        montage,
        target=128,  # WAVENET uses 128 Hz
        wavenet=True,
        pre_mask=channel_mask,
    )
    
    # Remove artifact channels
    art_channel_mask = seizure_pre.loc[180*fs:,:].abs().max() <= (np.median(seizure_pre.loc[180*fs:,:].abs().max())*50)
    seizure_nart = seizure_pre.loc[:,art_channel_mask]
    
    if len(seizure_nart.columns) == 0:
        raise ValueError(f"No channels kept for {patient} {onset_run}")
    
    # Initialize and fit WAVENET model
    model = WVNT(fs=fs, w_size=1, w_stride=0.5, model_path='', verbose=False, batch_size=512)
    model.fit(seizure_nart.iloc[: fs * 120])
    
    # Generate probabilities
    sz_prob = model(seizure_nart)
    sz_prob_times = model.get_win_times(len(seizure_nart))
    
    # Save probability file
    os.makedirs(prob_dir, exist_ok=True)
    sz_prob_df = pd.concat((sz_prob, pd.Series(sz_prob_times, name='time')), axis=1)
    out_path = ospj(prob_dir, f"{patient}_task-ictal{onset_run}_run-{run}_mdl-WVNT_sz_prob.pkl")
    sz_prob_df.to_pickle(out_path)
    
    return sz_prob, sz_prob_times


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


def wideform_preds(element, all_labels):
    """Convert element list to boolean array"""
    element_set = set(element) if not isinstance(element, (set, np.ndarray)) else set(element)
    return np.array([label in element_set for label in all_labels])


def calculate_phi(pred_labels, true_bool, all_labels):
    """Calculate Matthews Correlation Coefficient (Phi)"""
    pred_bool = wideform_preds(pred_labels, all_labels)
    
    if len(pred_bool) == 0 or len(true_bool) == 0:
        return np.nan
    
    return matthews_corrcoef(true_bool, pred_bool)


def calculate_confidence_intervals(curves, x_common):
    """
    Calculate mean and confidence intervals for curves.
    
    Parameters:
    -----------
    curves : list of tuples
        List of (x, y) tuples for each curve
    x_common : np.array
        Common x values to interpolate to
        
    Returns:
    --------
    mean_y : np.array
        Mean y values
    ci_lower : np.array
        Lower confidence interval (2.5th percentile)
    ci_upper : np.array
        Upper confidence interval (97.5th percentile)
    """
    interp_ys = []
    
    for x, y in curves:
        # Interpolate to common x values
        interp_y = np.interp(x_common, x, y)
        interp_ys.append(interp_y)
    
    if len(interp_ys) == 0:
        return np.array([]), np.array([]), np.array([])
    
    interp_ys = np.array(interp_ys)
    mean_y = np.mean(interp_ys, axis=0)
    ci_lower = np.percentile(interp_ys, 2.5, axis=0)
    ci_upper = np.percentile(interp_ys, 97.5, axis=0)
    
    return mean_y, ci_lower, ci_upper


def main():
    """Main validation analysis pipeline for WAVENET model"""
    
    print("Loading metadata and annotations...")
    
    # Load seizure metadata - filter for split 2, HUP patients, not stim-induced
    seizures_df = pd.read_csv(ospj(metapath, "metadata_v7_BIDS.csv"))
    seizures_df = seizures_df[
        (seizures_df.split == 2) & 
        # (seizures_df.Patient.str.contains('HUP', na=False)) & 
        (seizures_df.stim == 0)
    ]
    
    print(f"Found {len(seizures_df)} test seizures")
    
    # Load clinical annotations
    annotations_df = pd.read_pickle(ospj(repopath, "PROCESSED_DATA", "dataset_consensus.pkl"))
    annotations_df = annotations_df[annotations_df.stim == 0]
    
    # Storage for results
    # Onset metrics
    all_onset_roc_data = []  # List of (fpr, tpr) tuples per patient
    all_onset_prc_data = []  # List of (recall, precision) tuples per patient
    all_onset_aucs = []
    all_onset_auprcs = []
    
    # Spread metrics
    all_spread_roc_data = []  # List of (fpr, tpr) tuples per patient
    all_spread_prc_data = []  # List of (recall, precision) tuples per patient
    all_spread_aucs = []
    all_spread_auprcs = []
    
    # Combined metrics (for black mean curves)
    all_combined_roc_data = []  # Combined onset+spread for mean curve
    all_combined_prc_data = []  # Combined onset+spread for mean curve
    
    all_onset_kappas = []
    all_spread_kappas = []
    all_onset_interrater_kappas = []
    all_spread_interrater_kappas = []
    
    # Store patient IDs and seizure info for LME analysis
    all_patients_list = []
    all_seizure_ids = []
    
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
        ueo_annotators = annot_row['ueo']
        
        # Get spread labels if available
        sec_consensus = annot_row.get('sec_consensus', None)
        sec_annotators = annot_row.get('sec', None)
        
        # Get onset labels from consensus
        onset_labels = [ch for ch, is_onset in zip(all_chs, ueo_consensus) if is_onset]
        
        # Calculate inter-rater reliability for onset
        onset_inter_rater = calculate_inter_rater_reliability(ueo_annotators, all_chs)
        
        # Calculate inter-rater reliability for spread if available
        if sec_annotators is not None and len(sec_annotators) > 0:
            spread_inter_rater = calculate_inter_rater_reliability(sec_annotators, all_chs)
        else:
            spread_inter_rater = np.nan
        
        try:
            # Load or generate WAVENET probabilities
            sz_prob, sz_prob_times = load_or_generate_wavenet_probabilities(
                patient, onset_run, onset_labels, overwrite=OVERWRITE
            )
            
            # Apply smoothing
            sz_prob_smooth = pd.DataFrame(
                sc.ndimage.uniform_filter1d(sz_prob, size=20, mode='nearest', axis=0, origin=0),
                columns=sz_prob.columns
            )
            
            # Calculate temporal alignment
            time_diff = consensus_time - approx_onset
            onset_idx = int(np.argmin(np.abs((sz_prob_times - 180) + time_diff)))
            
            # Calculate spread index (offset by SPREAD_TIME_OFFSET seconds)
            spread_idx = int(np.argmin(np.abs((sz_prob_times - 180) + time_diff + SPREAD_TIME_OFFSET)))
            
            # Ensure spread_idx is after onset_idx and within bounds
            if spread_idx <= onset_idx:
                spread_idx = min(onset_idx + 10, len(sz_prob_times) - 1)
            if spread_idx >= len(sz_prob_times):
                spread_idx = len(sz_prob_times) - 1
            
            # Extract first contacts and create masks
            prob_chs = np.array([ch.split('-')[0] for ch in sz_prob_smooth.columns])
            onset_labels_set = set(onset_labels)
            onset_mask = np.array([ch in onset_labels_set for ch in prob_chs])
            
            # Create spread mask if spread labels are available
            if sec_consensus is not None:
                spread_labels = [ch for ch, is_spread in zip(all_chs, sec_consensus) if is_spread]
                spread_labels_set = set(spread_labels)
                spread_mask = np.array([ch in spread_labels_set for ch in prob_chs])
            else:
                # If no spread labels, use onset mask (for evaluation purposes)
                spread_mask = onset_mask.copy()
            
            if len(np.unique(onset_mask)) < 2:
                print(f"Skipping {patient} {onset_run}: insufficient positive/negative channels")
                continue
            
            # Get probabilities at onset window (5 timepoints)
            onset_probs = sz_prob_smooth.iloc[onset_idx:onset_idx+5, :].mean(axis=0).values
            
            # Get probabilities at spread window (5 timepoints)
            spread_window_end = min(spread_idx + 5, len(sz_prob_smooth))
            spread_probs = sz_prob_smooth.iloc[spread_idx:spread_window_end, :].mean(axis=0).values
            
            # Calculate ROC curves
            fpr_onset, tpr_onset, _ = roc_curve(onset_mask, onset_probs)
            all_onset_roc_data.append((fpr_onset, tpr_onset))
            all_combined_roc_data.append((fpr_onset, tpr_onset))
            
            fpr_spread, tpr_spread, _ = roc_curve(onset_mask, spread_probs)
            all_spread_roc_data.append((fpr_spread, tpr_spread))
            all_combined_roc_data.append((fpr_spread, tpr_spread))
            
            # Calculate PRC curves
            precision_onset, recall_onset, _ = precision_recall_curve(onset_mask, onset_probs)
            all_onset_prc_data.append((recall_onset, precision_onset))
            all_combined_prc_data.append((recall_onset, precision_onset))
            
            precision_spread, recall_spread, _ = precision_recall_curve(onset_mask, spread_probs)
            all_spread_prc_data.append((recall_spread, precision_spread))
            all_combined_prc_data.append((recall_spread, precision_spread))
            
            # Calculate AUCs
            try:
                onset_auc = roc_auc_score(onset_mask, onset_probs)
                all_onset_aucs.append(onset_auc)
            except:
                all_onset_aucs.append(np.nan)
            
            try:
                spread_auc = roc_auc_score(onset_mask, spread_probs)
                all_spread_aucs.append(spread_auc)
            except:
                all_spread_aucs.append(np.nan)
            
            # Calculate AUPRCs
            try:
                onset_auprc = auc(recall_onset, precision_onset)
                all_onset_auprcs.append(onset_auprc)
            except:
                all_onset_auprcs.append(np.nan)
            
            try:
                spread_auprc = auc(recall_spread, precision_spread)
                all_spread_auprcs.append(spread_auprc)
            except:
                all_spread_auprcs.append(np.nan)
            
            # Calculate kappa at hardcoded threshold for onset
            threshold = WAVENET_THRESHOLD
            predicted_chs_onset = prob_chs[onset_probs > threshold]
            onset_kappa = calculate_phi(predicted_chs_onset, onset_mask, prob_chs)
            all_onset_kappas.append(onset_kappa)
            
            # Calculate kappa at hardcoded threshold for spread
            predicted_chs_spread = prob_chs[spread_probs > threshold]
            spread_kappa = calculate_phi(predicted_chs_spread, spread_mask, prob_chs)
            all_spread_kappas.append(spread_kappa)
            
            # Store patient and seizure ID for LME analysis (only if we successfully calculated kappas)
            seizure_id = f"{patient}_{onset_run}"
            all_patients_list.append(patient)
            all_seizure_ids.append(seizure_id)
            all_onset_interrater_kappas.append(onset_inter_rater)
            all_spread_interrater_kappas.append(spread_inter_rater)
            
        except Exception as e:
            print(f"Error processing {patient} {onset_run}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Calculate average AUC values for legend
    valid_onset_aucs = [a for a in all_onset_aucs if not np.isnan(a)]
    valid_spread_aucs = [a for a in all_spread_aucs if not np.isnan(a)]
    valid_onset_auprcs = [a for a in all_onset_auprcs if not np.isnan(a)]
    valid_spread_auprcs = [a for a in all_spread_auprcs if not np.isnan(a)]
    
    mean_onset_auc = np.mean(valid_onset_aucs) if len(valid_onset_aucs) > 0 else np.nan
    mean_spread_auc = np.mean(valid_spread_aucs) if len(valid_spread_aucs) > 0 else np.nan
    mean_onset_auprc = np.mean(valid_onset_auprcs) if len(valid_onset_auprcs) > 0 else np.nan
    mean_spread_auprc = np.mean(valid_spread_auprcs) if len(valid_spread_auprcs) > 0 else np.nan
    
    # Separate figure for black mean ROC curve (onset only)
    fig, ax = plt.subplots(figsize=(5,5))
    
    # Calculate and plot average ROC curve for onset (black)
    if len(all_onset_roc_data) > 0:
        mean_fpr = np.linspace(0, 1, 100)
        mean_tprs = []
        
        for fpr, tpr in all_onset_roc_data:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            mean_tprs.append(interp_tpr)
        
        mean_tpr = np.mean(mean_tprs, axis=0)
        mean_tpr[-1] = 1.0
        
        # Plot average ROC curve as solid black line
        label_text = f'Mean ROC (AUROC = {mean_onset_auc:.3f})'
        ax.plot(mean_fpr, mean_tpr, color='black', linewidth=4, label=label_text)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_ylim(-0.025, 1.025)
    ax.set_xlim(-0.025, 1.025)
    ax.legend(loc='lower right')
    sns.despine()
    
    roc_onset_path = ospj(figpath, 'WAVENET_ROC_curve_onset.pdf')
    plt.savefig(roc_onset_path, bbox_inches='tight')
    print(f"ROC curve (onset) saved to {roc_onset_path}")
    plt.close()
    
    # Separate figure for black mean PRC curve (onset only)
    fig, ax = plt.subplots(figsize=(5,5))
    
    # Calculate and plot average PRC curve for onset (black)
    if len(all_onset_prc_data) > 0:
        mean_recall = np.linspace(0, 1, 100)
        mean_precisions = []
        
        for recall, precision in all_onset_prc_data:
            # Reverse to ensure increasing recall
            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            mean_precisions.append(interp_precision)
        
        mean_precision = np.mean(mean_precisions, axis=0)
        
        # Plot average PRC curve as solid black line
        label_text = f'Mean PRC (AUPRC = {mean_onset_auprc:.3f})'
        ax.plot(mean_recall, mean_precision, color='black', linewidth=4, label=label_text)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim(-0.025, 1.025)
    ax.set_xlim(-0.025, 1.025)
    ax.legend(loc='lower left')
    sns.despine()
    
    prc_onset_path = ospj(figpath, 'WAVENET_PRC_curve_onset.pdf')
    plt.savefig(prc_onset_path, bbox_inches='tight')
    print(f"PRC curve (onset) saved to {prc_onset_path}")
    plt.close()
    
    # Combined ROC Curve Plot (onset and spread, no confidence intervals)
    fig, ax = plt.subplots(figsize=(5,5))
    
    # Plot onset mean ROC curve (red)
    if len(all_onset_roc_data) > 0:
        mean_fpr = np.linspace(0, 1, 100)
        mean_tprs = []
        
        for fpr, tpr in all_onset_roc_data:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            mean_tprs.append(interp_tpr)
        
        mean_tpr_onset = np.mean(mean_tprs, axis=0)
        mean_tpr_onset[-1] = 1.0
        
        label_text = f'Onset (AUROC = {mean_onset_auc:.3f})'
        ax.plot(mean_fpr, mean_tpr_onset, color='red', linewidth=4, label=label_text)
    
    # Plot spread mean ROC curve (purple)
    if len(all_spread_roc_data) > 0:
        mean_fpr = np.linspace(0, 1, 100)
        mean_tprs = []
        
        for fpr, tpr in all_spread_roc_data:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            mean_tprs.append(interp_tpr)
        
        mean_tpr_spread = np.mean(mean_tprs, axis=0)
        mean_tpr_spread[-1] = 1.0
        
        label_text = f'Spread (AUROC = {mean_spread_auc:.3f})'
        ax.plot(mean_fpr, mean_tpr_spread, color='purple', linewidth=4, label=label_text)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_ylim(-0.025, 1.025)
    ax.set_xlim(-0.025, 1.025)
    ax.legend(loc='lower right')
    sns.despine()
    
    roc_path = ospj(figpath, 'WAVENET_ROC_curve.pdf')
    plt.savefig(roc_path, bbox_inches='tight')
    print(f"ROC curve saved to {roc_path}")
    plt.close()
    
    # Combined PRC Curve Plot (onset and spread, no confidence intervals)
    fig, ax = plt.subplots(figsize=(5,5))
    
    # Plot onset mean PRC curve (red)
    if len(all_onset_prc_data) > 0:
        mean_recall = np.linspace(0, 1, 100)
        mean_precisions = []
        
        for recall, precision in all_onset_prc_data:
            # Reverse to ensure increasing recall
            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            mean_precisions.append(interp_precision)
        
        mean_precision_onset = np.mean(mean_precisions, axis=0)
        
        label_text = f'Onset (AUPRC = {mean_onset_auprc:.3f})'
        ax.plot(mean_recall, mean_precision_onset, color='red', linewidth=4, label=label_text)
    
    # Plot spread mean PRC curve (purple)
    if len(all_spread_prc_data) > 0:
        mean_recall = np.linspace(0, 1, 100)
        mean_precisions = []
        
        for recall, precision in all_spread_prc_data:
            # Reverse to ensure increasing recall
            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            mean_precisions.append(interp_precision)
        
        mean_precision_spread = np.mean(mean_precisions, axis=0)
        
        label_text = f'Spread (AUPRC = {mean_spread_auprc:.3f})'
        ax.plot(mean_recall, mean_precision_spread, color='purple', linewidth=4, label=label_text)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim(-0.025, 1.025)
    ax.set_xlim(-0.025, 1.025)
    ax.legend(loc='lower left')
    sns.despine()
    
    prc_path = ospj(figpath, 'WAVENET_PRC_curve.pdf')
    plt.savefig(prc_path, bbox_inches='tight')
    print(f"PRC curve saved to {prc_path}")
    plt.close()
    
    # Stripplot and boxplot for kappa
    print("\nGenerating kappa comparison plot...")
    
    # Prepare data for plotting
    plot_data = []
    
    # Add WAVENET onset kappa values
    for kappa in all_onset_kappas:
        if not np.isnan(kappa):
            plot_data.append({'Type': 'WAVENET', 'Phase': 'Onset', 'kappa': kappa})
    
    # Add WAVENET spread kappa values
    for kappa in all_spread_kappas:
        if not np.isnan(kappa):
            plot_data.append({'Type': 'WAVENET', 'Phase': 'Spread', 'kappa': kappa})
    
    # Add Interrater onset kappa values
    for kappa in all_onset_interrater_kappas:
        if not np.isnan(kappa):
            plot_data.append({'Type': 'Interrater', 'Phase': 'Onset', 'kappa': kappa})
    
    # Add Interrater spread kappa values
    for kappa in all_spread_interrater_kappas:
        if not np.isnan(kappa):
            plot_data.append({'Type': 'Interrater', 'Phase': 'Spread', 'kappa': kappa})
    
    if len(plot_data) > 0:
        plot_df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(2,4.8))
        
        # Create stripplot first (individual dots) - behind boxplots
        sns.stripplot(
            data=plot_df,
            x='Phase',
            y='kappa',
            hue='Type',
            dodge=True,
            palette=['blue', 'gray'],
            alpha=0.4,
            ax=ax,
            order=['Onset', 'Spread'],
            hue_order=['WAVENET', 'Interrater'],
            legend=False
        )
        
        # Create boxplot second (on top) - fill=False, same colors
        # Use gap parameter to add spacing between groups (requires seaborn 0.13.0+)
        sns.boxplot(
            data=plot_df,
            x='Phase',
            y='kappa',
            hue='Type',
            palette=['blue', 'gray'],
            dodge=True,
            fill=False,
            gap=0.2,
            ax=ax,
            order=['Onset', 'Spread'],
            hue_order=['WAVENET', 'Interrater'],
            legend=True
        )
        
        # Set legend from stripplot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title='Type', loc='best')
        
        ax.set_ylabel('Agreement (κ)')
        ax.set_xlabel('')
        sns.despine()
        
        kappa_path = ospj(figpath, 'WAVENET_kappa_comparison.pdf')
        plt.savefig(kappa_path, bbox_inches='tight')
        print(f"Kappa comparison plot saved to {kappa_path}")
        plt.close()
    
    # Perform Linear Mixed Effects (LME) analysis
    print("\nPerforming Linear Mixed Effects analysis...")
    
    # Prepare data for LME analysis (paired comparison)
    # Calculate seizure-level differences: Interrater - WAVENET
    lme_data = []
    
    # Add onset data with differences
    for i, (patient, seizure_id, wavenet_kappa, interrater_kappa) in enumerate(
        zip(all_patients_list, all_seizure_ids, all_onset_kappas, all_onset_interrater_kappas)
    ):
        if not np.isnan(wavenet_kappa) and not np.isnan(interrater_kappa):
            difference = interrater_kappa - wavenet_kappa
            lme_data.append({
                'patient': patient,
                'seizure_id': seizure_id,
                'phase': 'Onset',
                'difference': difference  # Interrater - WAVENET
            })
    
    # Add spread data with differences
    for i, (patient, seizure_id, wavenet_kappa, interrater_kappa) in enumerate(
        zip(all_patients_list, all_seizure_ids, all_spread_kappas, all_spread_interrater_kappas)
    ):
        if not np.isnan(wavenet_kappa) and not np.isnan(interrater_kappa):
            difference = interrater_kappa - wavenet_kappa
            lme_data.append({
                'patient': patient,
                'seizure_id': seizure_id,
                'phase': 'Spread',
                'difference': difference  # Interrater - WAVENET
            })
    
    lme_df = pd.DataFrame(lme_data)
    
    # LME results storage
    lme_results = {}
    
    # Run LME model for Onset
    if len(lme_df[lme_df['phase'] == 'Onset']) > 0:
        onset_lme_df = lme_df[lme_df['phase'] == 'Onset'].copy()
        
        try:
            # Fit mixed effects model: difference ~ 1 + (1|patient)
            # Intercept represents mean difference (Interrater - WAVENET)
            # Testing if intercept is significantly different from zero
            model_onset = smf.mixedlm(
                "difference ~ 1",
                onset_lme_df,
                groups=onset_lme_df["patient"]
            )
            result_onset = model_onset.fit(reml=False)
            
            # Extract statistics
            intercept_coef = result_onset.params['Intercept']  # Mean difference: Interrater - WAVENET
            intercept_se = result_onset.bse['Intercept']
            intercept_tstat = result_onset.tvalues['Intercept']
            intercept_pval = result_onset.pvalues['Intercept']
            
            # Confidence intervals
            ci_onset = result_onset.conf_int()
            intercept_ci_lower = ci_onset.loc['Intercept', 0]
            intercept_ci_upper = ci_onset.loc['Intercept', 1]
            
            lme_results['LME_onset'] = {
                'mean_difference': float(intercept_coef),  # Interrater - WAVENET
                'SE': float(intercept_se),
                'tstat': float(intercept_tstat),
                'pvalue': float(intercept_pval),
                'CI_lower': float(intercept_ci_lower),
                'CI_upper': float(intercept_ci_upper),
                'n_seizures': int(len(onset_lme_df)),
                'n_patients': int(onset_lme_df['patient'].nunique()),
                'AIC': float(result_onset.aic),
                'BIC': float(result_onset.bic),
                'log_likelihood': float(result_onset.llf)
            }
            
            print(f"\nLME Results for Onset (Paired Comparison):")
            print(f"  Mean difference (Interrater - WAVENET): {intercept_coef:.4f} ± {intercept_se*1.96:.4f} (p={intercept_pval:.4f})")
            print(f"  95% CI: [{intercept_ci_lower:.4f}, {intercept_ci_upper:.4f}]")
            print(f"  n_seizures: {len(onset_lme_df)}, n_patients: {onset_lme_df['patient'].nunique()}")
            
        except Exception as e:
            print(f"Error fitting LME model for Onset: {e}")
            import traceback
            traceback.print_exc()
            lme_results['LME_onset'] = {'error': str(e)}
    
    # Run LME model for Spread
    if len(lme_df[lme_df['phase'] == 'Spread']) > 0:
        spread_lme_df = lme_df[lme_df['phase'] == 'Spread'].copy()
        
        try:
            # Fit mixed effects model: difference ~ 1 + (1|patient)
            # Intercept represents mean difference (Interrater - WAVENET)
            model_spread = smf.mixedlm(
                "difference ~ 1",
                spread_lme_df,
                groups=spread_lme_df["patient"]
            )
            result_spread = model_spread.fit(reml=False)
            
            # Extract statistics
            intercept_coef = result_spread.params['Intercept']  # Mean difference: Interrater - WAVENET
            intercept_se = result_spread.bse['Intercept']
            intercept_tstat = result_spread.tvalues['Intercept']
            intercept_pval = result_spread.pvalues['Intercept']
            
            # Confidence intervals
            ci_spread = result_spread.conf_int()
            intercept_ci_lower = ci_spread.loc['Intercept', 0]
            intercept_ci_upper = ci_spread.loc['Intercept', 1]
            
            lme_results['LME_spread'] = {
                'mean_difference': float(intercept_coef),  # Interrater - WAVENET
                'SE': float(intercept_se),
                'tstat': float(intercept_tstat),
                'pvalue': float(intercept_pval),
                'CI_lower': float(intercept_ci_lower),
                'CI_upper': float(intercept_ci_upper),
                'n_seizures': int(len(spread_lme_df)),
                'n_patients': int(spread_lme_df['patient'].nunique()),
                'AIC': float(result_spread.aic),
                'BIC': float(result_spread.bic),
                'log_likelihood': float(result_spread.llf)
            }
            
            print(f"\nLME Results for Spread (Paired Comparison):")
            print(f"  Mean difference (Interrater - WAVENET): {intercept_coef:.4f} ± {intercept_se*1.96:.4f} (p={intercept_pval:.4f})")
            print(f"  95% CI: [{intercept_ci_lower:.4f}, {intercept_ci_upper:.4f}]")
            print(f"  n_seizures: {len(spread_lme_df)}, n_patients: {spread_lme_df['patient'].nunique()}")
            
        except Exception as e:
            print(f"Error fitting LME model for Spread: {e}")
            import traceback
            traceback.print_exc()
            lme_results['LME_spread'] = {'error': str(e)}
    
    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    
    # Filter out NaN values for statistics
    valid_onset_aucs = [a for a in all_onset_aucs if not np.isnan(a)]
    valid_spread_aucs = [a for a in all_spread_aucs if not np.isnan(a)]
    valid_onset_auprcs = [a for a in all_onset_auprcs if not np.isnan(a)]
    valid_spread_auprcs = [a for a in all_spread_auprcs if not np.isnan(a)]
    valid_onset_kappas = [k for k in all_onset_kappas if not np.isnan(k)]
    valid_spread_kappas = [k for k in all_spread_kappas if not np.isnan(k)]
    valid_onset_interrater_kappas = [k for k in all_onset_interrater_kappas if not np.isnan(k)]
    valid_spread_interrater_kappas = [k for k in all_spread_interrater_kappas if not np.isnan(k)]
    
    results = {
        'Onset_AUC': {
            'mean': np.mean(valid_onset_aucs) if len(valid_onset_aucs) > 0 else np.nan,
            'SE': np.std(valid_onset_aucs, ddof=1) / np.sqrt(len(valid_onset_aucs)) if len(valid_onset_aucs) > 1 else np.nan,
            'n': len(valid_onset_aucs)
        },
        'Spread_AUC': {
            'mean': np.mean(valid_spread_aucs) if len(valid_spread_aucs) > 0 else np.nan,
            'SE': np.std(valid_spread_aucs, ddof=1) / np.sqrt(len(valid_spread_aucs)) if len(valid_spread_aucs) > 1 else np.nan,
            'n': len(valid_spread_aucs)
        },
        'Onset_AUPRC': {
            'mean': np.mean(valid_onset_auprcs) if len(valid_onset_auprcs) > 0 else np.nan,
            'SE': np.std(valid_onset_auprcs, ddof=1) / np.sqrt(len(valid_onset_auprcs)) if len(valid_onset_auprcs) > 1 else np.nan,
            'n': len(valid_onset_auprcs)
        },
        'Spread_AUPRC': {
            'mean': np.mean(valid_spread_auprcs) if len(valid_spread_auprcs) > 0 else np.nan,
            'SE': np.std(valid_spread_auprcs, ddof=1) / np.sqrt(len(valid_spread_auprcs)) if len(valid_spread_auprcs) > 1 else np.nan,
            'n': len(valid_spread_auprcs)
        },
        'kappa_WAVENET_onset_consensus': {
            'mean': np.mean(valid_onset_kappas) if len(valid_onset_kappas) > 0 else np.nan,
            'SE': np.std(valid_onset_kappas, ddof=1) / np.sqrt(len(valid_onset_kappas)) if len(valid_onset_kappas) > 1 else np.nan,
            'n': len(valid_onset_kappas),
            'threshold': WAVENET_THRESHOLD
        },
        'kappa_WAVENET_spread_consensus': {
            'mean': np.mean(valid_spread_kappas) if len(valid_spread_kappas) > 0 else np.nan,
            'SE': np.std(valid_spread_kappas, ddof=1) / np.sqrt(len(valid_spread_kappas)) if len(valid_spread_kappas) > 1 else np.nan,
            'n': len(valid_spread_kappas),
            'threshold': WAVENET_THRESHOLD
        },
        'kappa_interrater_onset': {
            'mean': np.mean(valid_onset_interrater_kappas) if len(valid_onset_interrater_kappas) > 0 else np.nan,
            'SE': np.std(valid_onset_interrater_kappas, ddof=1) / np.sqrt(len(valid_onset_interrater_kappas)) if len(valid_onset_interrater_kappas) > 1 else np.nan,
            'n': len(valid_onset_interrater_kappas)
        },
        'kappa_interrater_spread': {
            'mean': np.mean(valid_spread_interrater_kappas) if len(valid_spread_interrater_kappas) > 0 else np.nan,
            'SE': np.std(valid_spread_interrater_kappas, ddof=1) / np.sqrt(len(valid_spread_interrater_kappas)) if len(valid_spread_interrater_kappas) > 1 else np.nan,
            'n': len(valid_spread_interrater_kappas)
        }
    }
    
    # Add LME results to results dictionary
    results.update(lme_results)
    
    # Save results to JSON
    results_path = ospj(figpath, 'WAVENET_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("\nSummary Statistics:")
    print(f"Onset AUC: {results['Onset_AUC']['mean']:.4f} ± {results['Onset_AUC']['SE']*1.96:.4f} (n={results['Onset_AUC']['n']})")
    print(f"Spread AUC: {results['Spread_AUC']['mean']:.4f} ± {results['Spread_AUC']['SE']*1.96:.4f} (n={results['Spread_AUC']['n']})")
    print(f"Onset AUPRC: {results['Onset_AUPRC']['mean']:.4f} ± {results['Onset_AUPRC']['SE']*1.96:.4f} (n={results['Onset_AUPRC']['n']})")
    print(f"Spread AUPRC: {results['Spread_AUPRC']['mean']:.4f} ± {results['Spread_AUPRC']['SE']*1.96:.4f} (n={results['Spread_AUPRC']['n']})")
    print(f"Kappa (WAVENET Onset vs Consensus): {results['kappa_WAVENET_onset_consensus']['mean']:.4f} ± {results['kappa_WAVENET_onset_consensus']['SE']*1.96:.4f} (n={results['kappa_WAVENET_onset_consensus']['n']}, threshold={WAVENET_THRESHOLD})")
    print(f"Kappa (WAVENET Spread vs Consensus): {results['kappa_WAVENET_spread_consensus']['mean']:.4f} ± {results['kappa_WAVENET_spread_consensus']['SE']*1.96:.4f} (n={results['kappa_WAVENET_spread_consensus']['n']}, threshold={WAVENET_THRESHOLD})")
    print(f"Kappa (Interrater Onset): {results['kappa_interrater_onset']['mean']:.4f} ± {results['kappa_interrater_onset']['SE']*1.96:.4f} (n={results['kappa_interrater_onset']['n']})")
    print(f"Kappa (Interrater Spread): {results['kappa_interrater_spread']['mean']:.4f} ± {results['kappa_interrater_spread']['SE']*1.96:.4f} (n={results['kappa_interrater_spread']['n']})")


if __name__ == "__main__":
    main()

