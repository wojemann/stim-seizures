#!/usr/bin/env python3
"""
ONCET validation analysis script.

Evaluates ONCET model performance on split 2 seizures from HUP patients that are not stim-induced.
Generates ROC and PRC curves, and calculates agreement metrics with consensus annotations.
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

# Get the project root for DynaSD
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

# Import models
from DynaSD.ONCET import ONCET
from config import Config
from utils import (
    clean_labels, 
    preprocess_for_detection, 
    get_data_from_bids
)

# Get paths from config 
datapath, prodatapath, figpath, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])

# Hardcoded threshold for ONCET predictions
ONCET_THRESHOLD = 0.69  # Change this value as needed

# Overwrite flag - set to True to regenerate probability files
OVERWRITE = False


def load_or_generate_oncet_probabilities(patient, onset_run, onset_labels, montage='bipolar', overwrite=False):
    """
    Load existing ONCET probability files or generate them if they don't exist.
    
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
    pattern = f"{patient}_task-ictal{onset_run}_run-*_mdl-ONCET_sz_prob.pkl"
    prob_paths = glob.glob(ospj(prob_dir, pattern))
    
    if prob_paths and not overwrite:
        # Load existing file
        sz_prob = pd.read_pickle(prob_paths[0])
        sz_prob_times = sz_prob.pop('time').values
        return sz_prob, sz_prob_times
    
    # Generate probability file
    print(f"Generating ONCET probabilities for {patient} {onset_run}...")
    
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
        target=fs_raw,
        wavenet=False,
        pre_mask=None,
    )
    
    # Preprocess seizure
    seizure_pre, fs = preprocess_for_detection(
        seizure,
        fs_raw,
        montage,
        target=fs_raw,
        wavenet=False,
        pre_mask=channel_mask,
    )
    
    # Remove artifact channels
    art_channel_mask = seizure_pre.loc[180*fs:,:].abs().max() <= (np.median(seizure_pre.loc[180*fs:,:].abs().max())*50)
    seizure_nart = seizure_pre.loc[:,art_channel_mask]
    
    if len(seizure_nart.columns) == 0:
        raise ValueError(f"No channels kept for {patient} {onset_run}")
    
    # Initialize and fit ONCET model
    model = ONCET(fs=fs, w_size=1, w_stride=0.5)
    model.fit(seizure_nart.iloc[: fs * 120])
    
    # Generate probabilities
    sz_prob = model(seizure_nart)
    sz_prob_times = model.get_win_times(len(seizure_nart))
    
    # Save probability file
    os.makedirs(prob_dir, exist_ok=True)
    sz_prob_df = pd.concat((sz_prob, pd.Series(sz_prob_times, name='time')), axis=1)
    out_path = ospj(prob_dir, f"{patient}_task-ictal{onset_run}_run-{run}_mdl-ONCET_sz_prob.pkl")
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


def main():
    """Main validation analysis pipeline for ONCET model"""
    
    print("Loading metadata and annotations...")
    
    # Load seizure metadata - filter for split 2, HUP patients, not stim-induced
    seizures_df = pd.read_csv(ospj(metapath, "metadata_v7_BIDS.csv"))
    seizures_df = seizures_df[
        (seizures_df.split == 2) & 
        (seizures_df.Patient.str.contains('HUP', na=False)) & 
        (seizures_df.stim == 0)
    ]
    
    print(f"Found {len(seizures_df)} test seizures")
    
    # Load clinical annotations
    annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v3.pkl"))
    annotations_df = annotations_df[annotations_df.stim == 0]
    
    # Storage for results
    all_roc_data = []  # List of (fpr, tpr) tuples per patient
    all_prc_data = []  # List of (recall, precision) tuples per patient
    all_aucs = []
    all_auprcs = []
    all_mccs = []
    all_interrater_mccs = []
    
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
        
        # Get onset labels from consensus
        onset_labels = [ch for ch, is_onset in zip(all_chs, ueo_consensus) if is_onset]
        
        # Calculate inter-rater reliability
        onset_inter_rater = calculate_inter_rater_reliability(ueo_annotators, all_chs)
        all_interrater_mccs.append(onset_inter_rater)
        
        try:
            # Load or generate ONCET probabilities
            sz_prob, sz_prob_times = load_or_generate_oncet_probabilities(
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
            
            # Extract first contacts and create masks
            prob_chs = np.array([ch.split('-')[0] for ch in sz_prob_smooth.columns])
            onset_labels_set = set(onset_labels)
            onset_mask = np.array([ch in onset_labels_set for ch in prob_chs])
            
            if len(np.unique(onset_mask)) < 2:
                print(f"Skipping {patient} {onset_run}: insufficient positive/negative channels")
                continue
            
            # Get probabilities at onset window (5 timepoints)
            onset_probs = sz_prob_smooth.iloc[onset_idx:onset_idx+5, :].mean(axis=0).values
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(onset_mask, onset_probs)
            all_roc_data.append((fpr, tpr))
            
            # Calculate PRC curve
            precision, recall, _ = precision_recall_curve(onset_mask, onset_probs)
            all_prc_data.append((recall, precision))
            
            # Calculate AUC
            try:
                auc_score = roc_auc_score(onset_mask, onset_probs)
                all_aucs.append(auc_score)
            except:
                all_aucs.append(np.nan)
            
            # Calculate AUPRC
            try:
                auprc_score = auc(recall, precision)
                all_auprcs.append(auprc_score)
            except:
                all_auprcs.append(np.nan)
            
            # Calculate MCC at hardcoded threshold
            threshold = ONCET_THRESHOLD
            predicted_chs = prob_chs[onset_probs > threshold]
            mcc = calculate_phi(predicted_chs, onset_mask, prob_chs)
            all_mccs.append(mcc)
            
        except Exception as e:
            print(f"Error processing {patient} {onset_run}: {e}")
            continue
    
    # Generate plots
    print("\nGenerating plots...")
    
    # ROC Curve
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot individual patient curves in light gray
    for fpr, tpr in all_roc_data:
        ax.plot(fpr, tpr, color='lightgray', alpha=0.5, linewidth=0.5)
    
    # Calculate and plot average ROC curve
    if len(all_roc_data) > 0:
        # Interpolate all curves to common FPR values
        mean_fpr = np.linspace(0, 1, 100)
        mean_tprs = []
        
        for fpr, tpr in all_roc_data:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            mean_tprs.append(interp_tpr)
        
        mean_tpr = np.mean(mean_tprs, axis=0)
        mean_tpr[-1] = 1.0
        
        # Plot average ROC curve as solid black line
        ax.plot(mean_fpr, mean_tpr, color='black', linewidth=2, label='Mean ROC')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - ONCET', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    sns.despine()
    
    roc_path = ospj(figpath, 'ONCET_ROC_curve.pdf')
    plt.savefig(roc_path, bbox_inches='tight')
    print(f"ROC curve saved to {roc_path}")
    plt.close()
    
    # PRC Curve
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot individual patient curves in light gray
    for recall, precision in all_prc_data:
        ax.plot(recall, precision, color='lightgray', alpha=0.5, linewidth=0.5)
    
    # Calculate and plot average PRC curve
    if len(all_prc_data) > 0:
        # Interpolate all curves to common recall values
        mean_recall = np.linspace(0, 1, 100)
        mean_precisions = []
        
        for recall, precision in all_prc_data:
            # Reverse to ensure increasing recall
            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            mean_precisions.append(interp_precision)
        
        mean_precision = np.mean(mean_precisions, axis=0)
        
        # Plot average PRC curve as solid black line
        ax.plot(mean_recall, mean_precision, color='black', linewidth=2, label='Mean PRC')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - ONCET', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    sns.despine()
    
    prc_path = ospj(figpath, 'ONCET_PRC_curve.pdf')
    plt.savefig(prc_path, bbox_inches='tight')
    print(f"PRC curve saved to {prc_path}")
    plt.close()
    
    # Stripplot and pointplot for MCC
    print("\nGenerating MCC comparison plot...")
    
    # Prepare data for plotting
    plot_data = []
    
    # Add interrater MCC values
    for mcc in all_interrater_mccs:
        if not np.isnan(mcc):
            plot_data.append({'Type': 'Interrater', 'MCC': mcc})
    
    # Add ONCET MCC values
    for mcc in all_mccs:
        if not np.isnan(mcc):
            plot_data.append({'Type': 'ONCET', 'MCC': mcc})
    
    if len(plot_data) > 0:
        plot_df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(4, 5))
        
        # Create stripplot for all data points
        sns.stripplot(
            data=plot_df,
            x='Type',
            y='MCC',
            color='gray',
            alpha=0.5,
            ax=ax,
            order=['ONCET', 'Interrater']
        )
        
        # Create black cross pointplot for ONCET only
        oncet_data = plot_df[plot_df['Type'] == 'ONCET']
        if len(oncet_data) > 0:
            sns.pointplot(
                data=oncet_data,
                x='Type',
                y='MCC',
                color='black',
                marker='x',
                markersize=15,
                linestyles='',
                errorbar='se',
                ax=ax,
                order=['ONCET']
            )
        
        # Create gray pointplot for Interrater only
        interrater_data = plot_df[plot_df['Type'] == 'Interrater']
        if len(interrater_data) > 0:
            sns.pointplot(
                data=interrater_data,
                x='Type',
                y='MCC',
                color='gray',
                marker='o',
                markersize=10,
                linestyles='',
                errorbar='se',
                ax=ax,
                order=['Interrater']
            )
        
        ax.set_ylabel('MCC', fontsize=12)
        ax.set_xlabel('', fontsize=12)
        sns.despine()
        
        mcc_path = ospj(figpath, 'ONCET_MCC_comparison.pdf')
        plt.savefig(mcc_path, bbox_inches='tight')
        print(f"MCC comparison plot saved to {mcc_path}")
        plt.close()
    
    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    
    # Filter out NaN values for statistics
    valid_aucs = [a for a in all_aucs if not np.isnan(a)]
    valid_auprcs = [a for a in all_auprcs if not np.isnan(a)]
    valid_mccs = [m for m in all_mccs if not np.isnan(m)]
    valid_interrater_mccs = [m for m in all_interrater_mccs if not np.isnan(m)]
    
    results = {
        'AUC': {
            'mean': np.mean(valid_aucs) if len(valid_aucs) > 0 else np.nan,
            'SE': np.std(valid_aucs, ddof=1) / np.sqrt(len(valid_aucs)) if len(valid_aucs) > 1 else np.nan,
            'n': len(valid_aucs)
        },
        'AUPRC': {
            'mean': np.mean(valid_auprcs) if len(valid_auprcs) > 0 else np.nan,
            'SE': np.std(valid_auprcs, ddof=1) / np.sqrt(len(valid_auprcs)) if len(valid_auprcs) > 1 else np.nan,
            'n': len(valid_auprcs)
        },
        'MCC_ONCET_consensus': {
            'mean': np.mean(valid_mccs) if len(valid_mccs) > 0 else np.nan,
            'SE': np.std(valid_mccs, ddof=1) / np.sqrt(len(valid_mccs)) if len(valid_mccs) > 1 else np.nan,
            'n': len(valid_mccs),
            'threshold': ONCET_THRESHOLD
        },
        'MCC_interrater': {
            'mean': np.mean(valid_interrater_mccs) if len(valid_interrater_mccs) > 0 else np.nan,
            'SE': np.std(valid_interrater_mccs, ddof=1) / np.sqrt(len(valid_interrater_mccs)) if len(valid_interrater_mccs) > 1 else np.nan,
            'n': len(valid_interrater_mccs)
        }
    }
    
    # Save results to JSON
    results_path = ospj(figpath, 'ONCET_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("\nSummary Statistics:")
    print(f"AUC: {results['AUC']['mean']:.4f} ± {results['AUC']['SE']:.4f} (n={results['AUC']['n']})")
    print(f"AUPRC: {results['AUPRC']['mean']:.4f} ± {results['AUPRC']['SE']:.4f} (n={results['AUPRC']['n']})")
    print(f"MCC (ONCET vs Consensus): {results['MCC_ONCET_consensus']['mean']:.4f} ± {results['MCC_ONCET_consensus']['SE']:.4f} (n={results['MCC_ONCET_consensus']['n']}, threshold={ONCET_THRESHOLD})")
    print(f"MCC (Interrater): {results['MCC_interrater']['mean']:.4f} ± {results['MCC_interrater']['SE']:.4f} (n={results['MCC_interrater']['n']})")


if __name__ == "__main__":
    main()

