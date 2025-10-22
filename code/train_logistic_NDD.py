# File system imports
import sys
import os
import glob
from os.path import join as ospj
from os.path import exists as ospe

# Scientific imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Utility imports
from utils import clean_labels

# Sklearn imports
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# Get the project root (parent directory of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

from DynaSD import LiNDDA, GIN, MINDD

from config import Config

# Get paths from config 
datapath, prodatapath, figpath, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])

def load_probability_files(patient, onset_run, model_dicts):
    """Load probability files for all model-sequence-forecast combinations for a given patient/seizure"""
    prob_files = {}
    
    for model_dict in model_dicts:
        model_class = model_dict['model']
        model_name = model_dict['model_name']
        sequence_length = model_dict['sequence_length']
        forecast_length = model_dict['forecast_length']
        # model_name = getattr(model_class, '__name__', 'UNKNOWN')
        
        for metric in ['mse']:
            # Create the key for this combination
            key = f"{model_name}_{metric}_sl{sequence_length}_fl{forecast_length}"
                
            # Find probability files
            prob_dir = ospj(prodatapath, 'sz_prob', patient)
            # if metric == 'prob':
            #     pattern = f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_sz_prob_forecast-{forecast}.pkl"
            # else:
            #     pattern = f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}_{metric}_prob_forecast-{forecast}.pkl"
            pattern = f"{patient}_task-ictal{onset_run}_mdl-{model_name}_seq-{sequence_length}"
            if metric == 'prob':
                pattern += f"_sz_prob_forecast-{forecast_length}.pkl"
            else:
                pattern += f"_{metric}_prob_forecast-{forecast_length}.pkl"
            prob_path = glob.glob(ospj(prob_dir, pattern))
            if prob_path:
                prob_files[key] = {
                    'data': pd.read_pickle(prob_path[0]),
                    'model_class': model_class,
                    'model_name': model_name,
                    'sequence_length': sequence_length,
                    'forecast_length': forecast_length,
                    'metric': metric,
                }
            else:
                # print(f"Warning: No probability file found for {patient} {onset_run} {key} {model_name} {sequence_length} {forecast_length} {metric}")
                pass
    return prob_files

# def main():
"""
Main seizure spread analysis pipeline.

Loads pre-computed probability matrices and applies model-specific thresholds
to analyze seizure onset and spread patterns.
"""

# Load seizure metadata
seizures_df = pd.read_csv(ospj(metapath, "metadata_v7_BIDS.csv"))

all_models = [
    {'model': LiNDDA, 'model_name': 'LiNDDA', 'sequence_length': 1, 'forecast_length': 1},
    {'model': GIN, 'model_name': 'GIN', 'sequence_length': 12, 'forecast_length': 1},
]

if ospe(ospj(prodatapath,'ndd_logistic_train_dataset.csv')):
    dataset = pd.read_csv(ospj(prodatapath,'ndd_logistic_train_dataset.csv'))
else:
    # Process each seizure
    pbar = tqdm(seizures_df[seizures_df.split == 1].iterrows(), total=len(seizures_df))
    dataset = []
    for _, row in pbar:
        patient = row.Patient
        onset_run = str(int(row.onset))
        onset_labels = clean_labels([l.strip() for l in row.SOZ.split(',')], patient)
        
        pbar.set_description(f"Patient: {patient} | Seizure: {onset_run}")
        
        # Load probability files for this seizure
        prob_files = load_probability_files(patient, onset_run, all_models)
        onset_time_sec = 180
        # Process each model-metric-sequence-forecast combination that has probability data
        for key, prob_info in prob_files.items():
            prob_data = prob_info['data']
            model_name = prob_info['model_name']
            metric = prob_info['metric']
            # Extract time array
            if 'time' in prob_data.columns:
                prob_times = prob_data.pop('time').values
            else:
                print(f"Warning: No time column found for {patient} {onset_run} {key}")
            onset_idx = int(np.argmin(np.abs(prob_times - onset_time_sec)))
            onset_odx = int(np.argmin(np.abs(prob_times - (onset_time_sec + 3))))
            onset_prob = pd.DataFrame(prob_data.iloc[onset_idx:onset_odx,:].mean(axis=0),columns=['probability'])
            onset_prob['label'] = [1 if l.split('-')[0] in onset_labels else 0 for l in onset_prob.index]
            onset_prob['patient'] = patient
            onset_prob['onset'] = onset_run
            onset_prob['model'] = model_name
            onset_prob['metric'] = metric
            dataset.append(onset_prob)
    dataset = pd.concat(dataset)
    dataset.to_csv(ospj(prodatapath,'ndd_logistic_train_dataset.csv'),index=False)

patients = dataset.patient.unique()
# n_splits = len(patients)-1
n_splits = 10
folds = KFold(n_splits=n_splits, shuffle=False)
cval_results = []
ensemble_models = {}  # Store models for ensemble: {model_name: [clf1, clf2, ...]}
dataset['probability'] = dataset['probability'].apply(lambda x: np.log(x))

for fold, (train_index, test_index) in tqdm(enumerate(folds.split(patients)), total=n_splits):
    train_patients = patients[train_index]
    test_patients = patients[test_index]
    train_dataset = dataset[dataset.patient.isin(train_patients)]
    test_dataset = dataset[dataset.patient.isin(test_patients)]
    
    for model_name in dataset.model.unique():
        train_data = train_dataset[train_dataset.model == model_name]
        test_data = test_dataset[test_dataset.model == model_name]
        clf = LogisticRegression(penalty=None, class_weight='balanced')
        clf.fit(train_data[['probability']], train_data['label'])
        
        # Store model for ensemble
        if model_name not in ensemble_models:
            ensemble_models[model_name] = []
        ensemble_models[model_name].append(clf)
        
        test_data['predicted_label'] = clf.predict(test_data[['probability']])
        test_data['predicted_probability'] = clf.predict_proba(test_data[['probability']])[:,1]
        cval_results.append({
            'model': model_name,
            'fold': fold,
            'train_accuracy': clf.score(train_data[['probability']], train_data['label']),
            'test_accuracy': clf.score(test_data[['probability']], test_data['label']),
            'train_f1': f1_score(train_data['label'], clf.predict(train_data[['probability']])),
            'test_f1': f1_score(test_data['label'], clf.predict(test_data[['probability']])),
            'train_phi': matthews_corrcoef(train_data['label'], clf.predict(train_data[['probability']])),
            'test_phi': matthews_corrcoef(test_data['label'], clf.predict(test_data[['probability']])),
            'train_sensitivity': recall_score(train_data['label'], clf.predict(train_data[['probability']])),
            'test_sensitivity': recall_score(test_data['label'], clf.predict(test_data[['probability']])),
            'train_precision': precision_score(train_data['label'], clf.predict(train_data[['probability']])),
            'test_precision': precision_score(test_data['label'], clf.predict(test_data[['probability']])),
            'train_recall': recall_score(train_data['label'], clf.predict(train_data[['probability']])),
            'test_recall': recall_score(test_data['label'], clf.predict(test_data[['probability']])),
            'train_auc': roc_auc_score(train_data['label'], clf.predict_proba(train_data[['probability']])[:,1]),
            'test_auc': roc_auc_score(test_data['label'], clf.predict_proba(test_data[['probability']])[:,1]),
        })

cval_results = pd.DataFrame(cval_results)
cval_results.to_csv(ospj(prodatapath,'ndd_logistic_train_results.csv'),index=False)

print(f"Ensemble models trained: {[(k, len(v)) for k, v in ensemble_models.items()]}")

# Load clinical annotations for test set
print("Loading clinical annotations for test set...")
annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v2.pkl"))

# Process test set
test_seizures = seizures_df[(seizures_df.split == 2) & (seizures_df.stim == 0)]
print(f"Processing {len(test_seizures)} test seizures...")

pbar = tqdm(test_seizures.iterrows(), total=len(test_seizures))
test_results = []

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
    
    # Get onset labels from consensus
    onset_labels = [ch for ch, is_onset in zip(all_chs, ueo_consensus) if is_onset]
    
    # Load probability files for this seizure
    prob_files = load_probability_files(patient, onset_run, all_models)
    
    # Process each model-metric-sequence-forecast combination that has probability data
    for key, prob_info in prob_files.items():
        prob_data = prob_info['data'].copy()
        model_name = prob_info['model_name']
        metric = prob_info['metric']
        
        # Skip if no ensemble models available for this model
        if model_name not in ensemble_models:
            print(f"Warning: No ensemble models found for {model_name}")
            continue
        
        # Extract time array
        if 'time' not in prob_data.columns:
            print(f"Warning: No time column found for {patient} {onset_run} {key}")
            continue
        
        prob_times = prob_data.pop('time').values
        prob_chs_raw = prob_data.columns.to_numpy()
        
        # Calculate temporal alignment using consensus time
        time_diff = consensus_time - approx_onset
        onset_idx = int(np.argmin(np.abs((prob_times - 180) + time_diff)))
        
        # Average over 5-timepoint window at onset
        onset_window = prob_data.iloc[onset_idx:onset_idx+2, :].mean(axis=0)
        onset_window = np.log(onset_window)
        
        # Extract first contacts and create labels
        prob_chs = np.array([ch.split('-')[0] for ch in prob_chs_raw])
        onset_mask = np.array([ch in onset_labels for ch in prob_chs])
        
        # Create dataframe for prediction
        onset_prob_df = pd.DataFrame({
            'probability': onset_window.values,
            'label': onset_mask.astype(int)
        })
        
        # Ensemble prediction: average predicted probabilities from all k-fold models
        ensemble_proba = []
        for clf in ensemble_models[model_name]:
            proba = clf.predict_proba(onset_prob_df[['probability']])[:, 1]
            ensemble_proba.append(proba)
        
        # Average probabilities across all models
        onset_pred_proba = np.mean(ensemble_proba, axis=0)
        
        # Get predictions by thresholding at 0.5
        onset_pred = (onset_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        try:
            test_phi = matthews_corrcoef(onset_prob_df['label'], onset_pred)
        except:
            test_phi = np.nan
        
        try:
            test_f1 = f1_score(onset_prob_df['label'], onset_pred)
        except:
            test_f1 = np.nan
        
        try:
            test_precision = precision_score(onset_prob_df['label'], onset_pred)
        except:
            test_precision = np.nan
        
        try:
            test_recall = recall_score(onset_prob_df['label'], onset_pred)
        except:
            test_recall = np.nan
        
        try:
            test_auc = roc_auc_score(onset_prob_df['label'], onset_pred_proba)
        except:
            test_auc = np.nan
        
        try:
            # Calculate accuracy manually for ensemble
            test_accuracy = np.mean(onset_prob_df['label'] == onset_pred)
        except:
            test_accuracy = np.nan
        
        try:
            test_sensitivity = recall_score(onset_prob_df['label'], onset_pred)
        except:
            test_sensitivity = np.nan
        
        test_results.append({
            'patient': patient,
            'onset': onset_run,
            'model': model_name,
            'metric': metric,
            'n_ensemble_models': len(ensemble_models[model_name]),
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_phi': test_phi,
            'test_sensitivity': test_sensitivity,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc,
        })

# Save test results
test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv(ospj(prodatapath, 'ndd_logistic_test_results.csv'), index=False)
print(f"Test results saved to {ospj(prodatapath, 'ndd_logistic_test_results.csv')}")
print(f"Total test results: {len(test_results_df)}")