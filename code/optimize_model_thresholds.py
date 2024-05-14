# iEEG imports
from ieeg.auth import Session

# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
import json
from scipy.linalg import hankel
from tqdm import tqdm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import cohen_kappa_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Imports for analysis
from seizure_detection_pipeline import prepare_segment, TRAIN_WIN, PRED_WIN

# OS imports
import os
from os.path import join as ospj
from os.path import exists as ospe
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')

plt.rcParams['image.cmap'] = 'magma'

def wideform_preds(element,all_labels):
    return [label in element for label in all_labels]

def shortform_preds(element,all_labels):
    a = np.array(all_labels)[element]
    return a
def apply_dice_score(row):
    all_chs = row.all_chs
    for col in ['ueo_consensus','ueo_any','sec_consensus','sec_any']:
        row[col+'_chs'] = shortform_preds(row[col],all_chs)
        for pred in ['strict','loose']:
            ch_preds = row[f'{col[:3]}_chs_{pred}']
            if (len(ch_preds) + len(row[col+'_chs'])) == 0:
                row[f'{col}_{pred}_dice'] = 0
            else:
                row[f'{col}_{pred}_dice'] = dice_score(row[col+'_chs'],ch_preds)
    return row

def apply_kappa_score(row):
    all_chs = row.all_chs
    for col in ['ueo_chs_strict','ueo_chs_loose','sec_chs_strict','ueo_chs_loose']:
        # need to turn model predictions into the wideform
        row[col+'_bool'] = wideform_preds(row[col],all_chs)
        for annot in ['consensus','any']:
            ch_preds = row[f'{col[:3]}_{annot}']
            # if (sum(ch_preds) + sum(row[col+'_bool'])) == 0:
            #     row[f'{col}_{annot}_kappa'] = 0
            # else:
            row[f'{col}_{annot}_kappa'] = cohen_kappa_score(row[col+'_bool'],ch_preds)
    return row

def apply_region_kappa(row):
    all_rs = row.all_rs
    for col in ['ueo_rs_strict','ueo_rs_loose','sec_rs_strict','ueo_rs_loose']:
        bool_col = wideform_preds(row[col],all_rs)
        for annot in ['consensus','any']:
            r_annots = wideform_preds(row[f'{col[:3]}_{annot}_rs'],all_rs)
            row[f'{col}_{annot}_kappa'] = cohen_kappa_score(bool_col,r_annots)
    return row

def main():
    _,_,datapath,prodatapath,figpath,patient_table,rid_hup,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config_unit.json'))





    # Loading in human annotations with consensus annotation already created
    annotations_df = pd.read_pickle(ospj(prodatapath,"stim_seizure_information_consensus.pkl"))
    annotations_df.columns = ['Patient' if c == 'patient' else c for c in annotations_df.columns]
    # Loading in predicted channels for all models from generate_model_annotations.py
    predicted_channels = pd.read_pickle(ospj(prodatapath,"predicted_channels.pkl"))
    predicted_channels = predicted_channels[predicted_channels.to_annotate == 1]
    # Sorting
    predicted_channels.sort_values('approximate_onset',inplace=True)
    annotations_df.sort_values('approximate_onset',inplace=True)
    # Creating a merged table with human and machine annotations based on approximate seizure onset time
    # pred_channels_wannots = pd.merge_asof(predicted_channels,
    #                                     annotations_df[['approximate_onset','Patient','all_chs','ueo_consensus','ueo_any','sec_consensus','sec_any']],
    #                                     on='approximate_onset',by='Patient',
    #                                     tolerance = 120,
    #                                     direction='nearest')
    # pred_channels_wannots.dropna(axis=0,subset='ueo_consensus',inplace=True)
    # pred_channels_wannots.sort_values(['Patient','iEEG_ID','approximate_onset'],inplace=True)
    # pred_channels_wdice = pred_channels_wannots.apply(apply_dice_score,axis=1)
    # pred_channels_wkappa = pred_channels_wannots.apply(apply_kappa_score,axis=1)

    # pred_channels_wdice.to_pickle(ospj(prodatapath,f"predicted_channels_wdice.pkl"))
    # pred_channels_wkappa.to_pickle(ospj(prodatapath,f"predicted_channels_wkappa.pkl"))
    annotations_df = annotations_df.copy()
    pt_groups = annotations_df.groupby('Patient')
    for pt,group in pt_groups:
        region_path = ospj(prodatapath,pt,'electrode_localizations_dkt.pkl')
        if not ospe(region_path):
            continue
        electrode_regions = pd.read_pickle(region_path)
        for idx,row in group.iterrows():
            for col in ['ueo_consensus','ueo_any','sec_consensus','sec_any']:
                chs = np.array(row.all_chs)[row[col]]
                electrode_locals = electrode_regions[electrode_regions.name.isin(chs)]['label'].unique()
                annotations_df.at[idx,col+'_rs'] = electrode_locals
            annotations_df.at[idx,'all_rs'] = electrode_regions[electrode_regions.name.isin(row['all_chs'])]['label'].unique()
    
    pt_groups = predicted_channels.groupby('Patient')
    for pt,group in pt_groups:
        region_path = ospj(prodatapath,pt,'electrode_localizations_dkt.pkl')
        if not ospe(region_path):
            continue
        electrode_regions = pd.read_pickle(region_path)
        for idx,row in group.iterrows():
            for col in ['ueo_chs_strict','ueo_chs_loose','sec_chs_strict','sec_chs_loose']:
                electrode_locals = electrode_regions[electrode_regions.name.isin(row[col])]['label'].unique()
                predicted_channels.at[idx,col.split('_')[0]+'_rs_'+col.split('_')[2]] = electrode_locals
    pred_regions_wannots = pd.merge_asof(predicted_channels,
                                        annotations_df[['approximate_onset','Patient','all_rs','ueo_consensus_rs','ueo_any_rs','sec_consensus_rs','sec_any_rs']],
                                        on='approximate_onset',by='Patient',
                                        tolerance = 120,
                                        direction='nearest')
    pred_regions_wannots.dropna(axis=0,subset='ueo_consensus_rs',inplace=True)
    pred_regions_wannots.sort_values(['Patient','iEEG_ID','approximate_onset','threshold','model'],inplace=True)
    pred_regions_wkappa = pred_regions_wannots.apply(apply_region_kappa,axis=1)

    # Now have to perform the same localization for the model predictions
    # Then perform the same join as above, might be able to combine this all into one?
    # Write new kappas function for region labels
        
if __name__ == "__main__":
    main()