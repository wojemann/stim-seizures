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

def main():
    _,_,datapath,prodatapath,figpath,patient_table,_,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'))

    annotations_df = pd.read_pickle(ospj(prodatapath,"stim_seizure_information_consensus.pkl"))
    predicted_channels = pd.read_pickle(ospj(prodatapath,"predicted_channels.pkl"))
    annotations_df.columns = ['Patient' if c == 'patient' else c for c in annotations_df.columns]
    # Iterating through each seizure that we have annotations for
    predicted_channels = predicted_channels[predicted_channels.to_annotate == 1]
    predicted_channels.sort_values('approximate_onset',inplace=True)
    annotations_df.sort_values('approximate_onset',inplace=True)
    pred_channels_wannots = pd.merge_asof(predicted_channels,
                                        annotations_df[['approximate_onset','Patient','all_chs','ueo_consensus','ueo_any','sec_consensus','sec_any']],
                                        on='approximate_onset',by='Patient',
                                        tolerance = 120,
                                        direction='nearest')
    pred_channels_wannots.dropna(axis=0,subset='ueo_consensus',inplace=True)
    pred_channels_wannots.sort_values(['Patient','iEEG_ID','approximate_onset'],inplace=True)
    pred_channels_wdice = pred_channels_wannots.apply(apply_dice_score,axis=1)
    pred_channels_wdice.to_pickle(ospj(prodatapath,f"predicted_channels_wdice.pkl"))

if __name__ == "__main__":
    main()