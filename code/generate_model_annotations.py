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

def main():
    _,_,datapath,prodatapath,figpath,patient_table,_,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config_unit.json'))

    seizures_df = pd.read_csv(ospj(datapath,"stim_seizure_information_BIDS.csv"))
    annotations_df = pd.read_pickle(ospj(prodatapath,"stim_seizure_information_consensus.pkl"))

    montage = 'bipolar'
    mdl_str = 'AbsSlp'
    clf_fs = 256
    # Iterating through each patient that we have annotations for
    predicted_channels = {'Patient': [],
                        'iEEG_ID': [],
                        'model':[],
                        'approximate_onset': [],
                        'ueo_time_consensus': [],
                        'threshold':[],
                        'ueo_chs_strict':[],
                        'ueo_chs_loose': [],
                        'sec_chs_strict': [],
                        'sec_chs_loose':[],
                        'to_annotate': []
                        }
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:
        pt = row.ptID
        pbar.set_description(desc=f"Patient: {pt}",refresh=True)
        if len(row.interictal_training) == 0:
            continue
        seizure_times = seizures_df[seizures_df.Patient == pt]
        qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),leave=False)
        
        for _,sz_row in qbar:
            _,_, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
            prob_path = f"probability_matrix_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.pkl"
            # clf_path = f"raw_preds_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.npy"
            sz_prob = pd.read_pickle(ospj(prodatapath,pt,prob_path))
            # sz_clf = np.load(ospj(prodatapath,pt,clf_path))
            time_wins = sz_prob.time.to_numpy()
            sz_prob.drop('time',axis=1,inplace=True)
            prob_chs = sz_prob.columns.to_numpy()
            sz_prob = sz_prob.to_numpy().T
            
            # Match seizure using approximate onset time in annotations, patient name, and task
            task_time = int(task[np.where([s.isnumeric() for s in task])[0][0]:])
            approx_time = sz_row.approximate_onset
            if task_time in annotations_df.approximate_onset.astype(int):
                annot_row = annotations_df[annotations_df.approximate_onset.astype(int) == task_time]
                consensus_time = annot_row.ueo_time_consensus.item()
            else:
                consensus_time = approx_time
            # all_chs = annot_row.all_chs.item()
            time_diff = consensus_time - approx_time
            
            # Find closest index to consensus onset time relative to actual onset time (consensus - approximate and find closest to 120 + diff)
            onset_index = np.argmin(np.abs((time_wins-60) + time_diff))
            # sweep threshold
            for final_thresh in np.linspace(0,1,int(1/0.05)+1):
                predicted_channels['Patient'].append(sz_row.Patient)
                predicted_channels['iEEG_ID'].append(sz_row.IEEGname)
                predicted_channels['model'].append(mdl_str)
                predicted_channels['approximate_onset'].append(sz_row.approximate_onset)
                predicted_channels['ueo_time_consensus'].append(consensus_time)
                predicted_channels['to_annotate'].append(sz_row.to_annotate)
                predicted_channels['threshold'].append(final_thresh)
                # append to dataframe with patient, ieeg file, approximate onset time, bids path, threshold, strict channels and liberal channels
                # Do the same for 10 second spread
                # Plot the matrices in each patient's figure folder

                # Here I will need to figure out when clinicians said the onset time was relative to the original time and then find that index in the time windows
                # Find out which channels were never seizing
                # first_detect = np.argmax(sz_prob>final_thresh,axis=1)
                # first_detect[first_detect == 0] = sz_prob.shape[1]
                # # first_onset_detect = np.argmax(sz_prob[:,(onset_index-10):]>final_thresh,axis=1)
                # ch_sorting = np.argsort(first_detect)

                # rejecting noisy/late channels
                # bottom_mask = np.sum(sz_clf[ch_sorting,:],axis=1) > 0
                # first_zero = np.where(~bottom_mask)[0][0].astype(int)
                # sz_clf[ch_sorting[first_zero:],:] = 0
                # sz_prob[ch_sorting[first_zero:],:] = 0

                # Here we want to loop through a list of thresholds and generate the following columns of data
                # bp channels in UEO, bp channels in UEO + 10
                sz_clf_final = sz_prob > final_thresh

                # Here this could be first seizing index, or it could be the time of the clinically defined UEO from the annotations
                # first_seizing_index = np.argmax(sz_clf_final.any(axis=0))
    
                mdl_ueo_idx = np.where(np.sum(sz_clf_final[:, onset_index:onset_index + 3], axis=1) > 0)[0]
                mdl_ueo_ch_bp = prob_chs[mdl_ueo_idx]
                mdl_ueo_ch_strict = np.array([s.split("-")[0] for s in mdl_ueo_ch_bp]).flatten()
                mdl_ueo_ch_loose = np.unique(np.array([s.split("-") for s in mdl_ueo_ch_bp]).flatten())
                predicted_channels['ueo_chs_strict'].append(mdl_ueo_ch_strict)
                predicted_channels['ueo_chs_loose'].append(mdl_ueo_ch_loose)
                mdl_sec_idx = np.where(np.sum(sz_clf_final[:, onset_index+10:onset_index + 13], axis=1) > 0)[0]
                mdl_sec_ch_bp = prob_chs[mdl_sec_idx]
                mdl_sec_ch_strict = np.array([s.split("-")[0] for s in mdl_sec_ch_bp]).flatten()
                mdl_sec_ch_loose = np.unique(np.array([s.split("-") for s in mdl_sec_ch_bp]).flatten())
                predicted_channels['sec_chs_strict'].append(mdl_sec_ch_strict)
                predicted_channels['sec_chs_loose'].append(mdl_sec_ch_loose)

            # In this section plot and save all of the plots that we generate in this section.
            os.makedirs(ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str),exist_ok=True)
            # plot_and_save_detection(sz_clf_final,
            #                         time_wins,
            #                         prob_chs,
            #                         ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str,f"{montage}_sz_clf_final.png"))
    predicted_channels = pd.DataFrame(predicted_channels)
    predicted_channels.to_pickle(ospj(prodatapath,"predicted_channels.pkl"))
    predicted_channels.to_csv(ospj(prodatapath,"predicted_channels.csv"))
if __name__ == "__main__":
    main()