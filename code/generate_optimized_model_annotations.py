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

# OS imports
import os
from os.path import join as ospj
from os.path import exists as ospe
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')

plt.rcParams['image.cmap'] = 'magma'

def main():
    _,_,datapath,prodatapath,metapath,_,patient_table,_,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'),None)

    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))
    montage = 'bipolar'
    mdl_str = 'NRG'
    clf_fs = 128
    final_thresh = 0.415
    first_sz_idx_offset = 113

    # Iterating through each patient that we have annotations for
    predicted_channels = {'Patient': [],
                        'iEEG_ID': [],
                        'model':[],
                        'stim':[],
                        'approximate_onset': [],
                        'offset': [],
                        'threshold':[],
                        'ueo_chs_strict':[],
                        'ueo_chs_loose': [],
                        'sec_chs_strict': [],
                        'sec_chs_loose':[],
                        'sz_chs': [],
                        'sz_times': [],
                        'all_channels': []
                        }
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:
        pt = row.ptID
        pbar.set_description(desc=f"Patient -- {pt}",refresh=True)
        if len(row.interictal_training) == 0:
            continue
        seizure_times = seizures_df[seizures_df.Patient == pt]
        qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),desc = 'Seizures',leave=False)
        
        for _,sz_row in qbar:
            _,_, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
            prob_path = f"probability_matrix_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.pkl"
            sz_prob = pd.read_pickle(ospj(prodatapath,pt,prob_path))
            
            time_wins = sz_prob.time.to_numpy()
            sz_prob.drop('time',axis=1,inplace=True)
            prob_chs = sz_prob.columns.to_numpy()
            sz_prob = sz_prob.to_numpy().T
            
            # Generate predicitons
            predicted_channels['Patient'].append(sz_row.Patient)
            predicted_channels['iEEG_ID'].append(sz_row.IEEGname)
            predicted_channels['model'].append(mdl_str)
            predicted_channels['stim'].append(sz_row.stim)
            predicted_channels['approximate_onset'].append(sz_row.approximate_onset)
            predicted_channels['offset'].append(sz_row.end)
            predicted_channels['threshold'].append(final_thresh)
            predicted_channels['all_channels'].append(np.array([s.split("-")[0] for s in prob_chs]).flatten())
            sz_clf_final = sz_prob > final_thresh

            # To save
            # First seizing index at the final threshold after offset
            # UEO channels at first seizing index + 3 indices
            # Seizing index and time since first seizing index for each channel
            sliced_data = sz_clf_final[:,first_sz_idx_offset:]
            first_sz_idxs = np.argmax(sliced_data,axis=1)
            seized_idxs = np.any(sliced_data,axis=1)
            first_sz_idxs += first_sz_idx_offset
            if sum(seized_idxs) > 0:
                sz_times_arr = time_wins[first_sz_idxs[seized_idxs]]
                sz_times_arr -= np.min(sz_times_arr)
                sz_ch_arr = prob_chs[seized_idxs]
                sz_ch_arr = np.array([s.split("-")[0] for s in sz_ch_arr]).flatten()
            else:
                sz_ch_arr = []
                sz_times_arr = []

            late = np.sum(sz_prob[:,-118:] > final_thresh,axis=1) > 30
            sz_prob_reject = sz_prob[~late,:]
            prob_chs_reject = prob_chs[~late]
            sz_clf_final = sz_prob_reject > final_thresh
            # Get channels
            sliced_data = sz_clf_final[:,first_sz_idx_offset:]
            df = pd.DataFrame(sliced_data).T
            seizing = df.rolling(window=5,closed='right').apply(lambda x: (x == 1).all())
            first_sz_idxs = seizing.idxmax().to_numpy() - 4
            seized_idxs = np.any(sliced_data,axis=1)
            first_sz_idxs += first_sz_idx_offset
            if sum(seized_idxs) > 0:
                sz_times_arr = time_wins[first_sz_idxs[seized_idxs]]
                sz_times_arr -= np.min(sz_times_arr)
                sz_ch_arr = prob_chs_reject[seized_idxs]
                sz_ch_arr = np.array([s.split("-")[0] for s in sz_ch_arr]).flatten()
            else:
                sz_ch_arr = []
                sz_times_arr = []
            predicted_channels['sz_chs'].append(sz_ch_arr)
            predicted_channels['sz_times'].append(sz_times_arr)

            onset_index = np.min(first_sz_idxs)
            spread_index = onset_index + 20

            mdl_ueo_idx = np.all(sz_prob_reject[:,onset_index:onset_index+5] > final_thresh,axis=1)
            mdl_ueo_ch_bp = prob_chs_reject[mdl_ueo_idx]
            mdl_ueo_ch_strict = np.array([s.split("-")[0] for s in mdl_ueo_ch_bp]).flatten()
            mdl_ueo_ch_loose = np.unique(np.array([s.split("-") for s in mdl_ueo_ch_bp]).flatten())
            predicted_channels['ueo_chs_strict'].append(mdl_ueo_ch_strict)
            predicted_channels['ueo_chs_loose'].append(mdl_ueo_ch_loose)

            mdl_sec_idx = np.all(sz_prob_reject[:,spread_index:spread_index+5] > final_thresh,axis=1)
            mdl_sec_ch_bp = prob_chs_reject[mdl_sec_idx]
            mdl_sec_ch_strict = np.array([s.split("-")[0] for s in mdl_sec_ch_bp]).flatten()
            mdl_sec_ch_loose = np.unique(np.array([s.split("-") for s in mdl_sec_ch_bp]).flatten())
            predicted_channels['sec_chs_strict'].append(mdl_sec_ch_strict)
            predicted_channels['sec_chs_loose'].append(mdl_sec_ch_loose)

    predicted_channels = pd.DataFrame(predicted_channels)
    predicted_channels.to_pickle(ospj(prodatapath,f"optimized_predicted_channels_{mdl_str}_{final_thresh}.pkl"))
    predicted_channels.to_csv(ospj(prodatapath,"optimized_predicted_channels.csv"))
if __name__ == "__main__":
    main()