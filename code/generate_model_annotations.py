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

def plot_and_save_detection(mat,win_times,yticks,fig_save_path,xlim = None):
    plt.subplots(figsize=(48,24))
    plt.imshow(mat)
    plt.axvline(120,linestyle = '--',color = 'white')
    plt.xlabel('Time (s)')
    plt.yticks(np.arange(len(yticks)),yticks,rotation=0,fontsize=10)
    plt.xticks(np.arange(0,len(win_times),10),win_times.round(1)[np.arange(0,len(win_times),10)]-60)
    if xlim is not None:
        plt.xlim(xlim)
    plt.savefig(fig_save_path)

def main():
    _,_,datapath,prodatapath,_,patient_table,_,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config_unit.json'))

    seizures_df = pd.read_csv(ospj(datapath,"stim_seizure_information_BIDS.csv"))

    montage = 'bipolar'
    mdl_str = 'LSTM'
    clf_fs = 256
    # Iterating through each patient that we have annotations for
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:
        pt = row.ptID
        pbar.set_description(desc=f"Patient: {pt}",refresh=True)
        if len(row.interictal_training) == 0:
            continue
        seizure_times = seizures_df[seizures_df.Patient == pt]
        qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),leave=False)
        for i,(_,sz_row) in enumerate(qbar):
            seizure,_, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
            prob_path = f"probability_matrix_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.pkl"
            clf_path = f"raw_preds_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.npy"
            sz_prob = pd.read_pickle(ospj(prodatapath,pt,prob_path))
            sz_clf = np.load(ospj(prodatapath,pt,clf_path))
            # Plot the matrices in each patient's figure folder


            # Here I will need to figure out when clinicians said the onset time was relative to the original time and then find that index in the time windows
            first_detect = np.argmax(sz_prob[:,115:]>.5,axis=1)
            first_detect[first_detect == 0] = sz_prob.shape[1]
            ch_sorting = np.argsort(first_detect)

            # rejecting noisy/late channels
            bottom_mask = np.sum(sz_clf[ch_sorting,:],axis=1) > 0
            first_zero = np.where(~bottom_mask)[0][0].astype(int)
            sz_clf[ch_sorting[first_zero:],:] = 0
            sz_prob[ch_sorting[first_zero:],:] = 0

            # Here we want to loop through a list of thresholds and generate the following columns of data
            # bp channels in UEO, bp channels in UEO + 10
            final_thresh = 0.5
            sz_clf_final = sz_prob > final_thresh

            # Here this could be first seizing index, or it could be the time of the clinically defined UEO from the annotations
            first_seizing_index = np.argmax(sz_clf_final.any(axis=0))
            mdl_ueo_idx = np.where(np.sum(sz_clf_final[:, first_seizing_index:first_seizing_index + 3], axis=1) > 0)[0]
            mdl_sec_idx = np.where(np.sum(sz_clf_final[:, first_seizing_index+10:first_seizing_index + 13], axis=1) > 0)[0]
            mdl_ueo_ch_bp = seizure.columns.to_numpy()[mdl_ueo_idx]
            mdl_ueo_ch = [s.split("-")[0] for s in mdl_ueo_ch_bp]
            all_ueo_preds["UEO_ch"].append(mdl_ueo_ch)
            all_ueo_preds["clinician"].append("MDL")
            all_ueo_preds["patient"].append(pt)
            all_ueo_preds["Seizure_ID"].append(sz_row.Seizure_ID)
            # (115,400)
            # In this section plot and save all of the plots that we generate in this section.
            if ~ospe(ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM")):
                os.makedirs(ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM"),exist_ok=True)
            plot_and_save_detection(sz_vals[ch_sorting,:],
                                    win_times,
                                    seizure.columns[ch_sorting],
                                    ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM",f"{montage}_loss_vals.png"))
            plot_and_save_detection(sz_prob[ch_sorting,:],
                                    win_times,
                                    seizure.columns[ch_sorting],
                                    ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM",f"{montage}_sz_prob.png"))
            
if __name__ == "__main__":
    main()