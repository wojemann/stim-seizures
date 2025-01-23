# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
from tqdm import tqdm
from kneed import KneeLocator

# Plotting
import matplotlib.pyplot as plt

# OS imports
from os.path import join as ospj
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')
sys.path.append('/users/wojemann/DSOSD/')
from DSOSD.model import NDD

plt.rcParams['image.cmap'] = 'magma'

def main():
    _,_,datapath,prodatapath,metapath,_,patient_table,_,_ = load_config(ospj('/mnt/sauce/littlab/users/wojemann/stim-seizures/code','config.json'),None)

    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))
    annotations_df = pd.read_pickle(ospj(prodatapath,"stim_seizure_information_consensus.pkl"))

    montage = 'bipolar'
    mdl_strs = ['LSTM','AbsSlp','WVNT']
    # Iterating through each patient that we have annotations for
    predicted_channels = {'Patient': [],
                        'iEEG_ID': [],
                        'model':[],
                        'stim':[],
                        'approximate_onset': [],
                        'ueo_time_consensus': [],
                        'threshold':[],
                        'ueo_chs_strict':[],
                        'ueo_chs_loose': [],
                        'sec_chs_strict': [],
                        'sec_chs_loose':[],
                        'to_annotate': []
                        }
    df_dict_list = []
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:
        pt = row.ptID
        pbar.set_description(desc=f"Patient -- {pt}",refresh=True)
        if len(row.interictal_training) == 0:
            continue
        ### ONLY PREDICTING FOR SEIZURES THAT HAVE BEEN ANNOTATED
        seizure_times = seizures_df[(seizures_df.Patient == pt) & (seizures_df.to_annotate == 1)]
        ###

        qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),desc = 'Seizures',leave=False)
        for _,sz_row in qbar:
            
            if (pt == 'CHOP037') & (sz_row.approximate_onset == 962082.12):
                    continue
            _,_, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
            # print(task)
            for mdl_str in mdl_strs:
                clf_fs = 128
                prob_path = f"pretrain_probability_matrix_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.pkl"
                sz_prob = pd.read_pickle(ospj(prodatapath,pt,prob_path))
                time_wins = sz_prob.time.to_numpy()
                sz_prob.drop('time',axis=1,inplace=True)
                prob_chs = sz_prob.columns.to_numpy()

                # Match seizure using approximate onset time in annotations, patient name, and task
                task_time = int(task[np.where([s.isnumeric() for s in task])[0][0]:])
                approx_time = sz_row.approximate_onset
                if task_time in annotations_df.approximate_onset.astype(int):
                    annot_row = annotations_df[annotations_df.approximate_onset.astype(int) == task_time]
                    consensus_time = annot_row.ueo_time_consensus.item()
                else:
                    consensus_time = approx_time
                
                # identifying difference between annotator and approximate time
                time_diff = consensus_time - approx_time

                onset_time = 120
                offset_time = 120

                # Find closest index to consensus onset time relative to actual onset time (consensus - approximate and find closest to 120 + diff)
                onset_index = np.argmin(np.abs((time_wins-onset_time) + time_diff))
                # Find closest index to consensus 10 second spread time
                spread_index = np.argmin(np.abs((time_wins-(onset_time+10)) + time_diff))
                for ch in prob_chs:
                    onset_ndd = sz_prob.loc[onset_index:onset_index+2,ch].mean()
                    spread_ndd = sz_prob.loc[spread_index:spread_index+2,ch].mean()
                    temp_dict = {
                        'patient': pt,
                        'iEEG_ID': sz_row.IEEGname,
                        'model':mdl_str,
                        'stim':sz_row.stim,
                        'approximate_onset': sz_row.approximate_onset,
                        'ueo_time_consensus': consensus_time,
                        'channel': ch,
                        'onset_ndd': onset_ndd,
                        'spread_ndd': spread_ndd,
                        'to_annotate': sz_row.to_annotate
                        }
                    df_dict_list.append(temp_dict)

    predicted_channels = pd.DataFrame(df_dict_list)
    predicted_channels.to_pickle(ospj(prodatapath,"NDD_soz_localizations.pkl"))

if __name__ == "__main__":
    main()