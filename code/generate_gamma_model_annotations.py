# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
from tqdm import tqdm
from kneed import KneeLocator
from scipy.stats import gamma

# Plotting
import matplotlib.pyplot as plt

# OS imports
from os.path import join as ospj
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')

plt.rcParams['image.cmap'] = 'magma'

def main():
    _,_,datapath,prodatapath,metapath,_,patient_table,_,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'),None)

    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))
    annotations_df = pd.read_pickle(ospj(prodatapath,"stim_seizure_information_consensus.pkl"))

    montage = 'bipolar'
    mdl_strs = ['LSTM','AbsSlp','NRG','WVNT']
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
            _,_, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
            for mdl_str in mdl_strs:
                # clf_fs = 128 if mdl_str == 'WVNT' else 256
                clf_fs = 128
                prob_path = f"pretrain_probability_matrix_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.pkl"
                sz_prob = pd.read_pickle(ospj(prodatapath,pt,prob_path))
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
                
                # identifying difference between annotator and approximate time
                time_diff = consensus_time - approx_time

                onset_time = 120

                # Find closest index to consensus onset time relative to actual onset time (consensus - approximate and find closest to 120 + diff)
                onset_index = np.argmin(np.abs((time_wins-onset_time) + time_diff))
                # Find closest index to consensus 10 second spread time
                spread_index = np.argmin(np.abs((time_wins-(onset_time+10)) + time_diff))
                # Fitting gamma distribution to seizure
                probabilities = sz_prob.flatten()
                shape, _, scale = gamma.fit(probabilities, floc=0)

                final_thresh = gamma.ppf(0.95, a=shape, scale=scale)

                predicted_channels['Patient'].append(sz_row.Patient)
                predicted_channels['iEEG_ID'].append(sz_row.IEEGname)
                predicted_channels['model'].append(mdl_str)
                predicted_channels['stim'].append(sz_row.stim)
                predicted_channels['approximate_onset'].append(sz_row.approximate_onset)
                predicted_channels['ueo_time_consensus'].append(consensus_time)
                predicted_channels['to_annotate'].append(sz_row.to_annotate)
                predicted_channels['threshold'].append(final_thresh)

                # get late szing mask
                sz_prob_reject = sz_prob
                prob_chs_reject = prob_chs
                # sz_clf_final = sz_prob > final_thresh

                # Here this could be first seizing index, or it could be the time of the clinically defined UEO from the annotations
                # first_seizing_index = np.argmax(sz_clf_final.any(axis=0))

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
    predicted_channels.to_pickle(ospj(prodatapath,"gamma_predicted_channels_nor.pkl"))
    predicted_channels.to_csv(ospj(prodatapath,"gamma_predicted_channels.csv"))
if __name__ == "__main__":
    main()