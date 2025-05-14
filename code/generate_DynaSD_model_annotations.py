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
sys.path.append('/mnt/sauce/littlab/users/wojemann/DSOSD/')
from DSOSD.model import NDD

plt.rcParams['image.cmap'] = 'magma'

def main():
    _,_,datapath,prodatapath,metapath,_,patient_table,_,_ = load_config(ospj('/mnt/sauce/littlab/users/wojemann/stim-seizures/code','config.json'),None)

    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))
    # annotations_df = pd.read_pickle(ospj(prodatapath,"stim_seizure_information_consensus.pkl"))
    annotations_df = pd.read_pickle(ospj(prodatapath,"threshold_tuning_consensus_v2.pkl"))


    montage = 'bipolar'
    threshold_str = 'automedian'
    mdl_strs = ['LSTM']
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
        # seizure_times = seizures_df[seizures_df.Patient == pt]

        ###

        qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),desc = 'Seizures',leave=False)
        for _,sz_row in qbar:
            if (pt == 'CHOP037') & (sz_row.approximate_onset == 962082.12):
                    continue
            _,_, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
            # print(task)
            for mdl_str in mdl_strs:
                clf_fs = 128
                # prob_path = f"pretrain_probability_matrix_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.pkl"
                prob_path = f"pretrain_probability_matrix_nosmooth_mdl-{mdl_str}_fs-{clf_fs}_montage-{montage}_task-{task}_run-{run}.pkl"
                sz_prob = pd.read_pickle(ospj(prodatapath,pt,prob_path))
                time_wins = sz_prob.time.to_numpy()
                sz_prob.drop('time',axis=1,inplace=True)
                prob_chs = sz_prob.columns.to_numpy()

                sz_prob = pd.DataFrame(sc.ndimage.uniform_filter1d(sz_prob,size=20,mode='nearest',axis=0,origin=0),columns=prob_chs)

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
                # Find closest index to seizure offset time
                offset_index = np.argmin(np.abs(time_wins - (np.max(time_wins)-offset_time)))

                model = NDD(fs = 128)
                threshold = model.get_gaussianx_threshold(sz_prob.iloc[:offset_index,:],noise_floor=threshold_str)
                
                if task_time == 89820: # This one seizure does not have sufficient duration to apply a 10 second moving median window.
                    sz_spread = model.get_onset_and_spread(sz_prob.iloc[onset_index:offset_index,:],
                    threshold=threshold,
                    filter_w=5,
                    rwin_size=5,
                    rwin_req=4,)
                else:
                    sz_spread = model.get_onset_and_spread(
                        sz_prob.iloc[onset_index:offset_index,:],
                        threshold=threshold,
                        filter_w=10,
                        rwin_size=5,
                        rwin_req=4,
                        )

                sz_spread -= np.min(sz_spread.min())

                predicted_channels['Patient'].append(sz_row.Patient)
                predicted_channels['iEEG_ID'].append(sz_row.IEEGname)
                predicted_channels['model'].append(mdl_str)
                predicted_channels['stim'].append(sz_row.stim)
                predicted_channels['approximate_onset'].append(sz_row.approximate_onset)
                predicted_channels['ueo_time_consensus'].append(consensus_time)
                predicted_channels['to_annotate'].append(sz_row.to_annotate)
                predicted_channels['threshold'].append(threshold)

                # get late szing mask
                szing_idxs = sz_spread.iloc[0,:].to_numpy()
                mdl_ueo_ch_bp = sz_spread.iloc[0, szing_idxs < 2].index
                mdl_sec_ch_bp = sz_spread.iloc[0,(szing_idxs >= 20) & (szing_idxs < 22)].index
                # Here this could be first seizing index, or it could be the time of the clinically defined UEO from the annotations
                
                mdl_ueo_ch_strict = np.array([s.split("-")[0] for s in mdl_ueo_ch_bp]).flatten()
                mdl_ueo_ch_loose = np.unique(np.array([s.split("-") for s in mdl_ueo_ch_bp]).flatten())
                predicted_channels['ueo_chs_strict'].append(mdl_ueo_ch_strict)
                predicted_channels['ueo_chs_loose'].append(mdl_ueo_ch_loose)
                
                mdl_sec_ch_strict = np.array([s.split("-")[0] for s in mdl_sec_ch_bp]).flatten()
                mdl_sec_ch_loose = np.unique(np.array([s.split("-") for s in mdl_sec_ch_bp]).flatten())
                predicted_channels['sec_chs_strict'].append(mdl_sec_ch_strict)
                predicted_channels['sec_chs_loose'].append(mdl_sec_ch_loose)

    predicted_channels = pd.DataFrame(predicted_channels)
    predicted_channels.to_pickle(ospj(prodatapath,f"DynaSD_gaussianx_{threshold_str}_predicted_channels_norp_valtuned_v2.pkl"))
if __name__ == "__main__":
    main()