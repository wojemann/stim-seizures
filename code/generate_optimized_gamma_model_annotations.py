# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
from tqdm import tqdm
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

    montage = 'bipolar'
    mdl_strs = ['LSTM','AbsSlp','NRG','WVNT']
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
        ### ONLY PREDICTING FOR SEIZURES THAT HAVE BEEN ANNOTATED
        # seizure_times = seizures_df[(seizures_df.Patient == pt) & (seizures_df.to_annotate == 1)]
        ###
        seizure_times = seizures_df[seizures_df.Patient == pt]

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
                sz_prob = sz_prob.to_numpy().T
                
                # identifying difference between annotator and approximate time
                onset_time = 120
                first_sz_idx_offset = np.argmin(np.abs(time_wins-onset_time))

                # Get KDE for all probability values
                probabilities = sz_prob.flatten()
                shape, _, scale = gamma.fit(probabilities, floc=0)
                final_thresh = gamma.ppf(0.95, a=shape, scale=scale)
        
                late = np.sum(sz_prob[:,-first_sz_idx_offset:] > final_thresh,axis=1) > (first_sz_idx_offset/2)
                # sz_prob_reject = sz_prob[~late,:]
                # prob_chs = prob_chs[~late]
                sz_prob_reject = sz_prob
                sz_clf_final = sz_prob_reject>final_thresh

                predicted_channels['Patient'].append(sz_row.Patient)
                predicted_channels['iEEG_ID'].append(sz_row.IEEGname)
                predicted_channels['model'].append(mdl_str)
                predicted_channels['stim'].append(sz_row.stim)
                predicted_channels['approximate_onset'].append(sz_row.approximate_onset)
                predicted_channels['offset'].append(sz_row.end)
                predicted_channels['threshold'].append(final_thresh)
                predicted_channels['all_channels'].append(np.array([s.split("-")[0] for s in prob_chs]).flatten())
                
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
                    sz_ch_arr = prob_chs[seized_idxs]
                    sz_ch_arr = np.array([s.split("-")[0] for s in sz_ch_arr]).flatten()
                    onset_index = np.min(first_sz_idxs[seized_idxs])
                else:
                    sz_ch_arr = []
                    sz_times_arr = []
                    onset_index = np.min(first_sz_idxs)
                predicted_channels['sz_chs'].append(sz_ch_arr)
                predicted_channels['sz_times'].append(sz_times_arr)

                new_onset_time = time_wins[onset_index]
                spread_index = np.argmin(np.abs(time_wins-(new_onset_time + 10)))

                mdl_ueo_idx = np.sum(sz_prob[:,onset_index:onset_index+10] > final_thresh,axis=1) > 6
                mdl_ueo_ch_bp = prob_chs[mdl_ueo_idx]
                mdl_ueo_ch_strict = np.array([s.split("-")[0] for s in mdl_ueo_ch_bp]).flatten()
                mdl_ueo_ch_loose = np.unique(np.array([s.split("-") for s in mdl_ueo_ch_bp]).flatten())
                predicted_channels['ueo_chs_strict'].append(mdl_ueo_ch_strict)
                predicted_channels['ueo_chs_loose'].append(mdl_ueo_ch_loose)

                mdl_sec_idx = np.sum(sz_prob[:,spread_index:spread_index+10] > final_thresh,axis=1) > 6
                mdl_sec_ch_bp = prob_chs[mdl_sec_idx]
                mdl_sec_ch_strict = np.array([s.split("-")[0] for s in mdl_sec_ch_bp]).flatten()
                mdl_sec_ch_loose = np.unique(np.array([s.split("-") for s in mdl_sec_ch_bp]).flatten())
                predicted_channels['sec_chs_strict'].append(mdl_sec_ch_strict)
                predicted_channels['sec_chs_loose'].append(mdl_sec_ch_loose)

    predicted_channels = pd.DataFrame(predicted_channels)
    predicted_channels.to_pickle(ospj(prodatapath,"gamma_predicted_channels_opt.pkl"))
    predicted_channels.to_csv(ospj(prodatapath,"gamma_predicted_channels.csv"))
if __name__ == "__main__":
    main()