import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from ieeg.auth import Session
from scipy import signal as sig
import scipy as sc
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import os
from os.path import join as ospj
from os.path import exists as ospe
import pathlib
from tqdm import tqdm
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')
from pioneer import Pioneer
import mne

# Loading metadata
with open('/mnt/leif/littlab/users/wojemann/stim-seizures/code/config.json','r') as f:
    CONFIG = json.load(f)
usr = CONFIG["paths"]["iEEG_USR"]
pass_path = CONFIG["paths"]["iEEG_PWD"]
datapath = CONFIG["paths"]["RAW_DATA"]
ieeg_list = CONFIG["patients"]
rid_hup = pd.read_csv(ospj(datapath,'rid_hup.csv'))
pt_list = np.unique(np.array([i.split("_")[0] for i in ieeg_list]))

np.random.seed(42)

# Iterate through each patient
for pt in [pt_list[0]]:
    print(f"Starting Seizure Preprocessing for {pt}")
    raw_datapath = ospj(datapath,pt)
    # load dataframe of seizure times
    seizure_times = pd.read_csv(ospj(raw_datapath,f"seizure_times_{pt}.csv"))
    # load electrode information
    if not os.path.exists(ospj(raw_datapath, "electrode_localizations.csv")):
        hup_no = pt[3:]
        rid = rid_hup[rid_hup.hupsubjno == hup_no].record_id.to_numpy()[0]
        recon_path = ospj('/mnt','leif','littlab','data',
                          'Human_Data','CNT_iEEG_BIDS',
                          f'sub-RID0{rid}','derivatives','ieeg_recon',
                          'module3/')
        electrode_localizations = optimize_localizations(recon_path,rid)
        electrode_localizations.to_csv(ospj(raw_datapath,"electrode_localizations.csv"))
    else:    
        electrode_localizations = pd.read_csv(ospj(raw_datapath,"electrode_localizations.csv"))
    ch_names = electrode_localizations[(electrode_localizations['index'] == 2) | (electrode_localizations['index'] == 3)]["name"]

    # loading seizures
    if not os.path.exists(ospj(raw_datapath, "seizures")):
        os.mkdir(ospj(raw_datapath, "seizures"))
    
    # Iterate through each seizure in pre-defined csv
    for i_sz,row in seizure_times.iterrows():
        print(f"Seizure number: {i_sz}")
        if os.path.exists(ospj(raw_datapath,"seizures",f"seizure_{i_sz}_stim_{row.stim}.csv")):
            seizure = pd.read_csv(ospj(raw_datapath,"seizures",f"seizure_{i_sz}_stim_{row.stim}.csv"))
            fs = seizure.pop('fs').to_numpy()[0]
        else:
            seizure,fs = get_iEEG_data(usr,pass_path,
                                        row.IEEGname,
                                        row.start*1e6,
                                        row.end*1e6,
                                        ch_names,
                                        force_pull = True)
            save_seizure = pd.concat((seizure,pd.DataFrame(np.ones(len(seizure),)*fs,columns=['fs'])),axis = 1)
            save_seizure.to_csv(ospj(raw_datapath,"seizures",f"seizure_{i_sz}_stim_{row.stim}.csv"),index=False)

        # # Preprocessing seizure
        t = np.arange(0,len(seizure)/fs,1/fs)
        ch_names_clean = seizure.columns.to_list()
        # Filtering
        notch_seizure = notch_filter(seizure.to_numpy(),fs)
        band_seizure = bandpass_filter(notch_seizure,fs)
        car_seizure = band_seizure - np.mean(band_seizure, axis=0)
        processed_seizure = pd.DataFrame(car_seizure,columns=seizure.columns)

        # Artifact Removal
        x = artifact_removal(processed_seizure.to_numpy(),fs,win_size = .05,
                 noise = np.mean(processed_seizure) + 10*np.std(processed_seizure))
        # Account for variation
        artifact_mask = sig.medfilt(x.any(1),5)
        stim_idxs = np.reshape(np.where(np.diff(artifact_mask,prepend=0)),(-1,2))
        for i_ch in range(len(ch_names_clean)):
            for win in stim_idxs:
                # Define windows
                win_len = win[1]-win[0]
                pre_idx = win[0] - win_len
                post_idx = win[1] + win_len

                # Interpolation parameters
                s = processed_seizure.to_numpy()
                pre_idxs = np.arange(pre_idx,win[0])
                post_idxs = np.arange(win[1],post_idx)
                fill_idxs = np.arange(win[0],win[1])

                # Interpolation
                interp_fn = sc.interpolate.interp1d(np.concatenate([t[pre_idxs],t[post_idxs]]),
                                        np.concatenate([s[pre_idxs,i_ch],s[post_idxs,i_ch]]))
                filled_s = interp_fn(t[fill_idxs])

                # Adding noise to linear interpolation
                sample_std = np.std(np.concatenate([s[pre_idxs,i_ch],s[post_idxs,i_ch]]))
                interp_samples = np.random.normal(filled_s,np.ones_like(filled_s)*sample_std)
                smoothed_samples = sc.ndimage.gaussian_filter1d(interp_samples,2)
                # assigning
                s[win[0]:win[1],i_ch] = smoothed_samples

        cols = ch_names_clean
        cols.append('fs')
        postrejection_seizure = pd.DataFrame(np.concatenate((s,np.ones((len(s,),1))*fs),axis=1), 
                                             columns = cols)
        postrejection_seizure.to_csv(ospj(raw_datapath,"seizures",f"processed_seizure_{i_sz}_stim_{row.stim}.csv"),index=False)


