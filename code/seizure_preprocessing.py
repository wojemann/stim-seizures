# Scientific imports
import numpy as np
import pandas as pd
from scipy import signal as sig
import scipy as sc

# System imports
from ieeg.auth import Session
from itertools import compress
import json
import os
from os.path import join as ospj
from utils import *
import sys

sys.path.append('/users/wojemann/iEEG_processing')

# Loading metadata
with open('/mnt/leif/littlab/users/wojemann/stim-seizures/code/config.json','r') as f:
    CONFIG = json.load(f)
usr = CONFIG["paths"]["iEEG_USR"]
passpath = CONFIG["paths"]["iEEG_PWD"]
datapath = CONFIG["paths"]["RAW_DATA"]
metadatapath = CONFIG["paths"]["META"]
ieeg_list = CONFIG["patients"]
rid_hup = pd.read_csv(ospj(datapath,'rid_hup.csv'))
# pt_list = np.unique(np.array([i.split("_")[0] for i in ieeg_list]))
pt_list = ["HUP235"]
np.random.seed(42)

metadata = pd.read_csv(ospj(metadatapath,'metadata_wchreject.csv'))
metadata.loc[:,'ieeg_id'] = 'HUP' + metadata.hupsubjno.apply(str) + '_phaseII'
metadata.loc[:,'ccep_id'] = 'HUP' + metadata.hupsubjno.apply(str) + '_CCEP'

# Iterate through each patient
for pt in pt_list:
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
        if not os.path.exists(recon_path):
            recon_path =  ospj('/mnt','leif','littlab','data',
                        'Human_Data','recon','BIDS_penn',
                        f'sub-RID0{rid}','derivatives','ieeg_recon',
                        'module3/')
        electrode_localizations = electrode_localization(recon_path,rid)
        electrode_localizations.to_csv(ospj(raw_datapath,"electrode_localizations.csv"))
    else:    
        electrode_localizations = pd.read_csv(ospj(raw_datapath,"electrode_localizations.csv"))
    # Take channels that are only white or gray matter
    ch_names = electrode_localizations[(electrode_localizations['index'] == 2) | (electrode_localizations['index'] == 3)]["name"].to_numpy()
    
    # Cleaning channel labels and dropping bad channels
    ch_names_clean = clean_labels(ch_names,pt)
    dirty_drop_electrodes = metadata[metadata.hupsubjno == int(pt[-3:])]["final_reject_channels"].str.split(',').to_list()[0]
    if isinstance(dirty_drop_electrodes,list):
        final_drop_electrodes = clean_labels(dirty_drop_electrodes,pt)
    else:
        final_drop_electrodes = []
    
    # Check intersection of channel names with iEEG.org
    with open(passpath, "r") as f:
        pwd = f.read()
    s = Session(usr, pwd)
    ds = s.open_dataset(pt+"_phaseII")
    all_channel_labels = ds.get_channel_labels()
    # making sure all channels are in iEEG file
    final_drop_electrodes += [ch for ch in ch_names_clean if ch not in all_channel_labels]
    # removing any channels that don't meet criteria
    ch_names_clean = [ch for ch in ch_names_clean if ch not in final_drop_electrodes]

    # Iterate through each seizure in pre-defined pkl file
    for i_sz,row in seizure_times.iterrows():
        # if i_sz != 11:
        #     continue
        print(f"Seizure number: {i_sz}")
        if os.path.exists(ospj(raw_datapath,"seizures",f"seizure_{i_sz}_stim_{row.stim}.pkl")):
            buffered_seizure = pd.read_pickle(ospj(raw_datapath,"seizures",f"seizure_{i_sz}_stim_{row.stim}.pkl"))
            fs = buffered_seizure.fs.to_numpy()[-1]
            cols = buffered_seizure.columns
            mask = (pd.isna(buffered_seizure.fs) | (buffered_seizure.fs == 0) | (np.isnan(buffered_seizure.fs)))
            buffer = buffered_seizure.loc[mask,ch_names_clean]
            seizure = buffered_seizure.loc[~mask,ch_names_clean]
            t = np.arange(0,len(seizure)/fs,1/fs)
        else:
            print(f"Could not find seizure number {i_sz}. Skipping")
            continue

        # # Preprocessing
        # Rejecting bad channels and update metadata list
        if row.stim != 1:
            print("channel rejection")
            reject_mask,dets = detect_bad_channels(buffer.to_numpy(),fs,row.stim == 1)
        else:
            print("stim seizure: no channel rejection")
            reject_mask = np.ones((seizure.shape[1],),dtype=bool)

        processed_seizure = seizure.loc[:,reject_mask]
        ch_names_rejected = list(compress(ch_names_clean,reject_mask))
        final_drop_electrodes += list(compress(ch_names_clean,~reject_mask))
        print(f"rejecting {list(compress(ch_names_clean,~reject_mask))}")
        if final_drop_electrodes:
            metadata.loc[metadata.hupsubjno == int(pt[-3:]),'final_reject_channels'] = str(list(set(final_drop_electrodes))).replace('\"','').replace("\'",'').replace('[','').replace(']','')

        
        # Detect segments with artifact
        art_idxs = artifact_removal(processed_seizure.to_numpy(),fs,win_size = .1,noise=5000)#,
                    # noise = np.mean(processed_seizure) + 10*np.std(processed_seizure))
        # Reject channels based on too much artifact
        art_ch_reject = list(compress(ch_names_rejected,np.sum(art_idxs,axis=0)/len(art_idxs) > 0.1))
        print(f"rejecting {art_ch_reject} for artifact")
        ch_names_rejected = [ch for ch in ch_names_rejected if ch not in art_ch_reject]
        final_drop_electrodes += art_ch_reject
        processed_seizure = processed_seizure.loc[:,ch_names_rejected]
        
        art_idxs = artifact_removal(processed_seizure.to_numpy(),fs,win_size = .1,
                    noise = np.mean(processed_seizure) + 10*np.std(processed_seizure))
        
        # Propogate artifact indices across all channels
        artifact_mask = sig.medfilt(art_idxs.any(1).astype(int),5)
        stim_idxs = np.reshape(np.where(np.diff(artifact_mask,prepend=0)),(-1,2))
        print("initiating artifact rejection")
        s = processed_seizure.to_numpy()
        for i_ch in range(s.shape[1]):
            for win in stim_idxs:
                # Define windows
                win_len = win[1] - win[0]
                pre_idx = win[0] - win_len
                post_idx = win[1] + win_len

                # Interpolation parameters
                pre_idxs = np.arange(pre_idx,win[0])
                post_idxs = np.arange(win[1],post_idx)
                fill_idxs = np.arange(win[0],win[1])

                # Interpolation
                # check for edge cases
                if post_idx >= len(s):
                        pre_win = s[pre_idxs,i_ch]
                        pre_t = t[pre_idxs]
                        post_win = pre_win
                        post_t = pre_t
                        interp_fn = lambda x: np.ones((len(x),))*s[win[0],i_ch]
                else:
                    if win[0] < win_len:
                        pre_t = np.arange((pre_idx)/fs,(win[0])/fs,1/fs)
                        post_t = t[post_idxs]
                        pre_win = buffer.loc[:,ch_names_clean[i_ch]].to_numpy()[pre_idxs]
                        post_win = s[post_idxs,i_ch]

                    else:
                        pre_win = s[pre_idxs,i_ch]
                        pre_t = t[pre_idxs]
                        post_win = s[post_idxs,i_ch]
                        post_t = t[post_idxs]
                    interp_fn = sc.interpolate.interp1d(np.concatenate([pre_t,post_t]),
                                            np.concatenate([pre_win,post_win]))
                # run interpolation
                filled_s = interp_fn(t[fill_idxs])

                # Adding noise to interpolation
                if post_idx < len(s):
                    sample_std = (np.std(pre_win) + np.std(s[post_idxs,i_ch]))/8
                else:
                    sample_std = np.std(pre_win)

                interp_samples = np.random.normal(filled_s,np.ones_like(filled_s)*sample_std)
                
                # assigning interpolated samples to artifact segment
                s[win[0]:win[1],i_ch] = interp_samples

        # Filtering
        notch_seizure = notch_filter(s,fs)
        band_seizure = bandpass_filter(notch_seizure,fs)
        cols = ch_names_rejected
        cols.append('fs')
        # CAR montage
        car_seizure = band_seizure - np.expand_dims(np.mean(band_seizure, axis = 1),1)
        
        # Saving processed seizure
        preprocessed_seizure = pd.DataFrame(np.concatenate((car_seizure,np.ones((len(s,),1))*fs),axis=1), 
                                                columns = cols)

        preprocessed_seizure.to_pickle(ospj(raw_datapath,"seizures",f"preprocessed_seizure_{i_sz}_stim_{row.stim}.pkl"))
    metadata.to_csv(ospj(metadatapath,"metadata_wchreject.csv"))