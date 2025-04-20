### SAVING SEIZURES AS BIDS FORMAT TO LEIF
import numpy as np
import pandas as pd
import json
import os
from os.path import join as ospj
from utils import *
import scipy as sc

from tqdm import tqdm


# BIDS imports
import mne
from mne_bids import BIDSPath, write_raw_bids

# Loading CONFIG
usr,passpath,datapath,prodatapath,metapath,figpath,patient_table,rid_hup,pt_list = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'),flag=None)

# Setting Seed
np.random.seed(171999)

TARGET = 512
OVERWRITE = False

def main():
    # Setting up BIDS targets
    bids_path_kwargs = {
        "root": ospj(datapath,'BIDS'),
        "datatype": "ieeg",
        "extension": ".edf",
        "suffix": "ieeg",
        "task": "ictal",
        "session": "clinical01",
    }
    bids_path = BIDSPath(**bids_path_kwargs)
    ieeg_kwargs = {
        "username": usr,
        "password_bin_file": passpath,
    }

    # Loading in all seizure data
    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information - LF_seizure_annotation.csv"))
    seizures_df.dropna(axis=0,how='all',inplace=True)
    seizures_df['approximate_onset'].fillna(seizures_df['UEO'],inplace=True)
    seizures_df['approximate_onset'].fillna(seizures_df['EEC'],inplace=True)
    seizures_df['approximate_onset'].fillna(seizures_df['Other_onset_description'],inplace=True)
    # drop HF stim induced seizures
    seizures_df = seizures_df[seizures_df.stim != 2]
    # adult_list = [pt for pt in pt_list if 'CHOP' not in pt]
    # seizures_df = seizures_df[seizures_df.Patient.isin(adult_list)]
    seizures_df = seizures_df[seizures_df.Patient.isin(pt_list)]
    bad_ch_dict = dict()
    buffer = 120 # seconds before and after seizure to save
    for pt, group in tqdm(
        seizures_df.groupby('Patient'),
        total=seizures_df.Patient.nunique(),
        desc="Patients",
        position=0,
    ):
        bad_ch_dict[pt] = set()
        ieegid = group.groupby('IEEGname').ngroup().astype(int)
        seizures_df.loc[ieegid.index,'IEEGID'] = ieegid
        group.loc[ieegid.index,'IEEGID'] = ieegid
        
        # sort by start time
        group = group.sort_values(["IEEGID","approximate_onset"])
        group.reset_index(inplace=True, drop=True)

        if pt != 'HUP275':
            continue
        
        for idx, row in tqdm(
            group.iterrows(), total=group.shape[0], desc="seizures", position=1, leave=False
        ):
            if row.stim == 2: # Skip high frequency induced seizures
                continue
            task_names = ['ictal','stim']
            onset = row.approximate_onset
            offset = row.end
            # get bids path
            sz_clip_bids_path = bids_path.copy().update(
                subject=pt,
                run=int(row["IEEGID"]),
                task=f"{task_names[int(row.stim)]}{int(onset)}",
            )

            # check if the file already exists, if so, skip
            if sz_clip_bids_path.fpath.exists() and not OVERWRITE:
                continue

            # CHOP037 has a seizure that's too large
            if (pt == 'CHOP037') & (onset == 962082.12):
                continue

            # HUP097 does not have an end time, so we'll just use 60 seconds from the start
            if np.isnan(offset):
                offset = onset + 60

            # get the duration and clip it to 5 mins
            duration = offset-onset

            data, fs = get_iEEG_data(
                iEEG_filename=row["IEEGname"],
                start_time_usec=(onset - buffer) * 1e6, # start buffer seconds before the seizure
                stop_time_usec=(offset + buffer) * 1e6,
                **ieeg_kwargs,
            )

            # channels with flat line may not save proprely, so we'll drop them
            data = data[data.columns[data.min(axis=0) != data.max(axis=0)]]

            # clean the labels
            data.columns = clean_labels(data.columns, pt=pt)
            
            # remove scalp and ekg electrodes
            no_scalp_labels = remove_scalp_electrodes(data.columns)
            data = data.loc[:,no_scalp_labels]

            # if there are duplicate labels, keep the first one in the table
            data = data.loc[:, ~data.columns.duplicated()]
            # get the channel types
            ch_types = check_channel_types(list(data.columns))
            ch_types.set_index("name", inplace=True, drop=True)

            # convert nan to 0
            data.fillna(0, inplace=True)

            # minimal preprocessing
            data_np = data.to_numpy().T
            data_np_notch = notch_filter(data_np,fs)
            # data_np_filt = bandpass_filter(data_np_notch,fs,order=3,lo=1,hi=100)
            signal_len = int(data_np_notch.shape[1]/fs*TARGET)
            data_np_ds = sc.signal.resample(data_np_notch,signal_len,axis=1)
            fs = TARGET

            # detect bad channels
            if row.stim == 0:
                ch_mask,_ = detect_bad_channels(data_np_ds.T,fs)
                bad_ch = data.columns[~ch_mask].to_list()
                bad_ch_dict[pt].update(bad_ch)

            # save the data
            # run is the iEEG file number
            # task is ictal with the start time in seconds appended
            data_info = mne.create_info(
                ch_names=list(data.columns), sfreq=fs, ch_types="eeg", verbose=False
            )
            raw = mne.io.RawArray(
                data_np_ds / 1e6,  # mne needs data in volts,
                data_info,
                verbose=False,
            )
            raw.set_channel_types(ch_types.type)
            annots = mne.Annotations(
                onset=[buffer], # seizure starts 60 seconds after the start of the clip
                duration=[duration],
                description=task_names[int(row.stim)],
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw.set_annotations(annots)

                write_raw_bids(
                    raw,
                    sz_clip_bids_path,
                    overwrite=OVERWRITE,
                    verbose=False,
                    allow_preload=True,
                    format="EDF",
                )
    seizures_df.to_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))
    # Save to a JSON file
    with open(ospj(metapath,'bad_ch_dict.pkl'), 'wb') as f:
        pickle.dump(bad_ch_dict, f)
if __name__ == "__main__":
    main()
