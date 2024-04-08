### SAVING INTERICTAL TRAINING DATA IN BIDS TO LEIF
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
usr,passpath,datapath,prodatapath,figpath,patient_table,rid_hup,pt_list = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'))

# Setting Seed
np.random.seed(171999)

TARGET = 512
OVERWRITE = False

# Setting up BIDS targets
bids_path_kwargs = {
    "root": ospj(datapath,'BIDS'),
    "datatype": "ieeg",
    "extension": ".edf",
    "suffix": "ieeg",
    "task": "interictal",
    "session": "clinical01",
}
bids_path = BIDSPath(**bids_path_kwargs)
ieeg_kwargs = {
    "username": usr,
    "password_bin_file": passpath,
}

# Loading in all seizure data
seizures_df = pd.read_csv(ospj(datapath,"stim_seizure_information_BIDS.csv"))

for _,row in tqdm(
    patient_table.iterrows(),
    total=len(patient_table),
    desc="Patients",
    position=0,
):
    if len(row.interictal_training) == 0:
        continue
    pt = row.ptID
    ieeg_name = row.interictal_training[0]
    onset = row.interictal_training[1]
    offset = onset + 10
    ieegid = int(seizures_df.loc[seizures_df.IEEGname == ieeg_name,'IEEGID'].mode())
    # get bids path
    clip_bids_path = bids_path.copy().update(
        subject=pt,
        run=ieegid,
        task=f"interictal{int(onset)}",
    )

    # check if the file already exists, if so, skip
    if clip_bids_path.fpath.exists() and not OVERWRITE:
        continue

    duration = offset-onset

    data, fs = get_iEEG_data(
        iEEG_filename=ieeg_name,
        start_time_usec= onset * 1e6, # start 30 seconds before the seizure
        stop_time_usec= offset * 1e6,
        **ieeg_kwargs,
    )

    # channels with flat line may not save proprely, so we'll drop them
    data = data[data.columns[data.min(axis=0) != data.max(axis=0)]]

    # clean the labels
    data.columns = clean_labels(data.columns, pt=pt)

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
    data_np_filt = bandpass_filter(data_np_notch,fs,order=3,lo=1,hi=100)
    factor = get_factor(fs,TARGET)
    data_np_ds = sc.signal.decimate(data_np_filt,factor)
    fs /= factor


    # save the data
    # run is the iEEG file number
    # task is ictal with the start time in seconds appended
    data_info = mne.create_info(
        ch_names=list(data.columns), sfreq=fs, ch_types="eeg", verbose=False
    )
    raw = mne.io.RawArray(
        data.to_numpy().T / 1e6,  # mne needs data in volts,
        data_info,
        verbose=False,
    )
    raw.set_channel_types(ch_types.type)
    annots = mne.Annotations(
        onset=0, # seizure starts 60 seconds after the start of the clip
        duration=[duration],
        description="interictal",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.set_annotations(annots)

        write_raw_bids(
            raw,
            clip_bids_path,
            overwrite=OVERWRITE,
            verbose=False,
            allow_preload=True,
            format="EDF",
        )
