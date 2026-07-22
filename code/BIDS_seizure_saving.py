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
from config import Config
OVERWRITE=False

def main():
    # Loading CONFIG
    usr,passpath,_,_,_,_,_,_,_ = Config.deal()
    prodatapath = "/Users/wojemann/local_data/stim_dataset_data/PROCESSED_DATA"
    datapath = "/Users/wojemann/local_data/stim_dataset_data/RAW_DATA"
    
    # Setting up BIDS targets
    bids_path_kwargs = {
        "root": ospj(datapath,'DATA'),
        "datatype": "ieeg",
        "extension": ".edf",
        "suffix": "ieeg",
        "task": "ictal",
        "session": "postimplant",
    }

    bids_path = BIDSPath(**bids_path_kwargs)
    ieeg_kwargs = {
        "username": usr,
        "password_bin_file": passpath,
    }

    stim_kwargs = {'Stim. Induced': 1, 'Spontaneous': 0}
    seizures_df = pd.read_csv(ospj(prodatapath,"annotations.csv"))
    buffer = 120 # seconds before and after seizure to save

    for pt, group in tqdm(
        seizures_df.groupby('patient'),
        total=seizures_df.patient.nunique(),
        desc="Patients",
        position=0,
    ):
        ieegid = group.groupby('iEEG_ID').ngroup().astype(int)
        seizures_df.loc[ieegid.index,'run'] = ieegid
        group.loc[ieegid.index,'run'] = ieegid
        
        # sort by start time
        group = group.sort_values(["run","approximate_onset"])
        group.reset_index(inplace=True, drop=True)
        for idx, row in tqdm(
            group.iterrows(), total=group.shape[0], desc="seizures", position=1, leave=False
        ):
            stim = stim_kwargs[row.stim]
            task_names = ['ictal','stim']
            onset = row.approximate_onset
            offset = row.end
            # get bids path
            sz_clip_bids_path = bids_path.copy().update(
                subject=pt,
                run=str(int(row["run"])).zfill(2),
                task=f"{task_names[stim]}{int(onset)}",
            )

            # check if the file already exists, if so, skip
            if sz_clip_bids_path.fpath.exists() and not OVERWRITE:
                continue

            # HUP097 does not have an end time, so we'll just use 60 seconds from the start
            if np.isnan(offset):
                offset = onset + 60

            # get the duration and clip it to 5 mins
            duration = offset-onset

            data, fs = get_iEEG_data(
                iEEG_filename=row["iEEG_ID"],
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

            # save the data
            # run is the iEEG file number
            # task is ictal with the start time in seconds appended
            data_info = mne.create_info(
                ch_names=list(data.columns), sfreq=fs, ch_types="eeg", verbose=False
            )
            raw = mne.io.RawArray(
                data_np / 1e6,  # mne needs data in volts,
                data_info,
                verbose=False,
            )
            raw.set_channel_types(ch_types.type)
            annots = mne.Annotations(
                onset=[buffer],
                duration=[duration],
                description=task_names[stim],
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
    seizures_df.to_csv(ospj(datapath,"DATA","annotations.tsv"),index=False,sep='\t')

if __name__ == "__main__":
    main()
