### SAVING INTERICTAL TRAINING DATA IN BIDS TO LEIF
from pickle import FALSE
import numpy as np
import pandas as pd
from os.path import join as ospj
from utils import clean_labels, check_channel_types, get_iEEG_data
import scipy as sc

from tqdm import tqdm
import warnings


# BIDS imports
import mne
from mne_bids import BIDSPath, write_raw_bids
from config import Config

# Loading CONFIG
usr,passpath,datapath,prodatapath,figpath,metapath,patient_table,rid_hup,pt_list = Config.deal(['usr','passpath','datapath','prodatapath','figpath','metapath','patient_table','rid_hup','pt_list'])

# Setting Seed
np.random.seed(171999)

TARGET = 256
OVERWRITE = False

def main():
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
    # seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))
    seizures_df = pd.read_csv(ospj(metapath,"metadata_v6_BIDS.csv"))
    # seizures_df = pd.read_csv(ospj(metapath,'validation_metadata','metadata_v6.csv'))
    patient_list = seizures_df.Patient.sort_values().unique()
    for pt in tqdm(
        patient_list,
        total=len(patient_list),
        desc="Patients",
        position=0,
    ):
        
        # if pt not in ['HUP074']:#('HUP065','HUP078','HUP126','HUP221','HUP276'):
        #     continue
        pt_seizures = seizures_df[seizures_df.Patient == pt].sort_values(by='onset')
        try:
            first_spaces = []
            id_count = 0
            while (len(first_spaces)==0) and (id_count < 10):
                ieeg_name = pt_seizures.IEEGname.iloc[id_count]
                if 'CCEP' in ieeg_name:
                    id_count += 1
                    continue
                ieeg_onsets = pt_seizures[pt_seizures.IEEGname == ieeg_name].onset.to_list()
                ieeg_onsets.insert(0,0)
                first_spaces = np.argwhere(np.diff(ieeg_onsets) > 2 * 60 * 60)[0]
                id_count += 1
            
            first_idx = first_spaces[0]
            if pt in ['HUP074','HUP070','HUP147','HUP206','HUP207']:
                first_idx = 1
            onset = (ieeg_onsets[first_idx] + ieeg_onsets[first_idx + 1])/2 # take time between the two seizures            
        except Exception as e:
            print(f"Failed to calculate onset time for patient {pt}: {str(e)}")
            onset = 10000
        if pt == 'HUP078':
            onset = 171114
        if pt == 'HUP215':
            ieeg_name = 'HUP215_phaseII_D03'
            onset = 371515
        offset = onset + 600

        task = f"interictal{int(onset)}"
        # if pt in ('HUP065','HUP078','HUP126','HUP221','HUP276'):
        #     offset = onset + 1200
        #     task = f"validation{int(onset)}"
        
        # Throwing error because there are no seizures that exist for this patient. So one option would be to save BIDS IEEGIDs into the config and access that from there.
        ieegid = int(seizures_df.loc[seizures_df.IEEGname == ieeg_name,'IEEGID'].mode())
        # get bids path
        clip_bids_path = bids_path.copy().update(
            subject=pt,
            run=ieegid,
            task=task,
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
        signal_len = int(data_np.shape[1]/fs*TARGET)
        data_np_ds = sc.signal.resample(data_np,signal_len,axis=1)
        fs = TARGET

        # save the data
        # run is the iEEG file number
        # task is ictal with the start time in seconds appended
        data_info = mne.create_info(
            ch_names=list(data.columns), sfreq=fs, ch_types= "seeg", verbose=False
        )
        
        raw = mne.io.RawArray(
            data_np_ds / 1e6,  # mne needs data in volts,
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

if __name__ == "__main__":
    main()