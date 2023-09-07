# Standard imports
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# OS imports
import os
from os.path import join as ospj
from os.path import exists as ospe
import pathlib
from tqdm import tqdm
from pqdm.processes import pqdm
from utils import *
from dtw_utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')
import mne

# loading config data
# Loading metadata
with open('/mnt/leif/littlab/users/wojemann/stim-seizures/code/config.json','r') as f:
    CONFIG = json.load(f)
usr = CONFIG["paths"]["iEEG_USR"]
passpath = CONFIG["paths"]["iEEG_PWD"]
datapath = CONFIG["paths"]["RAW_DATA"]
metadatapath = CONFIG["paths"]["META"]
prodatapath = CONFIG["paths"]["PROCESSED_DATA"]
ieeg_list = CONFIG["patients"]
rid_hup = pd.read_csv(ospj(datapath,'rid_hup.csv'))
pt_list = np.unique(np.array([i.split("_")[0] for i in ieeg_list]))

metadata = pd.read_csv(ospj(metadatapath,'metadata_wchreject.csv'))
# (metadata.cceps_hfs_seizure == 1) | 
metadata = metadata[(metadata.cceps_run1_sz == 1)].reset_index()
metadata.loc[:,'ieeg_id'] = 'HUP' + metadata.hupsubjno.apply(str) + '_phaseII'
metadata.loc[:,'ccep_id'] = 'HUP' + metadata.hupsubjno.apply(str) + '_CCEP'


alt_pt_list = [pt for pt in pt_list if pt not in ["HUP247"]]
# alt_pt_list = ["HUP235"]
for pt in alt_pt_list:
    print(f"Starting analysis for {pt}")

    raw_datapath = ospj(datapath,pt)
    processed_datapath = ospj(prodatapath,pt)
    if not ospe(processed_datapath):
        os.mkdir(processed_datapath)
    dirty_drop_electrodes = metadata[metadata.hupsubjno == int(pt[-3:])]["final_reject_channels"].str.split(',').to_list()[0]
    if isinstance(dirty_drop_electrodes,list):
            final_drop_electrodes = clean_labels(dirty_drop_electrodes,pt)
    else:
        final_drop_electrodes = []

    seizure_list = np.sort([s for s in os.listdir(ospj(raw_datapath, "seizures")) if 'preprocessed' in s])

    all_seizures = []
    all_ts = []
    for seizure_path in seizure_list:
        seizure_fs = pd.read_pickle(ospj(raw_datapath,"seizures",seizure_path))
        fs = seizure_fs.fs.to_numpy()[-1]
        seizure = seizure_fs.drop("fs",axis=1)
        cols = seizure.columns.to_list()
        clean_ch = [c for c in cols if c not in final_drop_electrodes]
        seizure = seizure.loc[:,clean_ch]
        all_seizures.append(seizure)

    indexed_seizures = [[i,all_seizures[i]] for i in range(len(all_seizures))]
    # pt_cohs = pqdm(indexed_seizures,calculate_coh_timeseries,n_jobs=32)
    pt_cohs = []
    for seizure in indexed_seizures:
        pt_cohs.append(parallel_coh_timeseries(seizure))
    with open(ospj(prodatapath,pt,f"{pt}_seizure_networks.pkl"),'wb') as f:
        pickle.dump({"seizure_list": seizure_list, "seizure_networks": pt_cohs},f)


    # pt_cohs = calculate_coh_timeseries(indexed_seizures[0])
    # with open(ospj(prodatapath,pt,f"script_test_networks.pkl"),'wb') as f:
    #     pickle.dump(pt_cohs,f)