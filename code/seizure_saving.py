### SAVING SEIZURES AS PICKLE FILES TO LEIF
import numpy as np
import pandas as pd
import json
import os
from os.path import join as ospj
from utils import *
import scipy as sc

# Loading CONFIG
with open('config.json','r') as f:
    CONFIG = json.load(f)
usr = CONFIG["paths"]["iEEG_USR"]
passpath = CONFIG["paths"]["iEEG_PWD"]
datapath = CONFIG["paths"]["RAW_DATA"]
figpath = CONFIG["paths"]["FIGURES"]
pt_table = pd.DataFrame(CONFIG["patients"]).sort_values('ptID')
rid_hup = pd.read_csv(ospj(datapath,'rid_hup.csv'))
pt_list = pt_table.ptID.to_numpy()
lf_pt_list = pt_list[pt_table.lf_stim==1]

np.random.seed(42)

# downsampling factor
def get_factor(fs,target=512):
    if fs%target != 0:
        print("FS not divisible by target, will perform \
              integer division and new fs may not match target")
    return fs // target
TARGET = 512

# Iterate through each patient
for pt in lf_pt_list:
    print(f"Starting Seizure Preprocessing for {pt}")
    try:
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
        ch_names = electrode_localizations[(electrode_localizations['index'] == 2) | (electrode_localizations['index'] == 3)]["name"].to_numpy()

        # loading seizures
        if not os.path.exists(ospj(raw_datapath, "seizures")):
            os.mkdir(ospj(raw_datapath, "seizures"))
        
        # Iterate through each seizure in pre-defined pkl file
        for i_sz,row in seizure_times.iterrows():
            print(f"Saving seizure number: {i_sz}")
            seizure,fs = get_iEEG_data(usr,passpath,
                                        row.IEEGname,
                                        row.start*1e6,
                                        row.end*1e6,
                                        ch_names,
                                        force_pull = True)
            factor = get_factor(fs,TARGET)
            fsd = fs//factor
            seizure_ds = pd.DataFrame(sc.signal.decimate(seizure.to_numpy(),FACTOR,axis=0),columns=ch_names)           
            seizure_ds.to_pickle(ospj(raw_datapath,"seizures",f"{fsd}_seizure_{i_sz}_stim_{row.stim}.pkl"))
    except:
        print(f"unable to save seizures for {pt}")
