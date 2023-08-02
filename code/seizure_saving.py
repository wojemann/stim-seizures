### SAVING SEIZURES AS PICKLE FILES TO LEIF
import numpy as np
import pandas as pd
import json
import os
from os.path import join as ospj
from utils import *

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
for pt in ['HUP224']:#pt_list:
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

            seizure,fs = get_iEEG_data(usr,pass_path,
                                        row.IEEGname,
                                        row.start*1e6,
                                        row.end*1e6,
                                        ch_names,
                                        force_pull = True)
            save_seizure = pd.concat((seizure,pd.DataFrame(np.ones(len(seizure),)*fs,columns=['fs'])),axis = 1)
            # adding buffer for pre-ictal analysis of onset
            print(f"Adding buffer for seizure number: {i_sz}")
            temp,_ = get_iEEG_data(usr,pass_path,
                                    row.IEEGname,(row.start-15)*1e6,
                                    row.start*1e6,
                                    ch_names,
                                    force_pull=True)
            temp_fs = pd.concat((temp,pd.DataFrame(np.zeros(len(temp),),columns=['fs'])),axis = 1)
            
            buffered_seizure = pd.concat((temp_fs,save_seizure),axis=0)
            buffered_seizure.to_pickle(ospj(raw_datapath,"seizures",f"seizure_{i_sz}_stim_{row.stim}.pkl"))
    except:
        print(f"unable to save seizures for {pt}")
