# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
from tqdm import tqdm
from kneed import KneeLocator

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Imports for deep learning
from sklearn.preprocessing import RobustScaler

# OS imports
from os.path import join as ospj
from utils import *
import sys
from seizure_detection_pipeline_pre_train import *

sys.path.append('/users/wojemann/iEEG_processing')
plt.rcParams['image.cmap'] = 'magma'

sys.path.append('/users/wojemann/DSOSD/')
from DSOSD.model import NDD

_,_,datapath,prodatapath,metapath,figpath,patient_table,rid_hup,_ = load_config(ospj('/mnt/sauce/littlab/users/wojemann/stim-seizures/code','config.json'))
seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))
consensus_annots = pd.read_pickle(ospj(prodatapath,"threshold_tuning_consensus.pkl"))
consensus_annots['Patient'] = consensus_annots['patient']
consensus_annots.sort_values('approximate_onset',inplace=True)

dict_list = []
pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
for _,row in pbar:
    try:
        pt = row.ptID
        pbar.set_description(desc=f"Patient: {pt}",refresh=True)
        
        # Skipping if no training data has been identified
        if len(row.interictal_training) == 0:
            continue

        # Preprocess the signal
        target=128

        ### ONLY PREDICTING FOR SPONTANEOUS SEIZURES THAT HAVE BEEN ANNOTATED
        seizure_times = seizures_df[(seizures_df.Patient == pt) & (seizures_df.to_annotate == 1) & (seizures_df.stim == 0)]
        ###

        sz_times_wannots = pd.merge_asof(seizure_times,
                                    consensus_annots[['approximate_onset','Patient','all_chs','ueo_consensus','ueo_any','sec_consensus','sec_any','ueo_time_consensus']],
                                    on='approximate_onset',by='Patient',
                                    tolerance = 240,
                                    direction='nearest')
                                    
        sz_times_wannots['ueo_chs'] = sz_times_wannots.apply(lambda x: clean_labels(np.array(x.all_chs)[x.ueo_consensus],''),axis=1)
        sz_times_wannots['sec_chs'] = sz_times_wannots.apply(lambda x: clean_labels(np.array(x.all_chs)[x.sec_consensus],''),axis=1)

        # Iterating through each seizure for that patient
        qbar = tqdm(sz_times_wannots.iterrows(),total=len(sz_times_wannots),leave=False)
        for i,(_,sz_row) in enumerate(qbar):
            if (pt == 'CHOP037') & (sz_row.approximate_onset == 962082.12):
                continue
            

            seizure_raw,fs_raw = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=False, verbose=0)

            chn_labels = remove_scalp_electrodes(seizure_raw.columns)
            seizure_neural = seizure_raw.loc[:,chn_labels]

            seizure_prep,fs = preprocess_for_detection(seizure_neural,fs_raw,
                                                wavenet=False,
                                                pre_mask=[],
                                                target=128)
            
            # seizure_prep.max()
            # noisy_channel_mask = seizure_prep.abs().max() <= (np.median(seizure_prep.abs().max())*50)
            # seizure_prep.columns[~noisy_channel_mask]

            scl = RobustScaler()
            scl.fit(seizure_prep.iloc[:fs*60,])
            
            seizure_z = pd.DataFrame(scl.transform(seizure_prep),columns=seizure_prep.columns)
            seizure_z.columns = [x.split('-')[0] for x in seizure_z.columns]

            approx_time = sz_row.approximate_onset
            consensus_time = sz_row.ueo_time_consensus
            
            # identifying difference between annotator and approximate time
            time_diff = consensus_time - approx_time

            onset_time = 120 + time_diff

            # Find closest index to consensus onset time relative to actual onset time (consensus - approximate and find closest to 120 + diff)

            for ch in sz_row.ueo_chs:
                if ch not in seizure_z.columns:
                    continue
                dict_list.append({
                    'Patient': sz_row.Patient,
                    'approximate_onset': sz_row.approximate_onset,
                    'channel': ch,
                    'type': 'ueo',
                    'signal': seizure_z.loc[int(onset_time*fs):int((onset_time+2)*fs),ch].to_numpy()
                    })

            for ch in sz_row.sec_chs:
                if ch not in seizure_z.columns:
                    continue
                dict_list.append({
                    'Patient': sz_row.Patient,
                    'approximate_onset': sz_row.approximate_onset,
                    'channel': ch,
                    'type': 'sec',
                    'signal': seizure_z.loc[int((onset_time+10)*fs):int((onset_time+12)*fs),ch].to_numpy()
                    })

            all_sig_chs = list(set(sz_row.ueo_chs + sz_row.sec_chs))
            for ch in all_sig_chs:
                if ch not in seizure_z.columns:
                    continue
                dict_list.append({
                    'Patient': sz_row.Patient,
                    'approximate_onset': sz_row.approximate_onset,
                    'channel': ch,
                    'type': 'inter',
                    'signal': seizure_z.loc[30*fs:32*fs,ch].to_numpy()
                    })
    except:
        print(f'Failed for {pt}')
        
viz_df = pd.DataFrame(dict_list)
viz_df.to_pickle(ospj(prodatapath,'onset_visualization_data.pkl'))