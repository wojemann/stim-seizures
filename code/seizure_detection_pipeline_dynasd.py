### SEIZURE DETECTION PIPELINE - PRE-TRAINED MODELS
"""
Script for seizure detection using pre-trained models (LSTM, AbsSlope, WaveNet).
Trains and applies multiple seizure detection algorithms to iEEG data in BIDS format.
Each model is trained on interictal data and tested on seizure recordings to generate
probability matrices for downstream analysis and visualization.

The pipeline supports three detection methods:
1. LSTM - Long Short-Term Memory autoregressive model
2. AbsSlope - Absolute slope feature-based detector  
3. WaveNet - Pre-trained convolutional neural network

Output: Probability matrices saved as pickled DataFrames with seizure onset visualizations.
"""

# Scientific computing imports
import os as _os
_os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARN messages
_os.environ['TF_TRT_DISABLED'] = '1'       # Silence TF-TRT warnings if TensorRT not installed

import numpy as np
import pandas as pd
from scipy.linalg import hankel
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Deep learning imports  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.models import load_model
from tensorflow.config.experimental import set_memory_growth, list_physical_devices
import tensorflow as tf
from absl import logging as absl_logging

# Suppress TensorFlow logging
absl_logging.set_verbosity(absl_logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Configure GPU memory growth to prevent allocation issues
try:
    for _gpu in list_physical_devices('GPU'):
        set_memory_growth(_gpu, True)
except Exception:
    pass

# File system and utility imports
import os
from os.path import join as ospj
from os.path import exists as ospe
from utils import *
from stim_seizure_preprocessing_utils import *

import sys
sys.path.append('/users/wojemann/DynaSD')
from DynaSD import NDD,GIN,LiNDDA,MINDA
from config import Config



# Get paths from config
datapath = Config.datapath
prodatapath = Config.prodatapath
figpath = Config.figpath


# Set default colormap for visualizations
plt.rcParams['image.cmap'] = 'magma'

# Global configuration
OVERWRITE = False  # Whether to overwrite existing probability matrix files

def plot_and_save_detection(mat,win_times,yticks,fig_save_path,xlim = None):
    """
    Create and save seizure detection heatmap with time axis and channel labels.
    
    Generates a comprehensive visualization showing seizure probability over time
    for each channel, with seizure onset marked and proper axis labeling.
    
    Parameters:
    -----------
    mat : numpy.ndarray
        Seizure probability matrix (channels x time_windows)
    win_times : numpy.ndarray
        Time stamps for each window
    yticks : list
        Channel labels for y-axis
    fig_save_path : str
        Path to save the generated figure
    xlim : tuple, optional
        X-axis limits for zooming
    """
    # plt.subplots(figsize=(48,24))
    plt.imshow(mat)
    plt.axvline(np.argwhere(np.ceil(win_times)==120)[0])

    plt.xlabel('Time (s)')
    plt.yticks(np.arange(len(yticks)),yticks,rotation=0,fontsize=10)
    plt.xticks(np.arange(0,len(win_times),10),win_times.round(1)[np.arange(0,len(win_times),10)]-120)
    if xlim is not None:
        plt.xlim(xlim)
    plt.clim([0,4])
    plt.savefig(fig_save_path)

def plot_and_save_detection_figure(mat,win_times,yticks,fig_save_path,xlim = None,cmap=False):
    # plt.subplots(figsize=(48,24))
    plot_onset_lower = np.argwhere(np.ceil(win_times)==120)[0]
    plot_onset_upper = np.argwhere(np.ceil(win_times)==210)[0] if max(win_times) > 210 else mat.shape[1]
    plt.imshow(mat[:,int(plot_onset_lower):int(plot_onset_upper)],cmap=cmap)

    plt.xticks([])
    plt.yticks([])
    if xlim is not None:
        plt.xlim(xlim)
    plt.clim([0,1])
    plt.savefig(fig_save_path,bbox_inches='tight')

def main():
    """
    Main seizure detection pipeline using pre-trained models.
    
    Workflow:
    1. Load configuration and seizure metadata from BIDS format
    2. Configure GPU settings for optimal performance  
    3. For each patient with available data:
       - Load interictal training data from BIDS
       - Clean electrode labels and localize neural channels
       - For each seizure recording:
         - Train model on interictal/early seizure data
         - Generate predictions across full seizure recording
         - Save probability matrices and visualizations
    4. Support three detection algorithms: LSTM, AbsSlope, WaveNet
    
    Models are trained patient-specifically on interictal data and applied
    to detect seizure onset patterns in ictal recordings.
    """
    # Configure GPU memory growth for TensorFlow/PyTorch compatibility
    gpus = list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Load configuration and paths
    _,_,datapath,prodatapath,metapath,figpath,patient_table,rid_hup,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'),None)

    # Load seizure metadata from BIDS processing
    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))

    # Detection parameters
    onset_time = 120          # Seizure onset time in recording (seconds)
    montage = 'bipolar'       # Electrode montage for preprocessing  
    train_win = 12           # Training window length (time points)
    pred_win = 1             # Prediction window length (time points)
    num_epochs = 10          # LSTM training epochs
    all_mdl_strs = ['AbsSlp','WVNT','LSTM']  # Models to run

    # Iterating through each patient that we have annotations for
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:
        pt = row.ptID
        pbar.set_description(desc=f"Patient: {pt}",refresh=True)
        
        # Skipping if no training data has been identified
        if len(row.interictal_training) == 0:
            continue

        # Loading data from bids
        inter_raw,fs_raw = get_data_from_bids(ospj(datapath,"BIDS"),pt,'interictal')

        # Pruning channels
        chn_labels = remove_scalp_electrodes(inter_raw.columns)
        inter_raw = inter_raw[chn_labels]

        try: # channel localization exception catch
            electrode_localizations,electrode_regions = electrode_wrapper(pt,rid_hup,datapath)
            if pt[:3] == 'CHO':
                suffix = ['CHOPR','CHOPM']
            else:
                suffix = ['dkt','atropos']

            electrode_localizations.name = clean_labels(electrode_localizations.name,pt) #don't end up using grey/white matter
            electrode_regions.name = clean_labels(electrode_regions.name,pt)
            electrode_localizations.to_pickle(ospj(prodatapath,pt,f'electrode_localizations_{suffix[1]}.pkl')) #don't end up using grey/white matter
            electrode_regions.to_pickle(ospj(prodatapath,pt,f'electrode_localizations_{suffix[0]}.pkl'))
            neural_channels = electrode_localizations.name[(electrode_localizations.name.isin(inter_raw.columns)) & ((electrode_localizations.label == 'white matter') | (electrode_localizations.label == 'gray matter'))]
        
        except:
            print(f"electrode localization failed for {pt}")
            neural_channels = chn_labels
        inter_neural = inter_raw.loc[:,neural_channels]
        
        # get baseline stds for stimulation artifact interpolation
        baseline_stds = inter_neural.std().to_numpy()
        wvcheck = mdl_str=='WVNT'
        # Preprocess the signal
        target=128

        seizure_times = seizures_df[seizures_df.Patient == pt]

        # Iterating through each seizure for that patient
        qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),leave=False)
        for i,(_,sz_row) in enumerate(qbar):
            if (pt == 'CHOP037') & (sz_row.approximate_onset == 962082.12):
                continue
            set_seed(1071999)
            qbar.set_description(f"{mdl_str} processing seizure {i}")
            # Load in seizure and metadata for BIDS path
            seizure,fs_raw, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)

            # Filter out bad channels from interictal clip
            seizure = seizure[neural_channels]

            # Interpolating stimulation artifact
            if sz_row.stim == 1:
                stim_chs = np.zeros((len(seizure.columns),),dtype=bool)
                for ch in sz_row.stim_channels.split('-'):
                    ch = clean_labels([ch],pt)[0]
                    stim_chs += np.array([ch == c for c in seizure.columns])
                pk_idxs,_ = stim_detect(seizure,threshold=baseline_stds*100,fs=fs_raw)
                seizure = barndoor(seizure,pk_idxs,fs_raw,plot=False)
                seizure = seizure.iloc[:,~stim_chs]
            inter_pre, fs, mask = preprocess_for_detection(inter_neural,fs_raw,montage,target=target,wavenet=False,pre_mask = None)
            # Preprocess seizure for seizure detection task
            seizure_pre, fs = preprocess_for_detection(seizure,fs_raw,montage,target=target,wavenet=False,pre_mask=mask)
            noisy_channel_mask = seizure_pre.loc[onset_time*fs:,:].abs().max() <= (np.median(seizure_pre.loc[onset_time*fs:,:].abs().max())*50)
            # noisy_channel_list = seizure_pre.columns[noisy_channel_mask].to_list()
            seizure_pre = seizure_pre.loc[:,noisy_channel_mask]
            if sz_row.stim == 1:    
                sz_train = inter_pre.loc[:,seizure_pre.columns]
            else:
                sz_train = seizure_pre.loc[:fs*60,:]
            for mdl in all_models:
                # Perform overwrite check
                prob_path = f"pretrain_probability_matrix_nosmooth_mdl-{str(mdl)}_fs-{int(fs)}_montage-{montage}_task-{task}_run-{run}.pkl"
                if (not OVERWRITE) and ospe(ospj(prodatapath,pt,prob_path)):
                    continue

                mdl = ...
                mdl.fit(sz_train.copy())
                sz_prob = mdl(seizure_pre.copy())

                sz_prob_df = pd.DataFrame(sz_prob.T,columns = seizure_pre.columns)
                time_df = pd.Series(time_wins,name='time')
                sz_prob_df = pd.concat((sz_prob_df,time_df),axis=1)
                os.makedirs(ospj(prodatapath,pt),exist_ok=True)
                sz_prob_df.to_pickle(ospj(prodatapath,pt,prob_path))
                
                ### Visualization
                detect_idx = np.argwhere(np.ceil(time_wins)==120)[0]
                first_detect = np.argmax(sz_prob[:,int(detect_idx):]>.5,axis=1)
                first_detect[first_detect == 0] = sz_prob.shape[1]
                ch_sorting = np.argsort(first_detect)
                colors = sns.color_palette("deep", len(all_mdl_strs))
                # Plot heatmaps for the first 4 colors
                cmap = LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1), colors[i_mdl]])
                os.makedirs(ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str),exist_ok=True)
                plot_and_save_detection_figure(sz_prob,
                                        time_wins,
                                        seizure.columns[ch_sorting],
                                        ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str,f"{montage}_sz_prob_colored.png"),
                                        cmap = cmap)
                plot_and_save_detection(sz_prob,
                                        time_wins,
                                        seizure.columns[ch_sorting],
                                        ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str,f"{montage}_sz_prob.png"),
                                        )
                del model
if __name__ == "__main__":
    main()