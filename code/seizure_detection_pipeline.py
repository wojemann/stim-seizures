# iEEG imports
from ieeg.auth import Session

# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
import json
from scipy.linalg import hankel
from tqdm import tqdm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import RobustScaler

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Imports for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# OS imports
import os
from os.path import join as ospj
from os.path import exists as ospe
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')

OVERWRITE = False
FACTOR = 2
TRAIN_WIN = 12
PRED_WIN = 1

# Functions for data formatting in autoregressive problem
# prepare_segment turns interictal/seizure clip into input and target data for autoregression
def prepare_segment(data, fs = 256,train_win = 12, pred_win = 1, w_size = 1, w_stride=0.5,ret_time=False):
    data_ch = data.columns.to_list()
    data_np = data.to_numpy()
    train_win = 12
    pred_win = 1
    j = int(fs-(train_win+pred_win)+1)
    nwins = num_wins(len(data_np[:,0]),fs,w_size,w_stride)
    data_mat = torch.zeros((nwins,j,(train_win+pred_win),len(data_ch)))
    for k in range(len(data_ch)): # Iterating through channels
        samples = MovingWinClips(data_np[:,k],fs,1,0.5)
        for i in range(samples.shape[0]):
            clip = samples[i,:]
            mat = torch.tensor(hankel(clip[:j],clip[-(train_win+pred_win):]))
            data_mat[i,:,:,k] = mat
    time_mat = MovingWinClips(np.arange(len(data))/fs,fs,1,0.5)
    win_times = time_mat[:,0]
    data_flat = data_mat.reshape((-1,train_win + pred_win,len(data_ch)))
    input_data = data_flat[:,:-1,:].float()
    target_data = data_flat[:,-1,:].float()
    if ret_time:
        return input_data, target_data, win_times
    else:
        return input_data, target_data

# predict_sz returns formatted data windows as distributions of MSE loss for each clip
def predict_sz(model, input_data, target_data,batch_size=1,ccheck=False):
    dataset = TensorDataset(input_data,target_data)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    ccheck = torch.cuda.is_available()
    if ccheck:
        model.cuda()
    with torch.no_grad():
        model.eval()
        mse_distribution = []
        for inputs, targets in dataloader:
            if ccheck:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            mse = (outputs-targets)**2
            mse_distribution.append(mse)
            del inputs, targets, outputs, mse
    return torch.cat(mse_distribution).cpu().numpy()

# repair_data turns the clip x sample output of predict_sz back into a channel x window time multivariate time series
def repair_data(outputs,data,fs=256,train_win=12,pred_win=1,w_size=1,w_stride=.5):
    nwins = num_wins(len(data.to_numpy()[:,0]),fs,w_size,w_stride)
    nchannels = data.shape[1]
    repaired = outputs.reshape((nwins,fs-(train_win + pred_win)+1,nchannels))
    return repaired

# Define LSTM and LR models
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1,:])
        return out
    def __str__(self):
         return "LSTM"
    
class LRModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Flatten the input along the 'sequence length' dimension
        x = x.squeeze()
        out = self.linear(x)
        return out
    def __str__(self):
        return "LR"

# Train the model instance using provided data
def train_model(model,dataloader,criterion,optimizer,num_epochs=100,ccheck=False):
        # Training loop
        tbar = tqdm(range(num_epochs),leave=False)
        for e in tbar:
            for inputs, targets in dataloader:
                if ccheck:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                del inputs, targets, outputs
            if e % 10 == 9:
                tbar.set_description(f"{loss.item():.4f}")
                del loss

# Prepare univariate features for classification by 0-1 normalizing excluding outliers
def scale_normalized(data,m=5):
    # takes in data and returns a flattened array with outliers removed based on distribution of entire tensor
    data_flat = data.flatten()
    d = np.abs(data_flat - np.median(data_flat))
    mdev = np.median(d)
    s = d / mdev
    scaler = np.max(data_flat[s<m])
    data_norm = data/scaler
    data_norm[data_norm > 1] = 1
    return data_norm

def main():
    # This pipeline assumes that the seizures have already been saved following BIDS file structure
    # Please run BIDS_seizure_saving.py and BIDS_interictal_saving.py to modify seizures for seizure detection.
    _,_,datapath,prodatapath,_,patient_table,_,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'))

    seizures_df = pd.read_csv(ospj(datapath,"stim_seizure_information_BIDS.csv"))

    montage = 'bipolar'
    train_win = TRAIN_WIN
    pred_win = PRED_WIN

    # Iterating through each patient that we have annotations for
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:
        pt = row.ptID
        pbar.set_description(desc=f"Patient: {pt}",refresh=True)
        if len(row.interictal_training) == 0:
            continue
        inter,fs = get_data_from_bids(ospj(datapath,"BIDS"),pt,'interictal')
        chn_labels = remove_scalp_electrodes(inter.columns)
        inter = inter[chn_labels]

        mask,_ = detect_bad_channels(inter.to_numpy(),fs)
        inter = inter.drop(inter.columns[~mask],axis=1)

        # Preprocess the signal
        inter, fs = preprocess_for_detection(inter,fs,montage,2)

        # Prepare input and target data for the LSTM
        input_data,target_data = prepare_segment(inter)

        dataset = TensorDataset(input_data, target_data)
        full_batch = len(dataset)
        dataloader = DataLoader(dataset, batch_size=full_batch, shuffle=False)

        # Instantiate the model
        input_size = input_data.shape[2]
        hidden_size = 100
        output_size = input_data.shape[2]

        # Check for cuda
        ccheck = torch.cuda.is_available()
        set_seed(1071999)
        # Initialize the model
        model = LSTMModel(input_size, hidden_size, output_size)
        if ccheck:
            model.cuda()
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the model, this will just modify the model object, no returns
        # print("Training patient specific model")
        train_model(model,dataloader,criterion,optimizer,ccheck=ccheck)

        # Creating classification thresholds
        # print("Generating loss decision threshold")
        input_data,target_data = prepare_segment(inter)
        inter_outputs = predict_sz(model,input_data,target_data,batch_size=full_batch,ccheck=ccheck)
        thresholds = np.percentile(inter_outputs,90,0)

        # Iterating through each seizure for that patient
        seizure_times = seizures_df[seizures_df.Patient == pt]
        qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),leave=False)
        for i,(_,sz_row) in enumerate(qbar):
            set_seed(1071999)
            qbar.set_description(f"Processing seizure {i}")
            seizure,fs_raw, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
            seizure = seizure[chn_labels]
            seizure = seizure.drop(seizure.columns[~mask],axis=1)
            prob_path = f"probability_matrix_mdl-{model}_fs-{int(fs_raw/FACTOR)}_montage-{montage}_task-{task}_run-{run}.pkl"
            if (not OVERWRITE) and ospe(ospj(prodatapath,pt,prob_path)):
                 continue
            seizure, fs = preprocess_for_detection(seizure,fs_raw,montage,factor=FACTOR)

            input_data, target_data,time_wins = prepare_segment(seizure,fs,train_win,pred_win,ret_time=True)
            outputs = predict_sz(model,input_data,target_data,batch_size=len(input_data)//2,ccheck=ccheck)
            seizure_mat = repair_data(outputs,seizure)

            # Getting raw predicted loss values for each window
            raw_sz_vals = np.mean(np.log(seizure_mat),1).T
            # Creating classifications
            sz_clf = (raw_sz_vals.T > np.log(thresholds)).T
            # Dropping channels with too many positive detections (bad channels)
            # This should be replaced with actual channel rejection
            rejection_mask = np.sum(sz_clf[:,:120],axis=1) > 60
            sz_clf[rejection_mask,:] = 0 # fake channel rejection

            # Creating probabilities by temporally smoothing classification
            sz_prob = sc.ndimage.uniform_filter1d(sz_clf.astype(float),10,axis=1)
            sz_prob_df = pd.DataFrame(sz_prob.T,columns = seizure.columns)
            time_df = pd.Series(time_wins,name='time')
            sz_prob_df = pd.concat((sz_prob_df,time_df),axis=1)
            os.makedirs(ospj(prodatapath,pt),exist_ok=True)
            sz_prob_df.to_pickle(ospj(prodatapath,pt,prob_path))
            # np.save(ospj(prodatapath,pt,prob_path),sz_prob)
            np.save(ospj(prodatapath,pt,f"raw_preds_mdl-{model}_fs-{fs}_montage-{montage}_task-{task}_run-{run}.npy"),sz_clf)
        del model
if __name__ == "__main__":
    main()