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
import random
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

plt.rcParams['image.cmap'] = 'magma'


# Functions for data formatting in autoregressive problem
# prepare_segment turns interictal/seizure clip into input and target data for autoregression
def prepare_segment(data, fs = 256,train_win = 12, pred_win = 1, w_size = 1, w_stride=0.5,ret_time=False):
    data_ch = data.columns.to_list()
    data_np = data.to_numpy()
    train_win = 12
    pred_win = 1
    j = fs-(train_win+pred_win)+1
    nwins = num_wins(data_np[:,0],fs,w_size,w_stride)
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
def predict_sz(model, input_data, target_data,batch_size=1):
    dataset = TensorDataset(input_data,target_data)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    ccheck = torch.cuda.is_available()
    if ccheck:
        model.cuda()
    with torch.no_grad():
        model.eval()
        mse_distribution = []
        for inputs, targets in tqdm(dataloader):
            if ccheck:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            mse = (outputs-targets)**2
            mse_distribution.append(mse)
    return torch.cat(mse_distribution).cpu().numpy()

# repair_data turns the clip x sample output of predict_sz back into a channel x window time multivariate time series
def repair_data(outputs,data,fs=256,train_win=12,pred_win=1,w_size=1,w_stride=.5):
    nwins = num_wins(data.to_numpy()[:,0],fs,w_size,w_stride)
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
    
class LRModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Flatten the input along the 'sequence length' dimension
        x = x.squeeze()
        out = self.linear(x)
        return out

# Train the model instance using provided data
def train_model(model,dataloader,criterion,optimizer,num_epochs=100):
        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
                if ccheck:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 9:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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

def load_config(config_path):
    with open('config.json','r') as f:
        CONFIG = json.load(f)
    usr = CONFIG["paths"]["iEEG_USR"]
    passpath = CONFIG["paths"]["iEEG_PWD"]
    datapath = CONFIG["paths"]["RAW_DATA"]
    prodatapath = CONFIG["paths"]["PROCESSED_DATA"]
    figpath = CONFIG["paths"]["FIGURES"]
    patient_table = pd.DataFrame(CONFIG["patients"]).sort_values('ptID')
    rid_hup = pd.read_csv(ospj(datapath,'rid_hup.csv'))
    pt_list = patient_table.ptID.to_numpy()
    return usr,passpath,datapath,prodatapath,figpath,patient_table,rid_hup,pt_list
def main():
    # This pipeline assumes that the seizures have already been saved following naming conventions
    # Please run XXXX.py to modify seizures for seizure detection. Future iterations may contain
    # preprocessing code to pull from a standardized saved seizure

    inter_times = {"HUP235": 307651,
                   "HUP238": 100011,
                   "HUP246": 100000}

    set_seed(5210)

    _,_,datapath,prodatapath,figpath,patient_table,pt_list = load_config('config.json')

    all_seizure_times = pd.read_csv(ospj(prodatapath,"consensus_annots.csv"))
    montage = 'bipolar'
    train_win = 12
    pred_win = 1
    # Iterating through each patient that we have annotations for
    for pt in all_seizure_times.patient.unique():
        seizure_times = all_seizure_times[all_seizure_times.patient == pt]

        raw_datapath = ospj(datapath,pt)
        if not os.path.exists(ospj(raw_datapath, "seizures")):
            os.mkdir(ospj(raw_datapath, "seizures"))
        
        fs = 256
        inter = pd.read_pickle(ospj(raw_datapath,"seizures",f"det{fs}_interictal_{montage}.pkl"))

        # Prepare input and target data for the LSTM
        input_data,target_data = prepare_segment(inter)

        dataset = TensorDataset(input_data, target_data)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

        # Instantiate the model
        input_size = input_data.shape[2]
        hidden_size = 10
        output_size = input_data.shape[2]

        # Initialize the model
        model = LSTMModel(input_size, hidden_size, output_size)
        print(model)
        ccheck = torch.cuda.is_available()
        if ccheck:
            model.cuda()
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the model, this will just modify the model object, no returns
        train_model(model,dataloader,criterion,optimizer)

        # Creating classification thresholds
        input_data,target_data = prepare_segment(inter)
        inter_outputs = predict_sz(model,input_data,target_data,batch_size=400)
        thresholds = np.percentile(inter_outputs,95,0)

        # Iterating through each seizure for that patient
        for _,sz_row in seizure_times.iterrows():
            i_sz = int(float(sz_row.Seizure_ID[7:]))
            seizure = pd.read_pickle(ospj(raw_datapath,"seizures",f"det{fs}_seizure_{i_sz}_stim_{int(sz_row.stim)}_{montage}.pkl"))
            mask,_ = detect_bad_channels(seizure.to_numpy(),fs)

            input_data, target_data, win_times = prepare_segment(seizure,fs,train_win,pred_win,ret_time=True)
            outputs = predict_sz(model,input_data,target_data,400)
            seizure_mat = repair_data(outputs,seizure)

            # Getting raw predicted loss values for each window
            raw_sz_vals = np.mean(np.log(seizure_mat),1).T
            # Creating classifications
            sz_clf = (raw_sz_vals.T > np.log(thresholds)).T
            # Dropping channels with too many positive detections (bad channels)
            # This should be replaced with actual channel rejection
            rejection_mask = np.sum(sz_clf[120],axis=1) > (sz_clf.shape[1]*3/5)
            sz_clf[rejection_mask,:] = 0 # fake channel rejection
            sz_clf[~mask,:] = 0 # real channel rejection

            # Normalizing values of the loss
            norm_sz_vals = scale_normalized(np.mean(np.log(seizure_mat),1).T)
            # Creating smoothed sz values
            sz_vals = sc.ndimage.uniform_filter1d(raw_sz_vals,10,axis=1)
            # Creating probabilities by temporally smoothing classification
            sz_prob = sc.ndimage.uniform_filter1d(sz_clf.astype(float),10,axis=1)
            # Sorting channels based on probability at sz onset
            first_detect = np.argmax(sz_prob[:,115:]>.5,axis=1)
            first_detect[first_detect == 0] = sz_prob.shape[1]
            ch_sorting = np.argsort(first_detect)

            # rejecting noisy/late channels
            bottom_mask = np.sum(sz_clf[ch_sorting,:],axis=1) > 0
            first_zero = np.where(~bottom_mask)[0][0].astype(int)
            sz_clf[ch_sorting[first_zero:],:] = 0
            sz_prob[ch_sorting[first_zero:],:] = 0

            sz_clf_final = sz_prob > 0.5
            first_seizing_index = np.argmax(sz_clf_final.any(axis=0))
            mdl_ueo_idx = np.where(np.sum(sz_clf_final[:, first_seizing_index:first_seizing_index + 3], axis=1) > 0)[0]
            mdl_ueo_ch_bp = seizure.columns.to_numpy()[mdl_ueo_idx]
            mdl_ueo_ch = [s.split("-")[0] for s in mdl_ueo_ch_bp]

            def plot_and_save_detection(mat,win_times,)
            


    

if __name__ == "__main__":
    main()