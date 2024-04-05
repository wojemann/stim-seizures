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

plt.rcParams['image.cmap'] = 'magma'


# Functions for data formatting in autoregressive problem
# prepare_segment turns interictal/seizure clip into input and target data for autoregression
def prepare_segment(data, fs = 256,train_win = 12, pred_win = 1, w_size = 1, w_stride=0.5,ret_time=False):
    data_ch = data.columns.to_list()
    data_np = data.to_numpy()
    train_win = 12
    pred_win = 1
    j = fs-(train_win+pred_win)+1
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
def train_model(model,dataloader,criterion,optimizer,num_epochs=100,ccheck=False):
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

def plot_and_save_detection(mat,win_times,yticks,fig_save_path,xlim = None):
                fig,ax = plt.subplots(figsize=(48,24))
                plt.imshow(mat)
                plt.axvline(120,linestyle = '--',color = 'white')
                plt.xlabel('Time (s)')
                plt.yticks(np.arange(len(yticks)),yticks,rotation=0,fontsize=10)
                plt.xticks(np.arange(0,len(win_times),10),win_times.round(1)[np.arange(0,len(win_times),10)]-60)
                if xlim is not None:
                    plt.xlim(xlim)
                plt.savefig(fig_save_path)

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
    # This pipeline assumes that the seizures have already been saved following naming conventions
    # Please run XXXX.py to modify seizures for seizure detection. Future iterations may contain
    # preprocessing code to pull from a standardized saved seizure

    set_seed(5210)

    _,_,datapath,prodatapath,figpath,_,_,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'))

    all_seizure_times = pd.read_csv(ospj(prodatapath,"consensus_annots.csv"))
    montage = 'bipolar'
    train_win = 12
    pred_win = 1
    all_ueo_preds = {"Seizure_ID": [],
                     "patient": [],
                     "clinician": [],
                     "UEO_ch": []}
    # Iterating through each patient that we have annotations for
    for pt in all_seizure_times.patient.unique():
        print(f"Starting seizure detection pipeline for {pt}")
        seizure_times = all_seizure_times[all_seizure_times.patient == pt]
        raw_datapath = ospj(datapath,pt)
        if not os.path.exists(ospj(raw_datapath, "seizures")):
            os.mkdir(ospj(raw_datapath, "seizures"))
        
        fs = 256
        inter = pd.read_pickle(ospj(raw_datapath,"seizures",f"det{fs}_interictal_{montage}.pkl"))
        mask,_ = detect_bad_channels(inter.to_numpy(),fs)
        inter = inter.drop(inter.columns[~mask],axis=1)

        # Prepare input and target data for the LSTM
        input_data,target_data = prepare_segment(inter)

        dataset = TensorDataset(input_data, target_data)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

        # Instantiate the model
        input_size = input_data.shape[2]
        hidden_size = 10
        output_size = input_data.shape[2]

        # Check for cuda
        ccheck = torch.cuda.is_available()

        # Initialize the model
        model = LSTMModel(input_size, hidden_size, output_size)
        if ccheck:
            model.cuda()
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the model, this will just modify the model object, no returns
        print("Training patient specific model")
        train_model(model,dataloader,criterion,optimizer,ccheck=ccheck)

        # Creating classification thresholds
        print("Generating loss decision threshold")
        input_data,target_data = prepare_segment(inter)
        inter_outputs = predict_sz(model,input_data,target_data,batch_size=400,ccheck=ccheck)
        thresholds = np.percentile(inter_outputs,95,0)

        # Iterating through each seizure for that patient
        for _,sz_row in seizure_times.iterrows():
            i_sz = int(float(sz_row.Seizure_ID.split('_')[-1]))

            print(f"Generating predictions for seizure {i_sz}")

            seizure = pd.read_pickle(ospj(raw_datapath,"seizures",f"det{fs}_seizure_{i_sz}_stim_{int(sz_row.stim)}_{montage}.pkl"))
            seizure = seizure.drop(seizure.columns[~mask],axis=1)

            input_data, target_data, win_times = prepare_segment(seizure,fs,train_win,pred_win,ret_time=True)
            outputs = predict_sz(model,input_data,target_data,400,ccheck=ccheck)
            seizure_mat = repair_data(outputs,seizure)

            # Getting raw predicted loss values for each window
            raw_sz_vals = np.mean(np.log(seizure_mat),1).T
            # Creating classifications
            sz_clf = (raw_sz_vals.T > np.log(thresholds)).T
            # Dropping channels with too many positive detections (bad channels)
            # This should be replaced with actual channel rejection
            rejection_mask = np.sum(sz_clf[:,:120],axis=1) > 80
            sz_clf[rejection_mask,:] = 0 # fake channel rejection

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

            final_thresh = 0.5
            sz_clf_final = sz_prob > final_thresh
            first_seizing_index = np.argmax(sz_clf_final.any(axis=0))
            mdl_ueo_idx = np.where(np.sum(sz_clf_final[:, first_seizing_index:first_seizing_index + 3], axis=1) > 0)[0]
            mdl_ueo_ch_bp = seizure.columns.to_numpy()[mdl_ueo_idx]
            mdl_ueo_ch = [s.split("-")[0] for s in mdl_ueo_ch_bp]
            all_ueo_preds["UEO_ch"].append(mdl_ueo_ch)
            all_ueo_preds["clinician"].append("MDL")
            all_ueo_preds["patient"].append(pt)
            all_ueo_preds["Seizure_ID"].append(sz_row.Seizure_ID)
            # (115,400)
            # In this section plot and save all of the plots that we generate in this section.
            if ~ospe(ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM")):
                os.makedirs(ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM"),exist_ok=True)
            plot_and_save_detection(sz_vals[ch_sorting,:],
                                    win_times,
                                    seizure.columns[ch_sorting],
                                    ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM",f"{montage}_loss_vals.png"))
            plot_and_save_detection(sz_prob[ch_sorting,:],
                                    win_times,
                                    seizure.columns[ch_sorting],
                                    ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM",f"{montage}_sz_prob.png"))
            plot_and_save_detection(sz_clf[ch_sorting,:],
                                    win_times,
                                    seizure.columns[ch_sorting],
                                    ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM",f"{montage}_sz_clf.png"),xlim=(115,400))
            plot_and_save_detection(sz_clf_final[ch_sorting,:],
                                    win_times,
                                    seizure.columns[ch_sorting],
                                    ospj(figpath,pt,"annotation_demo",sz_row.Seizure_ID,"LSTM",f"{montage}_sz_clf_final_{final_thresh}.png"))
    all_ueo_preds_df = pd.DataFrame(all_ueo_preds)
    all_ueo_preds_df.to_csv(ospj(prodatapath,f"annotation_demo_algorithm_{montage}.csv"),index=False)
            


    

if __name__ == "__main__":
    main()