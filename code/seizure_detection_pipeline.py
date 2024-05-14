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

OVERWRITE = True
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

# Define LSTM and LR models
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def fit_scaler(self, x):
        self.scaler = RobustScaler().fit(x)

    def scaler_transform(self, x):
        return self.scaler.transform(x)
    
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

class AbsSlope():
    def __init__(self, win_size = 1, stride = 0.5, fs = 256):
        self.function = lambda x: np.mean(np.abs(np.diff(x,axis=-1)),axis=-1)
        self.win_size = win_size
        self.stride = stride
        self.fs = fs
    
    def __str__(self) -> str:
        return "AbsSlp"
        
    def fit(self, x):
        # x should be samples x channels df
        self.scaler = RobustScaler().fit(x)
        nx = self.scaler.transform(x)
        self.nstds = np.std(nx,axis=0)

    def get_times(self, x):
        # x should be samples x channels df
        time_mat = MovingWinClips(np.arange(len(x))/self.fs,self.fs,self.win_size,self.stride)
        return time_mat[:,0]

    def forward(self, x):
        # x is samples x channels df
        self.data = x
        x = self.scaler.transform(x)
        x = x.T
        slopes = ft_extract(x, self.fs, self.function, self.win_size, self.stride)
        scaled_slopes = slopes.squeeze()/np.expand_dims(self.nstds,1)*self.fs
        scaled_slopes = scaled_slopes.squeeze()
        normalized_slopes = scale_normalized(scaled_slopes)
        return normalized_slopes
    
    def __call__(self, *args):
        return self.forward(*args)

# preprocessing function wrapper
def electrode_wrapper(pt,rid_hup):
    hup_no = pt[3:]
    rid = rid_hup[rid_hup.hupsubjno == hup_no].record_id.to_numpy()[0]
    rid = str(rid)
    if len(rid) < 4:
        rid = '0' + rid
    recon_path = ospj('/mnt','leif','littlab','data',
                        'Human_Data','CNT_iEEG_BIDS',
                        f'sub-RID{rid}','derivatives','ieeg_recon',
                        'module3/')
    if not os.path.exists(recon_path):
        recon_path =  ospj('/mnt','leif','littlab','data',
                        'Human_Data','recon','BIDS_penn',
                        f'sub-RID{rid}','derivatives','ieeg_recon',
                        'module3/')
    electrode_localizations,electrode_regions = optimize_localizations(recon_path,rid)
    return electrode_localizations,electrode_regions

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

def plot_and_save_detection(mat,win_times,yticks,fig_save_path,xlim = None):
    plt.subplots(figsize=(48,24))
    plt.imshow(mat)
    plt.axvline(120,linestyle = '--',color = 'white')
    plt.xlabel('Time (s)')
    plt.yticks(np.arange(len(yticks)),yticks,rotation=0,fontsize=10)
    plt.xticks(np.arange(0,len(win_times),10),win_times.round(1)[np.arange(0,len(win_times),10)]-60)
    if xlim is not None:
        plt.xlim(xlim)
    plt.savefig(fig_save_path)

def main():
    # This pipeline assumes that the seizures have already been saved following BIDS file structure
    # Please run BIDS_seizure_saving.py and BIDS_interictal_saving.py to modify seizures for seizure detection.
    _,_,datapath,prodatapath,figpath,patient_table,rid_hup,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'))

    seizures_df = pd.read_csv(ospj(datapath,"stim_seizure_information_BIDS.csv"))

    montage = 'bipolar'
    train_win = TRAIN_WIN
    pred_win = PRED_WIN
    mdl_str = 'LSTM'
    # normalize = True
    # smearing = 20

    # Iterating through each patient that we have annotations for
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:
        pt = row.ptID
        pbar.set_description(desc=f"Patient: {pt}",refresh=True)
        # Skipping if no training data has been identified
        if len(row.interictal_training) == 0:
            continue
        # Loading data from bids
        inter,fs = get_data_from_bids(ospj(datapath,"BIDS"),pt,'interictal')
        # Pruning channels
        chn_labels = remove_scalp_electrodes(inter.columns)
        inter = inter[chn_labels]
        try:
            electrode_localizations,electrode_regions = electrode_wrapper(pt,rid_hup)
            electrode_localizations.name = clean_labels(electrode_localizations.name,pt)
            electrode_regions.name = clean_labels(electrode_regions.name,pt)
            electrode_localizations.to_pickle(ospj(prodatapath,pt,'electrode_localizations_atropos.pkl'))
            electrode_regions.to_pickle(ospj(prodatapath,pt,'electrode_localizations_dkt.pkl'))
            neural_channels = electrode_localizations.name[(electrode_localizations.name.isin(inter.columns)) & ((electrode_localizations.label == 'white matter') | (electrode_localizations.label == 'gray matter'))]
        except:
            print(f"electrode localization failed for {pt}")
            neural_channels = chn_labels
        inter = inter.loc[:,neural_channels]

        # Detecting and removing excess noisy channels
        mask,_ = detect_bad_channels(inter.to_numpy(),fs)
        inter = inter.drop(inter.columns[~mask],axis=1)

        # Preprocess the signal
        inter, fs = preprocess_for_detection(inter,fs,montage,2)
        # Training selected model
        if mdl_str == 'LSTM':
            ###
            # Instantiate the model
            input_size = inter.shape[1]
            hidden_size = 10
            output_size = inter.shape[1]

            # Check for cuda
            # ccheck = torch.cuda.is_available()
            ccheck = False

            # Initialize the model
            model = LSTMModel(input_size, hidden_size, output_size)
            if ccheck:
                model.cuda()
            
            # Scale the training data
            model.fit_scaler(inter)
            inter_z = model.scaler_transform(inter)
            inter = pd.DataFrame(inter_z,columns=inter.columns)

            # Prepare input and target data for the LSTM
            input_data,target_data = prepare_segment(inter)

            dataset = TensorDataset(input_data, target_data)
            full_batch = len(dataset)
            dataloader = DataLoader(dataset, batch_size=full_batch, shuffle=False)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Train the model, this will just modify the model object, no returns
            # print("Training patient specific model")
            train_model(model,dataloader,criterion,optimizer,ccheck=ccheck)

            # Creating classification thresholds
            input_data,target_data = prepare_segment(inter)
            inter_outputs = predict_sz(model,input_data,target_data,batch_size=full_batch,ccheck=ccheck)
            thresholds = np.percentile(inter_outputs,90,0)
            ###
        elif mdl_str == 'AbsSlp':
            model = AbsSlope(1,.5, fs)
            model.fit(inter)
            
        # Iterating through each seizure for that patient
        seizure_times = seizures_df[seizures_df.Patient == pt]
        qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),leave=False)
        for i,(_,sz_row) in enumerate(qbar):
            set_seed(1071999)
            qbar.set_description(f"Processing seizure {i}")
            # Load in seizure and metadata for BIDS path
            seizure,fs_raw, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
            # Filter out bad channels from interictal clip
            seizure = seizure[neural_channels]
            seizure = seizure.drop(seizure.columns[~mask],axis=1)
            # Perform overwrite check
            prob_path = f"probability_matrix_mdl-{model}_fs-{int(fs_raw/FACTOR)}_montage-{montage}_task-{task}_run-{run}.pkl"
            if (not OVERWRITE) and ospe(ospj(prodatapath,pt,prob_path)):
                 continue
            # Preprocess seizure for seizure detection task
            seizure, fs = preprocess_for_detection(seizure,fs_raw,montage,factor=FACTOR)
            
            if mdl_str == 'LSTM':
                ###
                seizure_z = model.scaler_transform(seizure)
                seizure = pd.DataFrame(seizure_z,columns=seizure.columns)
                input_data, target_data,time_wins = prepare_segment(seizure,fs,train_win,pred_win,ret_time=True)
                # Generate seizure detection predictions for each window
                outputs = predict_sz(model,input_data,target_data,batch_size=len(input_data)//2,ccheck=ccheck)
                seizure_mat = repair_data(outputs,seizure)
                # Getting raw predicted loss values for each window
                raw_sz_vals = np.mean(np.log(seizure_mat),1).T
                # Creating classifications
                mdl_outs = (raw_sz_vals.T > np.log(thresholds)).T.astype(float)
                ###
            elif mdl_str == 'AbsSlp':
                mdl_outs = model(seizure)
                time_wins = model.get_times(seizure)

            # Creating probabilities by temporally smoothing classification
            sz_prob = sc.ndimage.uniform_filter1d(mdl_outs,20,axis=1)
            sz_prob_df = pd.DataFrame(sz_prob.T,columns = seizure.columns)
            time_df = pd.Series(time_wins,name='time')
            sz_prob_df = pd.concat((sz_prob_df,time_df),axis=1)
            os.makedirs(ospj(prodatapath,pt),exist_ok=True)
            sz_prob_df.to_pickle(ospj(prodatapath,pt,prob_path))
            # np.save(ospj(prodatapath,pt,prob_path),sz_prob)
            # np.save(ospj(prodatapath,pt,f"raw_preds_mdl-{model}_fs-{fs}_montage-{montage}_task-{task}_run-{run}.npy"),sz_clf)
            first_detect = np.argmax(sz_prob[:,115:]>0.5,axis=1)
            first_detect[first_detect == 0] = sz_prob.shape[1]
            ch_sorting = np.argsort(first_detect)
            
            os.makedirs(ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str),exist_ok=True)
            plot_and_save_detection(sz_prob[ch_sorting,:],
                                    time_wins,
                                    seizure.columns[ch_sorting],
                                    ospj(figpath,pt,"annotations",str(int(sz_row.approximate_onset)),mdl_str,f"{montage}_sz_prob.png"))
        del model
if __name__ == "__main__":
    main()