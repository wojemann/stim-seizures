#// "interictal_training": ["HUP253_phaseII",5783]
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
from tensorflow.keras.models import load_model
import tensorflow as tf

# OS imports
import os
from os.path import join as ospj
from os.path import exists as ospe
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')

# Setting Plotting parameters for heatmaps
plt.rcParams['image.cmap'] = 'magma'

OVERWRITE = True
TRAIN_WIN = 12
PRED_WIN = 1

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initialized

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

def prepare_wavenet_segment(data, fs = 128, w_size = 1, w_stride=0.5,ret_time=False):
    data_ch = data.columns.to_list()
    n_ch = len(data_ch)
    data_np = data.to_numpy()
    win_len_idx = w_size*fs
    nwins = num_wins(len(data_np[:,0]),fs,w_size,w_stride)
    data_mat = np.zeros((nwins,win_len_idx,len(data_ch)))
    for k in range(n_ch): # Iterating through channels
        samples = MovingWinClips(data_np[:,k],fs,w_size,w_stride)
        data_mat[:,:,k] = samples
    time_mat = MovingWinClips(np.arange(len(data))/fs,fs,w_size,w_stride)
    win_times = time_mat[:,0]
    data_flat = data_mat.transpose(0,2,1).reshape(-1,win_len_idx)
    if ret_time:
        return data_flat, win_times
    else:
        return data_flat
    
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
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

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
    
class NRG():
    def __init__(self, win_size = 1, stride = 0.5, fs = 256):
        self.win_size = win_size
        self.stride = stride
        self.fs = fs
        self.function = lambda x: np.sum(sig.welch(x,self.fs)[1],axis=-1)

    
    def __str__(self) -> str:
        return "NRG"
        
    def fit(self, x):
        # x should be samples x channels df
        self.scaler = RobustScaler().fit(x)
        self.inter = self.scaler.transform(x)
        self.nstds = np.std(self.inter,axis=0)

    def get_times(self, x):
        # x should be samples x channels df
        time_mat = MovingWinClips(np.arange(len(x))/self.fs,self.fs,self.win_size,self.stride)
        return time_mat[:,0]

    def forward(self, x):
        # x is samples x channels df
        self.data = x
        x = self.scaler.transform(x)
        x = x.T
        nrg = ft_extract(x, self.fs, self.function, self.win_size, self.stride)
        nrg = nrg.squeeze()
        normalized_nrg = scale_normalized(nrg)
        return normalized_nrg
    
    def __call__(self, *args):
        return self.forward(*args)

class WVNT():
    def __init__(self, mdl_path, win_size = 1, stride = 0.5, fs = 128):
        self.win_size = win_size
        self.stride = stride
        self.fs = fs
        self.mdl = load_model(mdl_path)
    
    def __str__(self) -> str:
        return "WVNT"
        
    def fit(self, x):
        # x should be samples x channels df
        self.scaler = RobustScaler().fit(x)

    def get_times(self, x):
        # x should be samples x channels df
        time_mat = MovingWinClips(np.arange(len(x))/self.fs,self.fs,self.win_size,self.stride)
        return time_mat[:,0]

    def forward(self, x):
        # x is samples x channels df
        chs = x.columns
        nwins = num_wins(len(x),self.fs,1,0.5)
        nch = len(chs)
        x = pd.DataFrame(self.scaler.transform(x),columns=chs)
        x = prepare_wavenet_segment(x)
        y = self.mdl.predict(x)[:,1]
        return y.reshape(nwins,nch).T
        
        
    
    def __call__(self, *args):
        return self.forward(*args)

class LSTMX(nn.Module):
    def __init__(self, num_channels, hidden_size):
        super(LSTMX, self).__init__()
        self.num_channels = num_channels
        self.lstms = nn.ModuleList([nn.LSTM(1, hidden_size, batch_first=True) for _ in range(num_channels)])
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_channels)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_channels):
            out, _ = self.lstms[i](x[:, :, i].unsqueeze(-1))  # LSTM input shape: (batch_size, seq_len, 1)
            out = self.fcs[i](out[:, -1, :])  # FC input shape: (batch_size, hidden_size)
            outputs.append(out.unsqueeze(1))  # Add channel dimension back

        # Concatenate outputs along channel dimension
        output = torch.cat(outputs, dim=1).squeeze()  # shape: (batch_size, num_channels, 1)
        return output
    
    def __str__(self):
         return "LSTMX"
    
    def fit_scaler(self, x):
        self.scaler = RobustScaler().fit(x)

    def scaler_transform(self, x):
        return self.scaler.transform(x)

# localization function wrapper
def electrode_wrapper(pt,rid_hup,datapath):
    if pt[:3] == 'HUP':
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
    else:
        recon_path = ospj(datapath,pt,f'{pt}_locations.xlsx')
        electrode_localizations,electrode_regions = choptimize_localizations(recon_path,pt)
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
    _,_,datapath,prodatapath,figpath,patient_table,rid_hup,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'),None)

    seizures_df = pd.read_csv(ospj(datapath,"stim_seizure_information_BIDS.csv"))

    montage = 'bipolar'
    train_win = TRAIN_WIN
    pred_win = PRED_WIN
    # pt_skip = True
    # Iterating through each patient that we have annotations for
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:   
        pt = row.ptID
        pbar.set_description(desc=f"Patient: {pt}",refresh=True)
        # Skipping if no training data has been identified
        # Creating code to test some patinets
        # if (pt != 'CHOP028') and pt_skip:
        #     continue
        # else:
        #     pt_skip = True
        # End patient test
        if len(row.interictal_training) == 0:
            continue
        # Loading data from bids
        inter,fs = get_data_from_bids(ospj(datapath,"BIDS"),pt,'interictal')
        # Pruning channels
        chn_labels = remove_scalp_electrodes(inter.columns)
        inter = inter[chn_labels]
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
            neural_channels = electrode_localizations.name[(electrode_localizations.name.isin(inter.columns)) & ((electrode_localizations.label == 'white matter') | (electrode_localizations.label == 'gray matter'))]
        except:
            print(f"electrode localization failed for {pt}")
            neural_channels = chn_labels
        inter = inter.loc[:,neural_channels]
        inter_nopre = inter.copy()
       
        for mdl_str in  ['LSTM','AbsSlp','NRG','WVNT']:
            wvcheck = mdl_str=='WVNT'
            # Preprocess the signal
            target=256
            inter, fs, mask = preprocess_for_detection(inter_nopre,fs,montage,target=target,wavenet=wvcheck,pre_mask = None)

            # Training selected model
            if mdl_str in ['LSTM','LSTMX']:
                ###
                # Instantiate the model
                input_size = inter.shape[1]
                hidden_size = 10

                # Check for cuda
                ccheck = torch.cuda.is_available()
                # ccheck = False

                # Initialize the model
                if mdl_str == 'LSTM':
                    model = LSTMModel(input_size, hidden_size)
                elif mdl_str == 'LSTMX':
                    model = LSTMX(input_size,hidden_size)
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
                thresholds = np.percentile(inter_outputs,85,0)
                ###
            elif mdl_str in ['NRG','AbsSlp','WVNT']:
                if mdl_str == 'AbsSlp':
                    model = AbsSlope(1,.5, fs)
                    model.fit(inter)
                elif mdl_str == 'NRG':
                    model = NRG(1,.5,fs)
                    model.fit(inter)
                elif mdl_str == 'WVNT':
                    model = WVNT(ospj(prodatapath,'WaveNet','v111.hdf5'),1,.5,fs)
                    model.fit(inter)
                
            # Iterating through each seizure for that patient
            seizure_times = seizures_df[seizures_df.Patient == pt]
            qbar = tqdm(seizure_times.iterrows(),total=len(seizure_times),leave=False)
            for i,(_,sz_row) in enumerate(qbar):
                set_seed(1071999)
                qbar.set_description(f"{mdl_str} processing seizure {i}")
                # Load in seizure and metadata for BIDS path
                seizure,fs_raw, _, _, task, run = get_data_from_bids(ospj(datapath,"BIDS"),pt,str(int(sz_row.approximate_onset)),return_path=True, verbose=0)
                # Filter out bad channels from interictal clip
                seizure = seizure[neural_channels]
                
                # Preprocess seizure for seizure detection task
                seizure, fs = preprocess_for_detection(seizure,fs_raw,montage,target=target,wavenet=wvcheck,pre_mask=mask)
                
                # Perform overwrite check
                prob_path = f"probability_matrix_mdl-{model}_fs-{int(fs)}_montage-{montage}_task-{task}_run-{run}.pkl"
                if (not OVERWRITE) and ospe(ospj(prodatapath,pt,prob_path)):
                    continue

                if mdl_str in ['LSTM','LSTMX']:
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
                elif mdl_str in ['NRG','AbsSlp','WVNT']:
                    mdl_outs = model(seizure)
                    time_wins = model.get_times(seizure)

                # Creating probabilities by temporally smoothing classification
                sz_prob = sc.ndimage.uniform_filter1d(mdl_outs,20,axis=1,mode='constant')
                sz_prob_df = pd.DataFrame(sz_prob.T,columns = seizure.columns)
                time_df = pd.Series(time_wins,name='time')
                sz_prob_df = pd.concat((sz_prob_df,time_df),axis=1)
                os.makedirs(ospj(prodatapath,pt),exist_ok=True)
                sz_prob_df.to_pickle(ospj(prodatapath,pt,prob_path))
                # np.save(ospj(prodatapath,pt,prob_path),sz_prob)
                # np.save(ospj(prodatapath,pt,f"raw_preds_mdl-{model}_fs-{fs}_montage-{montage}_task-{task}_run-{run}.npy"),sz_clf)
                first_detect = np.argmax(sz_prob[:,120:]>.75,axis=1)
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