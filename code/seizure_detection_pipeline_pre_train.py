# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
from scipy.linalg import hankel
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Imports for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.models import load_model
from tensorflow.config.experimental import set_memory_growth, list_physical_devices

# OS imports
import os
from os.path import join as ospj
from os.path import exists as ospe
from utils import *
from stim_seizure_preprocessing_utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Setting Plotting parameters for heatmaps
plt.rcParams['image.cmap'] = 'magma'

OVERWRITE = True

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
    
class LTI():
    def __init__(self, win_size = 1, stride = 0.5, fs = 256):
        self.win_size = win_size
        self.stride = stride
        self.fs = int(fs)
        self.model = LinearRegression(fit_intercept=False)

    def fit(self, x):
        # x should be samples x channels df
        self.scaler = RobustScaler().fit(x)
        nx = self.scaler.transform(x)
        self.model.fit(nx[:-1,:],nx[1:,:])
        
    def forward(self, x):
        ch_names = x.columns
        x = self.scaler.transform(x)
        y = self.model.predict(x[:-1,:])
        se = pd.DataFrame((x[1:,:]-y)**2,columns=ch_names)
        mse = se.rolling(self.win_size*self.fs,min_periods=self.win_size*self.fs,center=False).mean()
        mse_wins = mse.iloc[::int(self.stride*self.fs)].reset_index(drop=True)
        return mse_wins[~mse_wins.isna().any(axis=1)].to_numpy().T

    
    def get_times(self, x):
        # x should be samples x channels df
        time_arr = np.arange(len(x))/self.fs
        right_time_arr = time_arr[int((self.win_size*self.fs)-1)::int(self.stride*self.fs)]
        return right_time_arr
    
    def __str__(self):
        return "LTI"
    
    def __call__(self, *args):
        return self.forward(*args)

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
        self.inter = pd.DataFrame(nx,columns = x.columns)
        self.nstds = np.std(nx,axis=0)

    def get_times(self, x):
        # x should be samples x channels df
        time_mat = MovingWinClips(np.arange(len(x))/self.fs,self.fs,self.win_size,self.stride)
        return np.ceil(time_mat[:,-1])

    def forward(self, x):
        # x is samples x channels df
        self.data = x
        x = self.scaler.transform(x)
        x = x.T
        slopes = ft_extract(x, self.fs, self.function, self.win_size, self.stride)
        scaled_slopes = slopes.squeeze()/self.nstds.reshape(-1,1)*self.fs
        scaled_slopes = scaled_slopes.squeeze()
        # normalized_slopes = scale_normalized(scaled_slopes,15)
        # normalized_slopes = minmax_scale(scaled_slopes.reshape(-1,1)).reshape(scaled_slopes.shape)
        return scaled_slopes/1000
    
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
        return np.ceil(time_mat[:,-1])

    def forward(self, x):
        # x is samples x channels df
        self.data = x
        x = self.scaler.transform(x)
        x = x.T
        nrg = ft_extract(x, self.fs, self.function, self.win_size, self.stride)
        nrg = nrg.squeeze()
        # normalized_nrg = scale_normalized(nrg,15)
        # normalized_nrg = minmax_scale(nrg.reshape(-1,1)).reshape(nrg.shape)
        # return normalized_nrg
        return nrg/10
    def __call__(self, *args):
        return self.forward(*args)

class WVNT():
    def __init__(self, mdl, win_size = 1, stride = 0.5, fs = 128):
        self.win_size = win_size
        self.stride = stride
        self.fs = fs
        self.mdl = mdl
    
    def __str__(self) -> str:
        return "WVNT"
        
    def fit(self, x):
        # x should be samples x channels df
        self.scaler = RobustScaler().fit(x)

    def get_times(self, x):
        # x should be samples x channels df
        time_mat = MovingWinClips(np.arange(len(x))/self.fs,self.fs,self.win_size,self.stride)
        return np.ceil(time_mat[:,-1])

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
        recon_path = ospj('/mnt','sauce','littlab','data',
                            'Human_Data','CNT_iEEG_BIDS',
                            f'sub-RID{rid}','derivatives','ieeg_recon',
                            'module3/')
        if not os.path.exists(recon_path):
            recon_path =  ospj('/mnt','sauce','littlab','data',
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
    gpus = list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)  # Memory growth must be set before GPUs have been initialized
    # Please run BIDS_seizure_saving.py and BIDS_interictal_saving.py to modify seizures for seizure detection.
    _,_,datapath,prodatapath,metapath,figpath,patient_table,rid_hup,_ = load_config(ospj('/mnt/leif/littlab/users/wojemann/stim-seizures/code','config.json'),None)

    seizures_df = pd.read_csv(ospj(metapath,"stim_seizure_information_BIDS.csv"))

    onset_time = 120
    montage = 'bipolar'
    train_win = 12
    pred_win = 1
    all_mdl_strs = ['AbsSlp','LSTM','NRG','WVNT']
    # all_mdl_strs = ['LTI']

    if 'WVNT' in all_mdl_strs:
        wave_model = load_model(ospj(prodatapath,'WaveNet','v111.hdf5'))

    # Iterating through each patient that we have annotations for
    pbar = tqdm(patient_table.iterrows(),total=len(patient_table))
    for _,row in pbar:
        pt = row.ptID
        pbar.set_description(desc=f"Patient: {pt}",refresh=True)

        # if pt not in ['HUP238']:
        #     continue
       
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

        for i_mdl,mdl_str in  enumerate(all_mdl_strs):
            wvcheck = mdl_str=='WVNT'
            # Preprocess the signal
            target=128
            inter_pre, fs, mask = preprocess_for_detection(inter_neural,fs_raw,montage,target=target,wavenet=wvcheck,pre_mask = None)

            ### ONLY PREDICTING FOR SEIZURES THAT HAVE BEEN ANNOTATED
            # seizure_times = seizures_df[(seizures_df.Patient == pt) & (seizures_df.to_annotate == 1)]
            ###
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
                
                # Preprocess seizure for seizure detection task
                seizure_pre, fs = preprocess_for_detection(seizure,fs_raw,montage,target=target,wavenet=wvcheck,pre_mask=mask)
                
                noisy_channel_mask = seizure_pre.loc[onset_time*fs:,:].abs().max() <= (np.median(seizure_pre.loc[onset_time*fs:,:].abs().max())*50)
                # noisy_channel_list = seizure_pre.columns[noisy_channel_mask].to_list()
                seizure_pre = seizure_pre.loc[:,noisy_channel_mask]

                # Perform overwrite check
                prob_path = f"pretrain_probability_matrix_mdl-{mdl_str}_fs-{int(fs)}_montage-{montage}_task-{task}_run-{run}.pkl"
                
                if (not OVERWRITE) and ospe(ospj(prodatapath,pt,prob_path)):
                    continue
                if sz_row.stim == 1:    
                    sz_train = inter_pre.loc[:,seizure_pre.columns]
                else:
                    sz_train = seizure_pre.loc[:fs*60,:]

                if mdl_str in ['LSTM','LSTMX']:
                    ##############################
                    input_size = sz_train.shape[1]
                    hidden_size = 10
                    # Check for cuda
                    ccheck = torch.cuda.is_available()
                    # Initialize the model
                    if mdl_str == 'LSTM':
                        model = LSTMModel(input_size, hidden_size)
                    elif mdl_str == 'LSTMX':
                        model = LSTMX(input_size,hidden_size)
                    if ccheck:
                        model.cuda()

                    # Scale the training data
                    model.fit_scaler(sz_train)
                    sz_train_z = model.scaler_transform(sz_train)
                    sz_train_z = pd.DataFrame(sz_train_z,columns=sz_train.columns)

                    # Prepare input and target data for the LSTM
                    input_data,target_data = prepare_segment(sz_train_z,fs=fs)

                    dataset = TensorDataset(input_data, target_data)
                    full_batch = len(dataset)
                    dataloader = DataLoader(dataset, batch_size=full_batch, shuffle=False)
                    
                    # Define loss function and optimizer
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.01)

                    # Train the model, this will just modify the model object, no returns
                    train_model(model,dataloader,criterion,optimizer,ccheck=ccheck)
                    
                    ################################################
                    seizure_z = model.scaler_transform(seizure_pre)
                    seizure_z = pd.DataFrame(seizure_z,columns=seizure_pre.columns)
                    input_data, target_data,time_wins = prepare_segment(seizure_z,fs,train_win,pred_win,ret_time=True)
                    # Generate seizure detection predictions for each window
                    outputs = predict_sz(model,input_data,target_data,batch_size=len(input_data)//2,ccheck=ccheck)
                    seizure_mat = repair_data(outputs,seizure_z,fs=fs)
                    # Getting raw predicted loss values for each window
                    raw_sz_vals = np.sqrt(np.mean(seizure_mat,axis=1)).T
                    # Creating classifications
                    mdl_outs = raw_sz_vals
                    ###
                
                elif mdl_str in ['NRG','AbsSlp','WVNT','LTI']:
                    if mdl_str == 'AbsSlp':
                        model = AbsSlope(1,.5, fs)
                        model.fit(sz_train)
                    elif mdl_str == 'NRG':
                        model = NRG(1,.5,fs)
                        model.fit(sz_train)
                    elif mdl_str == 'WVNT':
                        model = WVNT(wave_model,1,.5,fs)
                        model.fit(sz_train)
                    elif mdl_str == 'LTI':
                        model = LTI(1,.5,fs)
                        model.fit(sz_train)
                    mdl_outs = model(seizure_pre)
                    time_wins = model.get_times(seizure_pre)

                # Creating probabilities by temporally smoothing classification
                sz_prob = sc.ndimage.uniform_filter1d(mdl_outs,20,axis=1,mode='constant')
                sz_prob_df = pd.DataFrame(sz_prob.T,columns = seizure_pre.columns)
                time_df = pd.Series(time_wins,name='time')
                sz_prob_df = pd.concat((sz_prob_df,time_df),axis=1)
                os.makedirs(ospj(prodatapath,pt),exist_ok=True)
                sz_prob_df.to_pickle(ospj(prodatapath,pt,prob_path))
                
                ### Visualization
                # np.save(ospj(prodatapath,pt,prob_path),sz_prob)
                # np.save(ospj(prodatapath,pt,f"raw_preds_mdl-{model}_fs-{fs}_montage-{montage}_task-{task}_run-{run}.npy"),sz_clf)
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