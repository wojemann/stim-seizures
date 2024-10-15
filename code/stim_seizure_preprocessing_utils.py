# Standard imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ieeg.auth import Session
from scipy import signal as sig
import scipy as sc
from itertools import compress

# OS imports
import os
from os.path import join as ospj

def stim_detect(data,threshold,fs):
    all_pks = np.zeros_like(data)
    for i, (_, ch) in enumerate(data.items()):
        pks,_ = sc.signal.find_peaks(np.abs(np.diff(ch.to_numpy())),
                                    height=threshold[i],
                                    distance=fs/4*3,
                                    )
        all_pks[pks,i] = 1
    
    pk_idxs,_ = sc.signal.find_peaks(all_pks.sum(axis=1),
                            distance=fs/4*3,
                            )
    stim_chs = all_pks.any(0)
    return pk_idxs,stim_chs

def barndoor(sz,pk_idxs,fs,pre=50e-3,post=100e-3,plot = False):
    data = sz.copy()
    pre_idx = np.floor(pre*fs).astype(int)
    post_idx = np.floor(post*fs).astype(int)
    win_idx = pre_idx + post_idx
    taper = np.linspace(0,1,win_idx)
    for idx in pk_idxs:
        sidx = int(idx-pre_idx)
        eidx = int(idx+post_idx)
        pre_data = data.iloc[sidx-win_idx:sidx,:].to_numpy()
        post_data = data.iloc[eidx:eidx+win_idx,:].to_numpy()
        data.iloc[sidx:eidx,:] = np.flip(pre_data,0) * np.flip(taper).reshape(-1,1) + np.flip(post_data,0) * taper.reshape(-1,1)
        if plot:
            _,axs = plt.subplots(4,1)
            axs[0].plot(sz.iloc[sidx-win_idx:eidx+win_idx,8])
            axs[1].plot(np.flip(pre_data[:,8],0))
            axs[1].plot(np.flip(post_data[:,8],0))
            axs[2].plot(np.flip(taper))
            axs[2].plot(taper)
            axs[3].plot(data.iloc[sidx-win_idx:eidx+win_idx,8])
            axs[3].axvline(sidx)
            axs[3].axvline(eidx)
            plt.show()
            fig = plt.figure()
            plt.plot(sz.iloc[sidx-win_idx:eidx+win_idx,8])
            plt.plot(data.iloc[sidx-win_idx:eidx+win_idx,8])
            plt.axvline(sidx,color='k')
            plt.axvline(eidx,color='k')
            plt.show()
            fig.savefig(ospj(figpath,'reconstruction_error.pdf'))
            plot=False

    return data
        