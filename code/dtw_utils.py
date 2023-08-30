import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.integrate import simpson
from scipy.signal import coherence
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

import scipy as sc
from bct.algorithms import community_louvain
plt.rcParams['image.cmap'] = 'BuPu'

# ML Imports
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from sklearn.utils import resample

# OS imports
import os
from os.path import join as ospj
from os.path import exists as ospe
import pathlib
from tqdm import tqdm
from pqdm.processes import pqdm
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')
import mne

def sammon(X, max_iter=1e10, tol=1e-9):
    '''
    MDS scaling
    X: np.array - samples by features
    '''
    pairwise_dissimilarities = squareform(pdist(X, metric='cityblock'))  # Using L1 distance
    plt.imshow(pairwise_dissimilarities)
    init = np.random.rand(X.shape[0], 2)  # Random 2D initialization
    
    def stress(Y):
        Y = Y.reshape((X.shape[0], 2))
        d = squareform(pdist(Y))
        d[d == 0] = 1  # Prevent division by zero
        ratio = pairwise_dissimilarities / d
        return np.sum((pairwise_dissimilarities - d)**2 * ratio)
    
    res = minimize(stress, init.ravel(), method='L-BFGS-B', tol=tol, options={'maxiter': max_iter, 'disp': True})
    return res.x.reshape((X.shape[0], 2))

def calculate_coh_timeseries(data,fs=1024,win_len=10,stride=1,factor=2,freq_bands=[(1, 4), (4, 8), (8, 13), (13, 30), (30, 80), (80, 150)], indexed = True):
        
    if indexed:
        index = data[0]
        data = data[1]
    cols = data.columns
    data = sig.decimate(data,factor,axis=0)
    data = pd.DataFrame(data,columns=cols)
    nfs = fs/factor

    # Simulate a DataFrame
    n_channels = data.shape[1]
    m_samples = data.shape[0]

    # Define frequency bands
    freq_bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 80), (80, 150)]

    # Window parameters
    window_length = int(win_len * nfs)
    stride = int(stride*nfs)

    # Initialize lists to hold results
    all_coherences = []

    # Loop through each window
    for start in tqdm(range(0, m_samples - window_length + 1, stride)):
        window_data = data.iloc[start:start + window_length]
        coherence_matrix_list = []
        
        for low_f, high_f in freq_bands:
            coherences = []
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    f, Cxy = coherence(window_data.iloc[:, i], window_data.iloc[:, j], fs=fs)
                    avg_coh = np.mean(Cxy[(f >= low_f) & (f <= high_f)])
                    coherences.append(avg_coh)
            
            coherence_matrix_list.append(np.array(coherences))
        
        concatenated_coherence = np.concatenate(coherence_matrix_list)
        all_coherences.append(concatenated_coherence)

    # Convert to NumPy array and L1 normalize
    all_coherences = np.array(all_coherences)
    all_coherences = normalize(all_coherences, norm='l1', axis=1)
    print(f"Finished Coh Calc")
    # Stability-based NMF
    n_bootstrap = 15  # Number of bootstrap samples
    max_components = 10  # Maximum number of components to test
    stability_scores = []

    for k in tqdm(range(1, max_components + 1)):
        W_list = []
        
        for _ in range(int(n_bootstrap)):
            # Bootstrap resampling
            bootstrap_data = resample(all_coherences)
            
            # Perform NMF on bootstrap sample
            model = NMF(n_components=k, init='random', random_state=0)
            W = model.fit_transform(bootstrap_data)
            
            # Sort each component by size and store
            W_list.append(np.sort(W, axis=0))
            
        # Calculate stability score across bootstrap samples
        W_array = np.array(W_list)
        stability_score = np.mean(np.std(W_array, axis=0) / np.mean(W_array, axis=0))
        stability_scores.append(stability_score)

    # Identify the number of components that gives the minimum stability score
    optimal_k = np.argmin(stability_scores) + 1

    # Perform NMF with optimal number of components
    model = NMF(n_components=optimal_k, init='random', random_state=0)
    W_optimal = model.fit_transform(all_coherences)
    H_optimal = model.components_

    recon_cohs = normalize(W_optimal@H_optimal,'l1',axis=1)

    if indexed:
        return [index,recon_cohs]
    else:
        return recon_cohs