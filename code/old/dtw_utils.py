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

def _dissimilarity(C,k):
    return 1/(2*k)*(2*k - np.sum(np.max(C,axis=0)) - np.sum(np.max(C,axis=1)))

def _ccorr(X,Y):
            s = X.shape[1]
            out_mat = np.zeros((s,s))
            for x_i in range(s):
                for y_i in range(s):
                    c = np.corrcoef(X[:,x_i],Y[:,y_i])
                    out_mat[x_i,y_i] = c[0,1]
            return out_mat

def stability_nmf(all_coherences,n_bootstrap=100,k_min = 5, k_max = 15):
    stability_scores = []

    for k in range(k_min, k_max + 1):
        D_list = []
        l_list = []
        for _ in range(int(n_bootstrap)):
            # Bootstrap resampling
            bootstrap_data = resample(all_coherences)
            
            # Perform NMF on bootstrap sample
            model = NMF(n_components=k, init='random', random_state=0)
            model.fit(bootstrap_data)
            D = model.components_
            l = model.reconstruction_err_

            # Sort each component by size and store
            D_list.append(D.T)
            l_list.append(l)
            
        D_array = np.array(D_list)
        # Calculate dissimilarity across Bootstraps
       
        # Cross-corr between columns of two different matrices
        
        # calculate dissimilarity for each combination of dictionaries at this k value
        diss_mat = np.zeros((n_bootstrap,n_bootstrap))
        for d_i in range(n_bootstrap):
            for d_j in range(n_bootstrap):
                C = _ccorr(D_array[d_i],D_array[d_j])
                diss_mat[d_i,d_j] = _dissimilarity(C,k)

        stability_score = np.mean(diss_mat[~np.eye(n_bootstrap,dtype=bool)])
        print(k,stability_score)
        stability_scores.append(stability_score)

    # Identify the number of components that gives the minimum stability score
    optimal_k = np.argmin(stability_scores) + 1
    model = NMF(n_components=optimal_k, init='random', random_state=0)
    W_optimal = model.fit_transform(all_coherences)
    H_optimal = model.components_
    recon_cohs = normalize(W_optimal@H_optimal,'l1',axis=1)
    return optimal_k,recon_cohs

def calculate_coh_timeseries(data,fs=1024,win_len=10,stride=1,factor=2,freq_bands=[(1, 4), (4, 8), (8, 13), (13, 30), (30, 80), (80, 150)], indexed = True):
        
    if indexed:
        index = data[0]
        data = data[1]
    cols = data.columns
    data = sig.decimate(data,factor,axis=0)
    data = pd.DataFrame(data,columns=cols)
    fs = fs/factor

    n_channels = data.shape[1]
    m_samples = data.shape[0]

    # Define frequency bands
    freq_bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 80), (80, 150)]

    # Window parameters
    window_length = int(win_len * fs)
    stride = int(stride*fs)

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
    optimal_k,recon_cohs = stability_nmf(all_coherences,n_bootstrap=100,k_min = 5, k_max = 20)

    if indexed:
        return [index,optimal_k,recon_cohs]
    else:
        return [optimal_k,recon_cohs]

def bootstrap_fun(k,bootstrap_data):
    # Perform NMF on bootstrap sample
    model = NMF(n_components=k, init='random', random_state=0)
    model.fit(bootstrap_data)
    D = model.components_
    return D

def pqdm_coh(window_data,fs=512,freq_bands=[(1, 4), (4, 8), (8, 13), (13, 30), (30, 80), (80, 150)]):
    n_channels = window_data.shape[1]
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
    return concatenated_coherence

def stability_fun(d_1,d_2,k):
    C = _ccorr(d_1,d_2)
    return _dissimilarity(C,k)

def pqdm_snmf(all_coherences,k_min=5,k_max=15):
    # def _dissimilarity(C,k):
    #     return 1/(2*k)*(2*k - np.sum(np.max(C,axis=0)) - np.sum(np.max(C,axis=1)))

    # def _ccorr(X,Y):
    #             s = X.shape[1]
    #             out_mat = np.zeros((s,s))
    #             for x_i in range(s):
    #                 for y_i in range(s):
    #                     c = np.corrcoef(X[:,x_i],Y[:,y_i])
    #                     out_mat[x_i,y_i] = c[0,1]
    #             return out_mat
    
    n_bootstrap=100
    
    stability_scores = []
    for k in range(k_min,k_max+1):
        args_dict = [[k,resample(all_coherences)]
                    for _ in range(n_bootstrap)]
        D_list=pqdm(args_dict,bootstrap_fun,n_jobs=48,argument_type='args')
        D_array = np.array(D_list)
        # calculate dissimilarity for each combination of dictionaries at this k value
        d_args = []
        for d_i in range(n_bootstrap):
            for d_j in range(d_i+1,n_bootstrap):
                d_args.append([D_array[d_i],D_array[d_j],k])                
        
        diss_mat = pqdm(d_args,stability_fun,n_jobs=48,argument_type='args')
        stability_score = np.mean(diss_mat)
        stability_scores.append(stability_score)
    
    return stability_scores

def parallel_coh_timeseries(data,fs=1024,win_len=10,stride=1,factor=2, indexed = True):
        
    if indexed:
        index = data[0]
        data = data[1]
    cols = data.columns
    data = sig.decimate(data,factor,axis=0)
    data = pd.DataFrame(data,columns=cols)
    fs = fs/factor
    m_samples = data.shape[0]

    # Window parameters
    window_length = int(win_len * fs)
    stride = int(stride*fs)

    # Initialize lists to hold results
    all_coherences = []
    
    # Loop through each window
    args_dict = [data.iloc[start:start + window_length]
                for start in 
                range(0, m_samples - window_length + 1, stride)]

    all_coherences = pqdm(args_dict,pqdm_coh,n_jobs=48)
    # Convert to NumPy array and L1 normalize
    all_coherences = np.array(all_coherences)
    all_coherences = normalize(all_coherences, norm='l1', axis=1)
    print(f"Finished Coh Calc")
  
    # k_min = 5; k_max = 15
    # stability_scores = pqdm_snmf(all_coherences,k_min,k_max)
    # optimal_k = np.argmin(stability_scores) + 1
    optimal_k = 6
    model = NMF(n_components=optimal_k, init='random', random_state=0)
    W_optimal = model.fit_transform(all_coherences)
    H_optimal = model.components_
    recon_cohs = normalize(W_optimal@H_optimal,'l1',axis=1)
    if indexed:
        return [index,optimal_k,recon_cohs]
    else:
        return [optimal_k,recon_cohs]