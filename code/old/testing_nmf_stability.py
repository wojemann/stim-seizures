# Standard imports
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# OS imports
import os
from os.path import join as ospj
from os.path import exists as ospe
import pathlib
from tqdm import tqdm
from pqdm.processes import pqdm
from utils import *
from dtw_utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')
import mne

# Loading metadata
with open('/mnt/leif/littlab/users/wojemann/stim-seizures/code/config.json','r') as f:
    CONFIG = json.load(f)
usr = CONFIG["paths"]["iEEG_USR"]
passpath = CONFIG["paths"]["iEEG_PWD"]
datapath = CONFIG["paths"]["RAW_DATA"]
metadatapath = CONFIG["paths"]["META"]
prodatapath = CONFIG["paths"]["PROCESSED_DATA"]
ieeg_list = CONFIG["patients"]




# Stability-based NMF
n_bootstrap = 100  # Number of bootstrap samples
max_components = 75  # Maximum number of components to test
stability_scores = []
k = np.arange(1,max_components+1,)
def fun(k):
    with open(ospj(prodatapath,f'all_coh_test.pkl'),'rb') as f:
        all_coherences = pickle.load(f)
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
    _diss = lambda C,k: 1/(2*k)*(2*k - np.sum(np.max(C,axis=0)) - np.sum(np.max(C,axis=1)))
    # Cross-corr between columns of two different matrices
    def _ccorr(X,Y):
        s = X.shape[1]
        out_mat = np.zeros((s,s))
        for x_i in range(s):
            for y_i in range(s):
                c = np.corrcoef(X[:,x_i],Y[:,y_i])
                out_mat[x_i,y_i] = c[0,1]
        return out_mat
    # calculate dissimilarity for each combination of dictionaries at this k value
    diss_mat = np.zeros((n_bootstrap,n_bootstrap))
    for d_i in range(n_bootstrap):
        for d_j in range(n_bootstrap):
            C = _ccorr(D_array[d_i],D_array[d_j])
            diss_mat[d_i,d_j] = _diss(C,k)

    stability_score = np.mean(diss_mat[~np.eye(n_bootstrap,dtype=bool)])
    return [k,stability_score]

stability_scores = pqdm(k,fun,n_jobs=48)
with open(ospj(prodatapath,f'stability_score_test.pkl'),'wb') as f:
    pickle.dump(stability_scores,f)