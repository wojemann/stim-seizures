# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd

from sklearn.metrics import matthews_corrcoef

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from statsmodels.stats.multitest import multipletests
# Imports for analysis

# OS imports
from os.path import join as ospj
from utils import *
import sys
sys.path.append('/users/wojemann/iEEG_processing')
plt.rcParams['image.cmap'] = 'magma'

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2

plt.rcParams['xtick.major.size'] = 5  # Change to your desired major tick size
plt.rcParams['ytick.major.size'] = 5  # Change to your desired major tick size
plt.rcParams['xtick.minor.size'] = 3   # Change to your desired minor tick size
plt.rcParams['ytick.minor.size'] = 3   # Change to your desired minor tick size

plt.rcParams['xtick.major.width'] = 2  # Change to your desired major tick width
plt.rcParams['ytick.major.width'] = 2  # Change to your desired major tick width
plt.rcParams['xtick.minor.width'] = 1  # Change to your desired minor tick width
plt.rcParams['ytick.minor.width'] = 1  # Change to your desired minor tick width
plt.rcParams['font.family'] = 'Verdana'

usr,passpath,datapath,prodatapath,metapath,figpath,patient_table,rid_hup,pt_list = load_config(ospj('/mnt/sauce/littlab/users/wojemann/stim-seizures/code','config.json'))

### Loading in clinician annotations
consensus_annots = pd.read_pickle(ospj(prodatapath,'threshold_tuning_consensus.pkl'))
consensus_annots.loc[consensus_annots.Seizure_ID == 'HUP249_3','stim'] = 1.0
consensus_annots = consensus_annots[consensus_annots.patient != 'CHOP038']

consensus_annots = consensus_annots[consensus_annots.clinician.apply(lambda x: len(x)!=2)]
spread_consensus_annots = consensus_annots.copy()
note_type = 'ueo'
for i_r, row in consensus_annots.iterrows():
    scores = 0
    annots = row[note_type]
    if len(annots) < 2: 
        continue
    for i_annot in range(len(annots)):
        for j_annot in range(i_annot+1,len(annots)):
            scores += matthews_corrcoef(annots[i_annot],annots[j_annot])
    consensus_annots.loc[i_r,'f1'] = scores/(len(annots)*(len(annots)-1)/2)
consensus_annots["model"] = ["Clin."]*len(consensus_annots)
consensus_annots["dice"] = consensus_annots["f1"]

### Loading model annotations

params = []
for epochs in [10,100]:
    for demin in [True,False]:
        for movtype in ['med','mean']:
            for movwin in [10,20]:
                for movdata in ['clf','prob']:
                    params.append((str(epochs),str(demin),movtype,str(movwin),movdata))

def plot_fun(params):
    epochs, demin, movtype, movwin, movdata = params
    print(f'Starting: {params}')

    anntype='consensus'
    mdl_anntype = 'strict'
    montage = 'bipolar'

    mdl_preds = pd.read_pickle(f'/mnt/sauce/littlab/users/wojemann/stim-seizures/PROCESSED_DATA/pretrain_predicted_channels_wmcc_epoch-{epochs}_min-{str(demin)}_mov-{movtype}-{str(movwin)}-{movdata}.pkl')
    mdl_preds = mdl_preds[mdl_preds.model != 'NRG']
    mdl_preds.loc[mdl_preds.approximate_onset == 439029.32,'stim'] = 1.0

    mdl_preds_sorted = mdl_preds.sort_values(['Patient','approximate_onset','threshold','model'])
    melt_cols = [f'ueo_chs_{mdl_anntype}_{anntype}_MCC',f'sec_chs_{mdl_anntype}_{anntype}_MCC']
    keep_cols = [c for c in mdl_preds_sorted.columns if c not in melt_cols]
    mdl_preds_long = mdl_preds_sorted.melt(id_vars=keep_cols,var_name='annot',value_name='dice')

    tune_preds_long = mdl_preds_long[mdl_preds_long.annot == f'ueo_chs_{mdl_anntype}_{anntype}_MCC']
    val_preds_long = mdl_preds_long[mdl_preds_long.annot == f'sec_chs_{mdl_anntype}_{anntype}_MCC']

    optimal_threshold_preds = tune_preds_long.loc[tune_preds_long.groupby(['Patient', 'approximate_onset','model'])['dice'].idxmax()][["model","dice","Patient","stim","approximate_onset",'threshold']]

    consensus_annots["model"] = ["Clin."]*len(consensus_annots)
    consensus_annots["dice"] = consensus_annots["f1"]
    all_plot_agreements = pd.concat([optimal_threshold_preds,consensus_annots[["model","dice","stim"]]]).dropna(subset='dice')

    all_plot_agreements = all_plot_agreements[all_plot_agreements.model != 'NRG']
    all_plot_agreements.loc[all_plot_agreements.model == 'LSTM',['model']] = 'NDD'
    all_plot_agreements.loc[all_plot_agreements.model == 'WVNT',['model']] = 'DL'
    all_plot_agreements.loc[all_plot_agreements.model == 'Clin.',['model']] = 'Interrater'


    colors = np.array(sns.color_palette("deep", 4))
    colors = colors[[0,3,1]]
    _,ax = plt.subplots(figsize=(4,5))
    sns.boxplot(all_plot_agreements,x='model',y='dice',hue='model',
                    palette=np.vstack([colors,[.5,.5,.5]]).tolist(),
                    order=['AbsSlp','DL','NDD','Interrater'],
                    width=.6,notch=True,fill=False,ax=ax)
    sns.swarmplot(all_plot_agreements,x='model',y='dice',color='gray',ax=ax,alpha=0.5)
    sns.pointplot(all_plot_agreements,x='model',y='dice', hue='model',
                palette=np.vstack([np.array(colors),[.5,.5,.5]]).tolist(),
                order = ['AbsSlp','DL','NDD','Interrater'],
                marker='_',markersize=50,errorbar=None,ax=ax,
                estimator='mean')
    sns.despine()
    plt.ylabel('MCC')
    plt.title("Onset Activity Encoding")
    plt.xticks(rotation=30,ha='right')
    plt.savefig(ospj(figpath,'smoothing',f'onset_tuned_onset_detection_{"-".join(params)}.png'),bbox_inches='tight')

_ = in_parallel(plot_fun,params)