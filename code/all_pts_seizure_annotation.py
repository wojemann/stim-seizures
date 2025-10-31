import os
import sys
from os.path import join as ospj
from os.path import exists as ospe
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm

# Get the project root (parent directory of examples/)
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)

from DynaSD import LiNDDA, GIN, WVNT

from config import Config
from utils import clean_labels, load_electrode_localizations, aggregate_prob_to_regions


def plot_seizure_spread(sz_prob, sz_spread, onset_idx, offset_idx, threshold, 
                         figpath, patient, onset_time, thresh_str, level='channel'):
    """
    Plot seizure spread visualization.
    
    Parameters:
    -----------
    sz_prob : pd.DataFrame
        Probability matrix (time x channels/regions)
    sz_spread : pd.DataFrame
        Spread dataframe (2 rows: indices and times)
    onset_idx : int
        Index of seizure onset
    offset_idx : int
        Index of seizure offset
    threshold : float
        Detection threshold
    figpath : str
        Path to save figure
    patient : str
        Patient ID
    onset_time : str
        Onset time string
    thresh_str : str
        Threshold string identifier
    level : str
        'channel' or 'region'
    """
    if len(sz_spread.columns) == 0:
        return
    
    plt_offset = 20
    plt.figure(figsize=(10, 6))
    plt.matshow(sz_prob.loc[onset_idx-plt_offset:offset_idx+plt_offset, sz_spread.columns].T,
                cmap='magma',
                interpolation='none',
                fignum=0)
    plt.clim([np.min(sz_prob.values), threshold])
    plt.plot(sz_spread.iloc[0, :] + plt_offset, np.arange(sz_spread.shape[1]), 
             color='black', linewidth=3, label='Spread onset')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Time (s)")
    plt.ylabel(f"{level.capitalize()}s")
    
    os.makedirs(ospj(figpath, patient), exist_ok=True)
    plt.savefig(ospj(figpath, patient, f'seizure-{onset_time}_{thresh_str}_{level}.png'), 
                bbox_inches='tight', dpi=150)
    plt.close()


thresh_str = 'pretrained'
datapath, prodatapath, figpath, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])
seizure_df = pd.read_csv(ospj(metapath, 'metadata_v7_BIDS.csv'))
seizure_df = seizure_df[seizure_df.stim == 0]

# model_dict = {
#     'model': GIN, 
#     'model_name': 'GIN', 
#     'sequence_length': 12, 
#     'forecast_length': 1, 
#     'suffix': '',
#     'metric': 'mse'
# }
# model_dict = {
#     'model': LiNDDA, 
#     'model_name': 'LiNDDA', 
#     'sequence_length': 5, 
#     'forecast_length': 4, 
#     'suffix': '',
#     'metric': 'mse'
# }
model_dict = {
    'model': WVNT,
    'model_name': 'WVNT',
    'sequence_length': None,
    'forecast_length': None,
    'suffix': '',
    'metric': 'mse'
}

# Process each patient/seizure
pbar = tqdm(seizure_df.iterrows(), total=len(seizure_df))
for _,row in pbar:
    try:
        patient = row.Patient
        onset_run = str(int(row.onset))
        pbar.set_description(f"Patient: {patient} | Seizure: {onset_run}")
        # Build path to probability file
        patient_path = ospj(prodatapath, 'sz_prob', patient)
        if model_dict['sequence_length'] is not None:
            sz_path = ospj(patient_path, 
                          f"{patient}_task-ictal{onset_run}_mdl-{model_dict['model_name']}_seq-{model_dict['sequence_length']}_"
                          f"{model_dict['metric']}_prob_forecast-{model_dict['forecast_length']}{model_dict['suffix']}.pkl")
        else:
            sz_path = ospj(patient_path, 
                          f"{patient}_task-ictal{onset_run}_run-*_mdl-{model_dict['model_name']}_sz_prob.pkl")
            potential_paths = glob.glob(sz_path)
            if len(potential_paths) == 0:
                print(f"No probability matrix found for {patient} {onset_run}")
                continue
            sz_path = potential_paths[0]
        if not os.path.exists(sz_path):
            print(f"Probability matrix not found for {patient} {onset_run}")
            continue
        
        # Load probability matrix
        sz_prob = pd.read_pickle(sz_path)
        sz_prob_times = sz_prob.pop('time').values  # Convert to numpy array for indexing
        
        # Initialize model and get threshold
        threshold_agg = 'manuscript'
        if model_dict['sequence_length'] is not None:
            model = model_dict['model'](fs=256, w_size=1, w_stride=0.5, 
                                            sequence_length=model_dict['sequence_length'],
                                            forecast_length=model_dict['forecast_length'],
                                            )
        else:
            model = model_dict['model'](fs=256, w_size=1, w_stride=0.5)
        threshold = model.get_threshold(sz_prob, method='pretrained', threshold_agg=threshold_agg)
        # Get channel-level spread
        initial_onset_idx = int(np.argmin(np.abs(sz_prob_times - 180)))
        sz_spread_ch, sz_clf = model.get_onset_and_spread(sz_prob.iloc[initial_onset_idx:,:], threshold=threshold, 
                                                   filter_w=10, rwin_size=5, rwin_req=4, ret_smooth_mat=True)
        
        if sz_spread_ch is None or len(sz_spread_ch.columns) == 0:
            print(f"No spread detected for {patient} {onset_run}")
            continue
        
        # Convert indices to times
        sz_spread_ch.loc[1, :] = sz_prob_times[sz_spread_ch.iloc[0, :].to_numpy().astype(int)]
        
        # Save channel-level spread
        save_path_ch = ospj(prodatapath, 'sz_spread', patient, 
                           f'seizure-{onset_run}_mdl-{model_dict["model_name"]}_seq-{model_dict["sequence_length"]}_'
                           f'forecast-{model_dict["forecast_length"]}{model_dict["suffix"]}_thresh-{model.threshold_agg}_ch-spread.pkl')
        os.makedirs(ospj(prodatapath, 'sz_spread', patient), exist_ok=True)
        sz_spread_ch.to_pickle(save_path_ch)
        
        # Calculate onset and offset indices for plotting
        onset_idx = int(sz_spread_ch.iloc[0, :].min()) + initial_onset_idx
        offset_idx = int(np.argmin(np.abs(sz_prob_times - (np.max(sz_prob_times) - 120))))
        
        # Plot channel-level spread
        plot_seizure_spread(sz_prob, sz_spread_ch, onset_idx, offset_idx, threshold,
                          figpath, patient, onset_run, thresh_str, level='channel')
        
        # Load electrode localizations for region mapping
        ch_to_region = load_electrode_localizations(patient, prodatapath)
        
        if ch_to_region is not None:
            # Aggregate probability matrix to regions
            sz_prob_region = aggregate_prob_to_regions(sz_prob, ch_to_region)
            
            if len(sz_prob_region.columns) > 0:
                # Get region-level spread by running detection on aggregated probabilities
                sz_spread_region = model.get_onset_and_spread(sz_prob_region, threshold=threshold,
                                                              filter_w=10, rwin_size=5, rwin_req=4)
                
                if sz_spread_region is not None and len(sz_spread_region.columns) > 0:
                    # Convert indices to times
                    sz_spread_region.loc[1, :] = sz_prob_times[sz_spread_region.iloc[0, :].to_numpy().astype(int)]
                    
                    # Save region-level spread
                    save_path_region = ospj(prodatapath, 'sz_spread', patient,
                                           f'seizure-{onset_run}_mdl-{model_dict["model_name"]}_seq-{model_dict["sequence_length"]}_'
                                           f'forecast-{model_dict["forecast_length"]}{model_dict["suffix"]}_thresh-{model.threshold_agg}_region-spread.pkl')
                    sz_spread_region.to_pickle(save_path_region)
                    
                    # Plot region-level spread
                    plot_seizure_spread(sz_prob_region, sz_spread_region, onset_idx, offset_idx, 
                                      threshold, figpath, patient, onset_run, thresh_str, level='region')
                else:
                    print(f"No region spread detected for {patient} {onset_run}")
            else:
                print(f"No regions found for {patient} {onset_run}")
        else:
            print(f"No electrode localizations found for {patient}")
    except Exception as e:
        print(f'Processing failed for {patient} {onset_run}: {e}')
        continue
