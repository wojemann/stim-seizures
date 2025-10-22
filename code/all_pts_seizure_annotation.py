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

from DynaSD import LiNDDA, GIN

from config import Config
from utils import clean_labels, load_electrode_localizations


def aggregate_prob_to_regions(sz_prob, ch_to_region):
    """
    Aggregate channel probability time series to region level by averaging.
    
    Parameters:
    -----------
    sz_prob : pd.DataFrame
        Channel probability dataframe (time x channels)
    ch_to_region : dict
        Mapping from first contact to region name
    
    Returns:
    --------
    sz_prob_region : pd.DataFrame
        Region probability dataframe (time x regions)
    """
    if sz_prob is None or len(sz_prob.columns) == 0:
        return pd.DataFrame()
    
    # Group channels by region
    region_prob_dict = {}
    
    for ch in sz_prob.columns:
        # Extract first contact from bipolar channel
        first_contact = ch.split('-')[0]
        region = ch_to_region.get(first_contact)
        
        if region is None:
            continue
        
        if region not in region_prob_dict:
            region_prob_dict[region] = []
        
        # Append this channel's time series to the region
        region_prob_dict[region].append(sz_prob[ch].values)
    
    if len(region_prob_dict) == 0:
        return pd.DataFrame()
    
    # Average probabilities within each region across channels
    region_prob_data = {}
    for region, ch_probs in region_prob_dict.items():
        # Average across all channels in this region (mean over axis 0)
        region_prob_data[region] = np.mean(ch_probs, axis=0)
    
    # Create DataFrame with same index as original
    sz_prob_region = pd.DataFrame(region_prob_data, index=sz_prob.index)
    
    return sz_prob_region


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

model_dict = {
    'model': GIN, 
    'model_name': 'GIN', 
    'sequence_length': 12, 
    'forecast_length': 1, 
    'suffix': 'nopass_nolayernorm',
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
        sz_path = ospj(patient_path, 
                      f"{patient}_task-ictal{onset_run}_mdl-{model_dict['model_name']}_seq-{model_dict['sequence_length']}_"
                      f"{model_dict['metric']}_prob_forecast-{model_dict['forecast_length']}{model_dict['suffix']}.pkl")
        
        if not os.path.exists(sz_path):
            print(f"Probability matrix not found for {patient} {onset_run}")
            continue
        
        # Load probability matrix
        sz_prob = pd.read_pickle(sz_path)
        sz_prob_times = sz_prob.pop('time').values  # Convert to numpy array for indexing
        
        # Initialize model and get threshold
        model = model_dict['model'](fs=256, w_size=1, w_stride=0.5, 
                                    sequence_length=model_dict['sequence_length'],
                                    forecast_length=model_dict['forecast_length'])
        threshold = model.get_threshold(sz_prob, method='pretrained')
        
        # Get channel-level spread
        sz_spread_ch = model.get_onset_and_spread(sz_prob, threshold=threshold, 
                                                   filter_w=10, rwin_size=5, rwin_req=4)
        
        if sz_spread_ch is None or len(sz_spread_ch.columns) == 0:
            print(f"No spread detected for {patient} {onset_run}")
            continue
        
        # Convert indices to times
        sz_spread_ch.loc[1, :] = sz_prob_times[sz_spread_ch.iloc[0, :].to_numpy().astype(int)]
        
        # Save channel-level spread
        save_path_ch = ospj(prodatapath, patient, 
                           f'seizure-{onset_run}_mdl-{model_dict["model_name"]}_seq-{model_dict["sequence_length"]}_'
                           f'forecast-{model_dict["forecast_length"]}{model_dict["suffix"]}_ch-spread.pkl')
        os.makedirs(ospj(prodatapath, patient), exist_ok=True)
        sz_spread_ch.to_pickle(save_path_ch)
        
        # Calculate onset and offset indices for plotting
        onset_idx = int(sz_spread_ch.iloc[0, :].min())
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
                    save_path_region = ospj(prodatapath, patient,
                                           f'seizure-{onset_run}_mdl-{model_dict["model_name"]}_seq-{model_dict["sequence_length"]}_'
                                           f'forecast-{model_dict["forecast_length"]}{model_dict["suffix"]}_region-spread.pkl')
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
