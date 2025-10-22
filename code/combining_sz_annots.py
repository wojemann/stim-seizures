import os
import sys
from os.path import join as ospj
from os.path import exists as ospe
import pandas as pd
import numpy as np
import glob
import re
from tqdm import tqdm

from config import Config
from utils import clean_labels, load_electrode_localizations

# Get paths from config
datapath, prodatapath, figpath, metapath = Config.deal(['datapath', 'prodatapath', 'figpath', 'metapath'])

# Model configuration - should match all_pts_seizure_annotation.py
model_dict = {
    'model_name': 'GIN', 
    'sequence_length': 12, 
    'forecast_length': 1, 
    'suffix': 'nopass_nolayernorm',
    'metric': 'mse'
}

thresh_str = 'pretrained'


def aggregate_prob_to_regions(sz_prob, ch_to_region):
    """Aggregate channel probability time series to region level by averaging."""
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
        region_prob_data[region] = np.mean(ch_probs, axis=0)
    
    # Create DataFrame with same index as original
    sz_prob_region = pd.DataFrame(region_prob_data, index=sz_prob.index)
    
    return sz_prob_region

# Initialize an empty list to collect each row for the final DataFrame
summary_data = []

# Load metadata to get seizure list
seizure_df = pd.read_csv(ospj(metapath, 'metadata_v7_BIDS.csv'))
seizure_df = seizure_df[seizure_df.stim == 0]

# Process each seizure
pbar = tqdm(seizure_df.iterrows(), total=len(seizure_df), desc="Processing seizures")
for _, row in pbar:
    patient = row.Patient
    onset_run = str(int(row.onset))
    pbar.set_description(f"Patient: {patient} | Seizure: {onset_run}")
    
    try:
        # Build file paths with new naming convention
        patient_path = ospj(prodatapath, patient)
        
        # Channel-level spread file
        spread_ch_file = ospj(patient_path,
                             f'seizure-{onset_run}_mdl-{model_dict["model_name"]}_seq-{model_dict["sequence_length"]}_'
                             f'forecast-{model_dict["forecast_length"]}{model_dict["suffix"]}_ch-spread.pkl')
        
        # Region-level spread file
        spread_region_file = ospj(patient_path,
                                 f'seizure-{onset_run}_mdl-{model_dict["model_name"]}_seq-{model_dict["sequence_length"]}_'
                                 f'forecast-{model_dict["forecast_length"]}{model_dict["suffix"]}_region-spread.pkl')
        
        # Probability file
        prob_file = ospj(prodatapath, 'sz_prob', patient,
                        f"{patient}_task-ictal{onset_run}_mdl-{model_dict['model_name']}_seq-{model_dict['sequence_length']}_"
                        f"{model_dict['metric']}_prob_forecast-{model_dict['forecast_length']}{model_dict['suffix']}.pkl")
        
        # Check if required files exist
        if not os.path.exists(spread_ch_file):
            print(f"Channel spread file not found for {patient} {onset_run}")
            continue
        
        if not os.path.exists(prob_file):
            print(f"Probability file not found for {patient} {onset_run}")
            continue
        
        # Load channel-level spread
        spread_ch_df = pd.read_pickle(spread_ch_file)
        seizing_times_ch = spread_ch_df.iloc[1, :].copy()
        seizing_times_ch -= np.min(seizing_times_ch)
        
        # Identify onset and spread channels
        onset_channels = seizing_times_ch[seizing_times_ch < 3].index.tolist()
        spread_channels = seizing_times_ch[seizing_times_ch < 10].index.tolist()
        
        # Load probability matrix
        prob_matrix = pd.read_pickle(prob_file)
        prob_times = prob_matrix.pop('time').values
        all_sz_channels = prob_matrix.columns.tolist()
        
        # Calculate onset index from spread data
        onset_offset = int(np.min(spread_ch_df.iloc[0, :]))
        
        # Calculate channel-level NDD metrics at different time windows
        onset_ndd_cum_list = []
        onset_ndd_ind_list = []
        time_windows = [1, 3, 5, 10, 15, 20, 25, 30, 40]
        
        for win in time_windows:
            # Cumulative: from onset to onset+win seconds (2 samples per second)
            end_idx = onset_offset + (win * 2)
            if end_idx < len(prob_matrix):
                onset_ndd_cum_list.append(prob_matrix.iloc[onset_offset:end_idx, :].mean())
            else:
                onset_ndd_cum_list.append(pd.Series())
        
        for win in [0, 1, 3, 5, 10, 15, 20, 25, 30, 40]:
            # Individual bins
            start_idx = onset_offset + (win * 2)
            end_idx = onset_offset + ((win + 1) * 2)
            if end_idx < len(prob_matrix):
                onset_ndd_ind_list.append(prob_matrix.iloc[start_idx:end_idx, :].mean())
            else:
                onset_ndd_ind_list.append(pd.Series())
        
        # Calculate spread ranks for channels
        spread_rank_ch = seizing_times_ch.rank(method='min')
        spread_rank_ch = spread_rank_ch.reindex(all_sz_channels).fillna(len(seizing_times_ch) + 1).astype(int)
        
        # Calculate percent channels seizing over time
        start = 0
        end = 60
        interval = 0.5
        num_points = int((end - start) / interval) + 1
        time_points = np.linspace(start, end, num_points)
        fraction_seizing_ch = np.array([np.sum(seizing_times_ch <= t) / len(all_sz_channels) for t in time_points])
        abs_seizing_ch = np.array([np.sum(seizing_times_ch <= t) for t in time_points])
        
        # Initialize region variables
        onset_regions = []
        spread_regions = []
        region_spread_rank = {}
        region_spread_time = {}
        fraction_seizing_region = None
        abs_seizing_region = None
        all_regions = []
        onset_ndd_cum_list_region = []
        onset_ndd_ind_list_region = []
        
        # Load electrode localizations for region mapping
        ch_to_region = load_electrode_localizations(patient, prodatapath)
        
        if ch_to_region is not None and os.path.exists(spread_region_file):
            # Load region-level spread
            spread_region_df = pd.read_pickle(spread_region_file)
            
            if len(spread_region_df.columns) > 0:
                seizing_times_region = spread_region_df.iloc[1, :].copy()
                seizing_times_region -= np.min(seizing_times_region)
                
                # Identify onset and spread regions
                onset_regions = seizing_times_region[seizing_times_region < 3].index.tolist()
                spread_regions = seizing_times_region[seizing_times_region < 10].index.tolist()
                
                # Aggregate probability matrix to regions
                prob_matrix_region = aggregate_prob_to_regions(prob_matrix, ch_to_region)
                all_regions = prob_matrix_region.columns.tolist()
                
                # Calculate region-level NDD metrics
                for win in time_windows:
                    end_idx = onset_offset + (win * 2)
                    if end_idx < len(prob_matrix_region):
                        onset_ndd_cum_list_region.append(prob_matrix_region.iloc[onset_offset:end_idx, :].mean())
                    else:
                        onset_ndd_cum_list_region.append(pd.Series())
                
                for win in [0, 1, 3, 5, 10, 15, 20, 25, 30, 40]:
                    start_idx = onset_offset + (win * 2)
                    end_idx = onset_offset + ((win + 1) * 2)
                    if end_idx < len(prob_matrix_region):
                        onset_ndd_ind_list_region.append(prob_matrix_region.iloc[start_idx:end_idx, :].mean())
                    else:
                        onset_ndd_ind_list_region.append(pd.Series())
                
                # Calculate spread ranks for regions
                spread_rank_region = seizing_times_region.rank(method='min')
                spread_rank_region = spread_rank_region.reindex(all_regions).fillna(len(seizing_times_region) + 1).astype(int)
                region_spread_rank = spread_rank_region.to_dict()
                region_spread_time = spread_region_df.iloc[1, :].to_dict()
                
                # Calculate percent regions seizing over time
                fraction_seizing_region = np.array([np.sum(seizing_times_region <= t) / len(all_regions) for t in time_points])
                abs_seizing_region = np.array([np.sum(seizing_times_region <= t) for t in time_points])
        
        # Append data to the summary list
        summary_data.append({
            # Basic info
            'patient': patient,
            'onset': float(onset_run),
            
            # Channel-level data
            'onset_channels': onset_channels,
            'spread_channels': spread_channels,
            'channel_spread_rank': spread_rank_ch.to_dict(),
            'channel_spread_time': spread_ch_df.iloc[1, :].to_dict(),
            'fraction_seizing_ch': fraction_seizing_ch,
            'abs_seizing_ch': abs_seizing_ch,
            'all_channels': all_sz_channels,
            'onset_ndd_cum_ch': onset_ndd_cum_list,
            'onset_ndd_ind_ch': onset_ndd_ind_list,
            
            # Region-level data
            'onset_regions': onset_regions,
            'spread_regions': spread_regions,
            'region_spread_rank': region_spread_rank,
            'region_spread_time': region_spread_time,
            'fraction_seizing_region': fraction_seizing_region,
            'abs_seizing_region': abs_seizing_region,
            'all_regions': all_regions,
            'onset_ndd_cum_region': onset_ndd_cum_list_region,
            'onset_ndd_ind_region': onset_ndd_ind_list_region,
        })
        
    except Exception as e:
        print(f"Error processing {patient} {onset_run}: {e}")
        continue

# Create a final dataframe
summary_df = pd.DataFrame(summary_data)

# Save to file with new naming convention
output_filename = (f"seizure_spread_summary_mdl-{model_dict['model_name']}_"
                  f"seq-{model_dict['sequence_length']}_"
                  f"forecast-{model_dict['forecast_length']}_"
                  f"thresh-{thresh_str}.pkl")

summary_df.to_pickle(ospj(prodatapath, output_filename))
print(f"\nSaved summary to {output_filename}")
print(f"Total seizures processed: {len(summary_df)}")
print(f"Seizures with region data: {summary_df['all_regions'].apply(lambda x: len(x) > 0).sum()}")
