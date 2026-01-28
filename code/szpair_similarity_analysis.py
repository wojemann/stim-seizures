import sys
import os
from os.path import join as ospj
from itertools import combinations
from sklearn.metrics import pairwise
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from config import Config
from utils import clean_labels, load_electrode_localizations

# Get paths from config
datapath, prodatapath, figpath, metapath = Config.deal(['datapath', 'prodatapath', 'figpath', 'metapath'])

# Model configuration - should match all_pts_seizure_annotation.py
# model_dict = {
#     'model_name': 'GIN', 
#     'sequence_length': 12, 
#     'forecast_length': 1, 
#     'suffix': '',
#     'metric': 'mse'
# }
model_dict = {
    'model_name': 'LiNDDA', 
    'sequence_length': 3, 
    'forecast_length': 2, 
    'suffix': '',
    'metric': 'mse'
}
# model_dict = {
#     'model_name': 'WVNT',
#     'sequence_length': None,
#     'forecast_length': None,
#     'suffix': '',
#     'metric': 'prob'
# }
threshold_agg = 'median'
thresh_str = f'pretrained_{threshold_agg}'

def dice_score(set1, set2):
    """Compute Dice score between two sets of onset channels"""
    intersection = len(set(set1) & set(set2))
    return (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0

# Load summary data with new naming convention
summary_filename = (f"seizure_spread_summary_mdl-{model_dict['model_name']}_"
                   f"seq-{model_dict['sequence_length']}_"
                   f"forecast-{model_dict['forecast_length']}_"
                   f"thresh-{thresh_str}.pkl")

summary_df = pd.read_pickle(ospj(prodatapath, summary_filename))
metadata_df = pd.read_csv(ospj(metapath,'metadata_v7_BIDS.csv'))
metadata_df['patient'] = metadata_df['Patient']
metadata_df.drop([col for col in metadata_df.columns if 'Unnamed' in col],axis=1,inplace=True)
metadata_df['notes'] = metadata_df['notes'].fillna('')
metadata_df = metadata_df[metadata_df['stim'] == 0]
# metadata_df = metadata_df[metadata_df['notes'].apply(lambda x: 'Nina' not in x)]
metadata_df['onset'] = metadata_df['onset'].astype(int).astype(float)
summary_df = summary_df.merge(metadata_df,how='inner',on=['patient','onset'])

# Initialize lists to store the results
dice_score_pairs = []
spearman_score_pairs = []
onset_distance_pairs = []
pair_patients = []
onset_1 = []
onset_2 = []
# count = 0
for patient, group in tqdm(summary_df.groupby('patient')):
    # count += 1
    # if count > 10:
    #     continue
    # Identify superset of all channels across the patient's seizures
    if len(group) < 2:
        continue
    patient_channel_superset = set()
    for spread_rank_dict in group['channel_spread_rank']:
        patient_channel_superset.update(spread_rank_dict.keys())
    
    # Calculate temporal distance between seizures
    onset_time_pairs = list(combinations(group['onset'],2))
    onset_distances = [np.abs(float(pair[0]) - float(pair[1])) for pair in onset_time_pairs]
    first_onset,second_onset = zip(*onset_time_pairs)

    # Calculate Dice similarity score for onset channels
    onset_channel_pairs = list(combinations(group['onset_channels'], 2))
    dice_similarities = [dice_score(pair[0], pair[1]) for pair in onset_channel_pairs]
    

    # Calculate Spearman correlation for spread ranks with superset channels added
    spread_rank_pairs = list(combinations(group['channel_spread_rank'], 2))
    spearman_correlations = []
    
    for rank1, rank2 in spread_rank_pairs:
        # Convert spread rank dictionaries to pandas Series
        rank1_series = pd.Series(rank1)
        rank2_series = pd.Series(rank2)
        
        # Add superset channels to each rank with last rank value
        last_rank_value = len(rank1_series) + 1
        for channel in patient_channel_superset:
            if channel not in rank1_series:
                rank1_series[channel] = last_rank_value
            if channel not in rank2_series:
                rank2_series[channel] = last_rank_value
        
        # Sort to align channels and calculate Spearman correlation
        rank1_series = rank1_series.sort_index()
        rank2_series = rank2_series.sort_index()
        spearman_corr, _ = spearmanr(rank1_series, rank2_series)
        spearman_correlations.append(spearman_corr)
    
    assert (len(dice_similarities) == len(spearman_correlations)) & (len(spearman_correlations) == len(onset_distances))

    dice_score_pairs.extend(dice_similarities)
    spearman_score_pairs.extend(spearman_correlations)
    onset_distance_pairs.extend(onset_distances)
    pair_patients.extend([patient]*len(onset_distances))
    onset_1.extend(first_onset)
    onset_2.extend(second_onset)

# Convert to DataFrame for easy viewing and analysis
dice_df = pd.DataFrame(np.array([pair_patients,onset_1,onset_2,dice_score_pairs]).T,columns=['patient','onset_1','onset_2','onset_dice'])
spearman_df = pd.DataFrame(np.array([pair_patients,onset_1,onset_2,spearman_score_pairs]).T,columns=['patient','onset_1','onset_2','spread_rank'])
temporal_df = pd.DataFrame(np.array([pair_patients,onset_1,onset_2,onset_distance_pairs]).T,columns=['patient','onset_1','onset_2','temporal_distance'])

# Merge results into a single DataFrame
temp_df = pd.merge(dice_df, spearman_df, on=['patient','onset_1','onset_2'])
similarity_results = pd.merge(temp_df,temporal_df, on=['patient','onset_1','onset_2'])

# Save results with new naming convention
output_filename = (f"pair_similarity_analysis_mdl-{model_dict['model_name']}_"
                  f"seq-{model_dict['sequence_length']}_"
                  f"forecast-{model_dict['forecast_length']}_"
                  f"thresh-{thresh_str}.pkl")

similarity_results.to_pickle(ospj(prodatapath, output_filename))
print(f"\nSaved pair similarity analysis to {output_filename}")
print(f"Total pairs analyzed: {len(similarity_results)}")