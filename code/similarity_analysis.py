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
# model_dict = {
#     'model_name': 'LiNDDA', 
#     'sequence_length': 3, 
#     'forecast_length': 2, 
#     'suffix': '',
#     'metric': 'mse'
# }
model_dict = {
    'model_name': 'WVNT',
    'sequence_length': None,
    'forecast_length': None,
    'suffix': '',
    'metric': 'prob'
}
threshold_agg = 'mean'
thresh_str = f'pretrained_{threshold_agg}'

def dice_score(set1, set2):
    """Compute Dice score between two sets"""
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
results_list = []

for patient, group in tqdm(summary_df.groupby('patient'), desc="Analyzing patients"):
    if len(group) < 2:
        continue
    
    # ==================== CHANNEL-LEVEL ANALYSIS ====================
    # Identify superset of all channels across the patient's seizures
    patient_channel_superset = set()
    for spread_rank_dict in group['channel_spread_rank']:
        patient_channel_superset.update(spread_rank_dict.keys())
    
    # Calculate Dice similarity score for onset channels
    onset_channel_pairs = list(combinations(group['onset_channels'], 2))
    dice_similarities_ch = [dice_score(pair[0], pair[1]) for pair in onset_channel_pairs]
    avg_dice_similarity_ch = np.mean(dice_similarities_ch) if dice_similarities_ch else None
    median_dice_similarity_ch = np.median(dice_similarities_ch) if dice_similarities_ch else None
    max_dice_similarity_ch = np.max(dice_similarities_ch) if dice_similarities_ch else None
    var_dice_similarity_ch = np.var(dice_similarities_ch) if dice_similarities_ch else None

    # Calculate NDD correlation for onset channels
    ndd_vals_ch = group['onset_ndd_ind_ch'].apply(lambda x: x[1] if len(x) > 1 else pd.Series())
    t_pear_ch = np.triu(ndd_vals_ch.T.corr(method='pearson'), 1)
    t_spear_ch = np.triu(ndd_vals_ch.T.corr(method='spearman'), 1)
    avg_ndd_pearson_ch = np.mean(t_pear_ch[t_pear_ch != 0]) if np.any(t_pear_ch != 0) else None
    avg_ndd_spearman_ch = np.mean(t_spear_ch[t_spear_ch != 0]) if np.any(t_spear_ch != 0) else None

    # Calculate Spearman correlation for spread ranks with superset channels added
    spread_rank_pairs_ch = list(combinations(group['channel_spread_rank'], 2))
    spearman_correlations_ch = []
    
    for rank1, rank2 in spread_rank_pairs_ch:
        rank1_series = pd.Series(rank1)
        rank2_series = pd.Series(rank2)
        
        # Add superset channels to each rank with last rank value
        last_rank_value = len(rank1_series) + 1
        for channel in patient_channel_superset:
            if channel not in rank1_series:
                rank1_series[channel] = last_rank_value
            if channel not in rank2_series:
                rank2_series[channel] = last_rank_value
        rank1_series = rank1_series.fillna(last_rank_value)
        rank2_series = rank2_series.fillna(last_rank_value)
        
        # Sort to align channels and calculate Spearman correlation
        rank1_series = rank1_series.sort_index()
        rank2_series = rank2_series.sort_index()
        spearman_corr, _ = spearmanr(rank1_series, rank2_series)
        if np.isnan(spearman_corr):
            spearman_corr = 0
        spearman_correlations_ch.append(spearman_corr)

    avg_spearman_ch = np.mean(spearman_correlations_ch) if spearman_correlations_ch else None
    median_spearman_ch = np.median(spearman_correlations_ch) if spearman_correlations_ch else None
    max_spearman_ch = np.max(spearman_correlations_ch) if spearman_correlations_ch else None
    var_spearman_ch = np.var(spearman_correlations_ch) if spearman_correlations_ch else None

    percent_seizing_ch = np.median(np.vstack(group.fraction_seizing_ch.to_numpy()), axis=0)

    # ==================== REGION-LEVEL ANALYSIS ====================
    # Initialize region variables
    avg_dice_similarity_region = None
    median_dice_similarity_region = None
    max_dice_similarity_region = None
    var_dice_similarity_region = None
    avg_ndd_pearson_region = None
    avg_ndd_spearman_region = None
    avg_spearman_region = None
    median_spearman_region = None
    max_spearman_region = None
    var_spearman_region = None
    percent_seizing_region = None
    
    # Check if region data is available
    if 'onset_regions' in group.columns and group['onset_regions'].apply(len).sum() > 0:
        # Identify superset of all regions across the patient's seizures
        patient_region_superset = set()
        for spread_rank_dict in group['region_spread_rank']:
            if isinstance(spread_rank_dict, dict):
                patient_region_superset.update(spread_rank_dict.keys())
        
        if len(patient_region_superset) > 0:
            # Calculate Dice similarity score for onset regions
            onset_region_pairs = list(combinations(group['onset_regions'], 2))
            dice_similarities_region = [dice_score(pair[0], pair[1]) for pair in onset_region_pairs if len(pair[0]) > 0 and len(pair[1]) > 0]
            if dice_similarities_region:
                avg_dice_similarity_region = np.mean(dice_similarities_region)
                median_dice_similarity_region = np.median(dice_similarities_region)
                max_dice_similarity_region = np.max(dice_similarities_region)
                var_dice_similarity_region = np.var(dice_similarities_region)

            # Calculate NDD correlation for onset regions
            ndd_vals_region = group['onset_ndd_ind_region'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else pd.Series())
            if not ndd_vals_region.apply(lambda x: len(x) == 0).all():
                t_pear_region = np.triu(ndd_vals_region.T.corr(method='pearson'), 1)
                t_spear_region = np.triu(ndd_vals_region.T.corr(method='spearman'), 1)
                avg_ndd_pearson_region = np.mean(t_pear_region[t_pear_region != 0]) if np.any(t_pear_region != 0) else None
                avg_ndd_spearman_region = np.mean(t_spear_region[t_spear_region != 0]) if np.any(t_spear_region != 0) else None

            # Calculate Spearman correlation for spread ranks with superset regions added
            spread_rank_pairs_region = [(r1, r2) for r1, r2 in combinations(group['region_spread_rank'], 2) 
                                       if isinstance(r1, dict) and isinstance(r2, dict) and len(r1) > 0 and len(r2) > 0]
            spearman_correlations_region = []
            
            for rank1, rank2 in spread_rank_pairs_region:
                rank1_series = pd.Series(rank1)
                rank2_series = pd.Series(rank2)
                
                # Add superset regions to each rank with last rank value
                last_rank_value = len(rank1_series) + 1
                for region in patient_region_superset:
                    if region not in rank1_series:
                        rank1_series[region] = last_rank_value
                    if region not in rank2_series:
                        rank2_series[region] = last_rank_value
                
                # Sort to align regions and calculate Spearman correlation
                rank1_series = rank1_series.sort_index()
                rank2_series = rank2_series.sort_index()
                spearman_corr, _ = spearmanr(rank1_series, rank2_series)
                spearman_correlations_region.append(spearman_corr)

            if spearman_correlations_region:
                avg_spearman_region = np.mean(spearman_correlations_region)
                median_spearman_region = np.median(spearman_correlations_region)
                max_spearman_region = np.max(spearman_correlations_region)
                var_spearman_region = np.var(spearman_correlations_region)

            # Percent regions seizing
            region_seizing_arrays = group['fraction_seizing_region'].dropna()
            if len(region_seizing_arrays) > 0:
                percent_seizing_region = np.median(np.vstack(region_seizing_arrays.to_numpy()), axis=0)

    # Store the results for this patient
    results_list.append({
        'patient': patient,
        # Channel-level metrics
        'avg_dice_similarity_ch': avg_dice_similarity_ch,
        'median_dice_similarity_ch': median_dice_similarity_ch,
        'max_dice_similarity_ch': max_dice_similarity_ch,
        'var_dice_similarity_ch': var_dice_similarity_ch,
        'avg_ndd_pearson_ch': avg_ndd_pearson_ch,
        'avg_ndd_spearman_ch': avg_ndd_spearman_ch,
        'avg_spearman_ch': avg_spearman_ch,
        'median_spearman_ch': median_spearman_ch,
        'max_spearman_ch': max_spearman_ch,
        'var_spearman_ch': var_spearman_ch,
        'percent_seizing_ch': percent_seizing_ch,
        # Region-level metrics
        'avg_dice_similarity_region': avg_dice_similarity_region,
        'median_dice_similarity_region': median_dice_similarity_region,
        'max_dice_similarity_region': max_dice_similarity_region,
        'var_dice_similarity_region': var_dice_similarity_region,
        'avg_ndd_pearson_region': avg_ndd_pearson_region,
        'avg_ndd_spearman_region': avg_ndd_spearman_region,
        'avg_spearman_region': avg_spearman_region,
        'median_spearman_region': median_spearman_region,
        'max_spearman_region': max_spearman_region,
        'var_spearman_region': var_spearman_region,
        'percent_seizing_region': percent_seizing_region,
    })

# Convert to DataFrame
similarity_results = pd.DataFrame(results_list)

# Save results with new naming convention
output_filename = (f"patient_similarity_analysis_mdl-{model_dict['model_name']}_"
                  f"seq-{model_dict['sequence_length']}_"
                  f"forecast-{model_dict['forecast_length']}_"
                  f"thresh-{thresh_str}.pkl")

similarity_results.to_pickle(ospj(prodatapath, output_filename))
print(f"\nSaved similarity analysis to {output_filename}")
print(f"Total patients analyzed: {len(similarity_results)}")
print(f"Patients with region data: {similarity_results['avg_dice_similarity_region'].notna().sum()}")