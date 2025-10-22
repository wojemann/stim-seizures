import sys
import os
from os.path import join as ospj
from itertools import combinations
from sklearn.metrics import pairwise
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
sys.path.append(os.path.abspath(ospj(os.path.dirname(__file__), '..')))

from DSOSD.utils import load_config
usr,passpath,datapath,prodatapath,metapath,figpath = load_config(ospj('/Users/wojemann/Documents/CNT/DSOSD/','config.json'))
thresh_str = 'cval'
# Function to compute Dice score between two sets of onset channels
def dice_score(set1, set2):
    intersection = len(set(set1) & set(set2))
    return (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0
summary_df = pd.read_pickle(ospj(prodatapath,f'seizure_spread_summary_{thresh_str}_sec-10.pkl'))
metadata_df = pd.read_csv(ospj(metapath,'metadata_v3.csv'))
metadata_df['patient'] = 'HUP' + metadata_df.patient.astype(int).astype(str).apply(lambda x: x.zfill(3))
metadata_df.drop([col for col in metadata_df.columns if 'Unnamed' in col],axis=1,inplace=True)
metadata_df['stim induced'] = metadata_df['stim induced'].fillna(0)
metadata_df['notes'] = metadata_df['notes'].fillna('')
metadata_df = metadata_df[(metadata_df['stim induced'] < 1)]
metadata_df = metadata_df[metadata_df['notes'].apply(lambda x: 'Nina' not in x)]
metadata_df['onset'] = metadata_df['onset'].astype(float)
summary_df = summary_df.merge(metadata_df,how='inner',on=['patient','onset'])

# Initialize lists to store the results
dice_scores = []
spearman_scores = []
percent_scores = []
metalist = []
for patient, group in tqdm(summary_df.groupby('patient')):
    if len(group) < 2:
        continue
    # Identify superset of all channels across the patient's seizures
    patient_channel_superset = set()
    for spread_rank_dict in group['channel_spread_rank']:
        patient_channel_superset.update(spread_rank_dict.keys())
    
    # Calculate Dice similarity score for onset channels
    onset_channel_pairs = list(combinations(group['onset_channels'], 2))
    dice_similarities = [dice_score(pair[0], pair[1]) for pair in onset_channel_pairs]
    avg_dice_similarity = np.mean(dice_similarities) if dice_similarities else None
    median_dice_similarity = np.median(dice_similarities) if dice_similarities else None
    max_dice_similarity = np.max(dice_similarities) if dice_similarities else None
    var_dice_similarity = np.var(dice_similarities) if dice_similarities else None

    # Calculate NDD correlation for onset channels
    ndd_vals = group['onset_ndd_ind'].apply(lambda x: x[1])
    t_pear = np.triu(ndd_vals.T.corr(method='pearson'),1)
    t_spear = np.triu(ndd_vals.T.corr(method='spearman'),1)
    avg_ndd_pearson = np.mean(t_pear)
    avg_ndd_spearman = np.mean(t_spear)

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

    avg_spearman = np.mean(spearman_correlations) if spearman_correlations else None
    median_spearman = np.median(spearman_correlations) if spearman_correlations else None
    max_spearman = np.max(spearman_correlations) if spearman_correlations else None
    var_spearman = np.var(spearman_correlations) if spearman_correlations else None

    percent_seizing = np.median(np.vstack(group.fraction_seizing.to_numpy()),axis=0)

    # Store the results for this patient
    dice_scores.append({
        'patient': patient,
        'avg_dice_similarity': avg_dice_similarity,
        'median_dice_similarity': median_dice_similarity,
        'max_dice_similarity': max_dice_similarity,
        'var_dice_similarity': var_dice_similarity
    })
    
    spearman_scores.append({
        'patient': patient,
        'avg_spearman': avg_spearman,
        'median_spearman': median_spearman,
        'max_spearman': max_spearman,
        'var_spearman': var_spearman
    })

    percent_scores.append({
        'patient': patient,
        'percent_seizing': percent_seizing
    })

# Convert to DataFrame for easy viewing and analysis
dice_df = pd.DataFrame(dice_scores)
spearman_df = pd.DataFrame(spearman_scores)
percent_df = pd.DataFrame(percent_scores)

# Merge results into a single DataFrame
temp_df = pd.merge(dice_df,spearman_df,on='patient')
similarity_results = pd.merge(temp_df, percent_df, on='patient')

# Display or save the final DataFrame
# similarity_results.to_csv(ospj(prodatapath,f"patient_similarity_analysis_{thresh_str}.csv"), index=False)
similarity_results.to_pickle(ospj(prodatapath,f"patient_similarity_analysis_{thresh_str}_sec-10.pkl"))