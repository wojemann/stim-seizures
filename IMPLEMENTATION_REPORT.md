# Region-Level Model Performance Implementation Report

## Project Overview
This report documents the implementation of a validation analysis script (`test_validation_analysis.py`) that evaluates NDD models (LiNDDA, GIN) and benchmark models (HFER, ABSSLP, IMPRINT, WVNT) on spontaneous seizures at both channel-level and region-level.

---

## COMPLETED WORK

### 1. Script Structure and Data Loading

**File:** `/mnt/sauce/littlab/users/wojemann/stim-seizures/code/test_validation_analysis.py`

**Completed Components:**

#### 1.1 Data Loading Functions
- ✅ `load_ndd_probability_files()`: Loads LiNDDA and GIN probability matrices from patient-specific pickle files
- ✅ `load_benchmark_probability_files()`: Loads HFER, ABSSLP, IMPRINT, WVNT probability matrices
- ✅ Clinical annotation loading from `threshold_tuning_consensus_v2.pkl`
- ✅ Seizure metadata loading from `seizures_df` with filtering for `split == 2` and `stim == 0`

**Key Data Structures:**
```python
# Seizure metadata
seizures_df = pd.read_csv(ospj(prodatapath, 'seizures.csv'))
test_seizures = seizures_df[(seizures_df['split'] == 2) & (seizures_df['stim'] == 0)]

# Clinical annotations
annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v2.pkl"))
# Merged using: seizures_df['approximate_onset'] == annotations_df['onset']

# Probability matrices (shape: n_channels × n_timepoints)
sz_prob = model.get_seizure_prob(...)  # Raw probabilities
sz_prob_smooth = sc.ndimage.uniform_filter1d(sz_prob, size=20, ...)  # 20-point smoothing
```

#### 1.2 Preprocessing
- ✅ 20-point uniform filter smoothing applied to all probability matrices (matching `val-generate_model_annotations.py`)
- ✅ Temporal alignment with clinical onset using `first_onset_idx`
- ✅ Channel label cleaning using `clean_labels()` from utils

#### 1.3 Clinical Annotation Extraction
- ✅ Consensus onset labels: `annot_row['ueo_consensus']` (unique earliest onset)
- ✅ Consensus spread labels: `annot_row['sec_consensus']` (10-second spread)
- ✅ Per-annotator onset labels: `annot_row['ueo_annot_X']` where X is annotator ID
- ✅ Per-annotator spread labels: `annot_row['sec_annot_X']`
- ✅ Handling of missing annotations with proper skip logic

---

### 2. Channel-Level Metric Calculation

**All metrics below are FULLY IMPLEMENTED and working:**

#### 2.1 Metrics Calculated Without Thresholding
- ✅ **Average SOZ NDD**: Mean probability of channels in onset labels
- ✅ **Average non-SOZ NDD**: Mean probability of channels not in onset labels
- ✅ **Onset AUC**: Area under ROC curve using onset labels as ground truth
- ✅ **Spread AUC**: Area under ROC curve using spread labels as ground truth

**Function:** `calculate_auc_and_probs(sz_prob, prob_chs, onset_labels, onset_idx)`

#### 2.2 Optimal Threshold Finding (Phi Maximization)
- ✅ Iterates through **all unique probability values** in the onset window (not a fixed grid)
- ✅ At each threshold, converts probabilities to binary predictions
- ✅ Calculates Matthews Correlation Coefficient (Phi) between predictions and consensus labels
- ✅ Tracks optimal threshold that maximizes Phi
- ✅ Calculates Phi with each individual annotator at optimal threshold
- ✅ Separate optimization for onset and spread tasks

**Function:** `find_optimal_thresholds(prob_data_onset, prob_data_spread, onset_mask, spread_mask, ...)`

**Implementation Detail:**
```python
# Extract unique probabilities for threshold candidates
unique_probs_onset = np.unique(prob_data_onset.values[onset_mask])
unique_probs_spread = np.unique(prob_data_spread.values[spread_mask])

# For each unique probability as threshold
for threshold in unique_probs_onset:
    predicted_labels = [ch for ch in all_chs if onset_pred[ch] >= threshold]
    phi = calculate_phi(predicted_labels, consensus_labels, all_chs)
    # Track maximum phi and corresponding threshold
```

#### 2.3 Phi at Learned Thresholds
- ✅ Loads pre-computed thresholds from external validation dataset:
  - `ndd_thresholds.csv`: Contains thresholds optimized for Phi, F1, IOU on external data
  - `benchmark_thresholds.csv`: Same for benchmark models
- ✅ Applies these learned thresholds to current test seizures
- ✅ Calculates Phi at each learned threshold (3 thresholds per model: phi_threshold, f1_threshold, iou_threshold)
- ✅ Stores per-annotator Phi values at learned thresholds

**Function:** `calculate_phi_at_learned_thresholds(..., learned_thresholds)`

#### 2.4 Inter-Rater Reliability
- ✅ Calculates average pairwise Phi between all annotator pairs
- ✅ Separate calculation for onset and spread tasks
- ✅ Returns `np.nan` for seizures with <2 annotators

**Function:** `calculate_inter_rater_reliability(annotators, all_labels)`

**Formula:**
```python
# For each pair of annotators (i, j):
phi_ij = calculate_phi(annotator_i_labels, annotator_j_labels, all_channels)

# Average across all pairs:
inter_rater_reliability = mean(phi_ij for all pairs)
```

---

### 3. Output Data Structure

**Completed:** Channel-level results stored in pandas DataFrame with following columns:

```python
results = []
for each seizure:
    for each model:
        result_dict = {
            # Identifiers
            'patient': patient,
            'seizure': onset_run,
            'approximate_onset': approximate_onset,
            'model_name': model_name,
            
            # Channel-level metrics (NO THRESHOLD)
            'onset_inter_rater_reliability': float,
            'spread_inter_rater_reliability': float,
            'avg_soz_prob': float,
            'avg_nsoz_prob': float,
            'onset_auc': float,
            'spread_auc': float,
            
            # Optimal threshold metrics
            'optimal_onset_threshold': float,
            'max_onset_phi': float,
            'onset_phi_annotators': list[float],  # One per annotator
            'optimal_spread_threshold': float,
            'max_spread_phi': float,
            'spread_phi_annotators': list[float],
            
            # Learned threshold metrics (from external dataset)
            'onset_phi_at_learned_phi_threshold': float,
            'onset_phi_annotators_at_learned_phi_threshold': list[float],
            'spread_phi_at_learned_phi_threshold': float,
            'spread_phi_annotators_at_learned_phi_threshold': list[float],
            # ... (same pattern for f1_threshold and iou_threshold)
            
            # Clinical annotations (for reference)
            'onset_consensus': list[str],  # Channel names
            'spread_consensus': list[str],
            'n_onset_annotators': int,
            'n_spread_annotators': int,
        }
```

**Saved to:** `results/test_validation_results.csv`

---

### 4. Example Figure Generation

- ✅ Created threshold vs. Phi plot for HUP238, seizure 290006, LiNDDA model
- ✅ Shows both onset and spread Phi curves across all unique probability thresholds
- ✅ Marks optimal thresholds with vertical lines

**Saved to:** `results/figures/threshold_vs_phi_HUP238_290006_LiNDDA.png`

---

## PENDING WORK: Region-Level Analysis

### Overview
The region-level analysis requires:
1. Loading electrode localization files
2. Mapping channels to brain regions (DKT atlas)
3. Aggregating probabilities within regions
4. Mapping clinical annotations to regions
5. Re-calculating all metrics at region level

---

### 5. Region Mapping Implementation (TO DO)

#### 5.1 Load Region Localization Files

**Location:** `/mnt/sauce/littlab/users/wojemann/dynasd_data/PROCESSED_DATA/electrode_localizations/{patient}.csv`

**Expected CSV Structure:**
```
channel,label,x,y,z,...
LA01,Left-Hippocampus,12.3,45.6,78.9,...
LA02,ctx-lh-fusiform,10.2,43.5,76.8,...
LA03,Left-Cerebral-White-Matter,8.1,41.4,74.7,...
```

**Function to implement:**
```python
def load_region_mapping(patient, prodatapath):
    """Load and clean region mapping for a patient"""
    csv_path = ospj(prodatapath, 'electrode_localizations', f'{patient}.csv')
    
    if not os.path.exists(csv_path):
        return None  # Set all region metrics to np.nan
    
    region_df = pd.read_csv(csv_path)
    
    # Filter out excluded regions
    excluded_keywords = ['white', 'ventricle', 'csf', 'outside', 'unknown', 'cerebral-white-matter']
    mask = True
    for keyword in excluded_keywords:
        mask &= ~region_df['label'].str.lower().str.contains(keyword)
    region_df = region_df[mask]
    
    # Clean channel labels to match probability matrix format
    region_df['channel_clean'] = [clean_labels([ch], patient)[0] for ch in region_df['channel']]
    
    return region_df
```

#### 5.2 Create Channel-to-Region Mapping

**Implementation:**
```python
# For bipolar montage, extract first contact
ch_to_region = {}
for idx, row in region_df.iterrows():
    ch_to_region[row['channel_clean']] = row['label']

# When mapping bipolar channels (e.g., "LA01-LA02"):
def get_region_for_bipolar(bipolar_ch, ch_to_region):
    """Map bipolar channel to region using first contact"""
    first_contact = bipolar_ch.split('-')[0]
    return ch_to_region.get(first_contact, None)
```

---

### 6. Probability Aggregation to Regions (TO DO)

#### 6.1 Group Channels by Region

**Implementation:**
```python
# For each seizure/model pair after smoothing
sz_prob_smooth  # Shape: (n_channels, n_timepoints)
prob_chs        # List of channel names

# Group probabilities by region
region_prob_dict = {}  # {region_name: list of channel probability arrays}

for i, ch in enumerate(prob_chs):
    region = get_region_for_bipolar(ch, ch_to_region)
    
    if region is None:
        continue  # Skip channels without region mapping
    
    if region not in region_prob_dict:
        region_prob_dict[region] = []
    
    region_prob_dict[region].append(sz_prob_smooth[i, :])  # Add full timeseries
```

#### 6.2 Average Probabilities Within Regions

**Implementation:**
```python
# Convert to region-level probability matrix
region_names = sorted(region_prob_dict.keys())
region_prob_list = []

for region in region_names:
    # Average across all channels in this region
    region_avg = np.mean(region_prob_dict[region], axis=0)  # Shape: (n_timepoints,)
    region_prob_list.append(region_avg)

region_prob_matrix = np.array(region_prob_list)  # Shape: (n_regions, n_timepoints)
```

**Key Principle:** Each region's probability at time t is the **average** of all its constituent channels' probabilities at time t.

---

### 7. Map Clinical Annotations to Regions (TO DO)

#### 7.1 Channel-to-Region Annotation Mapping

**Implementation:**
```python
def map_channels_to_regions(channel_list, ch_to_region, patient):
    """Convert list of channel names to list of region names"""
    region_set = set()
    
    for ch in channel_list:
        # Clean channel label
        ch_clean = clean_labels([ch], patient)[0]
        
        # Get region for first contact in bipolar pair
        first_contact = ch_clean.split('-')[0]
        region = ch_to_region.get(first_contact)
        
        if region is not None:
            region_set.add(region)
    
    return list(region_set)
```

#### 7.2 Apply Mapping to All Annotation Sets

**Implementation:**
```python
# Consensus annotations
onset_labels_regions = map_channels_to_regions(onset_labels, ch_to_region, patient)
ueo_consensus_regions = map_channels_to_regions(ueo_consensus, ch_to_region, patient)
sec_consensus_regions = map_channels_to_regions(sec_consensus, ch_to_region, patient)

# Per-annotator annotations
ueo_annotators_regions = [
    map_channels_to_regions(annotator_chs, ch_to_region, patient)
    for annotator_chs in ueo_annotators
]

sec_annotators_regions = [
    map_channels_to_regions(annotator_chs, ch_to_region, patient)
    for annotator_chs in sec_annotators
]
```

**Important:** Multiple channels can map to the same region. The region is labeled positive if **any** of its constituent channels were labeled positive by the clinician.

---

### 8. Calculate Region-Level Metrics (TO DO)

#### 8.1 Reuse Existing Functions

**Key Insight:** All existing metric calculation functions can be reused with region data:

```python
# EXISTING FUNCTION SIGNATURES (no changes needed):
def calculate_auc_and_probs(sz_prob, prob_chs, onset_labels, onset_idx):
def calculate_inter_rater_reliability(annotators, all_labels):
def find_optimal_thresholds(prob_data_onset, prob_data_spread, onset_mask, spread_mask, 
                            prob_times, onset_idx, spread_idx, all_chs, 
                            ueo_consensus, sec_consensus, ueo_annotators, sec_annotators):
def calculate_phi_at_learned_thresholds(...):
```

**Usage with region data:**
```python
# CHANNEL-LEVEL (current implementation):
onset_auc, avg_soz, avg_nsoz, spread_auc = calculate_auc_and_probs(
    sz_prob_smooth,     # (n_channels, n_timepoints)
    prob_chs,           # List of channel names
    onset_labels,       # List of onset channel names
    onset_idx
)

# REGION-LEVEL (to be implemented):
region_onset_auc, region_avg_soz, region_avg_nsoz, region_spread_auc = calculate_auc_and_probs(
    region_prob_matrix,      # (n_regions, n_timepoints) - AGGREGATED
    region_names,            # List of region names
    onset_labels_regions,    # List of onset region names - MAPPED
    onset_idx
)
```

#### 8.2 Complete Region-Level Calculation Pipeline

**Pseudocode:**
```python
# After loading sz_prob and smoothing...

# 1. Load region mapping
region_df = load_region_mapping(patient, prodatapath)

if region_df is None:
    # No region file found - set all region metrics to np.nan
    result_dict.update({
        'region_onset_inter_rater_reliability': np.nan,
        'region_spread_inter_rater_reliability': np.nan,
        'region_avg_soz_prob': np.nan,
        # ... all other region metrics set to np.nan
    })
    continue

# 2. Create channel-to-region mapping
ch_to_region = dict(zip(region_df['channel_clean'], region_df['label']))

# 3. Aggregate probabilities to regions
region_prob_matrix, region_names = aggregate_to_regions(sz_prob_smooth, prob_chs, ch_to_region)

# 4. Map clinical annotations to regions
onset_labels_regions = map_channels_to_regions(onset_labels, ch_to_region, patient)
ueo_consensus_regions = map_channels_to_regions(ueo_consensus, ch_to_region, patient)
sec_consensus_regions = map_channels_to_regions(sec_consensus, ch_to_region, patient)
ueo_annotators_regions = [map_channels_to_regions(a, ch_to_region, patient) for a in ueo_annotators]
sec_annotators_regions = [map_channels_to_regions(a, ch_to_region, patient) for a in sec_annotators]

# 5. Calculate region-level metrics (same functions, different inputs)
region_onset_auc, region_avg_soz, region_avg_nsoz, region_spread_auc = \
    calculate_auc_and_probs(region_prob_matrix, region_names, onset_labels_regions, onset_idx)

region_onset_inter_rater = calculate_inter_rater_reliability(ueo_annotators_regions, region_names)
region_spread_inter_rater = calculate_inter_rater_reliability(sec_annotators_regions, region_names)

region_optimal = find_optimal_thresholds(
    region_prob_matrix[:, onset_idx:],  # Onset window
    region_prob_matrix[:, spread_idx:], # Spread window
    onset_mask_regions,
    spread_mask_regions,
    prob_times[onset_idx:],
    onset_idx,
    spread_idx,
    region_names,  # all_labels = all regions
    ueo_consensus_regions,
    sec_consensus_regions,
    ueo_annotators_regions,
    sec_annotators_regions
)

region_learned_results = calculate_phi_at_learned_thresholds(
    # ... same pattern with region data
)

# 6. Update result_dict with region metrics
result_dict.update({
    'region_onset_inter_rater_reliability': region_onset_inter_rater,
    'region_spread_inter_rater_reliability': region_spread_inter_rater,
    'region_avg_soz_prob': region_avg_soz,
    'region_avg_nsoz_prob': region_avg_nsoz,
    'region_onset_auc': region_onset_auc,
    'region_spread_auc': region_spread_auc,
    'region_optimal_onset_threshold': region_optimal['optimal_onset_threshold'],
    'region_max_onset_phi': region_optimal['max_onset_phi'],
    'region_onset_phi_annotators_at_optimal': region_optimal['onset_phi_annotators'],
    # ... continue for all region metrics
})
```

---

### 9. Implementation Checklist

**Functions to Add:**
- [ ] `load_region_mapping(patient, prodatapath)` - Load and filter region CSV
- [ ] `get_region_for_bipolar(bipolar_ch, ch_to_region)` - Map bipolar channel to region
- [ ] `aggregate_to_regions(sz_prob, prob_chs, ch_to_region)` - Create region probability matrix
- [ ] `map_channels_to_regions(channel_list, ch_to_region, patient)` - Convert channel labels to region labels

**Main Loop Modifications:**
- [ ] After smoothing probabilities, check if region file exists
- [ ] If exists: perform region aggregation and annotation mapping
- [ ] Call all existing metric functions with region data
- [ ] Store region-level results in result_dict
- [ ] If not exists: set all region metrics to `np.nan`

**Output Updates:**
- [ ] Add region-level columns to results DataFrame (parallel to channel-level)
- [ ] Ensure CSV export includes all new region columns

---

### 10. Data Flow Summary

```
CHANNEL-LEVEL (✅ COMPLETE):
Raw Probabilities → Smoothing → Channel Metrics → Results DF

REGION-LEVEL (⏳ TO DO):
Raw Probabilities → Smoothing → [Aggregate to Regions] → Region Metrics → Results DF
                                      ↓
Clinical Annotations (channels) → [Map to Regions] ────────┘
```

---

### 11. Testing Recommendations

When implementing region-level analysis:

1. **Test with one seizure first:** HUP238, onset 290006 (already used for figure)
2. **Check region counts:** Verify reasonable number of regions (10-50 typical)
3. **Verify aggregation:** Ensure region probabilities are in [0, 1] range
4. **Compare channel vs. region:** Expect region metrics to be similar but smoother
5. **Handle edge cases:**
   - Seizures with no region mapping file → all region metrics = np.nan
   - Channels not in region file → excluded from region analysis
   - Regions with only 1 channel → still calculate normally
   - Empty region lists after filtering → handle gracefully

---

### 12. File Locations Reference

**Input Files:**
- Seizure metadata: `/mnt/sauce/littlab/users/wojemann/dynasd_data/PROCESSED_DATA/seizures.csv`
- Clinical annotations: `/mnt/sauce/littlab/users/wojemann/dynasd_data/PROCESSED_DATA/threshold_tuning_consensus_v2.pkl`
- NDD probabilities: `/mnt/sauce/littlab/users/wojemann/dynasd_data/PROCESSED_DATA/NDD_ICTAL_DATA/split_*/{patient}/`
- Benchmark probabilities: `/mnt/sauce/littlab/users/wojemann/dynasd_data/PROCESSED_DATA/NDD_ICTAL_DATA/{model}/split_*/{patient}/`
- Region mappings: `/mnt/sauce/littlab/users/wojemann/dynasd_data/PROCESSED_DATA/electrode_localizations/{patient}.csv` (⚠️ TO BE ADDED)
- Learned thresholds: 
  - `/mnt/sauce/littlab/users/wojemann/dynasd_data/PROCESSED_DATA/PERFORMANCE_CSVS/ndd_thresholds.csv`
  - `/mnt/sauce/littlab/users/wojemann/dynasd_data/PROCESSED_DATA/PERFORMANCE_CSVS/benchmark_thresholds.csv`

**Output Files:**
- Results CSV: `/mnt/sauce/littlab/users/wojemann/stim-seizures/results/test_validation_results.csv`
- Example figure: `/mnt/sauce/littlab/users/wojemann/stim-seizures/results/figures/threshold_vs_phi_HUP238_290006_LiNDDA.png`

**Code:**
- Main script: `/mnt/sauce/littlab/users/wojemann/stim-seizures/code/test_validation_analysis.py`

---

### 13. Expected Output Schema (After Region Implementation)

**Final DataFrame columns (~80 columns total):**

**Identifiers (5):**
- patient, seizure, approximate_onset, model_name, split

**Channel-Level Metrics (35):**
- onset_inter_rater_reliability, spread_inter_rater_reliability
- avg_soz_prob, avg_nsoz_prob, onset_auc, spread_auc
- optimal_onset_threshold, max_onset_phi, onset_phi_annotators
- optimal_spread_threshold, max_spread_phi, spread_phi_annotators
- onset_phi_at_learned_{phi|f1|iou}_threshold (×3)
- onset_phi_annotators_at_learned_{phi|f1|iou}_threshold (×3)
- spread_phi_at_learned_{phi|f1|iou}_threshold (×3)
- spread_phi_annotators_at_learned_{phi|f1|iou}_threshold (×3)

**Region-Level Metrics (35, same structure as channel):**
- region_onset_inter_rater_reliability, region_spread_inter_rater_reliability
- region_avg_soz_prob, region_avg_nsoz_prob, region_onset_auc, region_spread_auc
- region_optimal_onset_threshold, region_max_onset_phi, region_onset_phi_annotators
- ... (same pattern as channel-level with "region_" prefix)

**Clinical Annotations (5):**
- onset_consensus, spread_consensus
- n_onset_annotators, n_spread_annotators
- annotation_source

---

### 14. Key Implementation Notes

1. **No changes to existing functions:** All metric calculation functions work as-is
2. **Parallel structure:** Region metrics mirror channel metrics exactly
3. **Missing data handling:** Region file missing → all region metrics = np.nan
4. **Excluded regions filter:** Applied during region loading, not during analysis
5. **Bipolar mapping:** First contact of bipolar pair determines region assignment
6. **Aggregation method:** Simple mean across channels within each region
7. **Label propagation:** Region labeled positive if ANY constituent channel is positive

---

## Questions for Clarification

1. **Region file availability:** When will electrode localization CSVs be available at specified path?
2. **Region name standardization:** Are region labels already standardized (e.g., "ctx-lh-fusiform" vs "Left Fusiform")?
3. **Excluded regions:** Confirm list of exclusion keywords is complete
4. **Missing regions:** If some seizure has no region file, is np.nan for all region metrics acceptable?

---

## Estimated Implementation Time

- Add 4 helper functions: ~30 minutes
- Integrate into main loop: ~30 minutes
- Test on single seizure: ~15 minutes
- Debug and edge cases: ~30 minutes
- Full run on test dataset: ~2 hours (depending on size)

**Total: ~3-4 hours of work**

---

## Contact for Questions

**User:** wojemann
**Project:** stim-seizures
**Date:** October 16, 2025
**Python Environment:** stim-env (Python 3.6)

