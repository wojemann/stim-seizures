# Region-Level Analysis Implementation Summary

## Completed Changes to `test_validation_analysis.py`

### 1. Modified Prediction Logic ✅
**Changed from:** Threshold-then-count (≥4/5 timepoints above threshold)  
**Changed to:** Average-then-threshold (average of 5 timepoints ≥ threshold)

```python
# Old approach:
ueo_idx = np.sum(sz_clf[:, onset_idx:onset_idx+5], axis=1) >= 4

# New approach:
ueo_idx = sz_prob[:, onset_idx:onset_idx+5].mean(axis=1) > threshold
```

### 2. Added Helper Functions ✅

#### `load_electrode_localizations(patient, prodatapath)`
- Loads RID mapping from `electrode_localizations/summary.csv`
- Extracts HUP number and finds corresponding RID
- Loads electrode localization CSV: `electrode_localizations/sub-RID{rid}.csv`
- Filters out excluded regions (white matter, ventricle, CSF, outside, unknown, emptylabel)
- Returns dictionary mapping first contacts to region names
- Returns `None` if any step fails (gracefully handled)

#### `map_channels_to_regions(channel_list, ch_to_region)`
- Converts list of channel names (first contacts) to unique region names
- Used for mapping clinical annotations to regions

#### `aggregate_to_regions(sz_prob, prob_chs, ch_to_region)`
- Groups channels by region
- Averages probabilities across all channels within each region
- Returns region probability matrix (n_regions × n_timepoints) and region names
- Returns `None, None` if no valid regions found

### 3. Integration into Main Loop ✅

**Per-seizure setup (outside model loops):**
- Loads electrode localizations once per patient/seizure
- Maps onset and spread consensus annotations to regions
- Maps per-annotator annotations to regions
- Stores region annotations for use by all models

**Per-model processing (both NDD and benchmark):**
- Aggregates smoothed channel probabilities to regions
- Converts region annotation lists to boolean arrays
- Calculates all metrics at region level:
  - Inter-rater reliability
  - AUC (onset and spread)
  - Average SOZ/non-SOZ probabilities
  - Optimal threshold search (Phi maximization)
  - Phi at learned thresholds (phi, f1, iou)
  - Per-annotator Phi values
- Gracefully handles missing region data (sets all to np.nan)

### 4. Output Schema ✅

**New CSV file:** `test_validation_results_channels_and_regions.csv`

**Channel-level metrics (existing):**
- `onset_inter_rater_reliability`, `spread_inter_rater_reliability`
- `avg_onset_soz_prob`, `avg_onset_nsoz_prob`, `onset_auc`
- `avg_spread_soz_prob`, `avg_spread_nsoz_prob`, `spread_auc`
- `optimal_onset_threshold`, `max_onset_phi`, `onset_phi_annotators_at_optimal`
- `optimal_spread_threshold`, `max_spread_phi`, `spread_phi_annotators_at_optimal`
- Phi at learned thresholds (×3 for phi/f1/iou)

**Region-level metrics (new, parallel structure):**
- `region_onset_inter_rater_reliability`, `region_spread_inter_rater_reliability`
- `region_avg_onset_soz_prob`, `region_avg_onset_nsoz_prob`, `region_onset_auc`
- `region_avg_spread_soz_prob`, `region_avg_spread_nsoz_prob`, `region_spread_auc`
- `region_optimal_onset_threshold`, `region_max_onset_phi`, `region_onset_phi_annotators_at_optimal`
- `region_optimal_spread_threshold`, `region_max_spread_phi`, `region_spread_phi_annotators_at_optimal`
- Region phi at learned thresholds (×3 for phi/f1/iou)

### 5. Key Design Decisions ✅

1. **Single electrode localization load:** Loaded once per seizure, reused for all models
2. **Graceful degradation:** If region mapping unavailable, all region metrics = np.nan
3. **Region filtering:** Excludes white matter, ventricles, CSF, outside brain, unknown
4. **Bipolar channel mapping:** Uses first contact to determine region assignment
5. **Region aggregation:** Simple mean across all channels within each region
6. **Annotation mapping:** Region positive if ANY constituent channel is positive
7. **Minimal code duplication:** Region analysis reuses all existing metric functions

### 6. Data Flow

```
PER SEIZURE:
├─ Load clinical annotations (channels)
├─ Load electrode localizations → ch_to_region dict
├─ Map clinical annotations: channels → regions
└─ FOR EACH MODEL:
    ├─ Load and smooth channel probabilities
    ├─ CHANNEL-LEVEL:
    │   ├─ Calculate AUC, avg probs
    │   ├─ Find optimal threshold
    │   └─ Calculate Phi at learned thresholds
    ├─ REGION-LEVEL:
    │   ├─ Aggregate: channels → regions
    │   ├─ Calculate AUC, avg probs
    │   ├─ Find optimal threshold
    │   └─ Calculate Phi at learned thresholds
    └─ Store both channel and region metrics
```

### 7. Testing Recommendations

1. Verify RID mapping works for all patients in test set
2. Check that region counts are reasonable (typically 10-50 per patient)
3. Confirm region probabilities stay in [0, 1] range
4. Spot-check that region metrics are correlated but smoother than channel metrics
5. Verify np.nan handling for patients without electrode localizations

### 8. File Locations

**Input:**
- Electrode localizations: `{prodatapath}/electrode_localizations/sub-RID{rid}.csv`
- RID mapping: `{prodatapath}/electrode_localizations/summary.csv`

**Output:**
- Results: `{prodatapath}/test_validation_results_channels_and_regions.csv`

---

## Implementation Complete

All requirements from IMPLEMENTATION_REPORT.md have been fulfilled:
- ✅ Channel-to-region mapping
- ✅ Probability aggregation to regions
- ✅ Clinical annotation mapping to regions
- ✅ All metrics calculated at region level
- ✅ Parallel output structure (channel + region)
- ✅ Graceful handling of missing data
- ✅ Modified prediction logic (average-then-threshold)

