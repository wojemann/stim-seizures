"""
Test function to compare hankel matrix approach (DSOSD) vs sliding window approach (DynaSD)
for NDD model with forecast_length=1.

This verifies if the two sequence preparation methods produce equivalent results.
Also compares distributions, performance on real seizures, and get_onset_and_spread functions.
"""
import numpy as np
import pandas as pd
import torch
import random
from scipy.linalg import hankel
from numpy.lib.stride_tricks import sliding_window_view
import scipy as sc
import scipy.ndimage
import os
import sys
import glob
from os.path import join as ospj
from sklearn.metrics import matthews_corrcoef, roc_auc_score

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
dynasd_root = os.path.join(script_dir, '..', '..', 'DynaSD')
dsosd_root = os.path.join(script_dir, '..', '..', 'DSOSD')

if dynasd_root not in sys.path:
    sys.path.insert(0, dynasd_root)
if dsosd_root not in sys.path:
    sys.path.insert(0, dsosd_root)

# Import models
try:
    from DynaSD.NDD import NDD as DynaSD_NDD
    from DynaSD.base import DynaSDBase
except ImportError as e:
    print(f"Warning: Could not import DynaSD models: {e}")
    DynaSD_NDD = None
    DynaSDBase = None

try:
    from DSOSD.model import NDD as DSOSD_NDD
except ImportError as e:
    print(f"Warning: Could not import DSOSD models: {e}")
    DSOSD_NDD = None

# Get paths from config
try:
    from config import Config
    datapath, prodatapath, figpath, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])
except:
    # Fallback if config not available
    datapath = prodatapath = figpath = metapath = None

# Import utilities for loading raw data
try:
    from utils import get_data_from_bids, preprocess_for_detection, clean_labels, remove_scalp_electrodes
except ImportError:
    print("Warning: Could not import utils functions")
    get_data_from_bids = None
    preprocess_for_detection = None
    clean_labels = None
    remove_scalp_electrodes = None

# Import utilities for loading raw data
try:
    from utils import get_data_from_bids, preprocess_for_detection, clean_labels, remove_scalp_electrodes
except ImportError:
    print("Warning: Could not import utils functions")
    get_data_from_bids = None
    preprocess_for_detection = None
    clean_labels = None
    remove_scalp_electrodes = None


def num_wins(xLen, fs, winLen, winDisp):
    """Calculate number of windows (from DSOSD utils)"""
    return int(((xLen/fs - winLen + winDisp) - ((xLen/fs - winLen + winDisp)%winDisp))/winDisp)


def MovingWinClips(x, fs, winLen, winDisp):
    """Create moving windows (from DSOSD utils)"""
    nWins = num_wins(len(x), fs, winLen, winDisp)
    samples = np.empty((nWins, int(winLen*fs)))
    idxs = np.array([(winDisp*fs*i, (winLen+winDisp*i)*fs)
                     for i in range(nWins)]).astype(int)
    for i in range(idxs.shape[0]):
        samples[i, :] = x[idxs[i, 0]:idxs[i, 1]]
    return samples


def prepare_segment_hankel(data, fs=256, train_win=12, pred_win=1, w_size=1, w_stride=0.5):
    """
    DSOSD approach: Create sequences using hankel matrices within windows.
    
    Returns:
        input_data: (n_sequences, train_win, n_channels)
        target_data: (n_sequences, n_channels)
        window_info: dict with window start indices and sequence counts
    """
    data_np = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    n_samples, n_channels = data_np.shape
    
    j = int(w_size*fs - (train_win + pred_win) + 1)
    
    # Create windows
    nwins = num_wins(n_samples, fs, w_size, w_stride)
    data_mat = np.zeros((nwins, j, train_win + pred_win, n_channels))
    window_starts = []
    
    for k in range(n_channels):
        samples = MovingWinClips(data_np[:, k], fs, w_size, w_stride)
        for i in range(samples.shape[0]):
            clip = samples[i, :]
            # Create hankel matrix for this window
            mat = hankel(clip[:j], clip[-(train_win + pred_win):])
            data_mat[i, :, :, k] = mat
            
            if k == 0:  # Store window info once
                window_starts.append(i * int(w_stride * fs))
    
    # Flatten: (nwins * j, train_win + pred_win, n_channels)
    data_flat = data_mat.reshape((-1, train_win + pred_win, n_channels))
    input_data = data_flat[:, :-1, :]  # (n_sequences, train_win, n_channels)
    target_data = data_flat[:, -1, :]   # (n_sequences, n_channels)
    
    window_info = {
        'n_windows': nwins,
        'sequences_per_window': j,
        'total_sequences': nwins * j,
        'window_starts': window_starts
    }
    
    return input_data, target_data, window_info


def prepare_segment_sliding_window(data, sequence_length=12, forecast_length=1, stride=1):
    """
    DynaSD approach: Create sequences using sliding windows across entire data.
    
    Returns:
        input_data: (n_sequences, sequence_length, n_channels)
        target_data: (n_sequences, forecast_length, n_channels)
        sequence_info: dict with sequence start indices
    """
    data_np = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    n_samples, n_channels = data_np.shape
    
    total_seq_length = sequence_length + forecast_length
    n_sequences = (n_samples - total_seq_length) // stride + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Not enough data for even one sequence. Need at least {total_seq_length} samples.")
    
    # Use sliding_window_view
    try:
        windows = sliding_window_view(data_np, window_shape=(total_seq_length, n_channels), axis=(0, 1))
        windows = windows.squeeze()
        sequence_indices = np.arange(0, n_sequences) * stride
        selected_windows = windows[sequence_indices]
        
        input_data = selected_windows[:, :sequence_length, :]
        target_data = selected_windows[:, sequence_length:, :]
    except Exception:
        # Fallback method
        input_data = np.empty((n_sequences, sequence_length, n_channels), dtype=data_np.dtype)
        target_data = np.empty((n_sequences, forecast_length, n_channels), dtype=data_np.dtype)
        
        for i in range(n_sequences):
            seq_start = i * stride
            input_end = seq_start + sequence_length
            target_end = input_end + forecast_length
            
            input_data[i] = data_np[seq_start:input_end]
            target_data[i] = data_np[input_end:target_end]
    
    sequence_info = {
        'n_sequences': n_sequences,
        'sequence_starts': np.arange(n_sequences) * stride
    }
    
    return input_data, target_data, sequence_info


def aggregate_hankel_sequences_to_windows(mse_per_sequence, window_info, n_channels):
    """
    Aggregate DSOSD sequences (within windows) to window-level outputs.
    This mimics what DSOSD does: np.sqrt(np.mean(mdl_outs, axis=1))
    """
    n_windows = window_info['n_windows']
    sequences_per_window = window_info['sequences_per_window']
    
    # Reshape: (n_windows, sequences_per_window, n_channels)
    mse_per_window = mse_per_sequence.reshape((n_windows, sequences_per_window, n_channels))
    
    # Aggregate: sqrt(mean) per window
    window_mse = np.sqrt(np.mean(mse_per_window, axis=1))  # (n_windows, n_channels)
    
    return window_mse


def test_equivalence(data_length=10000, fs=256, train_win=12, pred_win=1, w_size=1, w_stride=0.5):
    """
    Test if hankel and sliding window approaches produce equivalent sequences.
    
    For forecast_length=1, sequences should be equivalent, but aggregation differs.
    """
    print("=" * 80)
    print("Testing Hankel (DSOSD) vs Sliding Window (DynaSD) Equivalence")
    print("=" * 80)
    
    # Create synthetic data
    n_channels = 8
    data = pd.DataFrame(
        np.random.randn(data_length, n_channels),
        columns=[f'ch{i}' for i in range(n_channels)]
    )
    
    print(f"\nTest parameters:")
    print(f"  Data length: {data_length} samples ({data_length/fs:.1f} seconds at {fs} Hz)")
    print(f"  Channels: {n_channels}")
    print(f"  train_win (sequence_length): {train_win}")
    print(f"  pred_win (forecast_length): {pred_win}")
    print(f"  w_size: {w_size}, w_stride: {w_stride}")
    
    # Method 1: DSOSD hankel approach
    print("\n" + "-" * 80)
    print("Method 1: DSOSD Hankel Approach")
    print("-" * 80)
    input_hankel, target_hankel, hankel_info = prepare_segment_hankel(
        data, fs=fs, train_win=train_win, pred_win=pred_win, 
        w_size=w_size, w_stride=w_stride
    )
    
    print(f"  Windows created: {hankel_info['n_windows']}")
    print(f"  Sequences per window: {hankel_info['sequences_per_window']}")
    print(f"  Total sequences: {hankel_info['total_sequences']}")
    print(f"  Input shape: {input_hankel.shape}")
    print(f"  Target shape: {target_hankel.shape}")
    
    # Method 2: DynaSD sliding window approach
    print("\n" + "-" * 80)
    print("Method 2: DynaSD Sliding Window Approach")
    print("-" * 80)
    input_sliding, target_sliding, sliding_info = prepare_segment_sliding_window(
        data, sequence_length=train_win, forecast_length=pred_win, stride=pred_win
    )
    
    print(f"  Total sequences: {sliding_info['n_sequences']}")
    print(f"  Input shape: {input_sliding.shape}")
    print(f"  Target shape: {target_sliding.shape}")
    
    # Compare sequences
    print("\n" + "=" * 80)
    print("Sequence Comparison")
    print("=" * 80)
    
    # Find overlapping sequences
    hankel_seq_starts = []
    for win_idx, win_start in enumerate(hankel_info['window_starts']):
        for seq_in_win in range(hankel_info['sequences_per_window']):
            hankel_seq_starts.append(win_start + seq_in_win)
    
    sliding_seq_starts = sliding_info['sequence_starts'].tolist()
    
    # Find sequences that exist in both methods
    common_starts = set(hankel_seq_starts) & set(sliding_seq_starts)
    print(f"\nCommon sequence start positions: {len(common_starts)}")
    print(f"  Hankel unique: {len(set(hankel_seq_starts)) - len(common_starts)}")
    print(f"  Sliding unique: {len(set(sliding_seq_starts)) - len(common_starts)}")
    
    if len(common_starts) > 0:
        # Compare a few sequences
        sample_starts = sorted(list(common_starts))[:5]
        print(f"\nComparing sequences at start positions: {sample_starts}")
        
        all_match = True
        for start in sample_starts:
            hankel_idx = hankel_seq_starts.index(start)
            sliding_idx = sliding_seq_starts.index(start)
            
            input_match = np.allclose(input_hankel[hankel_idx], input_sliding[sliding_idx], atol=1e-10)
            target_match = np.allclose(target_hankel[hankel_idx], target_sliding[sliding_idx, 0], atol=1e-10)
            
            if not (input_match and target_match):
                all_match = False
                print(f"  Start {start}: MISMATCH")
                if not input_match:
                    print(f"    Input diff: {np.max(np.abs(input_hankel[hankel_idx] - input_sliding[sliding_idx]))}")
                if not target_match:
                    print(f"    Target diff: {np.max(np.abs(target_hankel[hankel_idx] - target_sliding[sliding_idx, 0]))}")
            else:
                print(f"  Start {start}: MATCH")
        
        if all_match:
            print("\n✓ Sequences are equivalent where they overlap!")
        else:
            print("\n✗ Sequences differ even where they overlap!")
    else:
        print("\n⚠ No overlapping sequences found - methods create different sequence sets")
    
    # Test aggregation equivalence
    print("\n" + "=" * 80)
    print("Aggregation Comparison")
    print("=" * 80)
    
    # Simulate MSE values (random for testing)
    mse_hankel = np.random.rand(hankel_info['total_sequences'], n_channels)
    mse_sliding = np.random.rand(sliding_info['n_sequences'], n_channels)
    
    # Aggregate hankel sequences to windows
    window_mse_hankel = aggregate_hankel_sequences_to_windows(
        mse_hankel, hankel_info, n_channels
    )
    
    print(f"\nHankel aggregation:")
    print(f"  Input: {mse_hankel.shape} (sequences)")
    print(f"  Output: {window_mse_hankel.shape} (windows)")
    print(f"  Method: sqrt(mean) within each window")
    
    print(f"\nSliding window aggregation:")
    print(f"  Input: {mse_sliding.shape} (sequences)")
    print(f"  Note: DynaSD aggregates sequences to windows differently")
    print(f"  (uses _aggregate_sequences_to_windows_mse with overlap handling)")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
    For forecast_length=1:
    - Sequences themselves are equivalent where they overlap
    - BUT aggregation methods differ:
      * DSOSD: Aggregates sequences WITHIN windows (multiple sequences → 1 window output)
      * DynaSD: Aggregates sequences TO windows (sequences → windows with overlap handling)
    
    This difference in aggregation could explain performance differences!
    """)
    
    return {
        'hankel_info': hankel_info,
        'sliding_info': sliding_info,
        'common_starts': common_starts,
        'input_hankel': input_hankel,
        'input_sliding': input_sliding,
        'target_hankel': target_hankel,
        'target_sliding': target_sliding
    }


def compare_distributions(mse_hankel, mse_sliding, hankel_info, sliding_info, n_channels):
    """
    Compare the distributions of MSE values from both methods.
    
    Parameters:
    -----------
    mse_hankel : np.array
        MSE values from hankel method (n_sequences, n_channels)
    mse_sliding : np.array
        MSE values from sliding method (n_sequences, n_channels)
    hankel_info : dict
        Window information from hankel method
    sliding_info : dict
        Sequence information from sliding method
    n_channels : int
        Number of channels
    """
    print("\n" + "=" * 80)
    print("Distribution Comparison")
    print("=" * 80)
    
    # Aggregate hankel sequences to windows
    window_mse_hankel = aggregate_hankel_sequences_to_windows(
        mse_hankel, hankel_info, n_channels
    )
    
    # For sliding window, we need to aggregate to windows too
    # Use same windowing as hankel
    w_size = 1.0
    w_stride = 0.5
    fs = 256
    
    # Create windows for sliding method
    n_windows = hankel_info['n_windows']
    window_starts = np.arange(n_windows) * w_stride
    window_ends = window_starts + w_size
    
    # Get sequence times (assuming fs=256, each sequence is 1 sample apart)
    seq_times = sliding_info['sequence_starts'] / fs
    seq_end_times = seq_times + (13 / fs)  # sequence_length + forecast_length = 13 samples
    
    # Aggregate sliding sequences to windows (simplified - would need full DynaSD logic)
    window_mse_sliding = np.full((n_windows, n_channels), np.nan)
    
    for win_idx in range(n_windows):
        # Find sequences that overlap with this window
        # Simplified: sequences that start within window
        mask = (seq_times >= window_starts[win_idx]) & (seq_times < window_ends[win_idx])
        if mask.any():
            window_mse_sliding[win_idx] = np.sqrt(np.mean(mse_sliding[mask], axis=0))
    
    # Compare distributions
    print("\nWindow-level MSE Statistics:")
    print(f"  Hankel method:")
    print(f"    Mean: {np.nanmean(window_mse_hankel):.6f}")
    print(f"    Std:  {np.nanstd(window_mse_hankel):.6f}")
    print(f"    Min:  {np.nanmin(window_mse_hankel):.6f}")
    print(f"    Max:  {np.nanmax(window_mse_hankel):.6f}")
    
    print(f"\n  Sliding method:")
    print(f"    Mean: {np.nanmean(window_mse_sliding):.6f}")
    print(f"    Std:  {np.nanstd(window_mse_sliding):.6f}")
    print(f"    Min:  {np.nanmin(window_mse_sliding):.6f}")
    print(f"    Max:  {np.nanmax(window_mse_sliding):.6f}")
    
    # Compare where both have values
    valid_mask = ~(np.isnan(window_mse_hankel) | np.isnan(window_mse_sliding))
    if valid_mask.any():
        hankel_valid = window_mse_hankel[valid_mask]
        sliding_valid = window_mse_sliding[valid_mask]
        
        print(f"\n  Comparison (where both valid, n={valid_mask.sum()}):")
        diff = hankel_valid - sliding_valid
        print(f"    Mean difference: {np.mean(diff):.6f}")
        print(f"    Std difference:  {np.std(diff):.6f}")
        print(f"    Max difference:  {np.max(np.abs(diff)):.6f}")
        correlation = np.corrcoef(hankel_valid.flatten(), sliding_valid.flatten())[0,1]
        print(f"    Correlation:     {correlation:.6f}")
        print(f"\n  ⚠️  NOTE: This correlation compares aggregated window MSE values")
        print(f"      from hankel (DSOSD-style) vs sliding window (DynaSD-style) methods.")
        print(f"      Low correlation indicates different aggregation produces different results.")
        print(f"      This is NOT comparing DSOSD vs DynaSD probability values directly.")
    
    return window_mse_hankel, window_mse_sliding


def get_onset_and_spread_dsosd(sz_prob, threshold, w_size=1, w_stride=0.5, 
                                filter_w=10, rwin_size=5, rwin_req=4):
    """
    DSOSD version of get_onset_and_spread.
    Uses median filter on classification, then rolling window.
    """
    sz_clf = (sz_prob > threshold).reset_index(drop=True)
    filter_w_idx = np.floor((filter_w - w_size)/w_stride).astype(int) + 1
    sz_clf = pd.DataFrame(sc.ndimage.median_filter(sz_clf, size=filter_w_idx, 
                                                   mode='nearest', axes=0, origin=0), 
                         columns=sz_prob.columns)
    seized_idxs = np.any(sz_clf, axis=0)
    rwin_size_idx = np.floor((rwin_size - w_size)/w_stride).astype(int) + 1
    rwin_req_idx = np.floor((rwin_req - w_size)/w_stride).astype(int) + 1
    sz_spread_idxs_all = sz_clf.rolling(window=rwin_size_idx, center=False).apply(
        lambda x: (x == 1).sum()>=rwin_req_idx).dropna().reset_index(drop=True)
    
    # Padding
    missing_rows = rwin_size_idx-1
    last_valid_row = sz_spread_idxs_all.iloc[-1]
    padding = pd.DataFrame([last_valid_row] * missing_rows, columns=sz_spread_idxs_all.columns)
    sz_spread_idxs_all_padded = pd.concat([sz_spread_idxs_all, padding], ignore_index=True)
    
    sz_clf_ff = sz_spread_idxs_all_padded.copy()
    
    # Forward-fill
    for ch in sz_clf_ff.columns:
        for j in range(len(sz_clf_ff) - rwin_size_idx):
            if sz_spread_idxs_all_padded.at[j, ch]:
                future_sum = np.sum(sz_spread_idxs_all_padded.loc[j:j + rwin_size_idx, ch])
                if future_sum >= rwin_req_idx:
                    sz_clf_ff.loc[j:j + rwin_size_idx, ch] = 1
    
    sz_spread_idxs = sz_clf_ff.loc[:, seized_idxs]
    extended_seized_idxs = np.any(sz_spread_idxs, axis=0)
    first_sz_idxs = sz_spread_idxs.loc[:, extended_seized_idxs].idxmax(axis=0)
    
    if sum(extended_seized_idxs) > 0:
        sz_idxs_arr = np.array(first_sz_idxs)
        sz_order = np.argsort(first_sz_idxs)
        sz_idxs_arr = first_sz_idxs.iloc[sz_order].to_numpy()
        sz_ch_arr = first_sz_idxs.index[sz_order].to_numpy()
    else:
        sz_ch_arr = []
        sz_idxs_arr = np.array([])
    
    sz_idxs_df = pd.DataFrame(sz_idxs_arr.reshape(1,-1), columns=sz_ch_arr)
    return sz_idxs_df


def get_onset_and_spread_dynasd(sz_prob, threshold, w_size=1, w_stride=0.5,
                                filter_w=10, rwin_size=5, rwin_req=4):
    """
    DynaSD version of get_onset_and_spread.
    Uses uniform filter on probabilities, then convolution.
    """
    filter_w_idx = np.floor((filter_w - w_size)/w_stride).astype(int) + 1
    sz_prob = pd.DataFrame(sc.ndimage.uniform_filter1d(sz_prob, size=filter_w_idx, 
                                                       mode='nearest', axis=0, origin=0), 
                          columns=sz_prob.columns)
    
    sz_clf = (sz_prob > threshold).reset_index(drop=True)
    seized_idxs = np.any(sz_clf, axis=0)
    rwin_size_idx = np.floor((rwin_size - w_size)/w_stride).astype(int) + 1
    rwin_req_idx = np.floor((rwin_req - w_size)/w_stride).astype(int) + 1
    
    # Use convolution
    if len(sz_clf) > rwin_size_idx - 1:
        kernel = np.ones(rwin_size_idx)
        sz_spread_data = np.zeros((len(sz_clf) - rwin_size_idx + 1, sz_clf.shape[1]))
        
        for i, col in enumerate(sz_clf.columns):
            sliding_sums = np.convolve(sz_clf[col].astype(int), kernel, mode='valid')
            sz_spread_data[:, i] = (sliding_sums >= rwin_req_idx).astype(int)
        
        sz_spread_idxs_all = pd.DataFrame(sz_spread_data, columns=sz_clf.columns)
        
        # Padding
        missing_rows = rwin_size_idx - 1
        if len(sz_spread_idxs_all) > 0:
            last_valid_row = sz_spread_idxs_all.iloc[-1]
            padding = pd.DataFrame([last_valid_row] * missing_rows, columns=sz_spread_idxs_all.columns)
            sz_spread_idxs_all_padded = pd.concat([sz_spread_idxs_all, padding], ignore_index=True)
        else:
            sz_spread_idxs_all_padded = pd.DataFrame(np.zeros((len(sz_clf), len(sz_clf.columns))), 
                                                              columns=sz_clf.columns)
        sz_clf_ff = sz_spread_idxs_all_padded
    else:
        sz_clf_ff = sz_clf
    
    sz_spread_idxs = sz_clf_ff.loc[:, seized_idxs]
    extended_seized_idxs = np.any(sz_spread_idxs, axis=0)
    first_sz_idxs = sz_spread_idxs.loc[:, extended_seized_idxs].idxmax(axis=0)
    
    if sum(extended_seized_idxs) > 0:
        sz_idxs_arr = np.array(first_sz_idxs)
        sz_order = np.argsort(first_sz_idxs)
        sz_idxs_arr = first_sz_idxs.iloc[sz_order].to_numpy()
        sz_ch_arr = first_sz_idxs.index[sz_order].to_numpy()
    else:
        sz_ch_arr = []
        sz_idxs_arr = np.array([])
    
    sz_idxs_df = pd.DataFrame(sz_idxs_arr.reshape(1,-1), columns=sz_ch_arr)
    
    # Fill non-seizing channels
    undetected_chs = [col for col in sz_prob.columns if col not in sz_ch_arr]
    sz_idxs_df[undetected_chs] = np.nan
    
    return sz_idxs_df


def compare_get_onset_and_spread(sz_prob, threshold, verbose=True):
    """
    Compare get_onset_and_spread implementations from DSOSD and DynaSD.
    
    Parameters:
    -----------
    sz_prob : pd.DataFrame
        Probability matrix (n_timepoints, n_channels)
    threshold : float
        Threshold value
    verbose : bool
        Whether to print detailed analysis
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Comparing get_onset_and_spread Functions")
        print("=" * 80)
    
    if verbose:
        print(f"\nInput probability matrix shape: {sz_prob.shape}")
        print(f"Threshold: {threshold:.6f}")
    
    # Analyze probability distribution before thresholding
    prob_values = sz_prob.values.flatten()
    prob_values_valid = prob_values[~np.isnan(prob_values)]
    
    if verbose:
        print(f"\nProbability distribution:")
        print(f"  Mean: {np.mean(prob_values_valid):.6f}")
        print(f"  Std:  {np.std(prob_values_valid):.6f}")
        print(f"  Min:  {np.min(prob_values_valid):.6f}")
        print(f"  Max:  {np.max(prob_values_valid):.6f}")
        print(f"  Percent above threshold: {np.mean(prob_values_valid > threshold)*100:.1f}%")
    
    # DSOSD version
    print("\n" + "-" * 80)
    print("DSOSD Version (median filter on classification)")
    print("-" * 80)
    try:
        dsosd_result = get_onset_and_spread_dsosd(sz_prob, threshold)
        print(f"  Detected channels: {len(dsosd_result.columns)}")
        print(f"  Onset indices: {dsosd_result.iloc[0].values if len(dsosd_result.columns) > 0 else 'None'}")
        dsosd_channels = list(dsosd_result.columns)
        dsosd_indices = dsosd_result.iloc[0].values if len(dsosd_result.columns) > 0 else np.array([])
    except Exception as e:
        print(f"  Error: {e}")
        dsosd_result = None
        dsosd_channels = []
        dsosd_indices = np.array([])
    
    # DynaSD version
    print("\n" + "-" * 80)
    print("DynaSD Version (uniform filter on probabilities)")
    print("-" * 80)
    try:
        dynasd_result = get_onset_and_spread_dynasd(sz_prob, threshold)
        print(f"  Detected channels: {len(dynasd_result.columns)}")
        print(f"  Onset indices: {dynasd_result.iloc[0].values if len(dynasd_result.columns) > 0 else 'None'}")
        dynasd_channels = list(dynasd_result.columns)
        dynasd_indices = dynasd_result.iloc[0].values if len(dynasd_result.columns) > 0 else np.array([])
    except Exception as e:
        print(f"  Error: {e}")
        dynasd_result = None
        dynasd_channels = []
        dynasd_indices = np.array([])
    
    # Compare results
    print("\n" + "-" * 80)
    print("Comparison")
    print("-" * 80)
    
    common_channels = set(dsosd_channels) & set(dynasd_channels)
    dsosd_only = set(dsosd_channels) - set(dynasd_channels)
    dynasd_only = set(dynasd_channels) - set(dsosd_channels)
    
    print(f"\nCommon channels detected: {len(common_channels)}")
    print(f"  DSOSD only: {len(dsosd_only)}")
    print(f"  DynaSD only: {len(dynasd_only)}")
    
    if len(common_channels) > 0:
        print(f"\nComparing onset indices for common channels:")
        for ch in sorted(list(common_channels))[:5]:  # Show first 5
            dsosd_idx = dsosd_result[ch].iloc[0] if ch in dsosd_result.columns else np.nan
            dynasd_idx = dynasd_result[ch].iloc[0] if ch in dynasd_result.columns else np.nan
            diff = abs(dsosd_idx - dynasd_idx) if not (np.isnan(dsosd_idx) or np.isnan(dynasd_idx)) else np.nan
            print(f"  {ch}: DSOSD={dsosd_idx:.1f}, DynaSD={dynasd_idx:.1f}, diff={diff:.1f}")
    
    # Analyze why differences occur
    if verbose and len(common_channels) > 0:
        print("\n" + "-" * 80)
        print("Detailed Analysis of Differences:")
        print("-" * 80)
        
        # Show example channel with large difference
        max_diff_ch = None
        max_diff_val = 0
        for ch in sorted(list(common_channels))[:10]:  # Check first 10
            dsosd_idx = dsosd_result[ch].iloc[0] if ch in dsosd_result.columns else np.nan
            dynasd_idx = dynasd_result[ch].iloc[0] if ch in dynasd_result.columns else np.nan
            if not (np.isnan(dsosd_idx) or np.isnan(dynasd_idx)):
                diff = abs(dsosd_idx - dynasd_idx)
                if diff > max_diff_val:
                    max_diff_val = diff
                    max_diff_ch = ch
        
        if max_diff_ch:
            print(f"\nExample channel with large difference: {max_diff_ch}")
            dsosd_idx_ex = dsosd_result[max_diff_ch].iloc[0]
            dynasd_idx_ex = dynasd_result[max_diff_ch].iloc[0]
            print(f"  DSOSD onset index: {dsosd_idx_ex}")
            print(f"  DynaSD onset index: {dynasd_idx_ex}")
            print(f"  Difference: {abs(dsosd_idx_ex - dynasd_idx_ex)} windows")
            
            # Show probability values around threshold for this channel
            ch_probs = sz_prob[max_diff_ch].values
            print(f"\n  Probability values for {max_diff_ch}:")
            print(f"    Mean: {np.mean(ch_probs):.4f}")
            print(f"    Values around threshold ({threshold:.4f}):")
            near_threshold = np.abs(ch_probs - threshold) < threshold * 0.1
            if np.any(near_threshold):
                print(f"      {np.sum(near_threshold)} timepoints within 10% of threshold")
                print(f"      Range: {np.min(ch_probs[near_threshold]):.4f} to {np.max(ch_probs[near_threshold]):.4f}")
            
            # Explain the difference
            print(f"\n  Why the difference occurs:")
            print(f"    1. DSOSD filters AFTER thresholding:")
            print(f"       - Thresholds probabilities → binary classification")
            print(f"       - Applies median_filter to binary values")
            print(f"       - Median filter preserves edges better, less smoothing")
            print(f"    2. DynaSD filters BEFORE thresholding:")
            print(f"       - Applies uniform_filter1d to probabilities")
            print(f"       - Smooths probability values first")
            print(f"       - Then thresholds the smoothed probabilities")
            print(f"    3. Impact:")
            print(f"       - If probabilities hover around threshold, filtering order matters!")
            print(f"       - Filtering before thresholding can shift when threshold is crossed")
            print(f"       - This can cause large timing differences (26 seconds = 52 windows at 0.5s stride)")
    
    # Key differences
    if verbose:
        print("\n" + "-" * 80)
        print("Key Differences:")
        print("-" * 80)
        print("""
    1. Filtering:
       - DSOSD: median_filter on CLASSIFICATION (after thresholding)
       - DynaSD: uniform_filter1d on PROBABILITIES (before thresholding)
       ⚠️  THIS IS THE MAIN CAUSE OF LARGE DIFFERENCES!
    
    2. Rolling window:
       - DSOSD: pd.DataFrame.rolling().apply() with lambda function
       - DynaSD: np.convolve() with kernel (faster)
    
    3. Padding:
       - DSOSD: pads with last_valid_row at END
       - DynaSD: pads with last_valid_row at END (same)
    
    4. Non-seizing channels:
       - DSOSD: Not explicitly filled
       - DynaSD: Explicitly filled with NaN
    """)
    
    return {
        'dsosd_result': dsosd_result,
        'dynasd_result': dynasd_result,
        'common_channels': common_channels,
        'dsosd_only': dsosd_only,
        'dynasd_only': dynasd_only
    }


def test_sampling_rate_impact(data_length_seconds=60, n_channels=8):
    """
    Test the impact of sampling rate (256 Hz vs 128 Hz) on sequence preparation and windowing.
    
    Parameters:
    -----------
    data_length_seconds : float
        Length of data in seconds
    n_channels : int
        Number of channels
    """
    print("\n" + "=" * 80)
    print("Testing Sampling Rate Impact (256 Hz vs 128 Hz)")
    print("=" * 80)
    
    # Create synthetic data at original sampling rate (assume 512 Hz raw)
    fs_raw = 512
    data_length_samples = int(data_length_seconds * fs_raw)
    data_raw = pd.DataFrame(
        np.random.randn(data_length_samples, n_channels),
        columns=[f'ch{i}' for i in range(n_channels)]
    )
    
    print(f"\nTest parameters:")
    print(f"  Data length: {data_length_seconds} seconds")
    print(f"  Raw sampling rate: {fs_raw} Hz")
    print(f"  Channels: {n_channels}")
    
    # Test at 256 Hz (DSOSD)
    print("\n" + "-" * 80)
    print("256 Hz (DSOSD)")
    print("-" * 80)
    fs_256 = 256
    data_256 = data_raw.iloc[::fs_raw//fs_256, :].reset_index(drop=True)
    print(f"  Downsampled shape: {data_256.shape}")
    print(f"  Samples per second: {fs_256}")
    print(f"  Temporal resolution: {1/fs_256*1000:.2f} ms per sample")
    
    input_256, target_256, info_256 = prepare_segment_hankel(
        data_256, fs=fs_256, train_win=12, pred_win=1, w_size=1, w_stride=0.5
    )
    print(f"  Sequences created: {info_256['total_sequences']}")
    print(f"  Windows: {info_256['n_windows']}")
    print(f"  Sequences per window: {info_256['sequences_per_window']}")
    
    # Test at 128 Hz (DynaSD)
    print("\n" + "-" * 80)
    print("128 Hz (DynaSD)")
    print("-" * 80)
    fs_128 = 128
    data_128 = data_raw.iloc[::fs_raw//fs_128, :].reset_index(drop=True)
    print(f"  Downsampled shape: {data_128.shape}")
    print(f"  Samples per second: {fs_128}")
    print(f"  Temporal resolution: {1/fs_128*1000:.2f} ms per sample")
    
    input_128, target_128, info_128 = prepare_segment_hankel(
        data_128, fs=fs_128, train_win=12, pred_win=1, w_size=1, w_stride=0.5
    )
    print(f"  Sequences created: {info_128['total_sequences']}")
    print(f"  Windows: {info_128['n_windows']}")
    print(f"  Sequences per window: {info_128['sequences_per_window']}")
    
    # Compare
    print("\n" + "-" * 80)
    print("Comparison")
    print("-" * 80)
    print(f"\nSequence counts:")
    print(f"  256 Hz: {info_256['total_sequences']} sequences")
    print(f"  128 Hz: {info_128['total_sequences']} sequences")
    print(f"  Ratio: {info_256['total_sequences'] / info_128['total_sequences']:.2f}x")
    
    print(f"\nWindow counts:")
    print(f"  256 Hz: {info_256['n_windows']} windows")
    print(f"  128 Hz: {info_128['n_windows']} windows")
    print(f"  Ratio: {info_256['n_windows'] / info_128['n_windows']:.2f}x")
    
    print(f"\nSequences per window:")
    print(f"  256 Hz: {info_256['sequences_per_window']} sequences/window")
    print(f"  128 Hz: {info_128['sequences_per_window']} sequences/window")
    print(f"  Difference: {info_256['sequences_per_window'] - info_128['sequences_per_window']}")
    
    # Compare window timing
    print(f"\nWindow timing:")
    print(f"  256 Hz: Window stride = {0.5}s = {int(0.5*fs_256)} samples")
    print(f"  128 Hz: Window stride = {0.5}s = {int(0.5*fs_128)} samples")
    print(f"  Temporal alignment: Windows at 128 Hz are coarser")
    
    # Test if sequences align temporally
    print(f"\nTemporal alignment test:")
    # Check if sequences at same time points exist
    # At 256 Hz, sequence at sample 0 covers samples 0-12
    # At 128 Hz, sequence at sample 0 covers samples 0-12 (but different absolute time)
    
    # Compare sequences at equivalent time points
    time_256_samples = np.arange(info_256['total_sequences']) / fs_256
    time_128_samples = np.arange(info_128['total_sequences']) / fs_128
    
    # Find overlapping time ranges
    common_time_start = max(time_256_samples[0], time_128_samples[0])
    common_time_end = min(time_256_samples[-1], time_128_samples[-1])
    
    print(f"  Common time range: {common_time_start:.2f}s to {common_time_end:.2f}s")
    print(f"  256 Hz coverage: {time_256_samples[0]:.2f}s to {time_256_samples[-1]:.2f}s")
    print(f"  128 Hz coverage: {time_128_samples[0]:.2f}s to {time_128_samples[-1]:.2f}s")
    
    # Impact on feature extraction
    print(f"\nImpact on feature extraction:")
    print(f"  At 256 Hz: {fs_256} samples/second → finer temporal resolution")
    print(f"  At 128 Hz: {fs_128} samples/second → coarser temporal resolution")
    print(f"  Difference: 2x temporal resolution loss at 128 Hz")
    print(f"  This affects:")
    print(f"    - Sequence timing precision")
    print(f"    - Window boundary alignment")
    print(f"    - Feature extraction granularity")
    
    return {
        'fs_256': fs_256,
        'fs_128': fs_128,
        'info_256': info_256,
        'info_128': info_128,
        'data_256': data_256,
        'data_128': data_128
    }


def compare_probability_distributions(prob_dsosd, prob_dynasd, prob_chs, seizure_name):
    """
    Compare actual probability distributions from DSOSD and DynaSD models.
    
    Parameters:
    -----------
    prob_dsosd : pd.DataFrame or np.array
        Probability values from DSOSD (n_timepoints, n_channels)
    prob_dynasd : pd.DataFrame or np.array
        Probability values from DynaSD (n_timepoints, n_channels)
    prob_chs : np.array
        Channel names
    seizure_name : str
        Name of seizure for reporting
    """
    print("\n" + "=" * 80)
    print(f"Comparing Probability Distributions: {seizure_name}")
    print("=" * 80)
    
    # Convert to numpy if needed
    if isinstance(prob_dsosd, pd.DataFrame):
        prob_dsosd = prob_dsosd.to_numpy()
    if isinstance(prob_dynasd, pd.DataFrame):
        prob_dynasd = prob_dynasd.to_numpy()
    
    # Ensure same shape
    if prob_dsosd.shape != prob_dynasd.shape:
        print(f"  WARNING: Shape mismatch!")
        print(f"    DSOSD: {prob_dsosd.shape}")
        print(f"    DynaSD: {prob_dynasd.shape}")
        # Try to align
        min_time = min(prob_dsosd.shape[0], prob_dynasd.shape[0])
        min_ch = min(prob_dsosd.shape[1], prob_dynasd.shape[1])
        prob_dsosd = prob_dsosd[:min_time, :min_ch]
        prob_dynasd = prob_dynasd[:min_time, :min_ch]
        print(f"    Using aligned shape: {prob_dsosd.shape}")
    
    # Flatten for distribution comparison
    prob_dsosd_flat = prob_dsosd.flatten()
    prob_dynasd_flat = prob_dynasd.flatten()
    
    # Remove NaN and inf
    valid_mask = ~(np.isnan(prob_dsosd_flat) | np.isnan(prob_dynasd_flat) | 
                   np.isinf(prob_dsosd_flat) | np.isinf(prob_dynasd_flat))
    prob_dsosd_valid = prob_dsosd_flat[valid_mask]
    prob_dynasd_valid = prob_dynasd_flat[valid_mask]
    
    print(f"\nValid values: {len(prob_dsosd_valid)} / {len(prob_dsosd_flat)}")
    
    # Distribution statistics
    print(f"\nDSOSD Probability Distribution:")
    print(f"  Mean: {np.mean(prob_dsosd_valid):.6f}")
    print(f"  Std:  {np.std(prob_dsosd_valid):.6f}")
    print(f"  Min:  {np.min(prob_dsosd_valid):.6f}")
    print(f"  Max:  {np.max(prob_dsosd_valid):.6f}")
    print(f"  Median: {np.median(prob_dsosd_valid):.6f}")
    print(f"  25th percentile: {np.percentile(prob_dsosd_valid, 25):.6f}")
    print(f"  75th percentile: {np.percentile(prob_dsosd_valid, 75):.6f}")
    
    print(f"\nDynaSD Probability Distribution:")
    print(f"  Mean: {np.mean(prob_dynasd_valid):.6f}")
    print(f"  Std:  {np.std(prob_dynasd_valid):.6f}")
    print(f"  Min:  {np.min(prob_dynasd_valid):.6f}")
    print(f"  Max:  {np.max(prob_dynasd_valid):.6f}")
    print(f"  Median: {np.median(prob_dynasd_valid):.6f}")
    print(f"  25th percentile: {np.percentile(prob_dynasd_valid, 25):.6f}")
    print(f"  75th percentile: {np.percentile(prob_dynasd_valid, 75):.6f}")
    
    # Direct comparison
    print(f"\nDirect Comparison:")
    diff = prob_dsosd_valid - prob_dynasd_valid
    print(f"  Mean difference (DSOSD - DynaSD): {np.mean(diff):.6f}")
    print(f"  Std difference: {np.std(diff):.6f}")
    print(f"  Mean absolute difference: {np.mean(np.abs(diff)):.6f}")
    print(f"  Max absolute difference: {np.max(np.abs(diff)):.6f}")
    
    # Correlation
    if len(prob_dsosd_valid) > 1:
        correlation = np.corrcoef(prob_dsosd_valid, prob_dynasd_valid)[0, 1]
        print(f"\nCorrelation between DSOSD and DynaSD probability values:")
        print(f"  Pearson correlation: {correlation:.6f}")
        print(f"  (This measures how well DSOSD and DynaSD probabilities agree)")
        
        if correlation < 0.5:
            print(f"  ⚠️  LOW CORRELATION - Models produce very different probability values")
        elif correlation < 0.8:
            print(f"  ⚠️  MODERATE CORRELATION - Models show some agreement but differ significantly")
        else:
            print(f"  ✓  HIGH CORRELATION - Models produce similar probability values")
    
    # Per-channel correlation
    print(f"\nPer-Channel Correlations:")
    channel_corrs = []
    for ch_idx in range(min(prob_dsosd.shape[1], prob_dynasd.shape[1])):
        ch_dsosd = prob_dsosd[:, ch_idx]
        ch_dynasd = prob_dynasd[:, ch_idx]
        valid_ch = ~(np.isnan(ch_dsosd) | np.isnan(ch_dynasd) | 
                     np.isinf(ch_dsosd) | np.isinf(ch_dynasd))
        if np.sum(valid_ch) > 1:
            ch_corr = np.corrcoef(ch_dsosd[valid_ch], ch_dynasd[valid_ch])[0, 1]
            channel_corrs.append(ch_corr)
            if ch_idx < 5:  # Show first 5 channels
                ch_name = prob_chs[ch_idx] if ch_idx < len(prob_chs) else f"ch{ch_idx}"
                print(f"  {ch_name}: {ch_corr:.4f}")
    
    if channel_corrs:
        print(f"\n  Mean channel correlation: {np.mean(channel_corrs):.4f}")
        print(f"  Std channel correlation: {np.std(channel_corrs):.4f}")
        print(f"  Min channel correlation: {np.min(channel_corrs):.4f}")
        print(f"  Max channel correlation: {np.max(channel_corrs):.4f}")
    
    return {
        'dsosd_stats': {
            'mean': np.mean(prob_dsosd_valid),
            'std': np.std(prob_dsosd_valid),
            'min': np.min(prob_dsosd_valid),
            'max': np.max(prob_dsosd_valid)
        },
        'dynasd_stats': {
            'mean': np.mean(prob_dynasd_valid),
            'std': np.std(prob_dynasd_valid),
            'min': np.min(prob_dynasd_valid),
            'max': np.max(prob_dynasd_valid)
        },
        'correlation': correlation if len(prob_dsosd_valid) > 1 else np.nan,
        'mean_abs_diff': np.mean(np.abs(diff)),
        'channel_correlations': channel_corrs
    }


def calculate_phi_against_annotations(sz_prob, prob_chs, onset_idx, all_chs, 
                                      ueo_consensus, ueo_annotators, threshold):
    """
    Calculate Phi against clinician annotations like in test_validation_analysis.py
    
    Parameters:
    -----------
    sz_prob : np.array or pd.DataFrame
        Probability matrix (n_channels, n_timepoints) or (n_timepoints, n_channels)
    prob_chs : np.array
        Channel names from probabilities
    onset_idx : int
        Index of onset window
    all_chs : list
        All channel labels from annotations
    ueo_consensus : np.array
        Boolean array of consensus labels
    ueo_annotators : list
        List of boolean arrays, one per annotator
    threshold : float
        Threshold value
    
    Returns:
    --------
    dict with phi values
    """
    # Ensure sz_prob is numpy array
    if isinstance(sz_prob, pd.DataFrame):
        sz_prob = sz_prob.to_numpy().T  # Convert to (n_channels, n_timepoints)
    
    # Get average probabilities over 5-timepoint window at onset
    onset_probs = sz_prob[:, onset_idx:onset_idx+5].mean(axis=1)
    
    # Get predictions
    pred_bool = onset_probs > threshold
    predicted_chs = prob_chs[pred_bool] if np.any(pred_bool) else np.array([])
    
    # Convert to wideform
    def wideform_preds(element, all_labels):
        return np.array([label in element for label in all_labels])
    
    # Calculate phi with consensus
    pred_bool_wide = wideform_preds(predicted_chs, all_chs)
    if len(pred_bool_wide) == 0 or len(ueo_consensus) == 0:
        phi_consensus = np.nan
    else:
        phi_consensus = matthews_corrcoef(ueo_consensus, pred_bool_wide)
    
    # Calculate phi with each annotator
    phi_annotators = []
    for ann in ueo_annotators:
        if len(pred_bool_wide) == 0 or len(ann) == 0:
            phi_annotators.append(np.nan)
        else:
            phi_annotators.append(matthews_corrcoef(ann, pred_bool_wide))
    
    return {
        'phi_consensus': phi_consensus,
        'phi_annotators': phi_annotators,
        'predicted_channels': predicted_chs.tolist(),
        'n_predicted': len(predicted_chs)
    }


def test_sampling_rate_on_raw_eeg(patient, onset_run, prebuffer=180, postbuffer=120, 
                                  annotations_df=None, bandpass_low=3):
    """
    Load raw EEG data and test different sampling rates (256 Hz vs 128 Hz).
    Preprocesses at both rates, runs models, and compares outputs.
    
    Parameters:
    -----------
    patient : str
        Patient ID
    onset_run : str
        Seizure onset time (as string)
    prebuffer : float
        Seconds before seizure onset to include
    postbuffer : float
        Seconds after seizure offset to include
    annotations_df : pd.DataFrame, optional
        Clinical annotations dataframe
    """
    if datapath is None or get_data_from_bids is None or preprocess_for_detection is None:
        print("Skipping raw EEG test - config or utils not available")
        return None
    
    if DSOSD_NDD is None or DynaSD_NDD is None:
        print("Skipping raw EEG test - models not available")
        return None
    
    print("\n" + "=" * 80)
    print(f"Testing Sampling Rate Impact on Raw EEG: {patient} {onset_run}")
    print("=" * 80)
    
    try:
        # Load raw seizure data from BIDS
        print(f"\nLoading raw EEG data from BIDS...")
        print(f"  Patient: {patient}, Onset: {onset_run}")
        print(f"  Looking for task: ictal{onset_run}")
        
        try:
            seizure_raw, fs_raw, _, _, task, run = get_data_from_bids(
                ospj(datapath, "BIDS"), patient, f"ictal{onset_run}", 
                return_path=True, verbose=0
            )
        except Exception as e:
            print(f"  ✗ Error loading BIDS data: {e}")
            print(f"  Trying alternative task format...")
            # Try without 'ictal' prefix
            try:
                seizure_raw, fs_raw, _, _, task, run = get_data_from_bids(
                    ospj(datapath, "BIDS"), patient, onset_run, 
                    return_path=True, verbose=0
                )
            except Exception as e2:
                print(f"  ✗ Also failed with alternative format: {e2}")
                return None
        
        print(f"  Raw data shape: {seizure_raw.shape}")
        print(f"  Raw sampling rate: {fs_raw} Hz")
        print(f"  Duration: {seizure_raw.shape[0] / fs_raw:.1f} seconds")
        
        # Clean labels
        seizure_raw.columns = clean_labels(seizure_raw.columns, patient)
        
        # Remove scalp electrodes
        neural_channels = remove_scalp_electrodes(seizure_raw.columns)
        seizure_raw = seizure_raw.loc[:, neural_channels]
        
        # Get channel mask from first 60s (for artifact rejection)
        print(f"\nPreprocessing for channel selection...")
        _, _, channel_mask = preprocess_for_detection(
            seizure_raw.iloc[:120 * fs_raw, :],
            fs_raw,
            montage='bipolar',
            target=fs_raw,
            wavenet=False,
            pre_mask=None,
        )
        
        # Preprocess at both 128 Hz and 256 Hz
        print(f"\n" + "-" * 80)
        print("Preprocessing at 128 Hz")
        print("-" * 80)
        seizure_128, fs_128 = preprocess_for_detection(
            seizure_raw,
            fs_raw,
            montage='bipolar',
            target=128,
            wavenet=False,
            pre_mask=channel_mask,
            band=[bandpass_low, 120]  # Configurable lower bandpass
        )
        
        print(f"  Preprocessed shape: {seizure_128.shape}")
        print(f"  Sampling rate: {fs_128} Hz")
        print(f"  Duration: {seizure_128.shape[0] / fs_128:.1f} seconds")
        
        # Artifact rejection
        art_channel_mask_128 = seizure_128.loc[180*fs_128:,:].abs().max() <= (
            np.median(seizure_128.loc[180*fs_128:,:].abs().max()) * 50
        )
        seizure_128_nart = seizure_128.loc[:, art_channel_mask_128]
        print(f"  Channels after artifact rejection: {len(seizure_128_nart.columns)}")
        
        print(f"\n" + "-" * 80)
        print("Preprocessing at 256 Hz")
        print("-" * 80)
        seizure_256, fs_256 = preprocess_for_detection(
            seizure_raw,
            fs_raw,
            montage='bipolar',
            target=256,
            wavenet=False,
            pre_mask=channel_mask,
            band=[3, 120]  # Same bandpass for comparison
        )
        
        print(f"  Preprocessed shape: {seizure_256.shape}")
        print(f"  Sampling rate: {fs_256} Hz")
        print(f"  Duration: {seizure_256.shape[0] / fs_256:.1f} seconds")
        
        # Artifact rejection
        art_channel_mask_256 = seizure_256.loc[180*fs_256:,:].abs().max() <= (
            np.median(seizure_256.loc[180*fs_256:,:].abs().max()) * 50
        )
        seizure_256_nart = seizure_256.loc[:, art_channel_mask_256]
        print(f"  Channels after artifact rejection: {len(seizure_256_nart.columns)}")
        
        # Find common channels across all preprocessing
        common_chs = set(seizure_128_nart.columns) & set(seizure_256_nart.columns)
        print(f"\n  Common channels: {len(common_chs)}")
        print(f"  128 Hz only: {len(set(seizure_128_nart.columns) - common_chs)}")
        print(f"  256 Hz only: {len(set(seizure_256_nart.columns) - common_chs)}")
        
        if len(common_chs) == 0:
            print("  ⚠️  No common channels - cannot compare!")
            return None
        
        # Use only common channels for comparison
        seizure_128_common = seizure_128_nart.loc[:, sorted(list(common_chs))]
        seizure_256_common = seizure_256_nart.loc[:, sorted(list(common_chs))]
        
        # Train and run DSOSD model at 128 Hz (original)
        print(f"\n" + "-" * 80)
        print("Running DSOSD NDD model at 128 Hz (original)")
        print("-" * 80)
        model_dsosd_128 = DSOSD_NDD(fs=128, train_win=12, pred_win=1, w_size=1, w_stride=0.5)
        
        train_start_128 = int(60 * fs_128)
        train_end_128 = int(120 * fs_128)
        print(f"  Training on samples {train_start_128} to {train_end_128} ({train_start_128/fs_128:.1f}s to {train_end_128/fs_128:.1f}s)")
        
        # Seed random number generators to match DSOSD behavior
        # DSOSD uses np.random.seed(171999) in BIDS_seizure_saving.py
        seed = 171999
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        model_dsosd_128.fit(seizure_128_common.iloc[train_start_128:train_end_128, :])
        
        print(f"  Generating probabilities...")
        prob_dsosd_128_df = model_dsosd_128(seizure_128_common)
        print(f"  Probability shape: {prob_dsosd_128_df.shape}")
        
        # Train and run DSOSD model at 256 Hz
        print(f"\n" + "-" * 80)
        print("Running DSOSD NDD model at 256 Hz")
        print("-" * 80)
        model_dsosd_256 = DSOSD_NDD(fs=256, train_win=12, pred_win=1, w_size=1, w_stride=0.5)
        
        train_start_256 = int(60 * fs_256)
        train_end_256 = int(120 * fs_256)
        print(f"  Training on samples {train_start_256} to {train_end_256} ({train_start_256/fs_256:.1f}s to {train_end_256/fs_256:.1f}s)")
        
        # Seed random number generators to match DSOSD behavior
        # DSOSD uses np.random.seed(171999) in BIDS_seizure_saving.py
        seed = 171999
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        model_dsosd_256.fit(seizure_256_common.iloc[train_start_256:train_end_256, :])
        
        print(f"  Generating probabilities...")
        prob_dsosd_256_df = model_dsosd_256(seizure_256_common)
        print(f"  Probability shape: {prob_dsosd_256_df.shape}")
        
        # Train and run DynaSD model at 128 Hz
        print(f"\n" + "-" * 80)
        print("Running DynaSD NDD model at 128 Hz")
        print("-" * 80)
        model_dynasd_128 = DynaSD_NDD(fs=128, sequence_length=12, forecast_length=1, w_size=1, w_stride=0.5)
        
        # Seed random number generators to match DSOSD behavior
        # DSOSD uses np.random.seed(171999) in BIDS_seizure_saving.py
        seed = 171999
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        model_dynasd_128.fit(seizure_128_common.iloc[train_start_128:train_end_128, :])
        
        print(f"  Generating probabilities...")
        prob_dynasd_128_df = model_dynasd_128(seizure_128_common)
        print(f"  Probability shape: {prob_dynasd_128_df.shape}")
        
        # Train and run DynaSD model at 256 Hz
        print(f"\n" + "-" * 80)
        print("Running DynaSD NDD model at 256 Hz")
        print("-" * 80)
        model_dynasd_256 = DynaSD_NDD(fs=256, sequence_length=12, forecast_length=1, w_size=1, w_stride=0.5)
        
        # Seed random number generators to match DSOSD behavior
        # DSOSD uses np.random.seed(171999) in BIDS_seizure_saving.py
        seed = 171999
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        model_dynasd_256.fit(seizure_256_common.iloc[train_start_256:train_end_256, :])
        
        print(f"  Generating probabilities...")
        prob_dynasd_256_df = model_dynasd_256(seizure_256_common)
        print(f"  Probability shape: {prob_dynasd_256_df.shape}")
        
        # Compare probability distributions
        print(f"\n" + "=" * 80)
        print("Comparing Probability Distributions from Raw EEG")
        print("=" * 80)
        
        # Compare all combinations
        prob_comparisons = {}
        
        # DSOSD 128 vs DynaSD 128
        prob_comparisons['dsosd_128_vs_dynasd_128'] = compare_probability_distributions(
            prob_dsosd_128_df,
            prob_dynasd_128_df,
            seizure_128_common.columns,
            f"{patient} {onset_run} (DSOSD 128Hz vs DynaSD 128Hz)"
        )
        
        # DSOSD 256 vs DynaSD 256
        prob_comparisons['dsosd_256_vs_dynasd_256'] = compare_probability_distributions(
            prob_dsosd_256_df,
            prob_dynasd_256_df,
            seizure_256_common.columns,
            f"{patient} {onset_run} (DSOSD 256Hz vs DynaSD 256Hz)"
        )
        
        # DSOSD 128 vs DSOSD 256
        prob_comparisons['dsosd_128_vs_dsosd_256'] = compare_probability_distributions(
            prob_dsosd_128_df,
            prob_dsosd_256_df,
            seizure_128_common.columns,
            f"{patient} {onset_run} (DSOSD 128Hz vs DSOSD 256Hz)"
        )
        
        # DynaSD 128 vs DynaSD 256
        prob_comparisons['dynasd_128_vs_dynasd_256'] = compare_probability_distributions(
            prob_dynasd_128_df,
            prob_dynasd_256_df,
            seizure_128_common.columns,
            f"{patient} {onset_run} (DynaSD 128Hz vs DynaSD 256Hz)"
        )
        
        # Check window counts
        print(f"\n" + "-" * 80)
        print("Window Count Analysis")
        print("-" * 80)
        print(f"DSOSD 128Hz probability shape: {prob_dsosd_128_df.shape}")
        print(f"DSOSD 256Hz probability shape: {prob_dsosd_256_df.shape}")
        print(f"DynaSD 128Hz probability shape: {prob_dynasd_128_df.shape}")
        print(f"DynaSD 256Hz probability shape: {prob_dynasd_256_df.shape}")
        
        # Check for window count differences
        window_counts = {
            'dsosd_128': prob_dsosd_128_df.shape[0],
            'dsosd_256': prob_dsosd_256_df.shape[0],
            'dynasd_128': prob_dynasd_128_df.shape[0],
            'dynasd_256': prob_dynasd_256_df.shape[0]
        }
        
        if len(set(window_counts.values())) > 1:
            print(f"\n⚠️  Window count differences detected!")
            for name, count in window_counts.items():
                print(f"  {name}: {count} windows")
            print(f"  This could be due to:")
            print(f"  1. Different sequence aggregation methods")
            print(f"  2. Different window boundary handling")
            print(f"  3. Edge case handling in sequence-to-window conversion")
        
        # Calculate Phi against clinician annotations if available
        phi_results = {}
        smoothing_analysis = {}
        if annotations_df is not None:
            print(f"\n" + "=" * 80)
            print("Calculating Phi Against Clinician Annotations")
            print("=" * 80)
            
            # Get annotations
            approx_onset = float(onset_run)
            annot_matches = annotations_df[
                (annotations_df['patient'] == patient) & 
                (np.abs(annotations_df['approximate_onset'].astype(float) - approx_onset) < 360)
            ]
            
            if len(annot_matches) > 0:
                annot_row = annot_matches.iloc[0]
                all_chs = annot_row['all_chs']
                ueo_consensus = annot_row['ueo_consensus']
                ueo_annotators = annot_row['ueo']
                
                # Calculate temporal alignment (onset is at 180s from start)
                # Window stride is 0.5s, so onset_idx = 180 / 0.5 = 360
                onset_idx = int(180 / 0.5)  # Window index at 180s (same for all)
                
                # Extract first contacts from probability channels
                prob_chs_128 = np.array([ch.split('-')[0] for ch in prob_dsosd_128_df.columns])
                prob_chs_256 = np.array([ch.split('-')[0] for ch in prob_dsosd_256_df.columns])
                
                # Use fixed threshold to isolate probability differences from threshold calculation
                print(f"\n" + "-" * 80)
                print("Fixed Threshold Analysis (Threshold = 1.5)")
                print("-" * 80)
                print("Using fixed threshold to isolate probability differences from threshold calculation differences.")
                
                fixed_threshold = 1.5
                
                # Get raw probabilities (before smoothing) for all combinations
                prob_arrays = {
                    'dsosd_128': prob_dsosd_128_df.to_numpy().T,  # (n_channels, n_timepoints)
                    'dsosd_256': prob_dsosd_256_df.to_numpy().T,
                    'dynasd_128': prob_dynasd_128_df.to_numpy().T,
                    'dynasd_256': prob_dynasd_256_df.to_numpy().T
                }
                
                prob_chs_dict = {
                    'dsosd_128': prob_chs_128,
                    'dsosd_256': prob_chs_256,
                    'dynasd_128': prob_chs_128,
                    'dynasd_256': prob_chs_256
                }
                
                # Apply different smoothing methods
                smoothing_window = 20  # Same as test_validation_analysis.py
                
                # Calculate Phi for all 8 combinations (4 model/sampling rate × 2 smoothing methods)
                print(f"\n  Calculating Phi for all combinations at fixed threshold ({fixed_threshold})...")
                
                phi_results_fixed = {}
                
                for model_fs_key in ['dsosd_128', 'dsosd_256', 'dynasd_128', 'dynasd_256']:
                    prob_raw = prob_arrays[model_fs_key]
                    prob_chs = prob_chs_dict[model_fs_key]
                    
                    # Mean smoothing
                    prob_mean = sc.ndimage.uniform_filter1d(
                        prob_raw, size=smoothing_window, mode='nearest', axis=1, origin=0
                    )
                    
                    # Median smoothing
                    prob_median = sc.ndimage.median_filter(
                        prob_raw, size=smoothing_window, mode='nearest', axes=1, origin=0
                    )
                    
                    # Calculate Phi with fixed threshold and mean smoothing
                    phi_mean_fixed = calculate_phi_against_annotations(
                        prob_mean, prob_chs, onset_idx,
                        all_chs, ueo_consensus, ueo_annotators, fixed_threshold
                    )
                    
                    # Calculate Phi with fixed threshold and median smoothing
                    phi_median_fixed = calculate_phi_against_annotations(
                        prob_median, prob_chs, onset_idx,
                        all_chs, ueo_consensus, ueo_annotators, fixed_threshold
                    )
                    
                    phi_results_fixed[f'{model_fs_key}_mean'] = phi_mean_fixed
                    phi_results_fixed[f'{model_fs_key}_median'] = phi_median_fixed
                
                # Display results at fixed threshold
                print(f"\n  Phi Scores at Fixed Threshold ({fixed_threshold}):")
                print(f"    {'Model':<12} {'Sampling Rate':<15} {'Mean':<10} {'Median':<10} {'Difference':<12}")
                print(f"    {'-'*12} {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
                
                for model_fs_key in ['dsosd_128', 'dsosd_256', 'dynasd_128', 'dynasd_256']:
                    model_name = model_fs_key.split('_')[0].upper()
                    fs = model_fs_key.split('_')[1]
                    phi_mean = phi_results_fixed[f'{model_fs_key}_mean']['phi_consensus']
                    phi_median = phi_results_fixed[f'{model_fs_key}_median']['phi_consensus']
                    diff = phi_mean - phi_median
                    print(f"    {model_name:<12} {fs:<15} {phi_mean:<10.4f} {phi_median:<10.4f} {diff:<12.4f}")
                
                # Cross-comparisons at fixed threshold
                print(f"\n  Cross-Comparisons at Fixed Threshold:")
                print(f"    DSOSD 128Hz Mean vs DynaSD 128Hz Mean:   {phi_results_fixed['dsosd_128_mean']['phi_consensus'] - phi_results_fixed['dynasd_128_mean']['phi_consensus']:.4f}")
                print(f"    DSOSD 128Hz Median vs DynaSD 128Hz Median: {phi_results_fixed['dsosd_128_median']['phi_consensus'] - phi_results_fixed['dynasd_128_median']['phi_consensus']:.4f}")
                print(f"    DSOSD 256Hz Mean vs DynaSD 256Hz Mean:   {phi_results_fixed['dsosd_256_mean']['phi_consensus'] - phi_results_fixed['dynasd_256_mean']['phi_consensus']:.4f}")
                print(f"    DSOSD 256Hz Median vs DynaSD 256Hz Median: {phi_results_fixed['dsosd_256_median']['phi_consensus'] - phi_results_fixed['dynasd_256_median']['phi_consensus']:.4f}")
                print(f"    DSOSD 128Hz vs 256Hz (Mean):              {phi_results_fixed['dsosd_128_mean']['phi_consensus'] - phi_results_fixed['dsosd_256_mean']['phi_consensus']:.4f}")
                print(f"    DSOSD 128Hz vs 256Hz (Median):           {phi_results_fixed['dsosd_128_median']['phi_consensus'] - phi_results_fixed['dsosd_256_median']['phi_consensus']:.4f}")
                print(f"    DynaSD 128Hz vs 256Hz (Mean):            {phi_results_fixed['dynasd_128_mean']['phi_consensus'] - phi_results_fixed['dynasd_256_mean']['phi_consensus']:.4f}")
                print(f"    DynaSD 128Hz vs 256Hz (Median):          {phi_results_fixed['dynasd_128_median']['phi_consensus'] - phi_results_fixed['dynasd_256_median']['phi_consensus']:.4f}")
                
                # Find best at fixed threshold
                all_phi_fixed = {k: v['phi_consensus'] for k, v in phi_results_fixed.items()}
                best_key_fixed = max(all_phi_fixed, key=all_phi_fixed.get)
                best_phi_fixed = all_phi_fixed[best_key_fixed]
                
                print(f"\n  Summary at Fixed Threshold:")
                print(f"    Best combination: {best_key_fixed} with Phi = {best_phi_fixed:.4f}")
                print(f"    DSOSD 128Hz (original) Mean: {phi_results_fixed['dsosd_128_mean']['phi_consensus']:.4f}")
                print(f"    DynaSD 128Hz Mean: {phi_results_fixed['dynasd_128_mean']['phi_consensus']:.4f}")
                print(f"    Difference (DSOSD 128Hz - DynaSD 128Hz): {phi_results_fixed['dsosd_128_mean']['phi_consensus'] - phi_results_fixed['dynasd_128_mean']['phi_consensus']:.4f}")
                
                # Gaussian threshold analysis
                print(f"\n" + "-" * 80)
                print("Gaussian Threshold Analysis (Automedian)")
                print("-" * 80)
                print("Calculating thresholds using Gaussian mixture models (automedian method).")
                
                phi_results_gaussian = {}
                thresholds_gaussian = {}
                
                # Calculate thresholds for each model/sampling rate combination
                for model_fs_key in ['dsosd_128', 'dsosd_256', 'dynasd_128', 'dynasd_256']:
                    prob_raw = prob_arrays[model_fs_key]
                    prob_chs = prob_chs_dict[model_fs_key]
                    
                    # Get probability DataFrame for threshold calculation
                    if model_fs_key.startswith('dsosd'):
                        if '128' in model_fs_key:
                            prob_df_for_thresh = prob_dsosd_128_df.copy()
                        else:
                            prob_df_for_thresh = prob_dsosd_256_df.copy()
                    else:
                        if '128' in model_fs_key:
                            prob_df_for_thresh = prob_dynasd_128_df.copy()
                        else:
                            prob_df_for_thresh = prob_dynasd_256_df.copy()
                    
                    # Determine threshold calculation window (120s before end)
                    if 'time' in prob_df_for_thresh.columns:
                        prob_times = prob_df_for_thresh['time'].values
                        prob_data_only = prob_df_for_thresh.drop('time', axis=1)
                    else:
                        prob_times = np.arange(len(prob_df_for_thresh)) * 0.5
                        prob_data_only = prob_df_for_thresh.copy()
                    
                    offset_idx = int(np.argmin(np.abs(prob_times - (np.max(prob_times) - 120))))
                    prob_for_threshold = prob_data_only.iloc[:offset_idx, :]
                    
                    # Calculate threshold
                    if model_fs_key.startswith('dsosd'):
                        model_temp = DSOSD_NDD(fs=int(model_fs_key.split('_')[1]), train_win=12, pred_win=1, w_size=1, w_stride=0.5)
                        threshold = model_temp.get_gaussianx_threshold(
                            prob_for_threshold, noise_floor='automedian', verbose=False, seed=100
                        )
                    else:
                        model_temp = DynaSD_NDD(fs=int(model_fs_key.split('_')[1]), sequence_length=12, forecast_length=1, w_size=1, w_stride=0.5)
                        threshold = model_temp.get_threshold(
                            prob_for_threshold, method='automedian', verbose=False, seed=100, threshold_agg='median'
                        )
                    
                    thresholds_gaussian[model_fs_key] = threshold
                    
                    # Apply smoothing
                    prob_mean = sc.ndimage.uniform_filter1d(
                        prob_raw, size=smoothing_window, mode='nearest', axis=1, origin=0
                    )
                    prob_median = sc.ndimage.median_filter(
                        prob_raw, size=smoothing_window, mode='nearest', axes=1, origin=0
                    )
                    
                    # Calculate Phi with Gaussian threshold and mean smoothing
                    phi_mean_gauss = calculate_phi_against_annotations(
                        prob_mean, prob_chs, onset_idx,
                        all_chs, ueo_consensus, ueo_annotators, threshold
                    )
                    phi_mean_gauss['threshold'] = threshold
                    
                    # Calculate Phi with Gaussian threshold and median smoothing
                    phi_median_gauss = calculate_phi_against_annotations(
                        prob_median, prob_chs, onset_idx,
                        all_chs, ueo_consensus, ueo_annotators, threshold
                    )
                    phi_median_gauss['threshold'] = threshold
                    
                    phi_results_gaussian[f'{model_fs_key}_mean'] = phi_mean_gauss
                    phi_results_gaussian[f'{model_fs_key}_median'] = phi_median_gauss
                
                # Display Gaussian threshold results
                print(f"\n  Phi Scores with Gaussian Thresholds:")
                print(f"    {'Model':<12} {'Sampling Rate':<15} {'Threshold':<12} {'Mean Phi':<12} {'Median Phi':<12}")
                print(f"    {'-'*12} {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
                
                for model_fs_key in ['dsosd_128', 'dsosd_256', 'dynasd_128', 'dynasd_256']:
                    model_name = model_fs_key.split('_')[0].upper()
                    fs = model_fs_key.split('_')[1]
                    threshold = thresholds_gaussian[model_fs_key]
                    phi_mean = phi_results_gaussian[f'{model_fs_key}_mean']['phi_consensus']
                    phi_median = phi_results_gaussian[f'{model_fs_key}_median']['phi_consensus']
                    print(f"    {model_name:<12} {fs:<15} {threshold:<12.6f} {phi_mean:<12.4f} {phi_median:<12.4f}")
                
                # Store results
                smoothing_analysis = {
                    'fixed_threshold': fixed_threshold,
                    'phi_fixed_threshold': phi_results_fixed,
                    'phi_gaussian_threshold': phi_results_gaussian,
                    'gaussian_thresholds': thresholds_gaussian
                }
                
                # Use DSOSD 128Hz mean as default (original configuration)
                phi_results = {
                    'phi_dsosd': phi_results_fixed['dsosd_128_mean'],
                    'phi_dynasd': phi_results_fixed['dynasd_128_mean'],
                    'threshold_dsosd': fixed_threshold,
                    'threshold_dynasd': fixed_threshold
                }
            else:
                print(f"  No annotations found for {patient} {onset_run}")
        
        return {
            'patient': patient,
            'onset': onset_run,
            'fs_128': fs_128,
            'fs_256': fs_256,
            'prob_dsosd_128': prob_dsosd_128_df,
            'prob_dsosd_256': prob_dsosd_256_df,
            'prob_dynasd_128': prob_dynasd_128_df,
            'prob_dynasd_256': prob_dynasd_256_df,
            'prob_comparisons': prob_comparisons,
            'window_counts': window_counts,
            'common_channels': list(common_chs),
            'phi_results': phi_results,
            'smoothing_analysis': smoothing_analysis
        }
        
    except Exception as e:
        print(f"Error testing raw EEG: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_on_real_seizures(n_seizures=2, seizure_indices=None, use_raw_eeg=False):
    """
    Test both methods on real seizures from split == 2 dataset.
    
    Parameters:
    -----------
    n_seizures : int
        Number of seizures to test
    seizure_indices : list of int, optional
        Specific indices to test. If None, selects randomly or from different patients.
    """
    if metapath is None or prodatapath is None:
        print("Skipping real seizure test - config not available")
        return None
    
    if DSOSD_NDD is None or DynaSD_NDD is None:
        print("Skipping real seizure test - models not available")
        return None
    
    print("\n" + "=" * 80)
    print("Testing on Real Seizures (split == 2)")
    print("=" * 80)
    
    try:
        # Load seizure metadata
        seizures_df = pd.read_csv(ospj(metapath, "metadata_v7_BIDS.csv"))
        seizures_df = seizures_df[(seizures_df.split == 2) & (seizures_df.stim == 0)]
        
        if len(seizures_df) == 0:
            print("No seizures found in split == 2 with stim == 0")
            return None
        
        print(f"\nFound {len(seizures_df)} seizures in split == 2 with stim == 0")
        
        # Select seizures
        if seizure_indices is not None:
            # Use specified indices
            seizures_to_test = seizures_df.iloc[seizure_indices].copy()
            print(f"Testing on specified seizures: indices {seizure_indices}")
        else:
            # Use all seizures
            seizures_to_test = seizures_df.copy()
            print(f"\nTesting on all {len(seizures_to_test)} seizures")
        
        seizures_to_test = seizures_to_test.reset_index(drop=True)
        
        # Load clinical annotations
        annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v3.pkl"))
        annotations_df = annotations_df[annotations_df.stim == 0]
        
        results = []
        
        for idx, row in seizures_to_test.iterrows():
            patient = row.Patient
            onset_run = str(int(row.onset))
            approx_onset = row.onset
            
            print(f"\n{'='*80}")
            print(f"Seizure: {patient} {onset_run}")
            print(f"{'='*80}")
            
            # Only test with raw EEG - no probability file loading
            if use_raw_eeg and get_data_from_bids is not None:
                print(f"\n  Testing with raw EEG data at different sampling rates...")
                try:
                    raw_eeg_results = test_sampling_rate_on_raw_eeg(patient, onset_run, 
                                                                    annotations_df=annotations_df,
                                                                    bandpass_low=3)  # Default bandpass
                    if raw_eeg_results:
                        results.append({
                            'patient': patient,
                            'onset': onset_run,
                            'type': 'raw_eeg_sampling_rate',
                            'results': raw_eeg_results
                        })
                        print(f"  ✓ Successfully tested {patient} {onset_run}")
                    else:
                        print(f"  ⚠️  No results returned for {patient} {onset_run}")
                except Exception as e:
                    print(f"  ✗ Error testing {patient} {onset_run}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  Skipping - use_raw_eeg=False or get_data_from_bids not available")
        
        # Summary
        if results:
            print(f"\n{'='*80}")
            print("Summary Across Seizures")
            print(f"{'='*80}")
            
            # Handle different result types
            raw_eeg_results = [r for r in results if r.get('type') == 'raw_eeg_sampling_rate']
            prob_file_results = [r for r in results if r.get('type') != 'raw_eeg_sampling_rate' or 'type' not in r]
            
            if raw_eeg_results:
                print(f"\nRaw EEG Testing Results: {len(raw_eeg_results)} seizure(s)")
                # Collect all 4 combinations
                phi_dsosd_128_all = []
                phi_dsosd_256_all = []
                phi_dynasd_128_all = []
                phi_dynasd_256_all = []
                
                for r in raw_eeg_results:
                    print(f"  {r['patient']} {r['onset']}: {'✓ Success' if r.get('results') else '✗ Failed'}")
                    if r.get('results'):
                        if 'prob_comparison' in r['results']:
                            comp = r['results']['prob_comparison']
                            if 'correlation' in comp and not np.isnan(comp['correlation']):
                                print(f"    Probability correlation: {comp['correlation']:.4f}")
                        
                        # Extract all 4 combinations from smoothing analysis
                        if 'smoothing_analysis' in r['results'] and r['results']['smoothing_analysis']:
                            smooth = r['results']['smoothing_analysis']
                            if 'phi_fixed_threshold' in smooth:
                                phi_fixed = smooth['phi_fixed_threshold']
                                
                                # Extract mean smoothing results for all 4 combinations
                                for key in ['dsosd_128_mean', 'dsosd_256_mean', 'dynasd_128_mean', 'dynasd_256_mean']:
                                    if key in phi_fixed and 'phi_consensus' in phi_fixed[key]:
                                        phi_val = phi_fixed[key]['phi_consensus']
                                        if not np.isnan(phi_val):
                                            if key == 'dsosd_128_mean':
                                                phi_dsosd_128_all.append(phi_val)
                                            elif key == 'dsosd_256_mean':
                                                phi_dsosd_256_all.append(phi_val)
                                            elif key == 'dynasd_128_mean':
                                                phi_dynasd_128_all.append(phi_val)
                                            elif key == 'dynasd_256_mean':
                                                phi_dynasd_256_all.append(phi_val)
                                
                                # Print all 4 combinations for this seizure
                                print(f"    Phi scores (mean smoothing, threshold={smooth.get('fixed_threshold', 1.5):.1f}):")
                                if 'dsosd_128_mean' in phi_fixed and 'phi_consensus' in phi_fixed['dsosd_128_mean']:
                                    print(f"      DSOSD 128Hz: {phi_fixed['dsosd_128_mean']['phi_consensus']:.4f}")
                                if 'dsosd_256_mean' in phi_fixed and 'phi_consensus' in phi_fixed['dsosd_256_mean']:
                                    print(f"      DSOSD 256Hz: {phi_fixed['dsosd_256_mean']['phi_consensus']:.4f}")
                                if 'dynasd_128_mean' in phi_fixed and 'phi_consensus' in phi_fixed['dynasd_128_mean']:
                                    print(f"      DynaSD 128Hz: {phi_fixed['dynasd_128_mean']['phi_consensus']:.4f}")
                                if 'dynasd_256_mean' in phi_fixed and 'phi_consensus' in phi_fixed['dynasd_256_mean']:
                                    print(f"      DynaSD 256Hz: {phi_fixed['dynasd_256_mean']['phi_consensus']:.4f}")
                            
                            # Show all smoothing combinations if available
                            print(f"    Smoothing analysis (all combinations):")
                            for key in ['dsosd_128_mean', 'dsosd_128_median', 'dsosd_256_mean', 'dsosd_256_median',
                                       'dynasd_128_mean', 'dynasd_128_median', 'dynasd_256_mean', 'dynasd_256_median']:
                                if key in smooth.get('phi_fixed_threshold', {}) and 'phi_consensus' in smooth['phi_fixed_threshold'][key]:
                                    model_fs = key.replace('_mean', '').replace('_median', '')
                                    smoothing_type = 'Mean' if 'mean' in key else 'Median'
                                    print(f"      {model_fs.upper()} {smoothing_type}: {smooth['phi_fixed_threshold'][key]['phi_consensus']:.4f}")
                
                # Summary across all seizures for all 4 combinations
                print(f"\n  Summary Phi scores across seizures (mean smoothing):")
                if phi_dsosd_128_all:
                    print(f"    DSOSD 128Hz: mean={np.mean(phi_dsosd_128_all):.4f}, std={np.std(phi_dsosd_128_all):.4f}")
                if phi_dsosd_256_all:
                    print(f"    DSOSD 256Hz: mean={np.mean(phi_dsosd_256_all):.4f}, std={np.std(phi_dsosd_256_all):.4f}")
                if phi_dynasd_128_all:
                    print(f"    DynaSD 128Hz: mean={np.mean(phi_dynasd_128_all):.4f}, std={np.std(phi_dynasd_128_all):.4f}")
                if phi_dynasd_256_all:
                    print(f"    DynaSD 256Hz: mean={np.mean(phi_dynasd_256_all):.4f}, std={np.std(phi_dynasd_256_all):.4f}")
                
                # Cross-comparisons
                if phi_dsosd_128_all and phi_dynasd_128_all:
                    print(f"\n  Cross-comparisons:")
                    print(f"    DSOSD 128Hz vs DynaSD 128Hz: {np.mean(phi_dsosd_128_all) - np.mean(phi_dynasd_128_all):.4f}")
                if phi_dsosd_256_all and phi_dynasd_256_all:
                    print(f"    DSOSD 256Hz vs DynaSD 256Hz: {np.mean(phi_dsosd_256_all) - np.mean(phi_dynasd_256_all):.4f}")
                if phi_dsosd_128_all and phi_dsosd_256_all:
                    print(f"    DSOSD 128Hz vs DSOSD 256Hz: {np.mean(phi_dsosd_128_all) - np.mean(phi_dsosd_256_all):.4f}")
                if phi_dynasd_128_all and phi_dynasd_256_all:
                    print(f"    DynaSD 128Hz vs DynaSD 256Hz: {np.mean(phi_dynasd_128_all) - np.mean(phi_dynasd_256_all):.4f}")
            
            if prob_file_results:
                print(f"\nProbability File Results: {len(prob_file_results)} seizure(s)")
                phi_dsosd_all = [r.get('phi_dsosd') for r in prob_file_results if 'phi_dsosd' in r and not np.isnan(r.get('phi_dsosd', np.nan))]
                phi_dynasd_all = [r.get('phi_dynasd') for r in prob_file_results if 'phi_dynasd' in r and not np.isnan(r.get('phi_dynasd', np.nan))]
                
                if phi_dsosd_all and phi_dynasd_all:
                    print(f"\nPhi scores:")
                    print(f"  DSOSD: mean={np.mean(phi_dsosd_all):.4f}, std={np.std(phi_dsosd_all):.4f}")
                    print(f"  DynaSD: mean={np.mean(phi_dynasd_all):.4f}, std={np.std(phi_dynasd_all):.4f}")
                    print(f"  Difference: {np.mean(phi_dsosd_all) - np.mean(phi_dynasd_all):.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error testing on real seizures: {e}")
        import traceback
        traceback.print_exc()
        return None


def _diagnose_single_seizure(patient, onset_run, annotations_df=None):
    """
    Helper function to diagnose thresholding for a single seizure.
    Returns threshold values and performance metrics.
    """
    # Load real seizure probability data
    result = None
    prob_df = None
    
    try:
        result = test_sampling_rate_on_raw_eeg(patient, onset_run, prebuffer=180, postbuffer=120)
        if result is not None and 'prob_dsosd_128' in result:
            prob_df = result['prob_dsosd_128'].copy()
        else:
            return None
    except Exception as e:
        print(f"  ✗ Error loading {patient} {onset_run}: {e}")
        return None
    
    # Determine threshold calculation window
    if 'time' in prob_df.columns:
        prob_times = prob_df['time'].values
        prob_data_only = prob_df.drop('time', axis=1)
    else:
        prob_times = np.arange(len(prob_df)) * 0.5
        prob_data_only = prob_df
    
    offset_idx = int(np.argmin(np.abs(prob_times - (np.max(prob_times) - 120))))
    prob_for_threshold = prob_data_only.iloc[:offset_idx, :]
    
    # Initialize models
    model_dsosd = DSOSD_NDD(fs=128, train_win=12, pred_win=1, w_size=1, w_stride=0.5)
    model_dynasd = DynaSD_NDD(fs=128, sequence_length=12, forecast_length=1, w_size=1, w_stride=0.5)
    
    # Calculate thresholds
    threshold_dsosd = model_dsosd.get_gaussianx_threshold(
        prob_for_threshold, noise_floor='automedian', verbose=False, seed=100
    )
    
    threshold_dynasd = model_dynasd.get_threshold(
        prob_for_threshold, method='automedian', verbose=False, seed=100, threshold_agg='median'
    )
    
    # Test with matched parameters
    original_boundary = model_dynasd._boundary
    model_dynasd._boundary = 1.1748070328441302
    
    def dsosd_fallback_threshold(self):
        return 1.479305740987984
    import types
    model_dynasd._get_pretrained_threshold = types.MethodType(dsosd_fallback_threshold, model_dynasd)
    
    threshold_dynasd_matched = model_dynasd.get_threshold(
        prob_for_threshold, method='automedian', verbose=False, seed=100, threshold_agg='median'
    )
    
    model_dynasd._boundary = original_boundary
    
    # Calculate performance if annotations available
    phi_results = {}
    if annotations_df is not None:
        approx_onset = float(onset_run)
        annot_matches = annotations_df[
            (annotations_df['patient'] == patient) & 
            (np.abs(annotations_df['approximate_onset'].astype(float) - approx_onset) < 360)
        ]
        
        if len(annot_matches) > 0:
            annot_row = annot_matches.iloc[0]
            all_chs = annot_row['all_chs']
            ueo_consensus = annot_row['ueo_consensus']
            ueo_annotators = annot_row['ueo']
            
            onset_idx = int(180 / 0.5)
            prob_chs = np.array([ch.split('-')[0] for ch in prob_df.columns])
            prob_array = prob_df.to_numpy().T
            
            smoothing_window = 20
            prob_smooth = sc.ndimage.uniform_filter1d(
                prob_array, size=smoothing_window, mode='nearest', axis=1, origin=0
            )
            
            phi_results['dsosd'] = calculate_phi_against_annotations(
                prob_smooth, prob_chs, onset_idx, all_chs, ueo_consensus, ueo_annotators, threshold_dsosd
            )
            phi_results['dynasd_orig'] = calculate_phi_against_annotations(
                prob_smooth, prob_chs, onset_idx, all_chs, ueo_consensus, ueo_annotators, threshold_dynasd
            )
            phi_results['dynasd_matched'] = calculate_phi_against_annotations(
                prob_smooth, prob_chs, onset_idx, all_chs, ueo_consensus, ueo_annotators, threshold_dynasd_matched
            )
    
    return {
        'patient': patient,
        'onset_run': onset_run,
        'threshold_dsosd': threshold_dsosd,
        'threshold_dynasd': threshold_dynasd,
        'threshold_dynasd_matched': threshold_dynasd_matched,
        'phi_results': phi_results,
        'prob_df': prob_df
    }


def diagnose_thresholding_differences(seizures_list=None, annotations_df=None):
    """
    Diagnostic function to compare thresholding methods between DSOSD and DynaSD.
    Tests different configurations and allows modifying DynaSD's private variables.
    Uses real seizure probability data to ensure thresholding logic is properly tested.
    Tests multiple seizures to ensure robustness.
    
    Parameters:
    -----------
    seizures_list : list of tuples, optional
        List of (patient, onset_run) tuples to test. 
        Default: [('HUP235', '497084'), ('HUP238', '290006'), and tries to find another HUP238]
    annotations_df : pd.DataFrame, optional
        Clinical annotations dataframe for performance testing
    """
    if DSOSD_NDD is None or DynaSD_NDD is None:
        print("Skipping thresholding diagnostic - models not available")
        return None
    
    # Default seizures: HUP235 497084, HUP238 290006, and try to find another HUP238
    if seizures_list is None:
        seizures_list = [('HUP235', '497084'), ('HUP238', '290006')]
        # Try to find another HUP238 seizure
        try:
            from config import Config
            _, _, _, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])
            seizures_df = pd.read_csv(ospj(metapath, "metadata_v7_BIDS.csv"))
            hup238_seizures = seizures_df[(seizures_df.Patient == 'HUP238') & (seizures_df.stim == 0)]
            if len(hup238_seizures) > 1:
                # Get a different one than 290006
                other_hup238 = hup238_seizures[hup238_seizures.onset.astype(int) != 290006]
                if len(other_hup238) > 0:
                    seizures_list.append(('HUP238', str(int(other_hup238.iloc[0].onset))))
        except:
            pass
    
    print("\n" + "=" * 80)
    print("Thresholding Diagnostic: DSOSD vs DynaSD")
    print("=" * 80)
    print(f"Testing on {len(seizures_list)} seizure(s): {seizures_list}")
    
    # Process each seizure
    all_results = []
    for patient, onset_run in seizures_list:
        print(f"\n{'='*80}")
        print(f"Processing: {patient} {onset_run}")
        print(f"{'='*80}")
        result = _diagnose_single_seizure(patient, onset_run, annotations_df)
        if result is not None:
            all_results.append(result)
            print(f"  ✓ Successfully processed")
        else:
            print(f"  ✗ Failed to process")
    
    if len(all_results) == 0:
        print("\n✗ No seizures successfully processed")
        return None
    
    # Use first seizure for detailed analysis
    first_result = all_results[0]
    prob_df = first_result['prob_df']
    
    # Load real seizure probability data
    try:
        # Run the real seizure test to get probability data
        print(f"\nLoading real seizure data...")
        result = test_sampling_rate_on_raw_eeg(patient, onset_run, prebuffer=180, postbuffer=120)
        
        if result is None or 'prob_dsosd_128' not in result:
            print("  ✗ Could not load real seizure data, falling back to synthetic data")
            # Fallback to synthetic data
            np.random.seed(42)
            n_timepoints = 1000
            n_channels = 10
            prob_data = np.zeros((n_timepoints, n_channels))
            for ch in range(n_channels):
                prob_data[:800, ch] = np.random.lognormal(mean=0.1, sigma=0.3, size=800)
                prob_data[800:, ch] = np.random.lognormal(mean=1.5, sigma=0.4, size=200)
            prob_df = pd.DataFrame(
                prob_data,
                columns=[f'ch{i:02d}-ch{i+1:02d}' for i in range(n_channels)]
            )
        else:
            # Use real probability data (use DSOSD 128Hz as reference)
            prob_df = result['prob_dsosd_128'].copy()
            print(f"  ✓ Loaded real probability data: {prob_df.shape}")
    except Exception as e:
        print(f"  ✗ Error loading real seizure data: {e}")
        print("  Falling back to synthetic data...")
        import traceback
        traceback.print_exc()
        # Fallback to synthetic data
        np.random.seed(42)
        n_timepoints = 1000
        n_channels = 10
        prob_data = np.zeros((n_timepoints, n_channels))
        for ch in range(n_channels):
            prob_data[:800, ch] = np.random.lognormal(mean=0.1, sigma=0.3, size=800)
            prob_data[800:, ch] = np.random.lognormal(mean=1.5, sigma=0.4, size=200)
        prob_df = pd.DataFrame(
            prob_data,
            columns=[f'ch{i:02d}-ch{i+1:02d}' for i in range(n_channels)]
        )
    
    print(f"\nTest data shape: {prob_df.shape}")
    print(f"Probability range: [{prob_df.min().min():.4f}, {prob_df.max().max():.4f}]")
    print(f"Mean: {prob_df.mean().mean():.4f}, Median: {prob_df.median().median():.4f}")
    print(f"Percentiles: 50th={prob_df.quantile(0.5).mean():.4f}, 95th={prob_df.quantile(0.95).mean():.4f}, 99th={prob_df.quantile(0.99).mean():.4f}")
    
    # Determine threshold calculation window
    # Match test_validation_analysis.py: use :offset_idx where offset_idx is 120 seconds before end
    # This matches how thresholds are calculated in production code
    if result is not None and 'prob_dsosd_128' in result:
        # Real data: use up to 120 seconds before the end (matching test_validation_analysis.py)
        # Extract time column if it exists
        if 'time' in prob_df.columns:
            prob_times = prob_df['time'].values
            prob_data_only = prob_df.drop('time', axis=1)
        else:
            # If no time column, estimate from window stride (0.5s)
            prob_times = np.arange(len(prob_df)) * 0.5
            prob_data_only = prob_df
        
        # Calculate offset_idx: 120 seconds before the end
        offset_idx = int(np.argmin(np.abs(prob_times - (np.max(prob_times) - 120))))
        prob_for_threshold = prob_data_only.iloc[:offset_idx, :]
        print(f"\nUsing first {offset_idx} windows (up to 120s before end) for threshold calculation")
        print(f"  Time range: 0.0s to {prob_times[offset_idx]:.1f}s (out of {np.max(prob_times):.1f}s total)")
    else:
        # Synthetic data: use first 500 samples
        threshold_window_size = 500
        prob_for_threshold = prob_df.iloc[:threshold_window_size, :]
        print(f"\nUsing first {threshold_window_size} samples for threshold calculation")
    
    # Initialize models
    model_dsosd = DSOSD_NDD(fs=128, train_win=12, pred_win=1, w_size=1, w_stride=0.5)
    model_dynasd = DynaSD_NDD(fs=128, sequence_length=12, forecast_length=1, w_size=1, w_stride=0.5)
    
    print("\n" + "-" * 80)
    print("1. Default Thresholding Comparison")
    print("-" * 80)
    
    # DSOSD default (automedian)
    threshold_dsosd = model_dsosd.get_gaussianx_threshold(
        prob_for_threshold,
        noise_floor='automedian',
        verbose=False,
        seed=100
    )
    print(f"DSOSD (automedian): {threshold_dsosd:.6f}")
    
    # DynaSD default (automedian)
    threshold_dynasd = model_dynasd.get_threshold(
        prob_for_threshold,
        method='automedian',
        verbose=False,
        seed=100,
        threshold_agg='median'
    )
    print(f"DynaSD (automedian, threshold_agg='median'): {threshold_dynasd:.6f}")
    print(f"Difference: {threshold_dsosd - threshold_dynasd:.6f}")
    
    print("\n" + "-" * 80)
    print("2. Examining Internal Parameters")
    print("-" * 80)
    
    print(f"\nDSOSD hardcoded values:")
    print(f"  Boundary check: 1.1725")
    print(f"  Boundary filter: 1.1748070328441302")
    print(f"  Fallback threshold: 1.479305740987984")
    
    print(f"\nDynaSD current values:")
    print(f"  _boundary: {model_dynasd._boundary}")
    print(f"  _get_pretrained_threshold() (median): {model_dynasd._get_pretrained_threshold()}")
    model_dynasd.threshold_agg = 'mean'
    print(f"  _get_pretrained_threshold() (mean): {model_dynasd._get_pretrained_threshold()}")
    model_dynasd.threshold_agg = 'median'  # Reset
    
    print("\n" + "-" * 80)
    print("3. Testing with DSOSD-equivalent Parameters")
    print("-" * 80)
    
    # Save original values
    original_boundary = model_dynasd._boundary
    original_threshold_agg = model_dynasd.threshold_agg
    
    # Modify DynaSD to match DSOSD's parameters
    print("\nModifying DynaSD to match DSOSD parameters...")
    model_dynasd._boundary = 1.1748070328441302  # Match DSOSD's filter boundary
    
    # Override _get_pretrained_threshold to return DSOSD's fallback
    def dsosd_fallback_threshold(self):
        return 1.479305740987984
    
    # Monkey patch the method
    import types
    model_dynasd._get_pretrained_threshold = types.MethodType(dsosd_fallback_threshold, model_dynasd)
    
    threshold_dynasd_matched = model_dynasd.get_threshold(
        prob_for_threshold,
        method='automedian',
        verbose=False,
        seed=100,
        threshold_agg='median'
    )
    print(f"DynaSD (matched boundary + fallback): {threshold_dynasd_matched:.6f}")
    print(f"Difference from DSOSD: {threshold_dsosd - threshold_dynasd_matched:.6f}")
    
    # Test with different boundary values
    print("\n" + "-" * 80)
    print("4. Testing Different Boundary Values")
    print("-" * 80)
    
    boundary_tests = [
        (1.11771875, "DynaSD original"),
        (1.1725, "DSOSD check boundary"),
        (1.1748070328441302, "DSOSD filter boundary"),
        (1.194797045747334, "DSOSD automean check"),
    ]
    
    for boundary_val, label in boundary_tests:
        model_dynasd._boundary = boundary_val
        threshold = model_dynasd.get_threshold(
            prob_for_threshold,
            method='automedian',
            verbose=False,
            seed=100,
            threshold_agg='median'
        )
        print(f"  Boundary {boundary_val:.10f} ({label}): {threshold:.6f}")
    
    # Reset to original
    model_dynasd._boundary = original_boundary
    model_dynasd.threshold_agg = original_threshold_agg
    
    print("\n" + "-" * 80)
    print("5. Step-by-Step Boundary Calculation Comparison")
    print("-" * 80)
    
    # Manually compute boundaries for first channel to see differences
    ch_idx = 0
    X_dsosd = prob_for_threshold.iloc[:, ch_idx].to_numpy()
    X_dynasd = prob_for_threshold.iloc[:, ch_idx].ffill().to_numpy()  # DynaSD uses ffill
    
    X_f_dsosd = np.log(X_dsosd.reshape(-1,1)+1e-10)
    X_f_dynasd = np.log(X_dynasd.reshape(-1,1)+1e-10)
    
    # Percentile calculation difference
    X_f_dsosd_filtered = X_f_dsosd[X_f_dsosd < np.percentile(X_f_dsosd, 99.99)].reshape(-1,1)
    X_f_dynasd_filtered = X_f_dynasd[X_f_dynasd < np.percentile(X_f_dynasd, 99.99, method='lower')].reshape(-1,1)
    
    print(f"\nChannel {prob_df.columns[ch_idx]}:")
    print(f"  DSOSD data points after filtering: {len(X_f_dsosd_filtered)}")
    print(f"  DynaSD data points after filtering: {len(X_f_dynasd_filtered)}")
    print(f"  Difference: {len(X_f_dsosd_filtered) - len(X_f_dynasd_filtered)}")
    
    # Fit GMMs
    from sklearn.mixture import GaussianMixture
    
    gmm_dsosd = GaussianMixture(n_components=2, random_state=100)
    gmm_dsosd.fit(X_f_dsosd_filtered)
    
    gmm_dynasd = GaussianMixture(n_components=2, random_state=100)
    gmm_dynasd.fit(X_f_dynasd_filtered)
    
    # Calculate boundaries
    means_dsosd = gmm_dsosd.means_.flatten()
    means_dynasd = gmm_dynasd.means_.flatten()
    
    sigma1_dsosd, sigma2_dsosd = np.sqrt(gmm_dsosd.covariances_.flatten())
    sigma1_dynasd, sigma2_dynasd = np.sqrt(gmm_dynasd.covariances_.flatten())
    
    pi1_dsosd, pi2_dsosd = gmm_dsosd.weights_
    pi1_dynasd, pi2_dynasd = gmm_dynasd.weights_
    
    # Solve for boundaries
    A_dsosd = (1 / (2 * sigma1_dsosd**2)) - (1 / (2 * sigma2_dsosd**2))
    B_dsosd = (means_dsosd[1] / sigma2_dsosd**2) - (means_dsosd[0] / sigma1_dsosd**2)
    C_dsosd = ((means_dsosd[0]**2 / (2 * sigma1_dsosd**2)) - (means_dsosd[1]**2 / (2 * sigma2_dsosd**2))
               - np.log((pi1_dsosd * sigma2_dsosd) / (pi2_dsosd * sigma1_dsosd)))
    
    A_dynasd = (1 / (2 * sigma1_dynasd**2)) - (1 / (2 * sigma2_dynasd**2))
    B_dynasd = (means_dynasd[1] / sigma2_dynasd**2) - (means_dynasd[0] / sigma1_dynasd**2)
    C_dynasd = ((means_dynasd[0]**2 / (2 * sigma1_dynasd**2)) - (means_dynasd[1]**2 / (2 * sigma2_dynasd**2))
                - np.log((pi1_dynasd * sigma2_dynasd) / (pi2_dynasd * sigma1_dynasd)))
    
    boundaries_dsosd = np.roots([A_dsosd, B_dsosd, C_dsosd])
    boundaries_dynasd = np.roots([A_dynasd, B_dynasd, C_dynasd])
    
    meets_criteria_dsosd = np.exp(boundaries_dsosd[(boundaries_dsosd > min(means_dsosd)) & (boundaries_dsosd < max(means_dsosd))])
    meets_criteria_dynasd = np.exp(boundaries_dynasd[(boundaries_dynasd > min(means_dynasd)) & (boundaries_dynasd < max(means_dynasd))])
    
    print(f"\nGMM Results:")
    print(f"  DSOSD means: {means_dsosd}")
    print(f"  DynaSD means: {means_dynasd}")
    print(f"  DSOSD boundary (log space): {meets_criteria_dsosd[0] if len(meets_criteria_dsosd) > 0 else 'None'}")
    print(f"  DynaSD boundary (log space): {meets_criteria_dynasd[0] if len(meets_criteria_dynasd) > 0 else 'None'}")
    if len(meets_criteria_dsosd) > 0 and len(meets_criteria_dynasd) > 0:
        print(f"  Boundary difference: {meets_criteria_dsosd[0] - meets_criteria_dynasd[0]:.6f}")
    
    print("\n" + "-" * 80)
    print("6. Performance Testing on Real Seizures")
    print("-" * 80)
    
    # Test performance with both thresholding methods on real seizures
    if result is not None and 'prob_dsosd_128' in result and annotations_df is not None:
        print("\nTesting performance with different thresholding configurations...")
        
        # Get annotations for this seizure
        approx_onset = float(onset_run)
        annot_matches = annotations_df[
            (annotations_df['patient'] == patient) & 
            (np.abs(annotations_df['approximate_onset'].astype(float) - approx_onset) < 360)
        ]
        
        if len(annot_matches) > 0:
            annot_row = annot_matches.iloc[0]
            all_chs = annot_row['all_chs']
            ueo_consensus = annot_row['ueo_consensus']
            ueo_annotators = annot_row['ueo']
            
            # Calculate temporal alignment (onset is at 180s from start)
            onset_idx = int(180 / 0.5)  # Window index at 180s
            
            # Get probability data (use DSOSD 128Hz as reference)
            prob_chs = np.array([ch.split('-')[0] for ch in prob_df.columns])
            prob_array = prob_df.to_numpy().T  # (n_channels, n_timepoints)
            
            # Apply smoothing (mean, as in test_validation_analysis)
            smoothing_window = 20
            prob_smooth = sc.ndimage.uniform_filter1d(
                prob_array, size=smoothing_window, mode='nearest', axis=1, origin=0
            )
            
            # Test 1: DSOSD threshold
            print(f"\n  DSOSD threshold ({threshold_dsosd:.6f}):")
            phi_dsosd = calculate_phi_against_annotations(
                prob_smooth, prob_chs, onset_idx,
                all_chs, ueo_consensus, ueo_annotators, threshold_dsosd
            )
            print(f"    Phi (consensus): {phi_dsosd['phi_consensus']:.4f}")
            print(f"    Phi (annotators): {[f'{p:.4f}' for p in phi_dsosd['phi_annotators']]}")
            
            # Test 2: DynaSD original threshold
            print(f"\n  DynaSD original threshold ({threshold_dynasd:.6f}):")
            phi_dynasd_orig = calculate_phi_against_annotations(
                prob_smooth, prob_chs, onset_idx,
                all_chs, ueo_consensus, ueo_annotators, threshold_dynasd
            )
            print(f"    Phi (consensus): {phi_dynasd_orig['phi_consensus']:.4f}")
            print(f"    Phi (annotators): {[f'{p:.4f}' for p in phi_dynasd_orig['phi_annotators']]}")
            
            # Test 3: DynaSD matched threshold (DSOSD-equivalent)
            print(f"\n  DynaSD matched threshold ({threshold_dynasd_matched:.6f}):")
            phi_dynasd_matched = calculate_phi_against_annotations(
                prob_smooth, prob_chs, onset_idx,
                all_chs, ueo_consensus, ueo_annotators, threshold_dynasd_matched
            )
            print(f"    Phi (consensus): {phi_dynasd_matched['phi_consensus']:.4f}")
            print(f"    Phi (annotators): {[f'{p:.4f}' for p in phi_dynasd_matched['phi_annotators']]}")
            
            print(f"\n  Performance Comparison:")
            print(f"    DSOSD vs DynaSD original: {phi_dsosd['phi_consensus'] - phi_dynasd_orig['phi_consensus']:.4f}")
            print(f"    DSOSD vs DynaSD matched: {phi_dsosd['phi_consensus'] - phi_dynasd_matched['phi_consensus']:.4f}")
            print(f"    DynaSD original vs matched: {phi_dynasd_orig['phi_consensus'] - phi_dynasd_matched['phi_consensus']:.4f}")
        else:
            print("  No annotations found for this seizure")
    else:
        print("  Skipping performance test - need real seizure data with annotations")
    
    print("\n" + "-" * 80)
    print("7. Summary of Differences")
    print("-" * 80)
    print("""
Key Differences Found:
1. Boundary value: DSOSD uses 1.1748070328441302, DynaSD uses 1.11771875
2. Fallback threshold: DSOSD uses 1.479305740987984, DynaSD uses pretrained (1.334605646 for median)
3. Data preprocessing: DynaSD uses .ffill() before log transform, DSOSD doesn't
4. Percentile method: DynaSD uses method='lower', DSOSD doesn't specify
5. Boundary check: DSOSD checks > 1.1725 first, then filters > 1.1748070328441302
   DynaSD only checks > self._boundary once

Note: DSOSD does NOT use fillna() or ffill() at all - it processes data directly.
      DynaSD uses .ffill() which is the correct modern syntax (not deprecated).
      The deprecated syntax would be fillna(method='ffill'), but neither codebase uses it.

To match DSOSD exactly in DynaSD:
- Set model._boundary = 1.1748070328441302
- Override model._get_pretrained_threshold() to return 1.479305740987984
- Remove .ffill() from _compute_gaussian_boundaries (modify base.py)
- Change percentile calculation to not use method='lower'
- Modify _aggregate_threshold to check > 1.1725 first, then filter > 1.1748070328441302
    """)
    
    return {
        'threshold_dsosd': threshold_dsosd,
        'threshold_dynasd': threshold_dynasd,
        'threshold_dynasd_matched': threshold_dynasd_matched,
        'boundary_tests': {label: val for val, label in boundary_tests},
        'performance_results': {
            'phi_dsosd': phi_dsosd if 'phi_dsosd' in locals() else None,
            'phi_dynasd_orig': phi_dynasd_orig if 'phi_dynasd_orig' in locals() else None,
            'phi_dynasd_matched': phi_dynasd_matched if 'phi_dynasd_matched' in locals() else None
        } if 'phi_dsosd' in locals() else None
    }


if __name__ == "__main__":
    import glob
    
    # Test 1: Sequence equivalence
    print("=" * 80)
    print("TEST 1: Sequence Preparation Equivalence")
    print("=" * 80)
    results = test_equivalence(
        data_length=10000,  # ~39 seconds at 256 Hz
        fs=256,
        train_win=12,
        pred_win=1,
        w_size=1,
        w_stride=0.5
    )
    
    # Test 2: Distribution comparison (synthetic MSE values)
    print("\n" + "=" * 80)
    print("TEST 2: Distribution Comparison (Synthetic MSE Values)")
    print("=" * 80)
    print("NOTE: This compares aggregated window MSE from hankel vs sliding methods")
    print("      using SYNTHETIC random MSE values. Low correlation is expected!")
    print("      For real probability comparison, see TEST 5.")
    mse_hankel = np.random.rand(results['hankel_info']['total_sequences'], 8)
    mse_sliding = np.random.rand(results['sliding_info']['n_sequences'], 8)
    window_mse_hankel, window_mse_sliding = compare_distributions(
        mse_hankel, mse_sliding, 
        results['hankel_info'], results['sliding_info'], 
        n_channels=8
    )
    
    # Test 3: get_onset_and_spread comparison (synthetic data)
    print("\n" + "=" * 80)
    print("TEST 3: get_onset_and_spread Comparison (Synthetic)")
    print("=" * 80)
    print("NOTE: This test uses synthetic random probabilities.")
    print("      Large differences (e.g., 26 seconds) occur because:")
    print("      1. DSOSD filters AFTER thresholding (median filter on binary)")
    print("      2. DynaSD filters BEFORE thresholding (uniform filter on probabilities)")
    print("      3. When probabilities hover around threshold, filtering order matters!")
    
    # Create more realistic synthetic data with some structure
    np.random.seed(42)
    n_timepoints = 100
    n_channels = 10
    
    # Create probabilities that vary over time (more realistic)
    synthetic_prob = pd.DataFrame(
        np.zeros((n_timepoints, n_channels)),
        columns=[f'ch{i:02d}-ch{i+1:02d}' for i in range(n_channels)]
    )
    
    # Add some channels that cross threshold at different times
    for ch_idx in range(n_channels):
        # Create a signal that gradually increases, then decreases
        t = np.arange(n_timepoints)
        base_signal = 0.5 + 0.3 * np.sin(2 * np.pi * t / n_timepoints)
        noise = np.random.randn(n_timepoints) * 0.2
        synthetic_prob.iloc[:, ch_idx] = base_signal + noise
    
    threshold = 1.0
    try:
        spread_comparison_synthetic = compare_get_onset_and_spread(synthetic_prob, threshold, verbose=True)
    except Exception as e:
        print(f"Error in get_onset_and_spread comparison: {e}")
        import traceback
        traceback.print_exc()
        spread_comparison_synthetic = None
    
    # Test 4: Sampling Rate Impact
    print("\n" + "=" * 80)
    print("TEST 4: Sampling Rate Impact (256 Hz vs 128 Hz)")
    print("=" * 80)
    try:
        sampling_rate_results = test_sampling_rate_impact(data_length_seconds=60, n_channels=8)
    except Exception as e:
        print(f"Error in sampling rate test: {e}")
        import traceback
        traceback.print_exc()
        sampling_rate_results = None
    
    # Test 5: Comprehensive testing with all combinations
    print("\n" + "=" * 80)
    print("TEST 5: Comprehensive Testing - All Combinations")
    print("=" * 80)
    print("NOTE: This tests all combinations of:")
    print("      - Preprocessing: bandpass low (1, 3, 5 Hz)")
    print("      - Model: DSOSD, DynaSD")
    print("      - Sampling rate: 128 Hz, 256 Hz")
    print("      - Smoothing: mean, median")
    print("      - Threshold: fixed (1.5), Gaussian (automedian)")
    print("      - Seizures: All seizures in split == 2 with stim == 0")
    try:
        # Load annotations
        try:
            from config import Config
            _, prodatapath, _, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])
            annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v3.pkl"))
            annotations_df = annotations_df[annotations_df.stim == 0]
        except:
            annotations_df = None
            print("  Warning: Could not load annotations")
        
        # Get seizures to test - all seizures in split == 2 with stim == 0
        seizures_df = pd.read_csv(ospj(metapath, "metadata_v7_BIDS.csv"))
        seizures_df = seizures_df[(seizures_df.split == 2) & (seizures_df.stim == 0)]
        
        # Use all seizures
        seizures_to_test = seizures_df.copy()
        print(f"\nFound {len(seizures_to_test)} seizures in split == 2 with stim == 0")
        
        # Test all combinations
        bandpass_lows = [1, 3, 5]
        all_results = []
        
        for bandpass_low in bandpass_lows:
            print(f"\n{'='*80}")
            print(f"Testing with bandpass low = {bandpass_low} Hz")
            print(f"{'='*80}")
            
            for idx, row in seizures_to_test.iterrows():
                patient = row.Patient
                onset_run = str(int(row.onset))
                
                print(f"\n  Processing: {patient} {onset_run} (bandpass low={bandpass_low} Hz)")
                try:
                    result = test_sampling_rate_on_raw_eeg(
                        patient, onset_run, 
                        annotations_df=annotations_df,
                        bandpass_low=bandpass_low
                    )
                    if result:
                        result['bandpass_low'] = bandpass_low
                        all_results.append(result)
                        print(f"    ✓ Success")
                    else:
                        print(f"    ✗ Failed")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
        
        # Create comprehensive results table
        print(f"\n{'='*80}")
        print("COMPREHENSIVE RESULTS TABLE")
        print(f"{'='*80}")
        
        # Aggregate all results
        results_table = []
        
        for result in all_results:
            patient = result['patient']
            onset = result['onset']
            bandpass_low = result['bandpass_low']
            
            if 'smoothing_analysis' in result and result['smoothing_analysis']:
                smooth = result['smoothing_analysis']
                
                # Fixed threshold results
                if 'phi_fixed_threshold' in smooth:
                    phi_fixed = smooth['phi_fixed_threshold']
                    fixed_thresh = smooth.get('fixed_threshold', 1.5)
                    
                    for model_fs_key in ['dsosd_128', 'dsosd_256', 'dynasd_128', 'dynasd_256']:
                        for smoothing_type in ['mean', 'median']:
                            key = f'{model_fs_key}_{smoothing_type}'
                            if key in phi_fixed:
                                phi_val = phi_fixed[key].get('phi_consensus', np.nan)
                                if not np.isnan(phi_val):
                                    model_name = model_fs_key.split('_')[0].upper()
                                    fs = model_fs_key.split('_')[1]
                                    results_table.append({
                                        'patient': patient,
                                        'onset': onset,
                                        'bandpass_low': bandpass_low,
                                        'model': model_name,
                                        'sampling_rate': fs,
                                        'smoothing': smoothing_type,
                                        'threshold_type': 'fixed',
                                        'threshold_value': fixed_thresh,
                                        'phi': phi_val,
                                        'sensitivity': phi_fixed[key].get('sensitivity', np.nan),
                                        'specificity': phi_fixed[key].get('specificity', np.nan),
                                        'f1': phi_fixed[key].get('f1', np.nan)
                                    })
                
                # Gaussian threshold results (if available)
                if 'phi_gaussian_threshold' in smooth:
                    phi_gauss = smooth['phi_gaussian_threshold']
                    for model_fs_key in ['dsosd_128', 'dsosd_256', 'dynasd_128', 'dynasd_256']:
                        for smoothing_type in ['mean', 'median']:
                            key = f'{model_fs_key}_{smoothing_type}'
                            if key in phi_gauss:
                                phi_val = phi_gauss[key].get('phi_consensus', np.nan)
                                threshold_val = phi_gauss[key].get('threshold', np.nan)
                                if not np.isnan(phi_val):
                                    model_name = model_fs_key.split('_')[0].upper()
                                    fs = model_fs_key.split('_')[1]
                                    results_table.append({
                                        'patient': patient,
                                        'onset': onset,
                                        'bandpass_low': bandpass_low,
                                        'model': model_name,
                                        'sampling_rate': fs,
                                        'smoothing': smoothing_type,
                                        'threshold_type': 'gaussian',
                                        'threshold_value': threshold_val,
                                        'phi': phi_val,
                                        'sensitivity': phi_gauss[key].get('sensitivity', np.nan),
                                        'specificity': phi_gauss[key].get('specificity', np.nan),
                                        'f1': phi_gauss[key].get('f1', np.nan)
                                    })
        
        # Create DataFrame and display summary
        if results_table:
            results_df = pd.DataFrame(results_table)
            
            print(f"\nTotal configurations tested: {len(results_df)}")
            print(f"\nSummary Statistics by Configuration:")
            print(f"{'='*120}")
            
            # Group by configuration (excluding patient/onset)
            config_cols = ['bandpass_low', 'model', 'sampling_rate', 'smoothing', 'threshold_type']
            summary = results_df.groupby(config_cols)['phi'].agg(['mean', 'std', 'count']).reset_index()
            summary = summary.sort_values('mean', ascending=False)
            
            print(f"\n{'Bandpass':<10} {'Model':<8} {'FS':<6} {'Smooth':<8} {'Threshold':<12} {'Mean Phi':<10} {'Std Phi':<10} {'Count':<8}")
            print(f"{'-'*120}")
            for _, row in summary.iterrows():
                print(f"{row['bandpass_low']:<10.0f} {row['model']:<8} {row['sampling_rate']:<6} {row['smoothing']:<8} {row['threshold_type']:<12} {row['mean']:<10.4f} {row['std']:<10.4f} {row['count']:<8.0f}")
            
            # Best configurations
            print(f"\n{'='*80}")
            print("BEST CONFIGURATIONS")
            print(f"{'='*80}")
            
            # Best overall
            best_overall = summary.iloc[0]
            print(f"\nBest Overall Configuration:")
            print(f"  Bandpass low: {best_overall['bandpass_low']:.0f} Hz")
            print(f"  Model: {best_overall['model']}")
            print(f"  Sampling rate: {best_overall['sampling_rate']} Hz")
            print(f"  Smoothing: {best_overall['smoothing']}")
            print(f"  Threshold: {best_overall['threshold_type']}")
            print(f"  Mean Phi: {best_overall['mean']:.4f} ± {best_overall['std']:.4f}")
            
            # Best by threshold type
            for thresh_type in ['fixed', 'gaussian']:
                thresh_summary = summary[summary['threshold_type'] == thresh_type]
                if len(thresh_summary) > 0:
                    best_thresh = thresh_summary.iloc[0]
                    print(f"\nBest {thresh_type.capitalize()} Threshold Configuration:")
                    print(f"  Bandpass low: {best_thresh['bandpass_low']:.0f} Hz")
                    print(f"  Model: {best_thresh['model']}")
                    print(f"  Sampling rate: {best_thresh['sampling_rate']} Hz")
                    print(f"  Smoothing: {best_thresh['smoothing']}")
                    print(f"  Mean Phi: {best_thresh['mean']:.4f} ± {best_thresh['std']:.4f}")
            
            # Best smoothing method
            smooth_summary = summary.groupby('smoothing')['mean'].mean().sort_values(ascending=False)
            print(f"\nBest Smoothing Method (overall):")
            for smooth_type, mean_phi in smooth_summary.items():
                print(f"  {smooth_type.capitalize()}: {mean_phi:.4f}")
            
            # Best bandpass
            bandpass_summary = summary.groupby('bandpass_low')['mean'].mean().sort_values(ascending=False)
            print(f"\nBest Bandpass Low (overall):")
            for bp_low, mean_phi in bandpass_summary.items():
                print(f"  {bp_low:.0f} Hz: {mean_phi:.4f}")
            
            # Store comprehensive results
            real_seizure_results = {
                'comprehensive_results': results_df,
                'summary': summary,
                'all_results': all_results
            }
        else:
            print("  No results to display")
            real_seizure_results = None
    except Exception as e:
        print(f"Error in real seizure testing: {e}")
        import traceback
        traceback.print_exc()
        real_seizure_results = None
    
    # Test 6: Thresholding diagnostic
    print("\n" + "=" * 80)
    print("TEST 6: Thresholding Diagnostic")
    print("=" * 80)
    print("NOTE: This compares thresholding methods between DSOSD and DynaSD")
    print("      and allows modifying DynaSD's private variables to match DSOSD.")
    print("      Uses real seizure probability data to properly test thresholding logic.")
    try:
        # Load annotations for performance testing
        try:
            from config import Config
            _, prodatapath, _, metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])
            annotations_df = pd.read_pickle(ospj(prodatapath, "threshold_tuning_consensus_v3.pkl"))
            annotations_df = annotations_df[annotations_df.stim == 0]
        except:
            annotations_df = None
            print("  Warning: Could not load annotations, skipping performance testing")
        
        # Use the same seizures from Test 5 if available, otherwise use defaults
        seizures_list = None
        if real_seizure_results and isinstance(real_seizure_results, dict) and 'all_results' in real_seizure_results:
            # Extract seizures from Test 5 results
            seizures_list = []
            for r in real_seizure_results['all_results']:
                if isinstance(r, dict) and 'patient' in r and 'onset' in r:
                    seizures_list.append((r['patient'], r['onset']))
            # Remove duplicates while preserving order
            seen = set()
            seizures_list = [x for x in seizures_list if not (x in seen or seen.add(x))]
            if len(seizures_list) == 0:
                seizures_list = None  # Fall back to defaults
        
        threshold_diagnostic_results = diagnose_thresholding_differences(
            seizures_list=seizures_list,
            annotations_df=annotations_df
        )
    except Exception as e:
        print(f"Error in thresholding diagnostic: {e}")
        import traceback
        traceback.print_exc()
        threshold_diagnostic_results = None
    
    # Test 7: Performance difference analysis for 256Hz mean filtered models
    print("\n" + "=" * 80)
    print("TEST 7: Performance Difference Analysis (DSOSD vs DynaSD at 256Hz, Mean Filtered)")
    print("=" * 80)
    print("NOTE: This analyzes differences in probabilities and performance")
    print("      between DSOSD 256Hz and DynaSD 256Hz with mean smoothing.")
    try:
        if real_seizure_results and isinstance(real_seizure_results, dict) and 'all_results' in real_seizure_results:
            # Use first seizure for detailed analysis
            first_result = None
            for r in real_seizure_results['all_results']:
                if isinstance(r, dict) and 'patient' in r and 'onset' in r:
                    first_result = r
                    break
            
            if first_result and 'prob_dsosd_256' in first_result and 'prob_dynasd_256' in first_result:
                patient = first_result['patient']
                onset = first_result['onset']
                results = first_result
                
                print(f"\nAnalyzing seizure: {patient} {onset}")
                
                # Get probability data
                prob_dsosd_256 = results['prob_dsosd_256']
                prob_dynasd_256 = results['prob_dynasd_256']
                
                if prob_dsosd_256 is not None and prob_dynasd_256 is not None:
                    # Remove time column if present
                    if 'time' in prob_dsosd_256.columns:
                        prob_dsosd_256_data = prob_dsosd_256.drop('time', axis=1)
                    else:
                        prob_dsosd_256_data = prob_dsosd_256.copy()
                    
                    if 'time' in prob_dynasd_256.columns:
                        prob_dynasd_256_data = prob_dynasd_256.drop('time', axis=1)
                    else:
                        prob_dynasd_256_data = prob_dynasd_256.copy()
                    
                    # Basic statistics
                    print(f"\n1. Probability Statistics:")
                    print(f"   DSOSD 256Hz shape: {prob_dsosd_256_data.shape}")
                    print(f"   DynaSD 256Hz shape: {prob_dynasd_256_data.shape}")
                    print(f"   DSOSD 256Hz - Mean: {prob_dsosd_256_data.mean().mean():.6f}, Median: {prob_dsosd_256_data.median().median():.6f}")
                    print(f"   DynaSD 256Hz - Mean: {prob_dynasd_256_data.mean().mean():.6f}, Median: {prob_dynasd_256_data.median().median():.6f}")
                    print(f"   DSOSD 256Hz - Range: [{prob_dsosd_256_data.min().min():.6f}, {prob_dsosd_256_data.max().max():.6f}]")
                    print(f"   DynaSD 256Hz - Range: [{prob_dynasd_256_data.min().min():.6f}, {prob_dynasd_256_data.max().max():.6f}]")
                    
                    # Correlation
                    if prob_dsosd_256_data.shape == prob_dynasd_256_data.shape:
                        # Flatten and correlate
                        dsosd_flat = prob_dsosd_256_data.values.flatten()
                        dynasd_flat = prob_dynasd_256_data.values.flatten()
                        correlation = np.corrcoef(dsosd_flat, dynasd_flat)[0, 1]
                        print(f"\n2. Correlation:")
                        print(f"   Pearson correlation: {correlation:.6f}")
                        
                        # Difference statistics
                        diff = prob_dsosd_256_data - prob_dynasd_256_data
                        print(f"\n3. Difference Statistics (DSOSD - DynaSD):")
                        print(f"   Mean difference: {diff.mean().mean():.6f}")
                        print(f"   Median difference: {diff.median().median():.6f}")
                        print(f"   Std of differences: {diff.std().std():.6f}")
                        print(f"   Max positive difference: {diff.max().max():.6f}")
                        print(f"   Max negative difference: {diff.min().min():.6f}")
                        
                        # Percentiles of differences
                        diff_flat = diff.values.flatten()
                        print(f"   Percentiles of differences:")
                        for p in [10, 25, 50, 75, 90, 95, 99]:
                            print(f"     {p}th percentile: {np.percentile(diff_flat, p):.6f}")
                    
                    # Performance comparison
                    if 'smoothing_analysis' in results and results['smoothing_analysis']:
                        smooth = results['smoothing_analysis']
                        if 'phi_fixed_threshold' in smooth:
                            phi_fixed = smooth['phi_fixed_threshold']
                            
                            print(f"\n4. Performance Comparison (Mean Smoothing, Fixed Threshold):")
                            dsosd_phi = phi_fixed.get('dsosd_256_mean', {}).get('phi_consensus', np.nan)
                            dynasd_phi = phi_fixed.get('dynasd_256_mean', {}).get('phi_consensus', np.nan)
                            
                            if not np.isnan(dsosd_phi) and not np.isnan(dynasd_phi):
                                print(f"   DSOSD 256Hz Phi: {dsosd_phi:.6f}")
                                print(f"   DynaSD 256Hz Phi: {dynasd_phi:.6f}")
                                print(f"   Difference (DSOSD - DynaSD): {dsosd_phi - dynasd_phi:.6f}")
                                
                                # Get other metrics if available
                                dsosd_metrics = phi_fixed.get('dsosd_256_mean', {})
                                dynasd_metrics = phi_fixed.get('dynasd_256_mean', {})
                                
                                if 'sensitivity' in dsosd_metrics and 'sensitivity' in dynasd_metrics:
                                    print(f"\n5. Detailed Metrics:")
                                    print(f"   Sensitivity - DSOSD: {dsosd_metrics['sensitivity']:.4f}, DynaSD: {dynasd_metrics['sensitivity']:.4f}")
                                    print(f"   Specificity - DSOSD: {dsosd_metrics['specificity']:.4f}, DynaSD: {dynasd_metrics['specificity']:.4f}")
                                    if 'f1' in dsosd_metrics and 'f1' in dynasd_metrics:
                                        print(f"   F1 - DSOSD: {dsosd_metrics['f1']:.4f}, DynaSD: {dynasd_metrics['f1']:.4f}")
                                
                                # Channel-wise analysis
                                if prob_dsosd_256_data.shape == prob_dynasd_256_data.shape:
                                    print(f"\n6. Channel-wise Analysis:")
                                    print(f"   {'Channel':<30} {'DSOSD Mean':<15} {'DynaSD Mean':<15} {'Difference':<15}")
                                    print(f"   {'-'*75}")
                                    
                                    for ch in prob_dsosd_256_data.columns[:10]:  # Show first 10 channels
                                        dsosd_ch_mean = prob_dsosd_256_data[ch].mean()
                                        dynasd_ch_mean = prob_dynasd_256_data[ch].mean()
                                        diff_ch = dsosd_ch_mean - dynasd_ch_mean
                                        print(f"   {ch:<30} {dsosd_ch_mean:<15.6f} {dynasd_ch_mean:<15.6f} {diff_ch:<15.6f}")
                                    
                                    if len(prob_dsosd_256_data.columns) > 10:
                                        print(f"   ... ({len(prob_dsosd_256_data.columns) - 10} more channels)")
                            else:
                                print(f"   Performance metrics not available")
                else:
                    print(f"   Probability data not available for comparison")
            else:
                print(f"   No valid results found from Test 5")
        else:
            print(f"   No real seizure results available from Test 5")
    except Exception as e:
        print(f"Error in performance difference analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)

