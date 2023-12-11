# standard imports
from os.path import join as ospj
import pickle
from numbers import Number
import time
from typing import Union
from glob import glob
import logging

# nonstandard imports
from ieeg.auth import Session
import pandas as pd
import numpy as np

# from scipy.integrate import simpson
import matplotlib.pyplot as plt
import seaborn as sns

## END IMPORTS
def create_pwd_file(username, password, fname=None):
  'Creates password in your current directory. You can also direct to absolute path in fname'
  if fname is None:
    fname = "{}_ieeglogin.bin".format(username[:3])
  with open(fname, 'wb') as f:
    f.write(password.encode())
  print("-- -- IEEG password file saved -- --\n")

## PULLING FROM IEEG
def _pull_iEEG(ds, start_usec, duration_usec, channel_ids):
    """
    Pull data while handling iEEGConnectionError
    """
    i = 0
    while True:
        if i == 50:
            logger = logging.getLogger()
            logger.error(
                f"failed to pull data for {ds.name}, {start_usec / 1e6}, {duration_usec / 1e6}, {len(channel_ids)} channels"
            )
            return None
        try:
            data = ds.get_data(start_usec, duration_usec, channel_ids)
            return data
        except Exception as _:
            time.sleep(1)
            i += 1


def get_iEEG_data(
    username: str,
    password_bin_file: str,
    iEEG_filename: str,
    start_time_usec: float,
    stop_time_usec: float,
    select_electrodes=None,
    ignore_electrodes=None,
    outputfile=None,
    force_pull = False
):
    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec

    with open(password_bin_file, "r") as f:
        pwd = f.read()

    iter = 0
    while True:
        try:
            if iter == 50:
                raise ValueError("Failed to open dataset")
            s = Session(username, pwd)
            ds = s.open_dataset(iEEG_filename)
            all_channel_labels = ds.get_channel_labels()
            break
            
        except Exception as e:
            time.sleep(1)
            iter += 1
    all_channel_labels = clean_labels(all_channel_labels, iEEG_filename)
    
    if select_electrodes is not None:
        if isinstance(select_electrodes[0], Number):
            channel_ids = select_electrodes
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(select_electrodes[0], str):
            select_electrodes = clean_labels(select_electrodes, iEEG_filename)
            if any([i not in all_channel_labels for i in select_electrodes]):
                if force_pull:
                    select_electrodes = [e for e in select_electrodes
                                          if e in all_channel_labels]
                else:
                    raise ValueError("Channel not in iEEG")

            channel_ids = [
                i for i, e in enumerate(all_channel_labels) if e in select_electrodes
            ]
            channel_names = select_electrodes
        else:
            print("Electrodes not given as a list of ints or strings")

    elif ignore_electrodes is not None:
        if isinstance(ignore_electrodes[0], int):
            channel_ids = [
                i
                for i in np.arange(len(all_channel_labels))
                if i not in ignore_electrodes
            ]
            channel_names = [all_channel_labels[e] for e in channel_ids]
        elif isinstance(ignore_electrodes[0], str):
            ignore_electrodes = clean_labels(ignore_electrodes, iEEG_filename)
            channel_ids = [
                i
                for i, e in enumerate(all_channel_labels)
                if e not in ignore_electrodes
            ]
            channel_names = [
                e for e in all_channel_labels if e not in ignore_electrodes
            ]
        else:
            print("Electrodes not given as a list of ints or strings")

    else:
        channel_ids = np.arange(len(all_channel_labels))
        channel_names = all_channel_labels

    # if clip is small enough, pull all at once, otherwise pull in chunks
    if (duration < 120 * 1e6) and (len(channel_ids) < 100):
        data = _pull_iEEG(ds, start_time_usec, duration, channel_ids)
    elif (duration > 120 * 1e6) and (len(channel_ids) < 100):
        # clip is probably too big, pull chunks and concatenate
        clip_size = 60 * 1e6

        clip_start = start_time_usec
        data = None
        while clip_start + clip_size < stop_time_usec:
            if data is None:
                data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
            else:
                new_data = _pull_iEEG(ds, clip_start, clip_size, channel_ids)
                data = np.concatenate((data, new_data), axis=0)
            clip_start = clip_start + clip_size

        last_clip_size = stop_time_usec - clip_start
        new_data = _pull_iEEG(ds, clip_start, last_clip_size, channel_ids)
        data = np.concatenate((data, new_data), axis=0)
    else:
        # there are too many channels, pull chunks and concatenate
        channel_size = 20
        channel_start = 0
        data = None
        while channel_start + channel_size < len(channel_ids):
            if data is None:
                data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
            else:
                new_data = _pull_iEEG(
                    ds,
                    start_time_usec,
                    duration,
                    channel_ids[channel_start : channel_start + channel_size],
                )
                data = np.concatenate((data, new_data), axis=1)
            channel_start = channel_start + channel_size

        last_channel_size = len(channel_ids) - channel_start
        new_data = _pull_iEEG(
            ds,
            start_time_usec,
            duration,
            channel_ids[channel_start : channel_start + last_channel_size],
        )
        data = np.concatenate((data, new_data), axis=1)

    df = pd.DataFrame(data, columns=channel_names)
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate  # get sample rate

    if outputfile:
        with open(outputfile, "wb") as f:
            pickle.dump([df, fs], f)
    else:
        return df, fs


def plot_iEEG_data(
    data: Union[pd.DataFrame, np.ndarray], t: np.ndarray, colors=None, dr=None, plot_color = 'k'
):
    """_summary_

    Args:
        data (Union[pd.DataFrame, np.ndarray]): _description_
        t (np.ndarray): _description_
        colors (_type_, optional): _description_. Defaults to None.
        dr (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if data.shape[0] != np.size(t):
        data = data.T
    n_rows = data.shape[1]
    duration = t[-1] - t[0]

    fig, ax = plt.subplots(figsize=(duration / 3, n_rows / 5))
    sns.despine()

    ticklocs = []
    ax.set_xlim(t[0], t[-1])
    dmin = data.min().min()
    dmax = data.max().min()

    if dr is None:
        dr = (dmax - dmin) * 0.8  # Crowd them a bit.

    y0 = dmin
    y1 = (n_rows - 1) * dr + dmax
    ax.set_ylim(y0, y1)

    segs = []
    for i in range(n_rows):
        if isinstance(data, pd.DataFrame):
            segs.append(np.column_stack((t, data.iloc[:, i])))
        elif isinstance(data, np.ndarray):
            segs.append(np.column_stack((t, data[:, i])))
        else:
            print("Data is not in valid format")

    for i in reversed(range(n_rows)):
        ticklocs.append(i * dr)

    offsets = np.zeros((n_rows, 2), dtype=float)
    offsets[:, 1] = ticklocs

    # # Set the yticks to use axes coordinates on the y axis
    ax.set_yticks(ticklocs)
    if isinstance(data, pd.DataFrame):
        ax.set_yticklabels(data.columns)

    if colors:
        for col, lab in zip(colors, ax.get_yticklabels()):
            lab.set_color(col)

    ax.set_xlabel("Time (s)")
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    ax.plot(t, data + ticklocs, color=plot_color, lw=0.4)

    return fig, ax

def artifact_removal(
    data: np.ndarray, fs: float, discon=1 / 12, noise=15000, win_size=1
) -> np.ndarray:
    """_summary_

    Args:
        data np.ndarray
        fs (float): sampling frequency
        discon (_type_, optional): Checking for nan values in signal. Considered disconnected if nan
        count above threshold. Defaults to 1/12.
        noise (int, optional): Threshold for noisy signal, value compared to absolute power. Defaults to 15000.
        win_size (int, optional): Rolling window size (no overlap). Defaults to 1s.

    Returns:
        np.ndarray: _description_
    """
    win_size = int(win_size * fs)
    
    n_wins = np.ceil(data.shape[0]/win_size)
    max_inds = n_wins*win_size
     
    all_inds = np.arange(max_inds)
    all_inds[data.shape[0]:] = np.nan
    ind_overlap = np.reshape(all_inds, (-1, int(win_size)))
    
    artifacts = np.empty_like(data)

    # mask indices with nan values
    artifacts = np.isnan(data)

    for win_inds in ind_overlap:
        win_inds = win_inds[~np.isnan(win_inds)].astype(int)
        is_disconnected = np.sum(np.abs(data[win_inds,:]), axis=0) < discon

        is_noise = (
            np.sqrt(np.sum(np.power(np.diff(data[win_inds,:], axis=0), 2), axis=0))
            > noise
        )

        artifacts[win_inds, :] = np.logical_or(
            artifacts[win_inds, :].any(axis=0), np.logical_or(is_disconnected, is_noise)
        )

    return artifacts

def detect_bad_channels(data,fs,lf_stim = False):
    '''
    data: raw EEG traces after filtering (i think)
    fs: sampling frequency
    channel_labels: string labels of channels to use
    '''
    values = data.copy()
    which_chs = np.arange(values.shape[1])
    ## Parameters to reject super high variance
    tile = 99
    mult = 10
    num_above = 1
    abs_thresh = 5e3

    ## Parameter to reject high 60 Hz
    percent_60_hz = 0.7

    ## Parameter to reject electrodes with much higher std than most electrodes
    mult_std = 10

    bad = []
    high_ch = []
    nan_ch = []
    zero_ch = []
    flat_ch = []
    high_var_ch = []
    noisy_ch = []
    all_std = np.empty((len(which_chs),1))
    all_std[:] = np.nan
    details = {}

    for i in range(len(which_chs)):       
        ich = which_chs[i]
        eeg = values[:,ich]
        bl = np.nanmedian(eeg)
        all_std[i] = np.nanstd(eeg)
        
        ## Remove channels with nans in more than half
        if sum(np.isnan(eeg)) > 0.5*len(eeg):
            bad.append(ich)
            nan_ch.append(ich)
            continue
        
        ## Remove channels with zeros in more than half
        if sum(eeg == 0) > (0.5 * len(eeg)):
            bad.append(ich)
            zero_ch.append(ich)
            continue

        ## Remove channels with extended flat-lining
        if (sum(np.diff(eeg,1) == 0) > (0.02 * len(eeg))) and (sum(abs(eeg - bl) > abs_thresh) > (0.02 * len(eeg))):
            bad.append(ich)
            flat_ch.append(ich)
        
        ## Remove channels with too many above absolute thresh
        if sum(abs(eeg - bl) > abs_thresh) > 10:
            if not lf_stim:
                bad.append(ich)
            high_ch.append(ich)
            continue

        ## Remove channels if there are rare cases of super high variance above baseline (disconnection, moving, popping)
        pct = np.percentile(eeg,[100-tile,tile])
        thresh = [bl - mult*(bl-pct[0]), bl + mult*(pct[1]-bl)]
        sum_outside = sum(((eeg > thresh[1]) + (eeg < thresh[0])) > 0)
        if sum_outside >= num_above:
            if not lf_stim:
                bad.append(ich)
            high_var_ch.append(ich)
            continue
        
        ## Remove channels with a lot of 60 Hz noise, suggesting poor impedance
        # Calculate fft
        Y = np.fft.fft(eeg-np.nanmean(eeg))
        
        # Get power
        P = abs(Y)**2
        freqs = np.linspace(0,fs,len(P)+1)
        freqs = freqs[:-1]
        
        # Take first half
        P = P[:np.ceil(len(P)/2).astype(int)]
        freqs = freqs[:np.ceil(len(freqs)/2).astype(int)]
        
        P_60Hz = sum(P[(freqs > 58) * (freqs < 62)])/sum(P)
        if P_60Hz > percent_60_hz:
            bad.append(ich)
            noisy_ch.append(ich)
            continue

    ## Remove channels for whom the std is much larger than the baseline
    median_std = np.nanmedian(all_std)
    higher_std = which_chs[(all_std > (mult_std * median_std)).squeeze()]
    bad_std = higher_std
    # for ch in bad_std:
    #     if ch not in bad:
    #         if ~lf_stim:
    #             bad.append(ch)
    channel_mask = np.ones((values.shape[1],),dtype=bool)
    channel_mask[bad] = False
    details['noisy'] = noisy_ch
    details['nans'] = nan_ch
    details['zeros'] = zero_ch
    details['flat'] = flat_ch
    details['var'] = high_var_ch
    details['higher_std'] = bad_std
    details['high_voltage'] = high_ch
    
    return channel_mask,details


def _num_wins(xLen, fs, winLen, winDisp):
  return int(((xLen/fs - winLen + winDisp) - ((xLen/fs - winLen + winDisp)%winDisp))/winDisp)