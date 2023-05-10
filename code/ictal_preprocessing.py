# %%
# pylint: disable=C0103
"""
    This script will preprocess the ictal clips from the common data folder and save them in the local data folder.   
"""
# imports
import os
from os.path import join as ospj
from fractions import Fraction

import numpy as np
import pandas as pd

from utils import (
    check_channel_types,
    bipolar_montage,
    notch_filter,
    bandpass_filter,
    clean_labels,
)

from scipy.signal import resample_poly, welch
from tqdm import tqdm
import mne
from config import CONFIG


def filename2starttime(filename):
    """
    This function will convert the filename to the start time of the clip
    """
    return int(filename.split("_")[2].split("-")[1][5:])


PLOT = False
OVERWRITE = True

# %%
# table
participants_tab = pd.read_csv(ospj(CONFIG.bids_dir, "participants.tsv"), sep="\t")
# %%
for ind, row in tqdm(participants_tab.iterrows(), total=participants_tab.shape[0]):
    subrid = row["participant_id"]
    # if subrid != "sub-RID0188":
        # continue

    scans_list = pd.read_csv(
        ospj(
            CONFIG.bids_dir,
            subrid,
            "ses-clinical01",
            f"{subrid}_ses-clinical01_scans.tsv",
        ),
        sep="\t",
    )
    scans_list = scans_list[scans_list.filename.str.contains("task-ictal")]

    for scan in scans_list["filename"]:
        # if "ictal" not in scan:
        #     continue
        # print(scan)
        scan_fname = scan.split("/")[-1]
        start_time = filename2starttime(scan_fname)

        fname = f"{subrid}_ses-clinical01_task-ictal{start_time}_ieeg_preprocessed.edf"
        fdir = ospj(CONFIG.data_dir, "patients", subrid)

        # check if the file already exists, if so, skip
        if os.path.exists(ospj(fdir, fname)) and not OVERWRITE:
            continue

        data = mne.io.read_raw_edf(
            ospj(CONFIG.bids_dir, subrid, "ses-clinical01", scan), verbose=False
        )

        # get the end time
        end_time = start_time + data.n_times / data.info["sfreq"]
        duration = end_time - start_time

        signals = data.get_data()  # (n_channels, n_samples)
        fs = data.info["sfreq"]
        ch_names = data.ch_names
        ch_names = clean_labels(ch_names, pt=None)
        ch_types = check_channel_types(ch_names)

        # notch filter at 60 Hz
        signals = notch_filter(signals.T, fs).T  # (n_channels, n_samples)

        # low pass filter with 10th order butterworth filter
        signals = bandpass_filter(
            signals, fs, order=10, lo=0.5, hi=120
        )  # (n_channels, n_samples)

        # apply bipolar montage
        signals, bipolar_ch_names = bipolar_montage(
            signals, ch_types
        )  # (n_channels, n_samples)
        signals = pd.DataFrame(
            signals, index=bipolar_ch_names.name
        ).T  # (n_samples, n_channels)

        # downsample to 200 Hz
        new_fs = 200
        frac = Fraction(new_fs, int(fs))
        signals = resample_poly(
            signals, up=frac.numerator, down=frac.denominator
        )  # (n_samples, n_channels)
        fs = new_fs
        signals = pd.DataFrame(
            signals, columns=bipolar_ch_names.name
        )  # (n_samples, n_channels)
        t = np.linspace(start_time, end_time, signals.shape[0])
        signals.index = t

        # get psd
        f, psd = welch(signals, fs, nperseg=fs, axis=0)

        # format signals in mne format
        processed_data = mne.io.RawArray(
            signals.T,
            mne.create_info(
                list(bipolar_ch_names.name.values),
                fs,
                ch_types=bipolar_ch_names.type.values,
            ),
            verbose=False,
        )

        # save in BIDS format in data/patients
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        mne.export.export_raw(
            ospj(CONFIG.data_dir, "patients", subrid, fname),
            raw=processed_data,
            fmt="EDF",
            overwrite=OVERWRITE,
            verbose=False,
        )

# %%
pd.set_option("display.max_rows", None, "display.max_columns", None)
# %%
from pennsieve import Pennsieve


# %%
