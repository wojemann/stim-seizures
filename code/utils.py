"""
This script has utility functions for the iEEG data analysis.
It has the following functions:
- get_iEEG_data
- clean_labels
- check_channel_types
- get_channel_types
- get_channel_labels
- get_channel_coords

Raises:
    ValueError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
"""
# %%
# pylint: disable-msg=C0103
# pylint: disable-msg=W0703

# standard imports
import os
from os.path import join as ospj
import pickle
from numbers import Number
import time
import re
from typing import Union
import itertools
from glob import glob
import logging
import warnings
import random
import json

# data IO imports
import mne
import mne_bids
from mne_bids import BIDSPath, read_raw_bids

# nonstandard imports
from ieeg.auth import Session
import pandas as pd
import numpy as np
import torch

from scipy.signal import iirnotch, sosfiltfilt, butter, welch, coherence, filtfilt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.integrate import simpson
import scipy.signal as sig
import scipy as sc
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF
from sklearn.utils import resample

# from scipy.integrate import simpson
import matplotlib.pyplot as plt
import seaborn as sns
from fooof import FOOOFGroup
import nibabel as nii

warnings.filterwarnings("ignore")

########################################## Data I/O ##########################################
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


def clean_labels(channel_li: list, pt: str) -> list:
    """This function cleans a list of channels and returns the new channels

    Args:
        channel_li (list): _description_

    Returns:
        list: _description_
    """

    new_channels = []
    for i in channel_li:
        i = i.replace("-", "")
        i = i.replace("GRID", "G")  # mne has limits on channel name size
        # standardizes channel names
        pattern = re.compile(r"([A-Za-z0-9]+?)(\d+)$")
        regex_match = pattern.match(i)

        if regex_match is None:
            new_channels.append(i)
            continue

        # if re.search('Cz|Fz|C3|C4|EKG',i):
        #     continue
        lead = regex_match.group(1).replace("EEG", "").strip()
        contact = int(regex_match.group(2))

        if pt in ("HUP75_phaseII", "HUP075", "sub-RID0065"):
            if lead == "Grid":
                lead = "G"

        if pt in ("HUP78_phaseII", "HUP078", "sub-RID0068"):
            if lead == "Grid":
                lead = "LG"

        if pt in ("HUP86_phaseII", "HUP086", "sub-RID0018"):
            conv_dict = {
                "AST": "LAST",
                "DA": "LA",
                "DH": "LH",
                "Grid": "LG",
                "IPI": "LIPI",
                "MPI": "LMPI",
                "MST": "LMST",
                "OI": "LOI",
                "PF": "LPF",
                "PST": "LPST",
                "SPI": "RSPI",
            }
            if lead in conv_dict:
                lead = conv_dict[lead]
        
        if pt in ("HUP93_phaseII", "HUP093", "sub-RID0050"):
            if lead.startswith("G"):
                lead = "G"
    
        if pt in ("HUP89_phaseII", "HUP089", "sub-RID0024"):
            if lead in ("GRID", "G"):
                lead = "RG"
            if lead == "AST":
                lead = "AS"
            if lead == "MST":
                lead = "MS"

        if pt in ("HUP99_phaseII", "HUP099", "sub-RID0032"):
            if lead == "G":
                lead = "RG"

        if pt in ("HUP112_phaseII", "HUP112", "sub-RID0042"):
            if "-" in i:
                new_channels.append(f"{lead}{contact:02d}-{i.strip().split('-')[-1]}")
                continue
        if pt in ("HUP116_phaseII", "HUP116", "sub-RID0175"):
            new_channels.append(f"{lead}{contact:02d}".replace("-", ""))
            continue

        if pt in ("HUP123_phaseII_D02", "HUP123", "sub-RID0193"):
            if lead == "RS": 
                lead = "RSO"
            if lead == "GTP":
                lead = "RG"
        
        new_channels.append(f"{lead}{contact:02d}")

        if pt in ("HUP189", "HUP189_phaseII", "sub-RID0520"):
            conv_dict = {"LG": "LGr"}
            if lead in conv_dict:
                lead = conv_dict[lead]
                
    return new_channels


def get_apn_dkt(
    fname="/mnt/leif/littlab/users/pattnaik/ictal_patterns/data/metadata/apn_dkt_labels.txt",
) -> dict:
    """Function to get antsPyNet DKT labels from text file

    Args:
        fname (str): AntsPyNet DKT labels file name

    Raises:
        ValueError: _description_

    Returns:
        dict: _description_
    """

    with open(fname, "r") as f:
        lines = f.readlines()

    dkt = {}
    for line in lines:
        if line.startswith("Label"):
            words = line.strip().split()
            reg_id = int(words[1][:-1])
            reg_name = " ".join(words[2:])
            dkt[reg_id] = reg_name

    return dkt


def check_channel_types(ch_list, threshold=24):
    """Function to check channel types

    Args:
        ch_list (_type_): _description_
        threshold (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
    """
    ch_df = []
    for i in ch_list:
        regex_match = re.match(r"([A-Za-z0-9]+)(\d{2})$", i)
        if regex_match is None:
            ch_df.append({"name": i, "lead": i, "contact": 0, "type": "misc"})
            continue
        lead = regex_match.group(1)
        contact = int(regex_match.group(2))
        ch_df.append({"name": i, "lead": lead, "contact": contact})
    ch_df = pd.DataFrame(ch_df)
    for lead, group in ch_df.groupby("lead"):
        if lead in ["ECG", "EKG"]:
            ch_df.loc[group.index, "type"] = "ecg"
            continue
        if lead in [
            "C",
            "Cz",
            "CZ",
            "F",
            "Fp",
            "FP",
            "Fz",
            "FZ",
            "O",
            "P",
            "Pz",
            "PZ",
            "T",
        ]:
            ch_df.loc[group.index.to_list(), "type"] = "eeg"
            continue
        if len(group) > threshold:
            ch_df.loc[group.index.to_list(), "type"] = "ecog"
        else:
            ch_df.loc[group.index.to_list(), "type"] = "seeg"
    return ch_df


def unnesting(df, explode, axis):
    """
    code that expands lists in a column in a dataframe.
    """
    if axis == 1:
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat(
            [pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1
        )
        df1.index = idx

        return df1.join(df.drop(explode, 1), how="right")
    else:
        df1 = pd.concat(
            [
                pd.DataFrame(df[x].tolist(), index=df.index).add_prefix(x)
                for x in explode
            ],
            axis=1,
        )
        return df1.join(df.drop(explode, 1), how="right")


def load_rid_forjson(rid):
    """
    load_rid_forjson loads the DKTantspynet output from IEEG_recon
    """
    dkt_directory = glob(
        f"/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS/{rid}/derivatives/ieeg_recon/module3/{rid}_ses-*_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv"
    )[0]
    brain_df = pd.read_csv(dkt_directory, index_col=0)
    brain_df["name"] = brain_df["name"].astype(str) + "-CAR"
    return brain_df


def label_fix(rid, threshold=0.25, return_old=False, df=None):
    """
    label_fix reassigns labels overlapping brain regions to "empty labels" in our DKTantspynet output from IEEG_recon
    input:  rid - name of patient. example: 'sub-RID0031'
            data_directory - directory containing CNT_iEEG_BIGS folder. (must end in '/')
            threshold - arbitrary threshold that r=2mm surround of electrode must overlap with a brain region. default: threshold = 25%, Brain region has a 25% or more overlap.
    output: relabeled_df - a dataframe that contains 2 extra columns showing the second most overlapping brain region and the percent of overlap.
    """
    if df is not None:
        brain_df = df
    else:
        brain_df = load_rid_forjson(rid)
    json_labels = glob(
        f"/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS/{rid}/derivatives/ieeg_recon/module3/{rid}_ses-*_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json"
    )[0]
    workinglabels = pd.read_json(json_labels, lines=True)

    empty = workinglabels[workinglabels["label"] == "EmptyLabel"]
    empty = unnesting(empty, ["labels_sorted", "percent_assigned"], axis=0)
    empty = empty[np.isnan(empty["percent_assigned1"]) == False]
    changed = empty[empty["percent_assigned1"] >= threshold]

    brain_df["name"] = brain_df["name"].str.replace("-CAR", "")
    relabeled_df = brain_df.merge(
        changed[["labels_sorted1", "percent_assigned1"]],
        left_on=brain_df["name"],
        right_on=changed["name"],
        how="left",
        indicator=True,
    )
    relabeled_df["final_label"] = relabeled_df["labels_sorted1"].fillna(
        relabeled_df["label"]
    )
    # relabeled_df['name'] = relabeled_df['name'].astype(str) + '-CAR' #added for this version for our analysis

    if return_old:
        return relabeled_df, brain_df
    return relabeled_df

def optimize_localizations(path_to_recon,RID):
    # /mnt/leif/littlab/data/Human_Data/recon/BIDS_penn/
    # python /mnt/leif/littlab/data/Human_Data/recon/code/run_penn_recons.py
    try:
        atropos_probs = pd.read_json(path_to_recon + f'sub-RID{RID}_ses-clinical01_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json',lines=True)
        dkt_probs = pd.read_json(path_to_recon + f'sub-RID{RID}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json',lines=True)
    except:
        dkt_probs = pd.read_json(path_to_recon + f'sub-RID{RID}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json')
        atropos_probs = pd.read_json(path_to_recon + f'sub-RID{RID}_ses-clinical01_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json')

    def _apply_matter_function(x):
        # look in labels sorted and see if it contains gray matter
        # if gray matter is greater than 5% then set label to gray matter
        x = pd.DataFrame(x).transpose()
        for i,label in enumerate(x['labels_sorted'].to_numpy()[0]):
            if (label == 'gray matter') and (x['percent_assigned'].to_numpy()[0][i] > 0.05):
                x['label'] = label
                x['index'] = 2
                break
            elif (label == 'white matter') and (x['percent_assigned'].to_numpy()[0][i] > 0.05):
                x['label'] = label
                x['index'] = 3
                break
        
        return x
    def _apply_region_function(x):
        # look in labels sorted and see if it contains gray matter
        # if gray matter is greater than 5% then set label to gray matter
        x = pd.DataFrame(x).transpose()
        for i,label in enumerate(x['labels_sorted'].to_numpy()[0]):
            if (label != 'EmptyLabel') and (x['percent_assigned'].to_numpy()[0][i] > 0.05):
                x['label'] = label
                break
        return x
        
    modified_atropos = atropos_probs.iloc[:,:].apply(lambda x: _apply_matter_function(x), axis = 1)
    modified_atropos_df = pd.DataFrame(np.squeeze(np.array(modified_atropos.to_list())),columns=atropos_probs.columns)

    modified_dkt = dkt_probs.iloc[:,:].apply(lambda x: _apply_region_function(x),axis = 1)
    modified_dkt_df = pd.DataFrame(np.squeeze(np.array(modified_dkt.to_list())),columns=dkt_probs.columns)
    modified_dkt_df[modified_atropos_df.label == 'white matter']['label'] = 'white matter'

    return modified_atropos_df,modified_dkt_df

def choptimize_localizations(recon_path,chopid):
    electrode_locals = pd.read_excel(recon_path,f'{chopid}_locations')
    electrode_locals.loc[:,'name'] = clean_labels(electrode_locals.full_label,chopid)
    electrode_locals.loc[:,'index'] = pd.NA
    electrode_locals.loc[electrode_locals.brain_area == 'Unknown','brain_area'] = 'EmptyLabel'
    electrode_regions = electrode_locals.copy()
    electrode_regions["label"] = electrode_locals["brain_area"]
    col_list = ["name","x","y","z","label","isgrey"]
    electrode_regions = electrode_regions[col_list]
    mapping = {"grey": "gray matter","white": "white matter","Unknown": "EmptyLabel"}
    electrode_locals["label"] = electrode_locals["matter"].replace(mapping)
    col_list = ["name","x","y","z","label","isgrey"]
    electrode_locals = electrode_locals[col_list]
    return electrode_locals,electrode_regions

def get_rpath(prodatapath,pt):
    if pt[:3] == 'CHO':
        region_path = ospj(prodatapath,pt,'electrode_localizations_CHOPR.pkl')
    else:
        region_path = ospj(prodatapath,pt,'electrode_localizations_dkt.pkl')
    return region_path

def surgical_parcellation(electrode_regions):
    for i,row in electrode_regions.iterrows():
        label = row.label
        if isinstance(label,float):
            label = "EmptyLabel"
        label = label.lower()
        if ("emptylabel" in label) or ("white" in label):
            surgical_label = "EmptyLabel"
        elif ("amygdala" in label) or ("hippocampus" in label):
            if "left" in label:
                surgical_label = 'left mesial temporal'
            else:
                surgical_label = 'right mesial temporal'
        elif ("temporal" in label) or ("fusiform" in label) or ("entorhinal" in label) or ("parahippocampal" in label):
            if "left" in label:
                surgical_label = 'left temporal neocortex'
            else:
                surgical_label = 'right temporal neocortex'
        else:
            if "left" in label:
                surgical_label = 'left other neocortex'
            else:
                surgical_label = 'right other neocortex'
        electrode_regions.loc[i,"label"] = surgical_label
    return electrode_regions
def surgical_parcelate(region_list):
    surgical_labels = []
    for label in region_list:
        if isinstance(label,float):
            label = "EmptyLabel"
        label = label.lower()
        if ("emptylabel" in label) or ("white" in label):
            surgical_label = "EmptyLabel"
        elif ("amygdala" in label) or ("hippocampus" in label):
            if "left" in label:
                surgical_label = 'left mesial temporal'
            else:
                surgical_label = 'right mesial temporal'
        elif ("temporal" in label) or ("fusiform" in label) or ("entorhinal" in label) or ("parahippocampal" in label):
            if "left" in label:
                surgical_label = 'left temporal neocortex'
            else:
                surgical_label = 'right temporal neocortex'
        else:
            if "left" in label:
                surgical_label = 'left other neocortex'
            else:
                surgical_label = 'right other neocortex'
        surgical_labels.append(surgical_label)
    return surgical_labels
######################## BIDS ########################
BIDS_DIR = "/mnt/leif/littlab/data/Human_Data/CNT_iEEG_BIDS"
BIDS_INVENTORY = "/mnt/leif/littlab/users/pattnaik/ieeg_recon/migrate/cnt_ieeg_bids.csv"


def get_cnt_inventory(bids_inventory=BIDS_INVENTORY):
    inventory = pd.read_csv(bids_inventory, index_col=0)
    inventory = inventory == "yes"
    return inventory


def get_pt_coords(pt):
    coords_path = glob(
        ospj(BIDS_DIR, pt, "derivatives", "ieeg_recon", "module3", "*DKTantspynet*csv")
    )[0]
    return pd.read_csv(coords_path, index_col=0)

def get_data_from_bids(root,subject,task_key,run=None,return_path=False,verbose=0):
    # setting subject
    all_subjects = mne_bids.get_entity_vals(root,'subject')
    ignore_subjects = [s for s in all_subjects if s != subject]

    # extracting task
    task_list = mne_bids.get_entity_vals(root, 'task', ignore_subjects=ignore_subjects)
    task = [t for t in task_list if task_key in t][0]
    ignore_tasks = [t for t in task_list if t != task]

    # Should be just one run per task
    if run is None:
        run = mne_bids.get_entity_vals(root, 'run', 
                                    ignore_tasks = ignore_tasks,
                                    ignore_subjects=ignore_subjects)[0]
    
    bidspath = BIDSPath(root=root,subject=subject,task=task,run=run,session='clinical01')
    data_raw = read_raw_bids(bidspath,verbose=verbose)
    data_df = data_raw.to_data_frame()
    fs = 1/data_df.time.diff().mode().item()
    if return_path:
        return data_df.drop('time',axis=1), int(fs), root, subject, task, run
    return data_df.drop('time',axis=1), int(fs)

################################################ Plotting and Visualization ################################################
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


def make_surf_transforms(sub_rid=None, overwrite=False, path=None):
    names = ["lh+rh_transform.pial", "rh_transform.pial", "lh_transform.pial"]
    assert (
        sub_rid is not None or path is not None
    ), "Either sub_rid or path must be specified"
    if sub_rid:
        fs_folder = ospj(BIDS_DIR, sub_rid, "derivatives/freesurfer")

        if (not overwrite) and os.path.exists(ospj(fs_folder, "surf", "lh+rh.pial")):
            return
    elif path:
        fs_folder = path

        if (not overwrite) and os.path.exists(ospj(fs_folder, "surf", "lh+rh.pial")):
            return
    
    if not overwrite:
        return [ospj(fs_folder, "surf", name) for name in names]
    
    (coords_R, simplices_R) = nii.freesurfer.io.read_geometry(
        ospj(fs_folder, "surf", "rh.pial")
    )
    (coords_L, simplices_L) = nii.freesurfer.io.read_geometry(
        ospj(fs_folder, "surf", "lh.pial")
    )

    simplices_bi = np.concatenate([simplices_R, simplices_L + len(coords_R)])
    coords_bi = np.concatenate([coords_R, coords_L])

    T1_file = nii.load(ospj(fs_folder, "mri", "T1.mgz"))
    vox_2_ras = T1_file.affine
    tkras = T1_file.header.get_vox2ras_tkr()

    return_paths = []
    for c, s, name in zip(
        [coords_bi, coords_R, coords_L],
        [simplices_bi, simplices_R, simplices_L],
        names,
    ):
        c_T = (
            vox_2_ras @ (np.linalg.inv(tkras) @ np.vstack([c.T, np.ones(c.shape[0])]))
        )[0:3, :].T
        nii.freesurfer.io.write_geometry(ospj(fs_folder, "surf", name), c_T, s)
        return_paths.append(ospj(fs_folder, "surf", name))

    return return_paths

def cohens_d(group1, group2):
    # Calculating means of the two groups
    mean1, mean2 = np.mean(group1), np.mean(group2)
     
    # Calculating pooled standard deviation
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
     
    # Calculating Cohen's d
    d = (mean1 - mean2) / pooled_std
     
    return d
################################################ Preprocessing ################################################
def notch_filter(data: np.ndarray, fs: float) -> np.array:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_

    Returns:
        np.array: _description_
    """
    # remove 60Hz noise
    # b, a = iirnotch(60, 15, fs)
    # d, c = iirnotch(120, 15, fs)
    b, a = butter(4,(58,62),'bandstop',fs=fs)
    d, c = butter(4,(118,122),'bandstop',fs=fs)

    data_filt = filtfilt(b, a, data, axis=-1)
    data_filt_filt = filtfilt(d, c, data_filt, axis = -1)
    # TODO: add option for causal filter
    # TODO: add optional argument for order

    return data_filt_filt


def bandpass_filter(data: np.ndarray, fs: float, order=3, lo=1, hi=150) -> np.array:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_
        order (int, optional): _description_. Defaults to 3.
        lo (int, optional): _description_. Defaults to 1.
        hi (int, optional): _description_. Defaults to 120.

    Returns:
        np.array: _description_
    """
    # TODO: add causal function argument
    # TODO: add optional argument for order
    sos = butter(order, [lo, hi], output="sos", fs=fs, btype="bandpass")
    data_filt = sosfiltfilt(sos, data, axis=-1)
    return data_filt


def artifact_removal(
    data: np.ndarray, fs: float, discon=1 / 12, noise=15000, win_size=1
) -> np.ndarray:
    """_summary_

    Args:
        data pandas
        fs (float): _description_
        discon (_type_, optional): _description_. Defaults to 1/12.
        noise (int, optional): _description_. Defaults to 15000.
        win_size (int, optional): _description_. Defaults to 1.

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


def num_wins(xLen, fs, winLen, winDisp):
  return int(((xLen/fs - winLen + winDisp) - ((xLen/fs - winLen + winDisp)%winDisp))/winDisp)


def bipolar_montage(data: np.ndarray, ch_types: pd.DataFrame) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray): _description_
        ch_types (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    n_ch = len(ch_types)
    new_ch_types = []
    for ind, row in ch_types.iterrows():
        # do only if type is ecog or seeg
        if row["type"] not in ["ecog", "seeg"]:
            continue

        ch1 = row["name"]

        ch2 = ch_types.loc[
            (ch_types["lead"] == row["lead"])
            & (ch_types["contact"] == row["contact"] + 1),
            "name",
        ]
        if len(ch2) > 0:
            ch2 = ch2.iloc[0]
            entry = {
                "name": ch1 + "-" + ch2,
                "type": row["type"],
                "idx1": ind,
                "idx2": ch_types.loc[ch_types["name"] == ch2].index[0],
            }
            new_ch_types.append(entry)

    new_ch_types = pd.DataFrame(new_ch_types)
    # apply montage to data
    new_data = np.empty((len(new_ch_types), data.shape[1]))
    for ind, row in new_ch_types.iterrows():
        new_data[ind, :] = data[row["idx1"], :] - data[row["idx2"], :]

    return new_data, new_ch_types

def ar_one(data):
    """
    The ar_one function fits an AR(1) model to the data and retains the residual as
    the pre-whitened data
    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates
    Returns
    -------
        data_white: ndarray, shape (T, N)
            Whitened signal with reduced autocorrelative structure
    """
    # Retrieve data attributes
    n_samp, n_chan = data.shape
    # Apply AR(1)
    data_white = np.zeros((n_samp-1, n_chan))
    for i in range(n_chan):
        win_x = np.vstack((data[:-1, i], np.ones(n_samp-1)))
        w = np.linalg.lstsq(win_x.T, data[1:, i], rcond=None)[0]
        data_white[:, i] = data[1:, i] - (data[:-1, i]*w[0] + w[1])
    return data_white

def preprocess_for_detection(data,fs,montage='bipolar',target=256, wavenet=False, pre_mask = None):
    # This function implements preprocessing steps for seizure detection
    chs = data.columns.to_list()
    ch_df = check_channel_types(chs)
    # Montage
    if montage == 'bipolar':
        data_bp_np,bp_ch_df = bipolar_montage(data.to_numpy().T,ch_df)
        bp_ch = bp_ch_df.name.to_numpy()
    elif montage == 'car':
        data_bp_np = (data.to_numpy().T - np.mean(data.to_numpy(),1))
        bp_ch = chs
    
    # Channel rejection
    if pre_mask is None:
        mask,info = detect_bad_channels(data_bp_np.T*1e3,fs)
        data_bp_np = data_bp_np[mask,:]
        bp_ch = bp_ch[mask]
    else:
        mask = np.atleast_1d([ch not in pre_mask for ch in bp_ch])
        data_bp_np = data_bp_np[mask,:]
        bp_ch = bp_ch[mask]
        
    # Filtering and autocorrelation
    if wavenet:
        target=128
        data_bp_notch = notch_filter(data_bp_np,fs)
        data_bp_filt = bandpass_filter(data_bp_notch,fs,lo=3,hi=127)
        signal_len = int(data_bp_filt.shape[1]/fs*target)
        data_bpd = sc.signal.resample(data_bp_filt,signal_len,axis=1).T
        fsd = int(target)
    else:
        # Bandpass filtering
        data_bp_notch = notch_filter(data_bp_np,fs)
        data_bp_filt = bandpass_filter(data_bp_notch,fs,lo=3,hi=100)
        # Down sampling
        signal_len = int(data_bp_filt.shape[1]/fs*target)
        data_bpd = sc.signal.resample(data_bp_filt,signal_len,axis=1).T
        fsd = int(target)
    data_white = ar_one(data_bpd)
    data_white_df = pd.DataFrame(data_white,columns = bp_ch)
    if pre_mask is None:
        return data_white_df,fsd,mask
    else:
        return data_white_df,fsd
    

def remove_scalp_electrodes(raw_labels):
    scalp_list = ['CZ','FZ','PZ',
                  'A01','A02',
                  'C03','C04',
                  'F03','F04','F07','F08',
                  'Fp01','Fp02',
                  'O01','O02',
                  'P03','P04',
                  'T03','T04','T05','T06',
                  'EKG01','EKG02',
                  'ROC','LOC',
                  'EMG01','EMG02',
                  'DC01','DC07'
                  ]
    chop_scalp = ['C1'+str(x) for x in range(19,29)]
    scalp_list += chop_scalp
    return [l for l in raw_labels if l.upper() not in scalp_list]
################################################ Feature Extraction ################################################


######################## Univariate, Time Domain ########################
def _timeseries_to_wins(
    data: np.ndarray, fs: float, win_size=2, win_stride=1
) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray)
        fs (float): _description_
        win_size (int, optional): _description_. Defaults to 2.
        win_stride (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    n_samples = data.shape[-1]

    idx = (
        np.arange(win_size * fs, dtype=int)[None, :]
        + np.arange(n_samples - win_size * fs + 1, dtype=int)[
            :: int(win_stride * fs), None
        ]
    )
    return data[:, idx]

def MovingWinClips(x,fs,winLen,winDisp):
  # calculate number of windows and initialize receiver
  nWins = num_wins(len(x),fs,winLen,winDisp)
  samples = np.empty((nWins,winLen*fs))
  # create window indices - these windows are left aligned
  idxs = np.array([(winDisp*fs*i,(winLen+winDisp*i)*fs)\
                   for i in range(nWins)],dtype=int)
  # apply feature function to each channel
  for i in range(idxs.shape[0]):
    samples[i,:] = x[idxs[i,0]:idxs[i,1]]
  
  return samples

def ll(x):
    return np.sum(np.abs(np.diff(x)), axis=-1)


def bandpower_fooof(x: np.ndarray, fs: float, lo=1, hi=120, relative=True, win_size=2, win_stride=1) -> np.array:
    """Use FOOOF to calculate bandpower

    Args:
        x (np.ndarray): _description_
        fs (float): _description_
        lo (int, optional): _description_. Defaults to 1.
        hi (int, optional): _description_. Defaults to 120.
        relative (bool, optional): _description_. Defaults to True.
        win_size (int, optional): _description_. Defaults to 2.
        win_stride (int, optional): _description_. Defaults to 1.

    Returns:
        np.array: _description_
    """
    # bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 80)}
    bands = {"broad":(1,100)}

    nperseg = int(win_size * fs)
    noverlap = int(win_stride * fs)

    freq, pxx = welch(x=x, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)

    # Initialize a FOOOF object
    fg = FOOOFGroup()

    # Set the frequency range to fit the model
    freq_range = [lo, hi]

    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fg.fit(freq, pxx, freq_range)
    fres = fg.get_results()

    def one_over_f(f, b0, b1):
        return b0 - np.log10(f ** b1)

    idx = np.logical_and(freq >= lo, freq <= hi)
    one_over_f_curves = np.array([one_over_f(freq[idx], *i.aperiodic_params) for i in fres])

    residual = np.log10(pxx[:, idx]) - one_over_f_curves
    freq = freq[idx]

    bandpowers = np.zeros((len(bands), pxx.shape[0]))
    for i_band, (lo, hi) in enumerate(bands.values()):
        if np.logical_and(60 >= lo, 60 <= hi):
            idx1 = np.logical_and(freq >= lo, freq <= 55)
            idx2 = np.logical_and(freq >= 65, freq <= hi)
            bp1 = simpson(
                y=residual[:, idx1],
                x=freq[idx1],
                dx=freq[1] - freq[0]
            )
            bp2 = simpson(
                y=residual[:, idx2],
                x=freq[idx2],
                dx=freq[1] - freq[0]
            )
            bandpowers[i_band] = bp1 + bp2
        else:
            idx = np.logical_and(freq >= lo, freq <= hi)
            bandpowers[i_band] = simpson(
                y=residual[:, idx],
                x=freq[idx],
                dx=freq[1] - freq[0]
            )
    return bandpowers.T

def bandpower(x: np.ndarray, fs: float, lo=1, hi=120, relative=True, win_size=2, win_stride=1) -> np.array:
    """
    Calculates the relative bandpower of a signal x, using a butterworth filter of order 'order'
    and bandpass filter between lo and hi Hz.

    Use scipy.signal.welch and scipy.signal.simpson
    """
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 80)}

    nperseg = int(win_size * fs)
    noverlap = nperseg - int(win_stride * fs)

    freq, pxx = welch(x=x, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)
    
    # log transform the power spectrum
    # pxx = 10*np.log10(pxx)
    
    all_bands = np.zeros((pxx.shape[0], len(bands)))
    for i, (band, (lo, hi)) in enumerate(bands.items()):
        idx_band = np.logical_and(freq >= lo, freq <= hi)
        bp = simpson(pxx[:, idx_band], dx=freq[1] - freq[0])
        # relative
        if relative:
            bp /= simpson(pxx, dx=freq[1] - freq[0])
        all_bands[:, i] = bp
    return all_bands
    # return bp
    # return data_filt


def ft_extract(
    data: np.ndarray, fs: float, ft: str, win_size=2, win_stride=1, fn_kwargs={}
) -> np.ndarray:
    """_summary_

    Args:
        data (mne.io.edf.edf.RawEDF): _description_
        ft (str): _description_
        win_size (int, optional): _description_. Defaults to 2.
        win_stride (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    wins = _timeseries_to_wins(data, fs, win_size, win_stride)
    wins = np.transpose(wins, (1, 0, 2))
    (n_wins, n_ch, _) = wins.shape

    # if ft is a list of features, then calculate both featurs and concatenate
    if isinstance(ft, list):
        assert len(ft) == len(fn_kwargs), "Incorrect number of feature arguments given"
        # ft_array = np.empty((n_ch, n_wins, len(ft)))
        ft_array = []
        for i, fn in enumerate(ft):
            # if f is not callable, then raise value error
            if not callable(fn):
                raise ValueError("Incorrect feature argument given")

            # if bandpower, then don't iterate over windows
            if fn is bandpower:
                # include win_size and win_stride in kwargs
                fn_kwargs[i]["win_size"] = win_size
                fn_kwargs[i]["win_stride"] = win_stride

                ft_array.append(fn(data, **(fn_kwargs[i])))
            else:
                for j, win in enumerate(wins):
                    # ft_array[:, j, i] = fn(win, **(fn_kwargs[i]))
                    ft_array.append(fn(win, **(fn_kwargs[i])))
        ft_array = np.array(ft_array)
        # transpose to n_ch x n_wins x n_ft
        ft_array = np.transpose(ft_array, (1, 0, 2))
        return ft_array

    elif callable(ft):
        # ft_array = np.empty((n_ch, n_wins))

        ft_array = []

        if ft is bandpower:
            # include win_size and win_stride in kwargs
            fn_kwargs["win_size"] = win_size
            fn_kwargs["win_stride"] = win_stride

            ft_array.append(ft(data, **fn_kwargs))
        else:
            for i, win in enumerate(wins):
                ft_array.append(ft(win, **fn_kwargs))
        
            # ft_array[:, i] = ft(win, **fn_kwargs)
        ft_array = np.array(ft_array)
        
        # convert 2 dim to 3 dim
        if ft_array.ndim == 2:
            ft_array = ft_array[:, :, None]
        # transpose to n_ch x n_wins x n_ft
        ft_array = np.transpose(ft_array, (1, 0, 2))

    else:
        raise ValueError("Incorrect feature type given")

    return ft_array


def _ll(x):
    return np.sum(np.abs(np.diff(x)), axis=-1)

def dice_score(x,y):
    num = 2*len(np.intersect1d(x,y))
    if len(x.shape) < 1:
        x = np.array([str(x)])
    if len(y.shape) < 1:
        y = np.array([str(y)])
    denom = len(x)+len(y)
    return num/denom

######################## Univariate, Spectral Domain ########################
bands = [
    [1, 4],  # delta
    [4, 8],  # theta
    [8, 12],  # alpha
    [12, 30],  # beta
    [30, 80],  # gamma
    [1, 80],  # broad
]
band_names = ["delta", "theta", "alpha", "beta", "gamma", "broad"]
N_BANDS = len(bands)

def _one_over_f(f: np.ndarray, b0: float, b1: float) -> np.ndarray:
    """_summary_

    Args:
        f (np.ndarray): _description_
        b0 (float): _description_
        b1 (float): _description_

    Returns:
        np.ndarray: _description_
    """
    return b0 - np.log10(f**b1)


def spectral_features(
    data: np.ndarray, fs: float, win_size=2, win_stride=1
) -> pd.DataFrame:
    """_summary_

    Args:
        data (np.ndarray): _description_
        fs (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    feature_names = [f"{i} power" for i in band_names] + ["b0", "b1"]

    freq, pxx = welch(
        x=data,
        fs=fs,
        window="hamming",
        nperseg=int(fs * win_size),
        noverlap=int(fs * win_stride),
        axis=0,
    )

    # Initialize a FOOOF object
    fg = FOOOFGroup(verbose=False)

    # Set the frequency range to fit the model
    freq_range = [0.5, 80]

    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fg.fit(freq, pxx.T, freq_range)
    fres = fg.get_results()

    idx = np.logical_and(freq >= freq_range[0], freq <= freq_range[1])
    one_over_f_curves = np.array(
        [_one_over_f(freq[idx], *i.aperiodic_params) for i in fres]
    )

    residual = np.log10(pxx[idx]).T - one_over_f_curves
    freq = freq[idx]

    bandpowers = np.zeros((len(bands), pxx.shape[-1]))
    for i_band, (lo, hi) in enumerate(bands):
        if np.logical_and(60 >= lo, 60 <= hi):
            idx1 = np.logical_and(freq >= lo, freq <= 55)
            idx2 = np.logical_and(freq >= 65, freq <= hi)
            bp1 = simpson(y=residual[:, idx1], x=freq[idx1], dx=freq[1] - freq[0])
            bp2 = simpson(y=residual[:, idx2], x=freq[idx2], dx=freq[1] - freq[0])
            bandpowers[i_band] = bp1 + bp2
        else:
            idx = np.logical_and(freq >= lo, freq <= hi)
            bandpowers[i_band] = simpson(
                y=residual[:, idx], x=freq[idx], dx=freq[1] - freq[0]
            )
    aperiodic_params = np.array([i.aperiodic_params for i in fres])
    clip_features = np.row_stack((bandpowers, aperiodic_params.T))

    return pd.DataFrame(clip_features, index=feature_names, columns=data.columns)


def coherence_bands(
    data: Union[pd.DataFrame, np.ndarray], fs: float, win_size=2, win_stride=1
) -> np.ndarray:
    """_summary_

    Args:
        data (Union[pd.DataFrame, np.ndarray]): _description_
        fs (float): _description_

    Returns:
        np.ndarray: _description_
    """
    _, n_channels = data.shape
    n_edges = sum(1 for i in itertools.combinations(range(n_channels), 2))
    n_freq = int(fs) + 1

    cohers = np.zeros((n_freq, n_edges))

    for i_pair, (ch1, ch2) in enumerate(itertools.combinations(range(n_channels), 2)):
        freq, pair_coher = coherence(
            data.iloc[:, ch1],
            data.iloc[:, ch2],
            fs=fs,
            window="hamming",
            nperseg=int(fs * win_size),
            noverlap=int(fs * win_stride),
        )

        cohers[:, i_pair] = pair_coher

    # keep only between originally filtered range
    filter_idx = np.logical_and(freq >= 0.5, freq <= 80)
    freq = freq[filter_idx]
    cohers = cohers[filter_idx]

    coher_bands = np.empty((N_BANDS, n_edges))
    coher_bands[-1] = np.mean(cohers, axis=0)

    # format all frequency bands
    for i_band, (lower, upper) in enumerate(bands[:-1]):
        filter_idx = np.logical_and(freq >= lower, freq <= upper)
        coher_bands[i_band] = np.mean(cohers[filter_idx], axis=0)

    return coher_bands

########################### Workspace Preparation ###########################
def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)

def load_config(config_path,flag='HUP'):
    with open(config_path,'r') as f:
        CONFIG = json.load(f)
    usr = CONFIG["paths"]["iEEG_USR"]
    passpath = CONFIG["paths"]["iEEG_PWD"]
    datapath = CONFIG["paths"]["RAW_DATA"]
    prodatapath = CONFIG["paths"]["PROCESSED_DATA"]
    figpath = CONFIG["paths"]["FIGURES"]
    metapath = CONFIG["paths"]["METADATA"]
    patient_table = pd.DataFrame(CONFIG["patients"]).sort_values('ptID').reset_index(drop=True)
    if flag == 'HUP':
        patient_table = patient_table[patient_table.ptID.apply(lambda x: x[:3]) == 'HUP']
    elif flag == 'CHOP':
        patient_table = patient_table[patient_table.ptID.apply(lambda x: x[:4]) == 'CHOP']
    rid_hup = pd.read_csv(ospj(metapath,'rid_hup.csv'))
    pt_list = patient_table.ptID.to_numpy()
    return usr,passpath,datapath,prodatapath,metapath,figpath,patient_table,rid_hup,pt_list
# %%
