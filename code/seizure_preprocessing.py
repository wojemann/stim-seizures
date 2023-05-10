import json

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

with open('config.json','r') as f:
    CONFIG = json.load(f)

datapath = CONFIG["paths"]["RAW_DATA"]
ieeg_list = CONFIG["patients"]
pt_list = np.unique(np.array([i.split("_")[0] for i in ieeg_list]))
print(pt_list)
# Iterate through each patient
for pt in tqdm(pt_list):
    raw_datapath = ospj(datapath,pt)
    # load dataframe of seizure times
    seizure_times = pd.read_csv(ospj(raw_datapath,f"seizure_times_{pt}.csv"))

