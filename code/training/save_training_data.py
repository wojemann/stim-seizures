# File system imports
import sys
import os
import glob
from os.path import join as ospj

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO/WARN messages
os.environ['TF_TRT_DISABLED'] = '1'       # Silence TF-TRT warnings if TensorRT not installed

# Scientific imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import RobustScaler

# Plotting imports
import matplotlib.pyplot as plt

# Deep learning imports
from tensorflow.config.experimental import set_memory_growth, list_physical_devices
import tensorflow as tf
from absl import logging as absl_logging

# HDF5 imports
import h5py

# Suppress TensorFlow logging
absl_logging.set_verbosity(absl_logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Configure GPU memory growth to prevent allocation issues
try:
    for _gpu in list_physical_devices('GPU'):
        set_memory_growth(_gpu, True)
except Exception:
    pass
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = ospj(script_dir, '..')
sys.path.insert(0, parent_dir)  # Insert parent dir at beginning of path

# Utility imports
from utils import preprocess_for_detection, get_data_from_bids, clean_labels, load_electrode_localizations

# Parallel execution
# from pqdm.threads import pqdm

# Import Config from parent directory's config.py
from config import Config

# Get paths from config 
datapath,prodatapath,figpath,metapath = Config.deal(['datapath','prodatapath','figpath','metapath'])


# Set default colormap for visualizations
plt.rcParams['image.cmap'] = 'magma'

# Global configuration
OVERWRITE = True  # Whether to overwrite existing probability matrix files
ONSET_TIME = 180
OFFSET_TIME = 120
SPREAD_TIME = 5
def save_training_data(patient, onset_run, onset_labels, montage, onset_time):
    electrode_localizations = load_electrode_localizations(patient, prodatapath,keep_white_matter=False)
    if electrode_localizations is None:
        print(f"Warning: No electrode localizations found for {patient}")
        return None
    # Load seizure recording (only if not all models exist)
    seizure, fs_raw, _, _, task, run = get_data_from_bids(
        ospj(datapath, "BIDS"), patient, onset_run, return_path=True, verbose=0
    )

    # Clean labels
    seizure_cols = [col for col in seizure.columns if col in electrode_localizations.keys()]
    seizure = seizure.loc[:,seizure_cols]

    # Initial mask from first 60s
    _, _, channel_mask = preprocess_for_detection(
        seizure.iloc[: 120 * fs_raw, :],
        fs_raw,
        montage,
        target=fs_raw,
        wavenet=False,
        pre_mask=None,
    )

    seizure_pre, fs = preprocess_for_detection(
        seizure,
        fs_raw,
        montage,
        target=fs_raw,
        wavenet=False,
        pre_mask=channel_mask,
    )

    scl = RobustScaler()
    scl.fit(seizure_pre.iloc[:fs_raw*120,])
    seizure_z = pd.DataFrame(scl.transform(seizure_pre),columns=seizure_pre.columns)
    art_channel_mask = seizure_z.loc[180*fs_raw:,:].abs().max() <= (np.median(seizure_z.loc[180*fs_raw:,:].abs().max())*50)
    art_chs = seizure_z.columns[art_channel_mask]
    art_chs = [ch for ch in art_chs if ch.split('-')[0] not in onset_labels]
    seizure_nart = seizure_z.loc[:,art_chs]

    onset_mask = np.array([ch.split('-')[0] in onset_labels for ch in seizure_nart.columns])

    # Get 256-sample clips
    def get_windowed_data(data, window_size, stride):
        data = data.to_numpy()
        data = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size, data.shape[1]))
        data = data[::stride, 0, :, :]
        data = data.transpose(0, 2, 1)
        data = data.reshape(-1, window_size)
        return data

    window_size = 256
    stride = window_size // 2  # 50% overlap
    
    # Class 1: Onset clips (from seizure onset channels)
    onset_clips = [{'patient': patient, 'onset': onset_run, 'class': 1, 'data': clip} 
                   for clip in get_windowed_data(
                       seizure_nart.loc[onset_time*fs_raw:(onset_time+5)*fs_raw, onset_mask],
                       window_size, stride)]
    
    # Class 2: Spread clips (from seizure onset channels, later in time)
    spread_clips = [{'patient': patient, 'onset': onset_run, 'class': 2, 'data': clip} 
                    for clip in get_windowed_data(
                        seizure_nart.loc[(onset_time+SPREAD_TIME)*fs_raw:(onset_time+SPREAD_TIME+15)*fs_raw, onset_mask],
                        window_size, stride)]
    
    # Class 0: Non-onset clips (from non-onset channels)
    # nonset_ch = np.random.choice(seizure_nart.columns[~onset_mask], size=sum(onset_mask), replace=False)
    nonset_ch = seizure_nart.columns[~onset_mask]
    non_onset_clips = [{'patient': patient, 'onset': onset_run, 'class': 0, 'data': clip} 
                       for clip in get_windowed_data(
                           seizure_nart.loc[10*fs_raw:20*fs_raw, nonset_ch],
                           window_size, stride)]
    return onset_clips + spread_clips + non_onset_clips

def main():
    """
    Extract and save training data from seizure recordings in HDF5 format.
    
    Workflow:
    1. Load seizure metadata from BIDS format
    2. For each patient's seizure recording:
       - Load and preprocess EEG data
       - Extract 256-sample clips with labels:
         * Class 0: Non-onset channels (background)
         * Class 1: Onset channels at seizure start
         * Class 2: Onset channels during spread phase
    3. Organize clips hierarchically by patient and seizure onset time
    4. Save to HDF5 file with structure:
       patient_name/onset_time/data (clips) and labels
    
    Output: seizure_training_data.h5 in PROCESSED_DATA directory
    """

    # Load seizure metadata from BIDS processing
    seizures_df = pd.read_csv(ospj(metapath,"metadata_v7_BIDS.csv"))
    seizures_df['stim'] = seizures_df['stim'].fillna(0)
    seizures_df = seizures_df[(seizures_df.stim == 0) & (seizures_df.split == 1)]
    seizures_df = seizures_df[seizures_df.notes.apply(lambda x: 'nina' not in str(x).lower())]
    seizures_df = seizures_df[seizures_df.source.apply(lambda x: 'nina' not in str(x).lower())]
    # seizures_df = seizures_df[(seizures_df.split )] # Filter for only seizures that have soft onset labels
    
    # Detection parameters
    onset_time = 180          # Seizure onset time in recording (seconds)
    montage = 'bipolar'       # Electrode montage for preprocessing
    # all_models = [WVNT]

    # Build all tasks across all patients and seizures (models are handled inside)
    tasks = []
    
    # Iterating through each patient that we have annotations for
    pbar = tqdm(seizures_df.iterrows(),total=len(seizures_df))
    for _,row in pbar:
        patient = row.Patient
        sz = row.onset
        offset = row.offset
        if (offset-sz) < 20:
            continue
        pbar.set_description(desc=f"Patient: {patient} | Seizure: {sz}",refresh=True)
        onset_run = str(int(row.onset))
        if not isinstance(row.SOZ, str):
            onset_labels = []
        else:
            onset_labels = clean_labels([l.strip() for l in row.SOZ.split(',')], patient)
        tasks.append((patient, onset_run, onset_labels, montage, onset_time))

    # Execute all tasks and collect clips
    # results_nested = pqdm(tasks, save_training_data, n_jobs=12)

    all_clips = []
    failed_tasks = []
    
    print("\nProcessing seizures and extracting clips...")
    for task in tqdm(tasks, total=len(tasks)):
        try:
            clips = save_training_data(*task)
            if clips:
                all_clips.extend(clips)
        except Exception as e:
            print(f"\nTask failed for {task[0]} seizure {task[1]}: {e}")
            failed_tasks.append(task)
            continue
    
    print(f"\nTotal clips extracted: {len(all_clips)}")
    print(f"Failed tasks: {len(failed_tasks)}")
    
    # Organize clips by patient and seizure onset
    patient_seizure_clips = {}
    for clip in all_clips:
        patient = clip['patient']
        onset = clip['onset']
        
        if patient not in patient_seizure_clips:
            patient_seizure_clips[patient] = {}
        if onset not in patient_seizure_clips[patient]:
            patient_seizure_clips[patient][onset] = []
        
        patient_seizure_clips[patient][onset].append({
            'data': clip['data'],
            'label': clip['class']
        })
    
    # Create H5 file with hierarchical structure
    output_file = ospj(prodatapath, "seizure_training_data_v2.h5")
    print(f"\nSaving data to {output_file}...")
    
    with h5py.File(output_file, 'w') as h5f:
        for patient in tqdm(sorted(patient_seizure_clips.keys()), desc="Saving patients"):
            patient_group = h5f.create_group(patient)
            
            for onset in sorted(patient_seizure_clips[patient].keys()):
                # Use onset time as int for the group name
                seizure_group = patient_group.create_group(str(int(onset)))
                
                clips_data = patient_seizure_clips[patient][onset]
                n_clips = len(clips_data)
                
                # Create datasets for data and labels
                data_array = np.array([clip['data'] for clip in clips_data], dtype=np.float32)
                label_array = np.array([clip['label'] for clip in clips_data], dtype=np.int8)
                
                seizure_group.create_dataset('data', data=data_array, 
                                            compression='gzip', compression_opts=4)
                seizure_group.create_dataset('labels', data=label_array)
                
                # Add metadata attributes
                seizure_group.attrs['n_clips'] = n_clips
                seizure_group.attrs['onset_time'] = int(onset)
                seizure_group.attrs['clip_length'] = 256
    
    print(f"\nData saved successfully!")
    print(f"Total patients: {len(patient_seizure_clips)}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    total_clips = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    for patient in patient_seizure_clips:
        for onset in patient_seizure_clips[patient]:
            clips = patient_seizure_clips[patient][onset]
            total_clips += len(clips)
            for clip in clips:
                class_counts[clip['label']] += 1
    
    print(f"Total clips: {total_clips}")
    print(f"Class 0 (non-onset): {class_counts[0]}")
    print(f"Class 1 (onset): {class_counts[1]}")
    print(f"Class 2 (spread): {class_counts[2]}")
    
if __name__ == "__main__":
    main()
