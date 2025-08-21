# stim-seizures
Analyzing the relationsihp between low-frequency electrical stimulation induced seizure and spontaneous seizure networks.
![alt text](https://github.com/wojemann/stim-seizures/stim_paper/stim_seizures_fig1.png "Intro Figure")

## ⚡️ Can electrical stimulation replace spontaneous seizures?
### 👨‍🍳 Authors
### 🏦 Affiliations

### 🧪 Preprint: will be made available upon preprint!
### 🔬 Manuscript: will be made available upon publication!

## 🤖 Prerequisites/Dependencies
- **OS**: Linux (tested). macOS should work; Windows is untested.
- **Python (stim-env)**: Python 3.10.x (tested with 3.10.16). Saved artifacts use pickle protocol 5, so use Python ≥3.8 (3.10 recommended) to ensure compatibility.
- **Environment setup**:
  - Create a virtual environment (recommended name: `stim-env`).
  - Install Python deps via the pinned `requirements.txt` at the repo root.
  - Additional packages not pinned in `requirements.txt` but required by the code:
    - `ieeg` (Python client for iEEG.org). Install from source at 
    - `DSOSD` (provides the `DSOSD.model.NDD` class used by the detection pipeline). Install from its source per that project’s instructions.
- **GPU (optional but recommended for deep models)**:
  - The pinned wheels target CUDA 12.x (e.g., torch 2.2.0 + cu12 and TensorFlow 2.16.1). Ensure a compatible NVIDIA driver and CUDA runtime if using GPU. CPU-only runs are supported but slower.
- **Jupyter**: JupyterLab 4.x is included in `requirements.txt` for running notebooks.
- **MNE-BIDS and neuroimaging**: `mne`, `mne-bids`, `nibabel`, `nilearn` are pinned and required for BIDS I/O and analyses.
- **Stats/ML libraries**: `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `pingouin`, `seaborn`, `matplotlib`, `fooof`, `kneed`, `bctpy`, and others are pinned in `requirements.txt`.
- **R (for mixed-effects models and stats in `code/*.R`)**: R ≥4.1 with the following packages installed:
  - `lme4`, `lmerTest`, `pbkrtest`, `dplyr`, `ggplot2`, `emmeans`, `multcomp`, `car`, `nlme`
- **Access to iEEG.org**:
  - An iEEG.org account is required to download raw EEG.
  - The config expects `IEEG_USR` (username) and an `IEEG_PWD` path to a binary file containing the password.

### Quick start
- Create and activate the environment, then install Python deps:
  ```bash
  python3.10 -m venv stim-env
  source stim-env/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install ieeg  # if not already present in your env
  # Install DSOSD from its source if your workflow uses NDD: DSOSD.model.NDD
  ```
- Install R dependencies (one-time):
  ```r
  install.packages(c("lme4","lmerTest","pbkrtest","dplyr","ggplot2","emmeans","multcomp","car","nlme"))
  ```
- Download data checkpoints
- Update config.json
    Config file
    There is an example config file with the required fields. you need to fill out the following paths:
    * RAW_DATA
    directory where the raw EEG recordings will be saved in iEEG-BIDS format
    * PROCESSED_DATA
    directory where derived metadata at the seizure level (annotations, probability matrices, etc.) and cohort level (annotations, checkpoints etc.)
    * METADATA
    directory where generated metadata such as seizure times, and raw manual annotations are stored
    * IEEG_USR
    username for accessing raw EEG files on iEEG.org
    * IEEG_PWD
    path to a binary file containing the password string for iEEG.org account
    * patients
    nested structure, list of dictionaries containing information about each patient in the cohort. The required fields are:
        * ptID
        patient name
        * ieeg_ids
        list of ieeg.org file ids associated with EEG recordings from that patient
        * interictal_training
        2 element list containing the ieeg_id and start time in that file (seconds) for the sample interictal time window (1 minute) to use as a baseline for stim seizure annotation
          
## Data
all raw EEG data is publically available on iEEG.org and will be uploaded in BIDS format as a publcially available dataset on pennsive.io upon publication. The script BIDS_seizure_saving.py contains code used to save the raw EEG data.

## Analysis pipeline
* annotation_analysis_and_consensys.ipynb
notebook to generate consensus thresholds and perform seizure similarity analysis (Figure SX). Run this first to generate threshold_tuning_consensus_v2.pkl, which contains the consensus annotations for each annotated seizure.
* BIDS_seizure_saving.py
script to pull seizure data from iEEG.org and save it in iEEG-BIDS format using mne python. Saves all seizures in seizure_information-LF that are spontaneous (stim == 0) or low-frequency stim-induced (stim == 1)
* BIDS_interictal_saving.py
script to pull the designated interictal data from iEEG.org and save it in iEEG-BIDS format using mne python. Saves all interictal clips as designated in the config, one for each patient.
* seizure_detection_pretrain.py
script to generate seizure detection values for each spontaneous and stimulation induced seizure using the NDD model and two benchmarks: Absolute slope, and a wavenet-based univariate seizure detector.
* val_generate_model_annotations.py
script to generate ueo channel and 10 second spread channel annotations for each seizure that had human annotations at each threshold in a parameter sweep.
* val_threshold_sweep_merging.py
script for merging human annotations and model annotations at each of the swept thresholds
* tuning_model_thresholds.ipynb
notebook containing code to visualize model performances and compare them to model performances
* lme_model_analysis.R
* val_generate_optimized_model_annotations.py
* model_prediction_analysis_cursor.ipynb
* lme_recruitment_analysis.R
* lme_semiology_mtle_analysis.R
* lme_spont_spread_analysis.R
* utils
    * utils.py
    * stim_seizure_preprocessing_utils.py
* visualizations
    * end2end_sandbox.ipynb
