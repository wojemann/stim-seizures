# stim-seizures
Analyzing the relationsihp between low-frequency electrical stimulation induced seizure and spontaneous seizure networks.

## Can electrical stimulation replace spontaneous seizures?
### authors
### affiliations

### Preprint: will be made available upon preprint!
### Manuscript: will be made available upon publication!

## Prerequisites/Dependencies

## Data
all raw EEG data is publically available on iEEG.org and will be uploaded in BIDS format as a publcially available dataset on pennsive.io upon publication. The script BIDS_seizure_saving.py contains code used to save the raw EEG data.

The following checkpoint files are provided in the pennsieve dataset and can be used to generate all main text and supplementary figures.
* stim_seizure_information - LF_seizure_annotation.csv
* stim_seizure_information_BIDS.csv
* stim_seizure_information - metadata-4.csv
* CHOP_metadata.csv
* threshold_tuning_consensus_v2.pkl
* pretrain_predicted_channels_wmcc_epoch-10_min-False_mov-mean-20-prob_v3.pkl
* optimized_predicted_channels_LSTM_tuned_thresholds_v4_sz-mean_pt-mean_smooth-med.pkl

## Analysis pipeline
Steps to replicate:
* Config file
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
* annotation_analysis_and_consensys.ipynb
notebook to generate consensus thresholds and perform seizure similarity analysis (Figure SX). Run this first to generate threshold_tuning_consensus_v2.pkl, which contains the consensus annotations for each annotated seizure.
* BIDS_seizure_saving.py
script to pull seizure data from iEEG.org and save it in iEEG-BIDS format using mne python. Saves all seizures in seizure_information-LF that are spontaneous (stim == 0) or low-frequency stim-induced (stim == 1)
* BIDS_interictal_saving.py
script to pull the designated interictal data from iEEG.org and save it in iEEG-BIDS format using mne python. Saves all interictal clips as designated in the config, one for each patient.
* seizure_detection_pretrain.py
script to generate seizure detection values for each spontaneous and stimulation induced seizure using the NDD model and two benchmarks: Absolute slope, and a wavenet-based univariate seizure detector.
* val_generate_model_annotations.py
script to generate ueo channel and 10 second spread channel annotations for each seizure that had human annotations. The annotations were generated for each threshold in [0,4,750] in order to fit thresholds.
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
