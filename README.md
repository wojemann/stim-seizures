# Multi-expert consensus annotations of spontaneous and stimulation-induced seizures in stereotactic EEG

Code to reproduce the technical validation and data overview analyses from our open sEEG dataset descriptor (\<link coming soon!>). The dataset provides multi-expert consensus annotations of seizure onset and 10-second spread channels for 78 seizures (44 spontaneous, 34 stimulation-induced) from 29 patients (19 HUP, 10 CHOP), and is hosted on Pennsieve in iEEG-BIDS format.

## Contents

- `code/` — analysis notebooks and scripts (run order in [`code/README.MD`](code/README.MD)):
  - `metadata_notebook.ipynb` — assemble seizure metadata and electrode localizations
  - `annotation_analysis_and_consensus.ipynb` — builds the majority-vote consensus annotations that every downstream analysis loads
  - `analyzing_annotator_reliability.ipynb` — inter-rater and consensus reliability (manuscript Figs. 3–4)
  - `WAVENET_validation_analysis.py` — example model benchmark against consensus (Fig. 5)
  - `BIDS_seizure_saving.py` — export seizures to iEEG-BIDS
  - `config.py` / `utils.py` — data paths and shared helpers (edit `config.py` for your local paths)
- `METADATA/`, `PROCESSED_DATA/`, `RAW_DATA/` — local data directories referenced by the code

If you use the raw data or any derivatives from this dataset, please cite the linked manuscript.
