# Quantitative Analysis of Stimulation-Induced Seizures

Analysis code for the multi-expert seizure annotation dataset (78 seizures, 29 patients:
19 HUP, 10 CHOP). See the top-level [README](../README.md) for an overview.

## Script order

1. **Configure local paths** — edit `config.py`, or (preferred) create a gitignored
   `config_local.py` overriding the paths (see the bottom of `config.py`).
2. `metadata_notebook.ipynb` — assemble seizure metadata and electrode localizations.
3. `BIDS_seizure_saving.py` — export seizure clips to iEEG-BIDS. *(requires iEEG.org access)*
4. `annotation_analysis_and_consensus.ipynb` — build the majority-vote consensus
   annotations saved to `PROCESSED_DATA/dataset_consensus.pkl`, which every downstream
   analysis loads. *(the channel-universe step requires iEEG.org access)*
5. `analyzing_annotator_reliability.ipynb` — inter-rater and consensus reliability
   analyses and figures.
6. `WAVENET_validation_analysis.py` — example single-channel model benchmark against the
   consensus annotations.

`utils.py` holds shared helpers used across these steps. Steps 5–6 run directly from the
shipped `dataset_consensus.pkl`; steps 2–4 regenerate it from the raw annotations.
