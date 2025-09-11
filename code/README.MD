# Unsupervised seizure detection with neural dynamic divergence
## Script order
* BIDS_seizure_saving
* * saving annotated seizures to compare models against interrater reliability
* BIDS_interictal_saving
* * saving interictal clips associated with the saved seizures for the stim seizures
* seizure_detection_pipeline_dynasd.py
* * running the different seizure detection algorithms on the set of annotated seizures with human annotations
