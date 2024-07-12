# SFARI-ASSR
EEG processing & analysis of SFARI ASSR paradigm (ASD, TD controls, unaffected siblings)
***
## [Main pre-processing script](main_preprocessing_ASSRoddball.py)
Reads in EEG data (in BDF files) and performs ICA, low/high-pass filtering, channel/epoch rejection before saving epoch data
- Calls functions from [pre-processing script](pre_processing_SFARI.py)
