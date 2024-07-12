# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:18:17 2024

@author: mdarrell
"""
filepath = 'C:/Users/theov/Dropbox (EinsteinMed)/1. CNL/6. EEG & MRI/Data analysis/Megan/Scripts/SFARI/'
#loading needed toolboxes 
import mne
import numpy as np
import pandas as pd
import os
import autoreject
import sys
sys.path.append(filepath)
import pre_processing_SFARI
from pre_processing_SFARI import read_in_data, detect_bad_chan, interpolate_bad_chan, ICA


paradigm = 'ASSR_oddball'

list_file = os.listdir(filepath+paradigm+'/')
file_path = filepath+paradigm

#%% READ IN DATA, DETECT/INTERPOLATE BAD CHANNELS, FILTER, EPOCH

# read in subject ID from list of files
Subject = 105

# set event dictionary
paradigm_conditions = ['40standard', '40target', '27standard', '27target']
event_dict_s1 = {
    "40standard": 11,
    "40target": 12,
    "27standard": 21,
    "27target": 22
}


# read in data, notch filter at 60/120 Hz to remove electronic noise
raw, events = read_in_data(Subject, file_path, list_file)
    
# detecting bad channels by NoisyChannels: deviation(), hfnoise(), correlation(), SNR(), ransac() if True
nd = detect_bad_chan(raw, True)

# refers to the percent of bad channels that can be interpolated without loss of data
per_allowed = 15
total_chan = 64
# interpolate bad channels previously detected with NoisyChannels
raw = interpolate_bad_chan(raw, nd, total_chan, per_allowed)
    
# high and low-pass filtration
lowpass_epochs = 80
highpass_epochs = 0.01
raw = raw.filter(l_freq=highpass_epochs, h_freq=lowpass_epochs)

# create epoch file without baseline correction
# -----EPOCHING EVENTS 
# -------- ISI = 500-800ms
# -------- stim = 500ms 
tmin_epochs = -0.2
tmax_epochs = 0.8
epochs = mne.Epochs(raw, events, tmin=tmin_epochs, tmax=tmax_epochs, event_id = event_dict_s1, baseline=None, detrend=1, preload=True, decim=1)

#%%#INDEPENDENT COMPONENTS ANALYSIS (ICA), low-pass filtered at 1
#------method: fastica
#------n_iterations: 1000

low_pass_filter = 1
ica = ICA(epochs, low_pass_filter)

#%% APPLY ICA AND SAVE FILES

# apply ICA
ica.apply(epochs)

# apply baseline correction after ICA
tmin = tmin_epochs
baseline = (tmin,0)
epochs.apply_baseline(baseline=baseline)

# Use of autoreject package after ICA to remove bads epochs for no PREP epochs
# ----utilizes a cross-validation metric to identify an optimal amplitude threshold
ar = autoreject.AutoReject(n_interpolate= [1, 4, 32],
                           n_jobs = 1,
                           verbose = True)
ar.fit(epochs)
epochs_clean, reject_log_all = ar.transform(epochs, return_log=True)


# Save epochs
save_path = filepath+paradigm+'/Epochs/'
if not os.path.exists(save_path):
   # Create the directory
   os.makedirs(save_path)

saving_file_1 = save_path+'/'+list_file[Subject]+'_epo.fif'
epochs_clean.save(saving_file_1)

# Save pre-processing information in CSV file
Info_pipprep = {'Channels interpolated': pd.Series(nd.get_bads()),
                     'Number ICA components rejected': pd.Series(len(ica.exclude)),
                     'Number epochs rejected with autoreject': pd.Series(len(np.where(reject_log_all.bad_epochs)[0])),
                     'Percent reject trial': (pd.Series(len(np.where(reject_log_all.bad_epochs)[0]))) / len(epochs)
                     }

for condition in paradigm_conditions:
    Info_pipprep['Number '+condition+' trial final'] = pd.Series((len(epochs_clean[condition])))
Info_pipprep = pd.DataFrame(Info_pipprep)

csv_path_all = save_path+"/"+list_file[Subject]+"_info_preprocessing"
Info_pipprep.to_csv(csv_path_all,  sep='\t', encoding='utf-8')

