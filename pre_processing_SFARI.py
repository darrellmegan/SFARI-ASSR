# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:39:11 2024

@author: mdarrell
"""
    

def read_in_data(Subject, file_path, list_file):
    
    import mne 
    import os
    import numpy as np

    list_file_eeg = os.listdir(file_path + '/' + list_file[Subject])
    
    # read in raw data in BDF format
    raw = mne.io.read_raw_bdf(file_path + '/' + list_file[Subject] + '/' + [s for s in list_file_eeg if '.bdf' in s][0],preload=True)
    events = mne.find_events(raw)

    # Setting EEG montage (order of the electrodes)
    montage = mne.channels.make_standard_montage('biosemi64')
    raw = raw.set_montage(montage, on_missing='ignore')
    
    # notch filter at 60 Hz (electronic noise) and respective harmonic
    freqs = (60, 120)
    raw = raw.notch_filter(freqs=freqs)
    
    
    # Identifying channels with NaN positions
    nan_channels = [ch['ch_name'] for ch in raw.info['chs'] if np.isnan(ch['loc'][:3]).any()]
    
    # Removing channels with NaN positions
    if nan_channels:
        print("Removing channels with NaN positions:", nan_channels)
        raw.drop_channels(nan_channels)
    else:
        print("No channels with NaN positions found.")
        
    return raw, events


def detect_bad_chan(raw,run_ransac):
    
    from pyprep.find_noisy_channels import NoisyChannels

    nd = NoisyChannels(raw, random_state=None) 
    
    # Functions = 
    #         self.find_bad_by_deviation()
    #         self.find_bad_by_hfnoise()
    #         self.find_bad_by_correlation()
    #         self.find_bad_by_SNR()
    #         self.find_bad_by_ransac(
    #             channel_wise=channel_wise, max_chunk_size=max_chunk_size
    #         )
    nd.find_all_bads(ransac = run_ransac, channel_wise = True) # Call all the functions to detect bad channels
    

    nd.get_bads()
    print('\n============\n\n',len(nd.get_bads()), 'BAD CHANNELS\n')
    
    return nd

def interpolate_bad_chan(raw, nd, total_chan, per_allowed):
    
    import math
    
    # define number of channels based on percent rejection threshold
    num_allowed = math.floor(per_allowed/100*total_chan)

    # interpolate only if number of bad channels is below threshold
    if len(nd.get_bads()) < num_allowed:
        raw.info["bads"] = nd.get_bads()
        raw = raw.interpolate_bads()
        print('Interpolating all bads channels:', nd.get_bads())
    # if within 5 channels of threshold, interpolate only abnormal amplitude channels
    elif (len(nd.get_bads()) > num_allowed) and (len(nd.get_bads()) < num_allowed+5):
        if len(nd.bad_by_SNR) <num_allowed:
            raw.info["bads"] = nd.bad_by_SNR
            raw = raw.interpolate_bads()
            print('Interpolating only abnormal amplitude channels:', nd.bad_by_SNR)
            
    return raw

def ICA(epochs, low_pass_filter):
    
    import mne 
    
    # First apply a lowpass of 1Hz, ICA is sensible to low frequency artifacts
    epochs_ica = epochs.filter(l_freq=low_pass_filter, h_freq=None)

    #Parameters ICA
    n_components = None 
    random_state = 42
    method = 'fastica' 
    fit_params = None 
    max_iter = 1000 

    ica = mne.preprocessing.ICA(n_components=n_components, method = method, max_iter = max_iter, fit_params= fit_params, random_state=random_state)

    ica.fit(epochs_ica)
    ica.plot_sources(epochs)
    ica.plot_components()

    # remove ICA components related to eye/muscle movement, cardiac activity by visual inspection 
    # - can assess just first 15 components
    #    EYE: electrode activity focused at front (blinks) or at front bilateral in opp magnitude (saccades)
    #    CARDIO: electrode activity concentrated bilaterally (symmetric) .. examine spectra for PQRS complex
    #    MUSCLE: densely focused electrode activity in one location (flat spectra)
    #       - can lose some true ERP information by removing muscle ICA, best to be conservative
    
    return(ica)