import numpy as np 
import mne
import sys, os, re


data_path = sys.argv[1] # EEG file in EDF format


# EEG_raw = mne.io.read_raw_edf(filename)
# channels = EEG_raw.info['ch_names']
# eeg_waveform = EEG_raw.to_data_frame().as_matrix()

def read_EDF(data_path, channels = None):
    if not os.path.isfile(data_path):
        raise Exception ('%s is not found'%data_path)

    try:
        EEG           = mne.io.read_raw_edf(data_path)
        channel_names = EEG.info['ch_names']
        Fs            = EEG.info['sfreq']
        eeg_waveform  = EEG.to_data_frame()
        eeg_waveform  = eeg_waveform.as_matrix() 

    except:
        print('could not read file')


    if channels is None:
        EEG_channel_ids = list(range(len(channel_names)))
    else:
        EEG_channel_ids = []
        for i in range(len(channels)):
            found = False

            for j in range(len(channel_names)):
                print(channel_names[j], str(channels[i]),j)
                if re.search(channel_names[j], channels[i]):
                    EEG_channel_ids.append(j)
                    found = True
                    break
            if not found:
                print('channel could not be found')

        eeg_waveform = eeg_waveform.T    
        print(eeg_waveform.shape)
        eeg_waveform = eeg_waveform[EEG_channel_ids,:]
    params  = {'Fs': Fs, 'EEG_channel_ids': EEG_channel_ids}

    return eeg_waveform,  params


EEG_channels = ['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']

EEG, params = read_EDF(data_path, EEG_channels)
print(EEG.shape, params)
