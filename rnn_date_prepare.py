# script to prepare data for RNN
import os.path, re, sys,h5py, glob
from datetime import datetime, timedelta
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from spectrum import pow2db
from data_loader import check_load_Yvonne_dataset
from multitaper_spectrogram import multitaper_spectrogram
from segment_EEG import *
from extract_features import *


EEG_files_path = '/Users/siddharthbiswal/Dropbox/sleep_analysis_notebooks/data/YvonneDataSet_Exported_1.mat'
label_files =  '/Users/siddharthbiswal/Dropbox/sleep_analysis_notebooks/data/YvonneDataSet_Labels_1.mat'
spect_files  = '/Users/siddharthbiswal/Dropbox/sleep_analysis_notebooks/data/Spect_YvonneDataSet_Exported_1.mat'


# plot spectrogram and label file
EEG_channels = ['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']
# EEG, sleep_stage, other_info = check_load_Yvonne_dataset(EEG_files_path, label_files, channels=EEG_channels)
# EEG = EEG.T
# sleep_stage[np.isnan(sleep_stage)]=-7

# ff1 = h5py.File(spect_files)
# print(ff1.keys())
# Spect,stimes, sfreqs = ff1['S'], ff1['stimes'], ff1['sfreqs']
# with h5py.File(spect_files) as f:
#     spects = [f[element[0]][:]  for element in f['S']]
# # print(spects[0].shape,stimes.shape)


def create_data_for_RNN():
    pass


if __name__ == "__main__":

    EEG_files_path = '/Users/siddharthbiswal/Dropbox/sleep_analysis_notebooks/data/YvonneDataSet_Exported_1.mat'
    label_files =  '/Users/siddharthbiswal/Dropbox/sleep_analysis_notebooks/data/YvonneDataSet_Labels_1.mat'
    spect_files  = '/Users/siddharthbiswal/Dropbox/sleep_analysis_notebooks/data/Spect_YvonneDataSet_Exported_1.mat'

    EEG_channels = ['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']
    EEG, sleep_stage, params = check_load_Yvonne_dataset(EEG_files_path,label_files, channels=EEG_channels)

    # create segment for EEG and EEG label and other information
    segs, sleep_stages, seg_times, seg_masks = segment_EEG(EEG, sleep_stage, 30,200,start_end_remove_epoch_num=2, amplitude_thres=500, to_remove_mean=False)
    print(len(segs), len(sleep_stages))
    Fs = 200

    features_, feature_names = extract_features(segs, EEG_channels, Fs)#, pxxs, freqs


