import pdb
import numpy as np
import pandas as pd
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import scipy.stats as stats
from joblib import Parallel, delayed
#from pyeeg import samp_entropy
# import pyrem.univariate as pu
from multitaper_spectrogram import *
from bandpower import *

NW = 2
total_freq_range = [0.5,20]  # [Hz]
window_length = 2  # [s]
window_step = 1  # [s]
band_names = ['delta','theta','alpha','sigma']
band_freq = [[0.5,4],[4,8],[8,12],[12,20]]  # [Hz]
band_num = len(band_freq)
combined_channel_names = ['F','C','O']
combined_channel_num = len(combined_channel_names)

def corr(data,type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    w,v = np.linalg.eig(C)
    
    x = np.sort(w)
    x = np.real(x)
    return x


def compute_features_each_seg(eeg_seg, seg_size, channel_num, combined_channel_num, band_num, NW, window_length, window_step, Fs, band_freq, total_freq_range, segi, seg_num):
    # psd estimation, size=(window_num, freq_point_num, channel_num)
    # frequencies, size=(freq_point_num,)
    # try:
    spec_mt, freq = multitaper_spectrogram(eeg_seg, Fs, NW, window_length, window_step)
    spec_mt[:,:,:-1] = (spec_mt[:,:,:-1]+spec_mt[:,:,1:])/2.0
    spec_mt = np.delete(spec_mt,[1,3,5],axis=2)
    spect_mt_avg = np.mean(spec_mt, axis=2)
    # print(spect_mt_avg.shape)

    #total_findex = [i for i in range(len(freq)) if total_freq_range[0]<=freq[i]<total_freq_range[1]]

    # relative band power using multitaper
    bandpower_mt, band_findex = bandpower(spec_mt, freq, band_freq, total_freq_range=total_freq_range, relative=True)
    #bandpower_mt = [bandpower_mt[i].squeeze() for i in range(band_num)]

    f1 = np.abs(np.diff(eeg_seg,axis=0)).sum(axis=0)*1.0/seg_size  # mean gradient
    f2 = stats.kurtosis(eeg_seg,axis=0,nan_policy='raise')  # kurtosis
    f3 = []

    # for ci in range(channel_num):
    #     f3.append(pu.samp_entropy(eeg_seg[:,ci],2,0.2,relative_r=True))  # sample entropy


    f4 = []
    f5 = []
    f6 = []
    f7 = []
    #f8 = []
    f9 = []
    for bi in range(band_num):
        if bi!=band_num-1: # no need for sigma band
            f4.extend(np.percentile(bandpower_mt[bi],95,axis=0))
            f5.extend(bandpower_mt[bi].min(axis=0))
            f6.extend(bandpower_mt[bi].mean(axis=0))
            f7.extend(bandpower_mt[bi].std(axis=0))

        spec_ravel = spec_mt[:,band_findex[bi],:].reshape(spec_mt.shape[0]*len(band_findex[bi]),spec_mt.shape[2])
        ###############f8.extend(stats.skew(spec_ravel, axis=0, nan_policy='raise'))  # skewness
        f9.extend(stats.kurtosis(spec_ravel, axis=0, nan_policy='raise'))  # kurtosis


    #band_names = ['delta','theta','alpha','sigma']
    f10 = []
    delta_theta = bandpower_mt[0]/(bandpower_mt[1]+1)
    f10.extend(np.percentile(delta_theta,95,axis=0))
    f10.extend(np.min(delta_theta,axis=0))
    f10.extend(np.mean(delta_theta,axis=0))
    f10.extend(np.std(delta_theta,axis=0))
    f11 = []
    delta_alpha = bandpower_mt[0]/(bandpower_mt[2]+1)
    f11.extend(np.percentile(delta_alpha,95,axis=0))
    f11.extend(np.min(delta_alpha,axis=0))
    f11.extend(np.mean(delta_alpha,axis=0))
    f11.extend(np.std(delta_alpha,axis=0))
    f12 = []
    theta_alpha = bandpower_mt[1]/(bandpower_mt[2]+1)
    f12.extend(np.percentile(theta_alpha,95,axis=0))
    f12.extend(np.min(theta_alpha,axis=0))
    f12.extend(np.mean(theta_alpha,axis=0))
    f12.extend(np.std(theta_alpha,axis=0))


    ##  new set of features 
    feat = []
    epoch = eeg_seg
    [nt, nc] = eeg_seg.shape
    fs = Fs

    lvl = np.array([0.1, 4, 8, 12, 30, 70, 180])  # Frequency levels in Hz
    lseg = np.round(nt/fs*lvl).astype('int')
    D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
    D[0,:]=0                                # set the DC component to zero
    D /= D.sum()                      # Normalize each channel               

    dspect = np.zeros((len(lvl)-1,nc))
    for j in range(len(dspect)):
        dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:], axis=0)

    # Find the shannon's entropy
    spentropy = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)

    # Find the spectral edge frequency
    sfreq = fs
    tfreq = 40
    ppow = 0.5

    topfreq = int(round(nt/sfreq*tfreq))+1
    A = np.cumsum(D[:topfreq,:])
    B = A - (A.max()*ppow)
    spedge = np.min(np.abs(B))
    spedge = (spedge - 1)/(topfreq-1)*tfreq


    # Calculate correlation matrix and its eigenvalues (b/w channels)
    data = pd.DataFrame(data=epoch)
    type_corr = 'pearson'
    lxchannels = corr(data, type_corr)

    # Calculate correlation matrix and its eigenvalues (b/w freq)
    data = pd.DataFrame(data=dspect)
    lxfreqbands = corr(data, type_corr)

    # Spectral entropy for dyadic bands
    # Find number of dyadic levels
    ldat = int(floor(nt/2.0))
    no_levels = int(floor(log(ldat,2.0)))
    seg = floor(ldat/pow(2.0, no_levels-1))

    # Find the power spectrum at each dyadic level
    dspect = np.zeros((no_levels,nc))
    for j in range(no_levels-1,-1,-1):
        dspect[j,:] = 2*np.sum(D[int(floor(ldat/2.0))+1:ldat,:], axis=0)
        ldat = int(floor(ldat/2.0))

    # Find the Shannon's entropy
    spentropyDyd = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)

    # Find correlation between channels
    data = pd.DataFrame(data=dspect)
    lxchannelsDyd = corr(data, type_corr)

    # Fractal dimensions
    no_channels = nc

    # Hjorth parameters
    # Activity
    activity = np.var(epoch, axis=0)

    mobility = np.divide(
                        np.std(np.diff(epoch, axis=0)), 
                        np.std(epoch, axis=0))

    complexity = np.divide(np.divide(
                                    # std of second derivative for each channel
                                    np.std(np.diff(np.diff(epoch, axis=0), axis=0), axis=0),
                                    # std of second derivative for each channel
                                    np.std(np.diff(epoch, axis=0), axis=0))
                           , mobility)

    sk = skew(epoch)

    # Kurtosis
    kurt = kurtosis(epoch)

    feat = np.concatenate((feat,
                   spentropy.ravel(),
                   spedge.ravel(),
                   lxchannels.ravel(),
                   lxfreqbands.ravel(),
                   spentropyDyd.ravel(),
                   lxchannelsDyd.ravel(),
                   #fd.ravel(),
                   activity.ravel(),
                   mobility.ravel(),
                   complexity.ravel(),
                   sk.ravel(),
                   kurt.ravel()
                    ))
    # print(feat.shape)
    temp = np.r_[f1,f2,f3,f4,f5,f6,f7,f9,f10,f11,f12]
    # print(temp.shape)
    temp1 = np.r_[temp, feat]
    # print(temp1.shape, spec_mt.shape)
    # print([temp1, spec_mt].shape)
    # temp2 = np.array([temp1, spect_mt_avg])
    # print(temp2.shape)
    # np.save('temp1', temp1)
    # np.save('spec_mt', spect_mt_avg)
    return spect_mt_avg, temp1

    # except:
    #     return np.array([])
    # return np.r_[f1,f2,f3,f4,f5,f6,f7,f9,f10,f11,f12]#f8


def extract_features(EEG_segs, channel_names, Fs, return_feature_names=False, process_num=None):
    """Extract features from EEG segments.

    Arguments:
    EEG_segs -- a list of EEG segments in numpy.ndarray type, size=(sample_point, channel_num)
    channel_names -- a list of channel names for each column of EEG_segs
    ##combined_channel_names -- a list of combined column_channels_names, for example from 'F3M2' and 'F4M1' to 'F'
    Fs -- sampling frequency in Hz

    Keyword arguments:
    process_num -- default None, number of parallel processes, if None, equals to 4x #CPU.

    Outputs:
    features from each segment in numpy.ndarray type, size=(seg_num, feature_num)
    a list of names of each feature
    psd estimation, size=(window_num, freq_point_num, channel_num), or a list of them for each band
    frequencies, size=(freq_point_num,), or a list of them for each band
    """


    if type(EEG_segs)!=list:
        raise TypeError('EEG segments should be list of numpy.ndarray, with size=(sample_point, channel_num).')

    seg_num = len(EEG_segs)
    if seg_num <= 0:
        return []

    seg_size = EEG_segs[0].shape[0]
    channel_num = EEG_segs[0].shape[1]

    features= Parallel(n_jobs=16,verbose=2)(delayed(compute_features_each_seg)(EEG_segs[segi], seg_size, channel_num, combined_channel_num, band_num, NW, window_length, window_step, Fs, band_freq, total_freq_range, segi, seg_num) for segi in range(seg_num))
    # print(features.shape)
    



    if return_feature_names:
        feature_names = ['mean_gradient_%s'%chn for chn in channel_names]
        feature_names += ['kurtosis_%s'%chn for chn in channel_names]
        feature_names += ['sample_entropy_%s'%chn for chn in channel_names]
        for ffn in ['max','min','mean','std','kurtosis']:#,'skewness'
            for bn in band_names:
                if ffn=='kurtosis' or bn!='sigma': # no need for sigma band
                    feature_names += ['%s_bandpower_%s_%s'%(bn,ffn,chn) for chn in combined_channel_names]

        power_ratios = ['delta/theta','delta/alpha','theta/alpha']
        for pr in power_ratios:
            feature_names += ['%s_max_%s'%(pr,chn) for chn in combined_channel_names]
            feature_names += ['%s_min_%s'%(pr,chn) for chn in combined_channel_names]
            feature_names += ['%s_mean_%s'%(pr,chn) for chn in combined_channel_names]
            feature_names += ['%s_std_%s'%(pr,chn) for chn in combined_channel_names]

    if return_feature_names:
        return np.array(features), feature_names#, pxx_mts, freqs
    else:
        return np.array(features)#, pxx_mts, freqs

