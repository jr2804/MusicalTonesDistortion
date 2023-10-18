# -*- coding: utf-8 -*-
"""
Created on Oct 16 2023 17:06

@author: Jan.Reimes
"""

import numpy as np
from scipy.signal import stft, istft

from . import SNR_MAX, SNR_MIN, NoiseType
from .p56.asl import calculateP56ASLEx
from .helper.noise import getNoise

def spectralSubtractionAlgo(s: np.ndarray, n: np.ndarray, fs: int, k: float, **kwargs):
    # parse arguments
    window = kwargs.get('window', 'hann')
    overlap = kwargs.get('overlap', 1-0.4)
    n_fft = kwargs.get('n_fft', round(0.025*fs))
    noverlap = round(overlap*n_fft)
    epsilon = kwargs.get('epsilon', 0.0)
    a: float = kwargs.get('a', 1)
    epsilon = np.maximum(epsilon, np.power(10, -90/(a*10)))

    # speech
    freq, time, X = stft(s+n, fs, window,  nperseg=n_fft, noverlap=noverlap, nfft=n_fft,)
    absXseg = np.power(np.abs(X), a)
    phiXseg = np.angle(X)

    # noise estimate
    _, _, N = stft(n, fs, window,  nperseg=n_fft, noverlap=noverlap, nfft=n_fft,)
    absNseg = np.power(np.abs(N), a)
    absWhatSeg = np.atleast_2d(np.mean(absNseg, axis=1)).T
    absWhatSeg = np.repeat(absWhatSeg, time.shape[0], axis=1)

    absYseg = np.maximum(absXseg - (k*absWhatSeg), epsilon)

    _, y = istft(np.power(absYseg, 1/a)*np.exp(1j*phiXseg), fs, window, n_fft, noverlap, n_fft)
    return y


def applySpecSub(signal: np.ndarray, fs: int, k: float, snr: float, speechLevel: float = None, **kwargs) -> np.ndarray:

    # check arguments
    if k < 0:
        raise ValueError(f"Parameter k must be >= 0!")

    if (snr < SNR_MIN) or (snr > SNR_MAX):
        raise ValueError(f"Parameter SNR must be between {SNR_MIN} and {SNR_MAX}")

    # parse arguments
    noiseType = kwargs.pop('noiseType', NoiseType.white)
    seed = kwargs.pop('seed', 280480)
    fmin = kwargs.pop('fmin', 20.0) # for pink noise
    speechLevelProc = kwargs.pop('speechLevelProc', -26.0)
    prefilterP56 = kwargs.pop('prefilterP56', 'FB')

    # calculate per channel
    signal = signal if len(signal.shape) > 1 else np.atleast_2d(signal).T
    output = np.zeros_like(signal)
    size = output.shape[0]
    for i in range(signal.shape[1]):
        # calibrate speech level
        speechLevelCurrent, _ = calculateP56ASLEx(signal[:,i], fs, prefilterP56)
        diff = speechLevelProc - speechLevelCurrent
        output[:,i] = signal[:, i].copy() * np.power(10, diff/20)

        # generate noise and scale acc. to SNR
        noise = getNoise(size, fs, noiseType, seed=seed, fmin=fmin, levelDb=snr - speechLevelProc)

        # run spectral subtraction
        sHat_plus = spectralSubtractionAlgo(output[:,i], noise, fs, k, **kwargs)
        sHat_minus = spectralSubtractionAlgo(output[:,i], -noise, fs, k, **kwargs)
        sHat = sHat_plus + sHat_minus
        L = np.minimum(sHat.shape[0], size)
        output[:L, i] = sHat[:L]

        # recalibrate after spectral subtraction
        speechLevelProcessed, _ = calculateP56ASLEx(output[:,i], fs, prefilterP56)
        if speechLevel is None:
            diff = speechLevelCurrent - speechLevelProcessed
        else:
            diff = speechLevel - speechLevelProcessed

        output[:,i] *= np.power(10, diff/20)

    return output.squeeze()

if __name__ == "__main__":
    pass
