# -*- coding: utf-8 -*-
"""
Created on Jan 04 2023 11:59

@author: Jan.Reimes
"""

from scipy.signal import welch
import numpy as np

DB_MIN = -100
DB_MIN_LIN = np.power(10, DB_MIN/10)
N_FFT = 8192
OVERLAP = 0.75
N_STEP = round((1-OVERLAP)*N_FFT)

def getSpectrumDb(s, fs, nperseg=N_FFT, noverlap=N_STEP):
    freq, S = welch(s, fs, 'hann', nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
    S = 10 * np.log10(np.maximum(S, DB_MIN_LIN))

    return freq, S

if __name__ == "__main__":
    pass
