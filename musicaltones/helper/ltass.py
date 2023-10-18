# -*- coding: utf-8 -*-
"""
Created on Jan 05 2023 10:31

@author: Jan.Reimes

Calculation of long-term average speech spectrum
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import freqz
from warnings import warn

from .coeffs import getCoeffsP50, FS

def ltassP50FB(freq, targetLevelDbPa=-4.7, **kwargs):
    minDb = kwargs.get('minDb', -100.0)
    b, a = getCoeffsP50()
    f, S = freqz(b, a, freq, fs=FS)
    S = 20*np.log10(np.maximum(S, np.power(10, minDb/20)))

    # scale to target level
    levelDbPa = 10 * np.log10(np.sum(np.power(10, S / 10) / freq.shape[0]))
    diff = targetLevelDbPa - levelDbPa
    return S + diff

def ltassP50(freq, freq_lower=None, freq_upper=None, fmin=100, fmax=8000, targetLevelDbPa=-4.7):
    warn('Function ltassP50() is deprecated - works only up to 8 kHz', DeprecationWarning, stacklevel=2)

    freq = np.maximum(freq, 1.0) # avoid f=0
    if freq_lower is None:
        freq_lower = freq[0]

    if freq_upper is None:
        freq_upper = freq[-1]

    # Spectral density, ITU-T P.50, clause 4.1, equation 4-1
    Sd = -376.44 + 465.439* np.log10(freq) - 157.745 * np.log10(freq)**2 + 16.7124 * np.log10(freq)**3
    Sd -= 94.0 # formula is defined for SPL?!

    # extrapolate for f < 100 and f > 8kHz (P.50 is not defined there)
    idxMin = np.argmin(np.abs(freq-fmin))
    idxMax = np.argmin(np.abs(freq-fmax))+1
    SdI = interp1d(np.log10(freq[idxMin:idxMax]), Sd[idxMin:idxMax], fill_value="extrapolate")
    Sd_i = SdI(np.log10(freq))

    delta_f = freq_upper - freq_lower
    corr_f = 10*np.log10(delta_f)
    S = Sd_i + corr_f

    # scale to target level
    levelDbPa = 10 * np.log10(np.sum(np.power(10, S / 10) / freq.shape[0]))
    diff = targetLevelDbPa - levelDbPa
    return S + diff


if __name__ == "__main__":
    pass
