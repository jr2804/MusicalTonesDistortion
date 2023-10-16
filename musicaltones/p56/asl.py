# -*- coding: utf-8 -*-
"""
Created on Jun 23 2021 13:04

@author: Jan.Reimes
"""

import numpy as np
from numba import jit
from scipy.signal import lfilter

from .prefilter import P56Prefilter, PrefilterP56, getFilter, applyFilters

class ASLException(Exception):
    pass

@jit(nopython=True)
def __bin_interp(upcount, lwcount, upthr, lwthr, Margin, tol):
    tol = np.abs(tol)

    # check if extreme counts are not already the true active value
    iterno = 1
    if abs(upcount - upthr - Margin) < tol:
        asl_ms_log = upcount
        cc = upthr
        return asl_ms_log, cc
    elif abs(lwcount - lwthr - Margin) < tol:
        asl_ms_log = lwcount
        cc = lwthr
        return asl_ms_log, cc
    else:
        # Initialize first middle for given (initial) bounds
        midcount = (upcount + lwcount) / 2.0
        midthr = (upthr + lwthr) / 2.0
        # Repeats loop until `diff' falls inside the tolerance (-tol<=diff<=tol)
        while 1:
            diff = midcount - midthr - Margin
            if np.abs(diff) <= tol:
                break
            # if tolerance is not met up to 20 iteractions, then relax the tolerance by 10%
            iterno = iterno + 1
            if iterno > 20:
                tol = tol * 1.1
            if diff > tol:  # then new bounds are ...
                midcount = (upcount + midcount) / 2.0
                # upper and middle activities
                midthr = (upthr + midthr) / 2.0
                # ... and thresholds
            elif diff < -tol:  # then new bounds are ...
                midcount = (midcount + lwcount) / 2.0
                # middle and lower activities
                midthr = (midthr + lwthr) / 2.0
                # ... and thresholds

        # Since the tolerance has been satisfied, midcount is selected
        # as the interpolated value with a tol [dB] tolerance.
        asl_ms_log = midcount
        cc = midthr

        return asl_ms_log, cc

def calculateP56ASL(x, fs, nbits=16, M = 15.9, H = 0.2, T = 0.03):
    '''
    This implements ITU P.56 method B.
    Usage:  asl_P56(x, fs, nbits)
        x             - the column vector of floating point speech data
        fs            - the sampling frequency
        nbits         - the number of bits (default: 16)
        M             - margin in dB of the difference between threshold and active speech level (default: 15.9)
        H             - hangover time in seconds (default: 0.2)
        T             - time constant of smoothing, in seconds (default:0.3
    Example call:
        asl_rms, asl, c0 = asl_P56(x, fs, nbits)
    References:
    [1] ITU-T (1993). Objective measurement of active speech level. ITU-T
        Recommendation P. 56
    Python implementation from MATLAB: Rui Cheng
    '''

    thres_no = nbits - 1  # number of thresholds, for 16 bit, it's 15
    eps = 2.2204e-16

    I = int(np.ceil(fs * H))  # hangover in samples
    g = np.exp(-1 / (fs * T))  # smoothing factor in enevlop detection
    c = np.array([pow(2, i) for i in range(-thres_no, thres_no - nbits + 1)])
    # vector with thresholds from one quantizing level up to half the maximum code, at a step of 2, in the case of 16bit samples, from 2^-15 to 0.5
    a = np.zeros(thres_no, dtype=int) # activity counter for each level threshold
    hang = I + np.zeros(thres_no, dtype=int)  # % hangover counter for each level threshold

    sq = np.sum(np.power(x, 2))  # long-term level square energy of x
    x_len = len(x)  # length of x

    # use a 2nd order IIR filter to detect the envelope q
    x_abs = np.abs(x)
    p = lfilter([1 - g], [1, -g], x_abs)
    q = lfilter([1 - g], [1, -g], p)

    @jit(nopython=True)
    def __getActivity(q, c, a, hang, thres_no, x_len):
        for k in range(x_len):
            for j in range(thres_no):
                if q[k] >= c[j]:
                    a[j] = a[j] + 1
                    hang[j] = 0
                elif hang[j] < I:
                    a[j] = a[j] + 1
                    hang[j] = hang[j] + 1
                else:
                    break

        return a, hang

    a, hang = __getActivity(q, c, a, hang, thres_no, x_len)

    # default result values
    activity = 0
    asl_dB = -100

    if a[0] == 0:
        raise ASLException('Could not detect any activity')
    else:
        AdB1 = 10 * np.log10(sq / a[0] + eps)

    CdB1 = 20 * np.log10(c[0] + eps)
    if AdB1 - CdB1 < M:
        msg = 'No frame above margin M=%.1f dB detected' % (float(M))
        print(msg)
        #raise ASLException(msg)
    else:
        AdB = np.zeros(thres_no) # [0 for i in range(thres_no)]
        CdB = np.zeros(thres_no) # [0 for i in range(thres_no)]
        Delta = np.zeros(thres_no) # [0 for i in range(thres_no)]
        AdB[0] = AdB1
        CdB[0] = CdB1
        Delta[0] = AdB1 - CdB1

        for j in range(1, thres_no):
            AdB[j] = 10 * np.log10(sq / (a[j] + eps) + eps)
            CdB[j] = 20 * np.log10(c[j] + eps)

        for j in range(1, thres_no):
            if a[j] != 0:
                Delta[j] = AdB[j] - CdB[j]
                if Delta[j] <= M:  # M = 15.9
                    # interpolate to find the asl
                    asl_dB, cl0 = __bin_interp(AdB[j], AdB[j - 1], CdB[j], CdB[j - 1], M, 0.5)
                    asl_lin = np.power(10, asl_dB / 10)
                    activity = (sq / x_len) / asl_lin
                    #c0 = pow(10, cl0 / 20)
                    break

    return asl_dB, activity

def calculateP56ASLEx(x, fs, preFilter: PrefilterP56='NoFilter', minAmplitude=0.1, maxAmplitude=1.0, **kwargs):
    # call calculateP56ASL() with additional pre-filter and range check of signal:
    x = np.array(x)
    maxAbsValue = np.abs(x).max()

    # apply pre-filter, if applicable
    preFilter = P56Prefilter(preFilter)
    if preFilter != P56Prefilter.NoFilter:
        coeffs = getFilter(preFilter, fs)
        y = applyFilters(x, coeffs)
    else:
        y = x

    # if max. amplitude is larger than maxAbsValue or smaller than minAbsValue, rescale signal
    if (maxAbsValue > maxAmplitude) or (maxAbsValue < minAmplitude):
        y = y / maxAbsValue
        offset_dB = 20*np.log10(maxAbsValue)
    else:
        offset_dB = 0.0

    asl, act = calculateP56ASL(y, fs, **kwargs)

    # compensate for scaling
    return asl+offset_dB, act

if __name__ == "__main__":
    pass
