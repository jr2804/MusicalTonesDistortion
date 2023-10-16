# -*- coding: utf-8 -*-
"""
Created on Jun 23 2021 18:06

@author: Jan.Reimes
"""

from typing import Union
from enum import Enum
from scipy import signal

class P56Prefilter(Enum):
    """
    Enum for pre-filter according to clause 10 / Annex B / Annex C of ITU-T P.56
    """
    NoFilter = 'NoFilter'
    NB = 'NB'
    SWB = 'SWB'
    FB = 'FB'

PrefilterP56 = Union[P56Prefilter, str]

def applyFilters(x, coeffs):
    # apply b-a-coeffs sequentially (filter cascade)
    y = x.copy()
    for ba in coeffs:
        b, a = ba
        y = signal.lfilter(b, a, y)

    return y

def getFilter(fltType: PrefilterP56, fs):
    # IIR filter design of pre-filters of P.56
    fltType = P56Prefilter(fltType)
    filters = []

    if fltType == P56Prefilter.NB:
        order, wn = signal.buttord(wp=[160.0, 7000.0], ws=[16.0, 23999.0], gpass=0.25, gstop=51, fs=fs)
        b, a = signal.butter(order, Wn=wn, btype='bandpass', output='ba', analog=False, fs=fs)
        filters.append((b, a))
    elif fltType == P56Prefilter.SWB:
        order, wn = signal.buttord(wp=[50.0, 14000.0], ws=[16.0, 23999.0], gpass=0.25, gstop=26, fs=fs)
        b, a = signal.butter(order, Wn=wn, btype='bandpass', output='ba', analog=False, fs=fs)
        filters.append((b, a))
        filters.append((b, a))
    elif fltType == P56Prefilter.FB:
        order, wn = signal.buttord(wp=[20.0, 20000.0], ws=[9.0, 23999.0], gpass=0.25, gstop=17.5, fs=fs)
        b, a = signal.butter(order, Wn=wn, btype='bandpass', output='ba', analog=False, fs=fs)
        filters.append((b, a))
        filters.append((b, a))
        filters.append((b, a))
    else:
        filters.append(([1], [1]))

    return filters


if __name__ == "__main__":
    pass
