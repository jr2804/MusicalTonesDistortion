# -*- coding: utf-8 -*-
"""
Created on Oct 16 2023 17:21

@author: Jan.Reimes
"""
from enum import IntEnum, auto
import colorednoise as cn
import numpy as np
from scipy.signal import filtfilt, butter, sosfiltfilt

from .coeffs import getCoeffsP50

class NoiseType(IntEnum):
    white = auto()
    pink = auto()
    p50fb = auto()


def getNoise(size: int, fs: int, noiseType: NoiseType, levelDb: float = 0.0, seed: int = 280480, **kwargs) -> np.ndarray:
    noiseType = NoiseType(noiseType) if isinstance(noiseType, int) else NoiseType[noiseType]
    fmin = kwargs.get('fmin', 20.0)
    hpOrder = kwargs.get('hpOrder', 6)
    dtype = kwargs.get('dtype', np.float32)

    if noiseType == NoiseType.pink:
        n = cn.powerlaw_psd_gaussian(1, size, fmin=0.5*fmin/fs, random_state=seed)
    elif noiseType in [NoiseType.white, NoiseType.p50fb]:
        rng = np.random.default_rng(seed)
        n = rng.normal(0, 1.0, size)
        if noiseType == NoiseType.p50fb:
            b, a = getCoeffsP50()
            n = filtfilt(b, a, n)
        else:
            sos = butter(hpOrder, fmin, 'highpass', fs=fs, output='sos')
            n = sosfiltfilt(sos, n)

    else:
        raise ValueError(f"unknown noise type: {noiseType}")

    # scale to level
    levelDbCurrent = 10*np.log10(np.mean(np.power(n,2)))
    diff = levelDb - levelDbCurrent
    n *= np.power(10, diff/20)
    return n.astype(dtype)


if __name__ == "__main__":
    pass
