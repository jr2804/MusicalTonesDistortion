# -*- coding: utf-8 -*-
"""
Created on Mar 18 2021 09:51

@author: Jan.Reimes

Important note:

This test can only be run with a valid license for ITU-T Rec. P.863
(aka POLQA, which is a commercial product from Opticom GmbH).
See https://polqa.info for further information

The following files must be located in the current subfolder <p863>
and MUST NOT be distributed in this repository:
PolqaOem64.dll
PolqaOemDemo64.exe
vcomp120.dll
"""

import subprocess
from enum import IntEnum
from pathlib import Path
import pandas
import re
import soundfile as sf
import tempfile, uuid

binPath = Path(__file__).parent
polqaExe = binPath / 'PolqaOemDemo64.exe'

class POLQAVersion(IntEnum):
    V1_1 = 1
    V2_4 = 2
    V3_0 = 3

def _createTmpCopy(wavFile, chNbr, timeRangeStart=0.0, timeRangeDuration=-1.0):
    tmpFile = Path(tempfile.gettempdir()) / Path("tmpPOLQACalc_%s.wav" % uuid.uuid4())
    s, fs = sf.read(wavFile)
    idxStart = int(fs * timeRangeStart)
    idxEnd = -1
    if timeRangeDuration > 0:
        idxEnd = idxStart + int(fs * timeRangeDuration)
    sf.write(tmpFile, s[idxStart:idxEnd,chNbr-1], fs, format='WAV', subtype='PCM_16')
    return tmpFile

def runPOLQA(wavFileDeg, wavFileRef, version=POLQAVersion.V3_0, highAccuracyMode=True,
             chNbrDeg=1, chNbrRef=1, timeRangeStart=0.0, timeRangeDuration=-1.0):

    # always copy to temp files
    tmpFiles = []
    wavFileDeg = _createTmpCopy(wavFileDeg, chNbrDeg, timeRangeStart, timeRangeDuration)
    tmpFiles.append(wavFileDeg)

    wavFileRef = _createTmpCopy(wavFileRef, chNbrRef, timeRangeStart, timeRangeDuration)
    tmpFiles.append(wavFileRef)

    version = POLQAVersion(version)

    cmdLineArgs = [str(polqaExe), '-LC SWB']
    if highAccuracyMode:
        cmdLineArgs += ['-EnableHaMode']
    cmdLineArgs += ["-Version %d" % (version.value)]
    cmdLineArgs += ['-Test "%s"' % wavFileDeg]
    cmdLineArgs += ['-Ref "%s"' % wavFileRef]

    res = subprocess.run(" ".join(cmdLineArgs), shell=True, capture_output=True)
    results = pandas.Series(dtype=float)
    warnings = []

    # parse results and warnings
    if res.returncode == 0:
        for resTitle in ['MOS-LQO', 'AVG  Delay', 'SNR Degraded', 'SNR Reference']:
            m = re.search(b"%s: ([+-]?\d+(?:\.\d+)?)" % (resTitle.encode('ascii')), res.stdout)
            if m:
                results[resTitle] = float(m.group(1).strip())

        for warn in re.findall(b"POLQA WARNING (.*)\r", res.stdout):
            warnings.append(warn.strip().decode('ascii'))

    # clean temp files
    for tmpFile in tmpFiles:
        if tmpFile.is_file():
            tmpFile.unlink()

    return results, warnings

if __name__ == "__main__":
    pass
