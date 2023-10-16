# -*- coding: utf-8 -*-
"""
Created on Oct 16 2023 16:55

@author: Jan.Reimes
"""

from pathlib import Path
from enum import Enum
import requests

dataPath = Path(__file__).parent
__baseURL = r'https://docbox.etsi.org/STQ/Open/TS%20103%20281%20Wave%20files/Annex_E%20speech%20data/'

# Test data from public ETSI server
class TestFilesETSI(Enum):
    German = 'German_P835_16_sentences_4convergence.wav'
    English = 'American_P835_16_sentences_4convergence.wav'
    Mandarin = 'FBMandarin_QCETSI_26dB.wav'


def downloadFile(fileURL, destination, proxy=None):
    s = requests.Session()
    try:
        if proxy:
            s.proxies = {"https": proxy, "http": proxy}

        r = s.get(fileURL)
        with open(destination, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    finally:
        s.close()

def downloadETSITestFile(file: TestFilesETSI, forceOverwrite=False, proxy=None):
    file = TestFilesETSI(file)
    fullURL = __baseURL + file.value
    targetFile = dataPath / file.value
    if not targetFile.is_file() or forceOverwrite:
        downloadFile(fullURL, targetFile, proxy=proxy)
    return targetFile

if __name__ == "__main__":
    pass

