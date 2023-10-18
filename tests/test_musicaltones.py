import unittest
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List
import soundfile as sf
from soxr import resample
#from concurrent.futures import ProcessPoolExecutor
from more_executors import Executors
from functools import partial

from tests import thisPath, resultsP863File, resultColumns, resultIndices, resultIdxRange
from tests.data import downloadETSITestFile, TestFilesETSI
from musicaltones.addMusicalNoise import applySpecSub, NoiseType
from musicaltones.p56.asl import calculateP56ASLEx
from musicaltones.helper import FS

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

class MusicalTonesTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = {
            'speechLevel': [-26],
            'n_fft': [400, 1200],
            #'snr': [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15,],
            'snr': [-30, 0, 15,],
            'k': [0.1, 0.5, 1.0], #np.arange(0.5, 5, step=1),
            #'noiseType': [nt.name for nt in NoiseType],
            'noiseType': ['white'],
                          }
        cls.parameters = pd.MultiIndex.from_product(list(cls.parameters.values()), names=cls.parameters.keys())

        cls.outputPath = thisPath / 'output'
        cls.outputPath.mkdir(exist_ok=True)
        cls.testFile = None

    @staticmethod
    def _process_sequence(s: np.ndarray, fs: int, outputFile: Path,
                          **kwargs) -> None:

        d = applySpecSub(s, fs, **kwargs)

        # store degraded and reference in one file
        signal = np.vstack((d, s)).T

        # use 16-bit (neeed for POLQA testing)
        sf.write(outputFile, signal, fs, subtype='PCM_16') #, format='FLAC')

    def test_specsub(self, maxWorkers: int = None, fs: int = FS):
        # run for all test signals
        testFiles = []
        for tstFile in TestFilesETSI:
            with self.subTest(testFile=tstFile.value):
                testFile = downloadETSITestFile(tstFile, proxy="http://127.0.0.1:3128")
                testFiles.append(testFile)

        # run synchroneous in debug mode:
        if debugger_is_active():
            PoolExecutor = partial(Executors.sync)
        else:
            maxWorkers = os.cpu_count() // 2 if maxWorkers is None else maxWorkers
            PoolExecutor = partial(Executors.process_pool, max_workers=maxWorkers)

        # storage for generated files
        df = pd.DataFrame(columns=resultColumns + resultIndices).set_index(resultIndices)
        if resultsP863File.is_file():
            dfOrig = pd.read_excel(resultsP863File, index_col=resultIdxRange)
        else:
            dfOrig = None

        # generate all samples via multiprocessing
        with PoolExecutor() as executor:
            # start tasks
            results = dict()
            for testFile in testFiles:
                # load & resample signal
                s, fs1 = sf.read(testFile)
                if fs1 != fs:
                    s = resample(s, fs1, fs)

                # iterate over internal pseudo-noise-reduction parameters:
                for p in self.parameters:
                    procArgs = dict(zip(self.parameters.names, p))
                    title = "_".join([f"{k}={v}" for k, v in procArgs.items()])

                    # run with given settings
                    outputFile = self.outputPath / Path(f'processed_{testFile.stem}_{title}.wav')

                    if not outputFile.is_file():
                        results[outputFile] = executor.submit(MusicalTonesTestCase._process_sequence, s, fs, outputFile,
                                                              **procArgs)

                    # store information for P.863 calculation in other unit test
                    key = str(outputFile)
                    if key not in df.index:
                        mos = -1.0
                        if (dfOrig is not None) and (key in dfOrig.index):
                            mos = dfOrig.loc[key, 'MOS-LQO']

                        df.loc[key, 'MOS-LQO'] = mos
                        df.loc[key, 'SourceFile'] = testFile.stem
                        df.loc[key, self.parameters.names] = p

            # wait for tasks
            nbrItems = df.shape[0]
            print(f"Waiting for {len(results)}/{nbrItems} items to complete...")
            for i, (key, futureResult) in enumerate(results.items()):
                with self.subTest(output=key):
                    e = futureResult.exception()
                    if e is None:
                        dfProc = futureResult.result()

                        # store in output
                        if dfProc is not None:
                            df = df.combine_first(dfProc)
                            df.to_excel(resultsP863File)
                    else:
                        print(str(e))

                    self.assertTrue(e is None, msg=str(e))

        # final check: remove all rows where the file does not exist
        remove = []
        for key, row in df.iterrows():
            if not Path(key).is_file():
                remove.append(key)

        df = df.drop(labels=remove)

        df.to_excel(resultsP863File)


if __name__ == '__main__':
    unittest.main()
