import unittest
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas
from concurrent.futures import ProcessPoolExecutor

from tests import resultsP863File, resultIdxRange
from musicaltones.p863 import runPOLQA

class P863CalcTestCase(unittest.TestCase):
    @staticmethod
    def _calculate_polqa(wavDeg: Path, wavRef: Path,
                         startTime: float, duration: float, nbrRanges: int,
                         chNbrRef=2, chNbrDeg=1,) -> "DataFrame":

        # calculation/average of multiple ranges
        dfPerFile = None
        try:
            for i in range(nbrRanges):
                res, _ = runPOLQA(wavDeg, wavRef, chNbrRef=chNbrRef, chNbrDeg=chNbrDeg,
                                  timeRangeStart=startTime + i * duration,
                                  timeRangeDuration=duration)
                res.name = i + 1
                if res.shape[0] > 0:
                    res = pandas.DataFrame(res).T
                    if dfPerFile is None:
                        dfPerFile = res
                    else:
                        dfPerFile = dfPerFile.combine_first(res)
        except:
            dfPerFile = None
        finally:
            return dfPerFile

    def test_calcP863(self, maxWorkers: int = 4):
        if not resultsP863File.is_file():
            self.assertFalse(True, 'Cannot find %s - please run test for generation of files first' % (resultsP863File.name))

        # analysis parameters for the ETSI test files (TODO: test files from other sources)
        duration = 8.0
        startTime = 16.0
        nbrRanges = 8

        # init storage and new column
        df = pandas.read_excel(resultsP863File, index_col=resultIdxRange)
        if not ('MOS-LQO' in df.columns):
            df['MOS-LQO'] = pandas.NA

        # calculate POLQA scores
        with ProcessPoolExecutor(max_workers=maxWorkers) as executor:

            # collect tasks and shuffle
            tasks = pandas.DataFrame(columns=['wavDeg', 'wavRef', 'startTime', 'duration', 'nbrRanges'])
            for i, (key, row) in enumerate(df.iterrows()):
                if pandas.isna(df.loc[key, 'MOS-LQO']) or (df.loc[key, 'MOS-LQO'] < 1.0):
                    tasks.loc[key] = (key, key, startTime, duration, nbrRanges)

            # shuffle
            tasks = tasks.sample(frac=1.0)

            # start tasks
            results = dict()
            for key, row in tasks.iterrows():
                results[key] = executor.submit(self._calculate_polqa, **row)

            # wait for tasks
            print(f"Waiting for {len(results)}/{df.shape[0]} items to complete...")
            for i, (key, futureResult) in enumerate(results.items()):
                print("[%d/%d] %s" % (i + 1, len(results), Path(key).name))
                e = futureResult.exception()
                if e is None:
                    dfPerFile = futureResult.result()

                    # store in output
                    if not dfPerFile is None:
                        df.loc[key, 'MOS-LQO'] = dfPerFile['MOS-LQO'].mean()
                        df.to_excel(resultsP863File)
                else:
                    df.loc[key, 'MOS-LQO'] = -1.0
                    print(str(e))

    def test_analyse_P863_results(self):
        # try to automatically select the four best noise reduction parameters that generate:
        # - equidistant MOS-LQO for anchoring (~1.0 / ~2.0 / ~3.0 / ~4.0 - 5.0/max is given by direct reference)
        # - most consistent results across language/source files

        if resultsP863File.is_file():
            # load
            df = pandas.read_excel(resultsP863File, index_col=resultIdxRange)
            df = df.dropna()
            df = df[df['MOS-LQO'] >= 1.0]

            # average across samples and languages
            df = df.groupby(['NFFT', 'Hop', 'SNR','OSF','TimeConst','PowExp']).agg({'MOS-LQO': ['mean', 'min', 'max', 'std']})

            # add delta column (max-min)
            df[('MOS-LQO', 'delta')] = df[('MOS-LQO', 'max')] - df[('MOS-LQO', 'min')]

            # evaluation criteria:
            # - delta and std (=multiplication) to be minimized -> score
            # - as close as possible to centre of bin
            df['score'] = df[('MOS-LQO', 'delta')] * df[('MOS-LQO', 'std')]
            # apply difference to centre
            df['score'] *= np.abs(df[('MOS-LQO', 'mean')] - df[('MOS-LQO', 'mean')].apply(round))

            # make histogram
            edges = np.arange(5) + 0.5 # "target MOS" in

            fig, ax = plt.subplots(1,1)
            ax.hist(df[('MOS-LQO', 'mean')], bins=edges)
            ax.set_xlabel('Avg. MOS-LQO')
            ax.set_ylabel('Count')
            ax.set_xlim([1, 5])
            ax.grid(True)
            fig.tight_layout()


            # iterate over edges
            for i in range(len(edges)-1):
                # filter items
                lower = edges[i]
                upper = edges[i+1]
                dfBin = df[(df[('MOS-LQO', 'mean')] >= lower) & (df[('MOS-LQO', 'mean')] < upper)]
                if dfBin.shape[0] > 0:

                    # sort and pick first
                    dfBin = dfBin.sort_values('score', ascending=True)
                    print(dfBin.iloc[0]['MOS-LQO'])

            plt.show()


if __name__ == '__main__':
    unittest.main()
