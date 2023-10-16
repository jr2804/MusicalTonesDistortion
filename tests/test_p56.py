import unittest
import soundfile as sf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from tests.data import downloadETSITestFile, TestFilesETSI
from musicaltones.p56.asl import calculateP56ASL, calculateP56ASLEx, getFilter, applyFilters

FS = 48000
x = np.random.randn(20*FS)

class P56TestCase(unittest.TestCase):
    showPlots: bool = True

    def test_p56_asl(self):
        # run P.56 ASL calculation on all three test files, should result in ~-26 dBov for each file
        for tstFile in TestFilesETSI:
            with self.subTest(testFile=tstFile.value):
                self.testFile = downloadETSITestFile(tstFile, proxy="https://127.0.0.1:3128")
                # load signal
                s, fs = sf.read(self.testFile)

                # test default P.56
                asl, act = calculateP56ASL(s, fs)
                self.assertAlmostEqual(asl, -26.0, delta=0.1)

                # test extended P.56 with signal check
                for scale in [0.001, 5.0]:
                    offset = 20*np.log10(scale)
                    asl, act = calculateP56ASLEx(s*scale, fs)
                    self.assertAlmostEqual(asl, -26.0+offset, delta=0.11)

    def _getTransferFunction(self, x, y, fs, N = 32768):
        wargs = dict(nperseg=N, noverlap=N*3/4, nfft=N, scaling='spectrum')
        freq, X = signal.welch(x, fs, **wargs)
        _, Y = signal.welch(y, fs, **wargs)
        H = 10 * np.log10(np.maximum(np.abs(Y) / np.abs(X), 1e-12))
        return freq, H

    def _plotTransferFunction(self, freq, H, title: str):
        plt.semilogx(freq, H)
        plt.title(f'Butterworth filter frequency response ({title})')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.grid(which='both', axis='both')
        plt.xlim([2, 22000])
        plt.ylim([-60, 5])
        plt.show()

    def test_p56_prefilter_nb(self):
        coeffs = getFilter('NB', fs=FS)
        y = applyFilters(x, coeffs)
        freq, H = self._getTransferFunction(x,y,FS)

        # check upper tolerances
        for f, limit in {16: -49.75, 160: 0.25, 7000: 0.25, 70000: -49.75}.items():
            idxF = np.argmin(np.abs(freq-f))
            self.assertLess(H[idxF], limit)

        # check lower tolerances
        for f, limit in {200: -0.25, 5500: -0.25}.items():
            idxF = np.argmin(np.abs(freq-f))
            self.assertGreater(H[idxF], limit)

        if self.showPlots:
            self._plotTransferFunction(freq, H, 'NB')

    def test_p56_prefilter_swb(self):
        coeffs = getFilter('SWB', fs=48000)
        y = applyFilters(x, coeffs)
        freq, H = self._getTransferFunction(x,y,FS)

        # check upper tolerances
        for f, limit in {16: -49.75, 50: 0.25, 14000: 0.25, 70000: -49.75}.items():
            idxF = np.argmin(np.abs(freq - f))
            self.assertLess(H[idxF], limit)

        # check lower tolerances
        for f, limit in {70: -0.25, 12000: -0.25}.items():
            idxF = np.argmin(np.abs(freq - f))
            self.assertGreater(H[idxF], limit)

        if self.showPlots:
            self._plotTransferFunction(freq, H, 'SWB')

    def test_p56_prefilter_fb(self):
        coeffs = getFilter('FB', fs=FS)

        y = applyFilters(x, coeffs)
        freq, H = self._getTransferFunction(x,y,FS)

        # check upper tolerances
        for f, limit in {9: -49.75, 20: 0.25, 20000: 0.25, 70000: -49.75}.items():
            idxF = np.argmin(np.abs(freq - f))
            self.assertLess(H[idxF], limit, msg=f'frequency = {f:.0f}')

        # check lower tolerances
        for f, limit in {30: -0.25, 18000: -0.25}.items():
            idxF = np.argmin(np.abs(freq - f))
            self.assertGreater(H[idxF], limit)

        if self.showPlots:
            self._plotTransferFunction(freq, H, 'FB')

if __name__ == '__main__':
    unittest.main()
