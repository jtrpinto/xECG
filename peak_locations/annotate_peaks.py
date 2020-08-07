'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: annotate_peaks.py
- Used to manually annotate the locations of R-peaks in PTB and UofTDB signals.


"Explaining ECG Biometrics: Is It All In The QRS?"
João Ribeiro Pinto and Jaime S. Cardoso
19th International Conference of the Biometrics Special Interest Group (BIOSIG 2020)

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import pickle as pk
from matplotlib import pyplot as pl
import numpy as np
from utils import bandpass_filter


db = 'ptb'  # or 'uoftdb'

with open('../results/interp_' + db + '_2s.pk', 'rb') as hf:
    results = pk.load(hf)

class PeakLabeller():
    def __init__(self):
        self.peaks = {}
        self.idx = 0
        self.max_idx = len(results['x'])
        for ii in range(self.max_idx):
            self.peaks[ii] = list()

    def getPeaks(self):
        fig = pl.figure()
        x = bandpass_filter(results['x'][self.idx], 200.0, [1, 30])
        pl.plot(np.linspace(0, len(x), len(x)), x)
        fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        fig.canvas.mpl_connect('close_event', self.__onClose__)
        pl.show()
        return 0
    
    def __onClose__(self, evt):
        if self.idx < self.max_idx - 1:
            self.idx += 1
            self.getPeaks()

    def __onclick__(self, click):
        r = int(click.xdata)
        r_m = np.argmax(results['x'][self.idx][r-10:r+10])
        r_m += r - 10
        self.peaks[self.idx].append(r_m / 200.0)
        print('click at', r_m/200.0, results['x'][self.idx][r_m-1:r_m+2])
    
a = PeakLabeller()
a.getPeaks() 


#%% Run this cell after peak annotation

peaks = a.peaks

with open('peaks_loc_' + db + '.pk', 'wb') as hf:
    pk.dump(peaks, hf)
