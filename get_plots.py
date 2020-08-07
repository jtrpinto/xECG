'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: get_plots.py
- Takes interpretations from interpret_ptb.py or interpret_uoftdb.py and builds
  segment explanation plots and ensemble heartbeat plots.


"Explaining ECG Biometrics: Is It All In The QRS?"
João Ribeiro Pinto and Jaime S. Cardoso
19th International Conference of the Biometrics Special Interest Group (BIOSIG 2020)

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pickle as pk
from matplotlib import pyplot as pl
from utils import plot_signal_attr, ensemble_attrs


db = 'ptb'  # Change here to get results for the other database

if db == 'uoftdb':
    id_sets = ['2', '5', '10', '20', '50', '100', '200', '500', '1019']
else:
    id_sets = ['2', '5', '10', '20', '50', '100', '290']

results = list()
for ids in id_sets:
    with open(os.path.join('results', 'interp_' + db + '_' + ids + 's.pk'), 'rb') as hf:
        results.append(pk.load(hf))

with open('peak_locations/peaks_loc_' + db + '.pk', 'rb') as hf:
    peaks = pk.load(hf)

pred = np.loadtxt(os.path.join('results', 'test_' + db + '_2s.csv'), delimiter=',')


# Plot segments ===============================================================

indices = np.argwhere(np.array(pred[:,1]) < 2)[:,0]
for idx in indices:  # index of the segment to be plotted
    fig, ax = pl.subplots(len(id_sets), 4, figsize=(12, 0.8*len(id_sets)), sharex=True, sharey=True)
    ax[0,0].set_title('Occlusion')
    ax[0,1].set_title('Saliency')
    ax[0,2].set_title('Gradient SHAP')
    ax[0,3].set_title('DeepLIFT')
    for rr in range(ax.shape[0]):
        ax[rr,0].set_ylabel(id_sets[rr] + ' id.')
        for cc in range(ax.shape[1]):
            for peak in peaks[idx]:
                ax[rr, cc].plot([peak, peak], [-10, 10], color='k', alpha=0.2)
        plot_signal_attr(fig, ax[rr, 0], results[rr]['occ'][idx], results[rr]['x'][idx], fs=200.0, lw=1.0)
        plot_signal_attr(fig, ax[rr, 1], results[rr]['sal'][idx], results[rr]['x'][idx], fs=200.0, lw=1.0)
        plot_signal_attr(fig, ax[rr, 2], results[rr]['gshap'][idx], results[rr]['x'][idx], fs=200.0, lw=1.0)
        plot_signal_attr(fig, ax[rr, 3], results[rr]['dlift'][idx], results[rr]['x'][idx], fs=200.0, lw=1.0)
    pl.subplots_adjust(wspace=0.05, hspace=0.1)
    pl.savefig(os.path.join('plots', db + '_segment' + str(idx) + '_id' + str(int(pred[idx, 1])) + '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    pl.close()

    
# Plot heartbeats =============================================================
    
for identity in ['0', '1']:  # identity to evaluate
    ensemble_x, ensemble_sal, ensemble_occ, ensemble_igrad, ensemble_gshap, ensemble_dlift = ensemble_attrs(results, pred, peaks, identity)
    fig, ax = pl.subplots(4, len(id_sets), figsize=(12, 3.5), sharex=True, sharey=True)
    ax[0,0].set_ylabel('Occlusion')
    ax[1,0].set_ylabel('Saliency')
    ax[2,0].set_ylabel('Gr.SHAP')
    ax[3,0].set_ylabel('DeepLIFT')
    for cc in range(ax.shape[1]):
        ax[0, cc].set_title(id_sets[cc] + ' id.')
        plot_signal_attr(fig, ax[0, cc], ensemble_occ[cc], ensemble_x[cc], fs=200.0, lw=1.5)
        plot_signal_attr(fig, ax[1, cc], ensemble_sal[cc], ensemble_x[cc], fs=200.0, lw=1.5)
        plot_signal_attr(fig, ax[2, cc], ensemble_gshap[cc], ensemble_x[cc], fs=200.0, lw=1.5)
        plot_signal_attr(fig, ax[3, cc], ensemble_dlift[cc], ensemble_x[cc], fs=200.0, lw=1.5)
    pl.subplots_adjust(wspace=0.05, hspace=0.1)
    pl.savefig(os.path.join('plots', db + '_heartbeat_id' + str(identity) + '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    pl.close()