'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: utils.py
- Contains some auxiliary functions for label normalisation [0, N-1], signal filtering,
  train-validation data division, and explanation signal plotting.


"Explaining ECG Biometrics: Is It All In The QRS?"
João Ribeiro Pinto and Jaime S. Cardoso
19th International Conference of the Biometrics Special Interest Group (BIOSIG 2020)

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def normalise_labels(X, y, N=None, key=None):
    # Converts class labels to [0, N-1] integers.
    # Can also select samples from the first N classes only.
    if key is None:
        if N is None:
            N = len(np.unique(y))
        key = {}
        classes = np.unique(y)
        for ii in range(N):
            key[classes[ii]] = ii
    
    X_norm = list()
    y_norm = list()
    for ss in range(len(y)):
        if y[ss] in key:
            X_norm.append(X[ss])
            y_norm.append(key[y[ss]])
    
    return np.array(X_norm), np.array(y_norm), key


def bandpass_filter(segment, fs, fc=[1, 40]):
    # Filters the signal with a butterworth bandpass filter with cutoff frequencies fc=[a, b]
    f0 = 2 * float(fc[0]) / float(fs)
    f1 = 2 * float(fc[1]) / float(fs)
    b, a = signal.butter(2, [f0, f1], btype='bandpass')
    return signal.filtfilt(b, a, segment)


def standard_normalisation(signal):
    # Returns signal with zero mean and unit variance
    return (signal - np.mean(signal)) / np.std(signal)


def stratified_train_validation_split(y, n_valid_per_class=1):
    # Divides the train dataset, assigning random n samples per class to the validation set.
    # n_valid_per_class can be >=1 (number of samples) or 0<n<1 (fraction of total identity samples)
    train_indices = list()
    valid_indices = list()
    for idd in np.unique(y):
        idd_indices = np.argwhere(y == idd)[:, 0]
        if n_valid_per_class >= 1:
            val_indices = np.random.choice(idd_indices, n_valid_per_class, replace=False)
        else:
            val_indices = np.random.choice(idd_indices, int(n_valid_per_class*len(idd_indices)), replace=False)
        for ii in idd_indices:
            if ii in val_indices:
                valid_indices.append(ii)
            else:
                train_indices.append(ii)
    return train_indices, valid_indices


def plot_signal_attr(fig, ax, attr, signal, fs=1.0, filter=True, lw=1.0):
    # Plots a signal with explanation strength as sample colors
    if filter:
        signal = bandpass_filter(signal, fs, fc=[1, 30])

    t = np.linspace(0, len(signal) / fs, len(signal))

    points = np.array([t, signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(attr.min(), attr.max())
    lc = LineCollection(segments, cmap='inferno_r', norm=norm)
    # Set the values used for colormapping
    lc.set_array(attr)
    lc.set_linewidth(lw)
    line = ax.add_collection(lc)

    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(signal.min(), signal.max())
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    

def ensemble_attrs(results, pred, peaks, identity):
    # Computes the average heartbeat signal and explanations for an identity
    ensemble_x = {}
    ensemble_sal = {}
    ensemble_occ = {}
    ensemble_igrad = {}
    ensemble_gshap = {}
    ensemble_dlift = {}
    indices = np.argwhere(pred[:,1] == int(identity))
    for rr in range(len(results)):
        ensemble_x[rr] = np.zeros((int(0.65 * 200.0),))
        ensemble_sal[rr] = np.zeros((int(0.65 * 200.0),))
        ensemble_occ[rr] = np.zeros((int(0.65 * 200.0),))
        ensemble_igrad[rr] = np.zeros((int(0.65 * 200.0),))
        ensemble_gshap[rr] = np.zeros((int(0.65 * 200.0),))
        ensemble_dlift[rr] = np.zeros((int(0.65 * 200.0),))
        for ii in indices[:,0]:
            for peak in peaks[ii]:
                try:
                    ensemble_x[rr] += results[rr]['x'][ii][int((peak - 0.25)*200.0):int((peak + 0.4)*200.0)]
                    ensemble_sal[rr] += results[rr]['sal'][ii][int((peak - 0.25)*200.0):int((peak + 0.4)*200.0)]
                    ensemble_occ[rr] += results[rr]['occ'][ii][int((peak - 0.25)*200.0):int((peak + 0.4)*200.0)]
                    ensemble_igrad[rr] += results[rr]['igrad'][ii][int((peak - 0.25)*200.0):int((peak + 0.4)*200.0)]
                    ensemble_gshap[rr] += results[rr]['gshap'][ii][int((peak - 0.25)*200.0):int((peak + 0.4)*200.0)]
                    ensemble_dlift[rr] += results[rr]['dlift'][ii][int((peak - 0.25)*200.0):int((peak + 0.4)*200.0)]
                except:
                    pass  # Ignoring incomplete heartbeats (at the start or end of segments)
        ensemble_x[rr] /= len(indices)
        ensemble_sal[rr] /= len(indices)
        ensemble_occ[rr] /= len(indices)
        ensemble_igrad[rr] /= len(indices)
        ensemble_gshap[rr] /= len(indices)   
        ensemble_dlift[rr] /= len(indices)              
    return ensemble_x, ensemble_sal, ensemble_occ, ensemble_igrad, ensemble_gshap, ensemble_dlift