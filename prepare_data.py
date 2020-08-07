'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: prepare_data.py
- Takes raw data from the UofTDB and PTB databases and prepares it to be used
  in this project, storing prepared data in uoftdb_data.pk and ptb_data.pk.


"Explaining ECG Biometrics: Is It All In The QRS?"
João Ribeiro Pinto and Jaime S. Cardoso
19th International Conference of the Biometrics Special Interest Group (BIOSIG 2020)

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import os
import sys
import glob
import warnings
import numpy as np
import pickle as pk
from scipy import io, signal
from utils import standard_normalisation, stratified_train_validation_split, normalise_labels


UOFTDB_DATA_DIR = 'UofTDB/txt'
PTB_DATA_DIR = 'PTB'

# Original sampling frequencies
FS_UOFTDB = 200.0
FS_PTB = 1000.0

SEGMENT_LENGTH = 5     # duration of each enrollment segment (seconds)
SKIP = SEGMENT_LENGTH  # separation between the start of the current and the start of the next segment 

TEST_SPLIT = 0.5       # fraction of data to reserve for testing 


# UofTDB ===============================================================

print('Preparing UofTDB data...')

X = list()
y = list()

counts = {'train': {}, 'test': {}}

files = sorted(os.listdir(UOFTDB_DATA_DIR))
for ff in files:
    subject_id = ff.split('_')[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.loadtxt(os.path.join(UOFTDB_DATA_DIR, ff))
    if len(data) > 0:
        for end_cut in range(SEGMENT_LENGTH, int(len(data)/FS_UOFTDB), SKIP):
            start = int((end_cut - SEGMENT_LENGTH)*FS_UOFTDB)
            end = int(end_cut*FS_UOFTDB)
            segment = standard_normalisation(data[start:end])
            X.append(segment)
            y.append(subject_id)
    sys.stdout.write("\r" + 'file ' + ff + '      ')
    sys.stdout.flush()

print()

train_indices, test_indices = stratified_train_validation_split(np.array(y).reshape(-1,1), n_valid_per_class=TEST_SPLIT)

X_train = np.array(X)[train_indices, :]
X_test = np.array(X)[test_indices, :]
y_train = np.array(y)[train_indices]
y_test = np.array(y)[test_indices]

X_train, y_train, key = normalise_labels(X_train, y_train)
X_test, y_test, _ = normalise_labels(X_test, y_test, key=key)

cs = np.unique(y_train, return_counts=True)
for ii in range(len(cs[0])):
    counts['train'][cs[0][ii]] = cs[1][ii]

cs = np.unique(y_test, return_counts=True)
for ii in range(len(cs[0])):
    counts['test'][cs[0][ii]] = cs[1][ii]

print('UOFTDB X_train', X_train.shape)
print('UOFTDB X_test ', X_test.shape)

with open('uoftdb_data.pk', 'wb') as hf:
    pk.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'counts': counts, 'key': key}, hf)

del X, X_train, X_test, y, y_train, y_test, counts, files


# PTB ===============================================================

print('Preparing PTB data...')

X = list()
y = list()

counts = {'train': {}, 'test': {}}

files = sorted(glob.glob(os.path.join(PTB_DATA_DIR, "*.mat")))
for ff in files:
    subject_id = os.path.basename(ff)[7:10]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = io.loadmat(ff)['val'][0].astype(np.float32)
    if len(data) > 0:
        for end_cut in range(SEGMENT_LENGTH, int(len(data)/FS_PTB), SKIP):
            start = int((end_cut - SEGMENT_LENGTH)*FS_PTB)
            end = int(end_cut*FS_PTB)
            segment = signal.resample(data[start:end], int(200.0 * SEGMENT_LENGTH))
            segment = standard_normalisation(segment)
            X.append(segment)
            y.append(subject_id)
    sys.stdout.write("\r" + 'file ' + ff + '      ')
    sys.stdout.flush()

print()

train_indices, test_indices = stratified_train_validation_split(np.array(y).reshape(-1,1), n_valid_per_class=TEST_SPLIT)

X_train = np.array(X)[train_indices, :]
X_test = np.array(X)[test_indices, :]
y_train = np.array(y)[train_indices]
y_test = np.array(y)[test_indices]

X_train, y_train, key = normalise_labels(X_train, y_train)
X_test, y_test, _ = normalise_labels(X_test, y_test, key=key)

cs = np.unique(y_train, return_counts=True)
for ii in range(len(cs[0])):
    counts['train'][cs[0][ii]] = cs[1][ii]

cs = np.unique(y_test, return_counts=True)
for ii in range(len(cs[0])):
    counts['test'][cs[0][ii]] = cs[1][ii]

print('PTB X_train', X_train.shape)
print('PTB X_test ', X_test.shape)

with open('ptb_data.pk', 'wb') as hf:
    pk.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'counts': counts, 'key': key}, hf)