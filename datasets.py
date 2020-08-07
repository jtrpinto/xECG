'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: dataset.py
- Contains the Dataset class used for training and testing with PyTorch.


"Explaining ECG Biometrics: Is It All In The QRS?"
João Ribeiro Pinto and Jaime S. Cardoso
19th International Conference of the Biometrics Special Interest Group (BIOSIG 2020)

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import torch
import numpy as np
import pickle as pk
from scipy.signal import resample
from torch.utils.data import Dataset
from utils import standard_normalisation


def get_subset(X, y, N=None):
    # Selects samples from the first N classes only
    if N is None:
        N = len(np.unique(y))
    indices = np.argwhere(y < N)
    X_norm = X[indices, :]
    y_norm = y[indices]
    return X_norm, np.array(y_norm)


class Dataset(Dataset):

    def __init__(self, data_file, fs, dataset='train', n_ids=None):
        with open(data_file, 'rb') as handle:
            data = pk.load(handle)
        if dataset == 'train' or dataset == 'validation':
            self.X, self.y = get_subset(data['X_train'], data['y_train'], N=n_ids)
            _, self.counts = np.unique(self.y, return_counts=True) 
            self.weights = np.array([1.0/i for i in self.counts])
        elif dataset == 'test':
            self.X, self.y = get_subset(data['X_test'], data['y_test'], N=n_ids)
            _, self.counts = np.unique(self.y, return_counts=True) 
            self.weights = np.array([1.0/i for i in self.counts])
        else:
            raise ValueError('Variable dataset must be \'train\', \'validation\', or \'test\'.')
        self.data_augment = dataset == 'train'
        self.fs = fs

    def __random_augment__(self, sample):
        # Applies a random augmentation procedure among a list of options
        functions = [self.__random_permutation__]
        idx = np.random.randint(0, high=len(functions))
        new_sample = functions[idx](sample)
        new_sample = standard_normalisation(new_sample)
        return new_sample

    def __random_permutation__(self, sample, split=5):
        # Applies random permutation of signal subsegments
        segments = np.array_split(sample, split)
        np.random.shuffle(segments)
        new_sample = np.concatenate(segments)
        return new_sample

    def __getitem__(self, index):
        x = self.X[index, 0, :]
        if self.data_augment:
            x = self.__random_augment__(x).reshape((1, self.X.shape[2]))
        else:
            x = x.reshape((1, self.X.shape[2]))
        x = x.astype(float)
        y = self.y[index, 0]
        return (x, y)

    def __len__(self):
        return self.X.shape[0]