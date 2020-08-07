'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: train_model_ptb.py
- Uses data from prepare_data.py and the Model class from models.py to train a model
  for biometric identification on the PTB database. The training routine can be found
  at trainers.py.


"Explaining ECG Biometrics: Is It All In The QRS?"
João Ribeiro Pinto and Jaime S. Cardoso
19th International Conference of the Biometrics Special Interest Group (BIOSIG 2020)

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import os
import torch
import numpy as np
import pickle as pk
from torch import nn
from torch import optim
from models import Model
from trainers import train_model
from torchvision import transforms
from datasets import Dataset
from utils import stratified_train_validation_split


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

DSET_FILE = '/ctm-hdd-pool01/jtrp/xECG/ptb_data.pk'  # Pickle file obtained with prepare_data.py
FS = 200.0                                           # Data sampling frequency
N_IDS = 2                                            # Number of identities

SAVE_MODEL = "models/ptb_" + str(N_IDS) + "s"  # Where to save the model

N_EPOCHS = 2500           # number of training epochs
BATCH_SIZE = N_IDS * 2    # number of samples to get from the dataset at each iteration
VALID_SPLIT = 0.1         # number of enrollment samples per subject to be used for validation
PATIENCE = 100            # for early stopping

DROPOUT = 0.2     
LEARN_RATE = 1e-3  
REG = 0   


# Building datasets
train_set = Dataset(DSET_FILE, FS, dataset='train', n_ids=N_IDS)
valid_set = Dataset(DSET_FILE, FS, dataset='validation', n_ids=N_IDS)


# creating data indices for training and validation splits
train_indices, valid_indices = stratified_train_validation_split(train_set.y, n_valid_per_class=VALID_SPLIT)

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=4,
                                           sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=4,
                                           sampler=valid_sampler)


# TRAINING THE MODEL ==============================================================================

print('\n ======= TRAINING MODEL ' + SAVE_MODEL + ' ======= \n')

model = Model(N=N_IDS, dropout=DROPOUT).to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=REG)

out = train_model(model, loss_fn, optimiser, train_loader, N_EPOCHS, DEVICE, patience=PATIENCE, valid_loader=valid_loader, filename=SAVE_MODEL)


# TESTING =========================================================================================

model.load_state_dict(torch.load(SAVE_MODEL + '.pth', map_location=DEVICE))
model = model.to(DEVICE)

test_set = Dataset(DSET_FILE, FS, dataset='test', n_ids=N_IDS)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

print('\n ======= TEST MODEL ' + SAVE_MODEL + ' ======= \n')

model.eval()
with torch.no_grad():
    test_loss = 0.
    t_corrects = 0
    t_total = 0
    for i, (X, y) in enumerate(test_loader):
        # copy the mini-batch to GPU
        X = X.float().to(DEVICE)
        y = y.to(DEVICE)
    
        ypred = model(X)                 # forward pass
        test_loss += loss_fn(ypred, y)  # accumulate the loss of the mini-batch
        t_corrects += (torch.argmax(ypred, 1) == y).float().sum()
        t_total += y.shape[0]
    test_loss /= i + 1
    t_idr = t_corrects / t_total
    print('....test loss: {:.4f} :: IDR {:.4f}'.format(test_loss.item(), t_idr))