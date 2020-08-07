'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: test_model_uoftdb.py
- Takes a model trained with train_model_uoftdb.py and runs it on the test set,
  printing the test loss and IDR (identification rate) results, and saving a
  .csv file on the results directory.


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
from models import Model
from datasets import Dataset


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

DSET_FILE = '/ctm-hdd-pool01/jtrp/xECG/uoftdb_data.pk'  # Pickle file obtained with prepare_data.py
FS = 200.0                                              # Data sampling frequency

MODEL = "models/uoftdb_2s"  # Weights of the model to be tested (without .pth)

N_IDS = int(MODEL[14:-1])   # Number of identities (getting it from filename, but you can change this)

# Building datasets
test_set = Dataset(DSET_FILE, FS, dataset='test', n_ids=N_IDS)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)


# PREPARING THE MODEL =============================================================================

model = Model(N=N_IDS, dropout=0.0).to(DEVICE)
model.load_state_dict(torch.load(MODEL + '.pth', map_location=DEVICE))
model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

# TESTING THE MODEL ==============================================================================

print('\n ======= TEST MODEL ' + MODEL + ' ======= \n')

results = list()

model.eval()
with torch.no_grad():
    test_loss = 0.
    t_corrects = 0
    t_total = 0
    for i, (X, y) in enumerate(test_loader):
        # copy the mini-batch to GPU
        X = X.float().to(DEVICE)
        y = y.to(DEVICE)
    
        ypred = model(X)                # forward pass
        test_loss += loss_fn(ypred, y)  # accumulate the loss of the mini-batch
        t_corrects += (torch.argmax(ypred, 1) == y).float().sum()
        t_total += y.shape[0]
        results.append(str(i) + ',' + str(y.cpu().numpy().item()) + ',' + str(torch.argmax(ypred, 1).cpu().numpy().item()) + ',' + str(np.max(nn.functional.softmax(ypred, dim=1).cpu().numpy())) + '\n')
    test_loss /= i + 1
    t_idr = t_corrects / t_total
    print('....test loss: {:.4f} :: IDR {:.4f}'.format(test_loss.item(), t_idr))


with open(os.path.join('results', 'test_' + os.path.basename(MODEL) + '.csv'), 'w') as g:
    for line in results:
        g.write(line)
