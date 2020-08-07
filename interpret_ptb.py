'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: interpret_ptb.py
- Runs a model trained using train_ptb.py, and obtains prediction explanations for
  the samples of identities #1 and #2, using several interpretability tools from
  Captum for PyTorch.


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
from captum.attr import Saliency, GradientShap, Occlusion, DeepLift, IntegratedGradients


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

DSET_FILE = '/ctm-hdd-pool01/jtrp/xECG/ptb_data.pk'  # Pickle file obtained with prepare_data.py
FS = 200.0                                           # Data sampling frequency

MODEL = "models/ptb_290s"  # Weights of the model to be tested (without .pth)

N_IDS = int(MODEL[11:-1])  # Number of identities (getting ot from filename, but you can change this)


print(' ========= Interpreting ' + MODEL + ' ========= ')

# Building datasets (uses the classkey generated during training)
test_set = Dataset(DSET_FILE, FS, dataset='test', n_ids=N_IDS)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=4)

dataiter = iter(test_loader)
signals, labels = dataiter.next()


# PREPARING THE MODEL =============================================================================

model = Model(N=N_IDS, dropout=0.0).to(DEVICE)
model.load_state_dict(torch.load(MODEL + '.pth', map_location=DEVICE))
model = model.to(DEVICE)


# INTERPRETABILITY ===============================================================================

grads_sal = list()
grads_igrad = list()
grads_occ = list()
grads_gshap = list()
grads_dlift = list()
signal = list()

for idx in range(36):
    x = signals[idx].float().unsqueeze(0)
    x.requires_grad = True

    model.eval()

    # Saliency
    saliency = Saliency(model)
    grads = saliency.attribute(x, target=labels[idx].item())
    grads_sal.append(grads.squeeze().cpu().detach().numpy())

    # Occlusion
    occlusion = Occlusion(model)
    grads = occlusion.attribute(x, strides = (1, int(FS / 100)), target=labels[idx].item(), sliding_window_shapes=(1, int(FS / 10)), baselines=0)
    grads_occ.append(grads.squeeze().cpu().detach().numpy())

    # Integrated Gradients
    integrated_gradients = IntegratedGradients(model)
    grads = integrated_gradients.attribute(x, target=labels[idx].item(), n_steps=1000)
    grads_igrad.append(grads.squeeze().cpu().detach().numpy())

    # Gradient SHAP
    gshap = GradientShap(model)
    baseline_dist = torch.cat([x*0, x*1])
    grads = gshap.attribute(x, n_samples=10, stdevs=0.1, baselines=baseline_dist, target=labels[idx].item())
    grads_gshap.append(grads.squeeze().cpu().detach().numpy())

    # DeepLIFT
    dlift = DeepLift(model)
    grads = dlift.attribute(x, x*0, target=labels[idx].item())
    grads_dlift.append(grads.squeeze().cpu().detach().numpy())

    signal.append(x.squeeze().cpu().detach().numpy())

with open(os.path.join('results', 'interp_' + os.path.basename(MODEL) + '.pk'), 'wb') as hf:
    pk.dump({'x': signal, 'sal': grads_sal, 'occ': grads_occ, 'igrad': grads_igrad, 'gshap': grads_gshap, 'dlift': grads_dlift}, hf)




