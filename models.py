'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: model.py
- Contains the Model class that includes the PyTorch implementation
  of the convolutional neural network used for biometric identification. 


"Explaining ECG Biometrics: Is It All In The QRS?"
João Ribeiro Pinto and Jaime S. Cardoso
19th International Conference of the Biometrics Special Interest Group (BIOSIG 2020)

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, N=100, dropout=.0):
        super(Model, self).__init__()
        fd = 108
        self.convnet = nn.Sequential(nn.Conv1d(1, 24, 5, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool1d(5),
                                     nn.Conv1d(24, 24, 5, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool1d(5),
                                     nn.Conv1d(24, 36, 5, stride=1, padding=0),
                                     nn.ReLU(),
                                     nn.MaxPool1d(5),
                                     nn.Conv1d(36, 36, 5, stride=1, padding=0),
                                     nn.ReLU()
                                    )

        self.fc = nn.Sequential(nn.Linear(fd, 100),
                                nn.ReLU(),
                                nn.Dropout(p=dropout),
                                nn.Linear(100, N)
                               )

    def forward(self, x):
        h = self.convnet(x)
        h = h.view(h.size()[0], -1)
        output = self.fc(h)
        return output
    
    def predict(self, X):
        logits = self.forward(X)
        probs = F.softmax(logits, dim=1)
        return probs