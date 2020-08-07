'''
xECG Project Repository (https://github.com/jtrpinto/xECG)

File: trainers.py
- Contains the model training routine. Used by train_model_ptb.py and
  train_model_uoftdb.py.


"Explaining ECG Biometrics: Is It All In The QRS?"
João Ribeiro Pinto and Jaime S. Cardoso
19th International Conference of the Biometrics Special Interest Group (BIOSIG 2020)

joao.t.pinto@inesctec.pt  |  https://jtrpinto.github.io
'''

import torch
import numpy as np
import sys
import pickle


def train_model(model, loss_fn, optimiser, train_loader, n_epochs, device, patience=np.inf, valid_loader=None, filename=None):
    # repeat training for the desired number of epochs
    train_hist = []
    train_idr = []
    valid_hist = []
    valid_idr = []

    # For early stopping:
    plateau = 0  
    best_valid_loss = None

    print(model)

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
    
        # training loop
        model.train()  # set model to training mode (affects dropout and batch norm.)
        for i, (X, y) in enumerate(train_loader):
            # copy the mini-batch to GPU
            X = X.float().to(device)
            y = y.to(device)
            
            ypred = model(X)          # forward pass
            loss = loss_fn(ypred, y)  # compute the loss
            optimiser.zero_grad()     # set all gradients to zero (otherwise they are accumulated)
            loss.backward()           # backward pass (i.e. compute gradients)
            optimiser.step()          # update the parameters
        
            # display the mini-batch loss
            sys.stdout.write("\r" + '........mini-batch no. {} loss: {:.4f}'.format(i+1, loss.item()))
            sys.stdout.flush()
        
            if torch.isnan(loss):
                print('NaN loss. Terminating train.')
                return [], []

        # compute the training and validation losses to monitor the training progress (optional)
        print()
        with torch.no_grad():  # now we are doing inference only, so we do not need gradients
            model.eval()       # set model to inference mode (affects dropout and batch norm.)
        
            train_loss = 0.
            t_corrects = 0
            t_total = 0
            for i, (X, y) in enumerate(train_loader):
                # copy the mini-batch to GPU
                X = X.float().to(device)
                y = y.to(device)
                
                ypred = model(X)                 # forward pass
                train_loss += loss_fn(ypred, y)  # accumulate the loss of the mini-batch
                t_corrects += (torch.argmax(ypred, 1) == y).float().sum()
                t_total += y.shape[0]
            train_loss /= i + 1
            train_hist.append(train_loss.item())
            t_idr = t_corrects / t_total
            train_idr.append(t_idr)
            print('....train loss: {:.4f} :: IDR {:.4f}'.format(train_loss.item(), t_idr))
        
            if valid_loader is None:
                print()
                continue
        
            valid_loss = 0.
            v_corrects = 0
            v_total = 0
            for i, (X, y) in enumerate(valid_loader):
                # copy the mini-batch to GPU
                X = X.float().to(device)
                y = y.to(device)
            
                ypred = model(X)                 # forward pass
                valid_loss += loss_fn(ypred, y)  # accumulate the loss of the mini-batch
                v_corrects += (torch.argmax(ypred, 1) == y).float().sum()
                v_total += y.shape[0]
            valid_loss /= i + 1
            valid_hist.append(valid_loss.item())
            v_idr = v_corrects / v_total
            valid_idr.append(v_idr)
            print('....valid loss: {:.4f} :: IDR {:.4f}'.format(valid_loss.item(), v_idr))

        if best_valid_loss is None:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), filename + '.pth')
            with open(filename + '_trainhist.pk', 'wb') as hf:
                pickle.dump({'loss': train_hist, 'idr': train_idr}, hf)
            with open(filename + '_validhist.pk', 'wb') as hf:
                pickle.dump({'loss': valid_hist, 'idr': valid_idr}, hf)
            print('....Saving...')
        elif valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), filename + '.pth')
            with open(filename + '_trainhist.pk', 'wb') as hf:
                pickle.dump({'loss': train_hist, 'idr': train_idr}, hf)
            with open(filename + '_validhist.pk', 'wb') as hf:
                pickle.dump({'loss': valid_hist, 'idr': valid_idr}, hf)
            plateau = 0
            print('....Saving...')
        else:
            plateau += 1
            if plateau >= patience:
                print('....Early stopping the train.')
                return train_hist, valid_hist

    return train_hist, valid_hist