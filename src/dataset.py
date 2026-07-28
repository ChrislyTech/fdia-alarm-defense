# -*- coding: utf-8 -*-
#dataset.py

import torch as t
import numpy as np
import scipy.io as sio
import os

def dataset(I):
    path = './TEdata'

    # train
    f=os.path.join(path, 'X.mat')
    train_data = sio.loadmat(f)['X']
    train_data=t.from_numpy(train_data).float()

    train_loader= train_data

    # val
    f = os.path.join(path, 'Xv.mat')
    val_data = sio.loadmat(f)['Xv']
    val_data = t.from_numpy(val_data).float()

    val_loader = val_data

    # test
    f = os.path.join(path, ''.join(('Xt', str(I), '.mat')))
    test_data = sio.loadmat(f)['Xt']
    test_data = t.from_numpy(test_data).float()

    test_loader = test_data

    return train_loader, val_loader, test_loader

