# -*- coding: utf-8 -*-
#model.py

import torch as t
from torch import nn
from torch.nn import functional as F

class AE(nn.Module):
    def __init__(self,opt):
        super(AE, self).__init__()

        # Encoder
        self.fcE1 = nn.Linear(opt.x_size, opt.h_size)
        self.bnE1 = nn.BatchNorm1d(opt.h_size, 1e-6)
        self.bnH = nn.BatchNorm1d(opt.h_size, affine=False)

        # Latent space
        self.fcU1 = nn.Linear(opt.h_size, opt.h_size)
        self.U1 = self.fcU1._parameters['weight'][:]

        # Decoder
        self.fcH = nn.Linear(opt.h_size, opt.h_size)
        self.bnD1 = nn.BatchNorm1d(opt.h_size, 1e-6)
        self.fcD1 = nn.Linear(opt.h_size, opt.x_size)

    def encoder(self, x):
        h = self.bnH(F.relu(self.bnE1(self.fcE1(x))))
        return h

    def latent(self, h, opt):
        V1 = t.triu(self.U1)
        A1 = V1 - V1.t()
        P = (t.eye(opt.h_size) - A1).mm(t.inverse(t.eye(opt.h_size) + A1))
        W = P[:,0:opt.T_size]
        T = h.mm(W)
        return T, W

    def decoder(self, T, W):
        hp = T.mm(W.t())
        xp = self.fcD1(self.bnD1(F.relu(self.fcH(hp))))
        return xp, hp

    def forward(self, x, opt):
        h = self.encoder(x)
        T, W = self.latent(h, opt)
        xp, hp = self.decoder(T, W)
        return xp, T, h, hp, W
