# -*- coding: utf-8 -*-
#utils.py

import numpy as np

def BICcom(q, J_Q, l):
    P_xN = np.exp(- q / J_Q)
    P_xF = np.exp(- J_Q / q)
    P_N = l
    P_F = 1 - l
    P_x = P_xN * P_N + P_xF * P_F
    P_Fx = P_xF * P_F / P_x
    return P_xF, P_Fx

def FDR(S, J, f):
    N = len(S)
    fdr = 0
    for i in range(f, N):
        if (S[i] > J):
            fdr += 1 / (N - f)
    return fdr

def FAR(S, J, f):
    far = 0
    for i in range(f):
        if (S[i] > J):
            far += 1 / f
    return far