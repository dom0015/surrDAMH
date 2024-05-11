#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:55:07 2018

@author: dom0015
"""

import numpy as np
import numpy.matlib as matlib
import numpy.typing as npt
from scipy.stats import norm


def lhs_normal(loc:  npt.ArrayLike, scale:  npt.NDArray | float = 1.0,
               n: int = 10, seed: int = 0):
    """
    Returns a Latin hypercube sample of size n from normal distribution
    with independent components N(loc,scale).

    Args:
        loc (np.ndarray): mean vector of length d
        scale (np.ndarray | float): standard deviation vector (or scalar, if constant)
        n (int): size of LHS sample (i.e. number of generated points from R^d)
        seed (int): random generator seed
    """
    loc = np.array(loc)
    no_parameters = len(loc)
    LHS_final = np.zeros([n, n])
    maxmin = 0
    RS = np.random.RandomState(seed)
    for i in range(5):
        R = RS.uniform(size=[n, no_parameters])
        P = np.zeros([n, no_parameters])

        for i in range(no_parameters):
            P[:, i] = RS.permutation(n)

        LHS_uni = (P+R)/n

        distances = np.zeros([n, n])
        for j in range(no_parameters):
            temp = matlib.repmat(np.reshape(LHS_uni[:, j], (1, n)), n, 1)-matlib.repmat(np.reshape(LHS_uni[:, j], (n, 1)), 1, n)
            distances = distances + np.multiply(temp, temp)

        quality = np.min(distances+np.eye(n)*1000)
        if quality > maxmin:
            LHS_final = LHS_uni

    LHS_norm = norm.ppf(LHS_final, loc=loc, scale=scale)

    return LHS_norm
