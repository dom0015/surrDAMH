#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:55:07 2018

@author: dom0015
"""

import numpy as np
import numpy.random
from scipy.stats import norm

def lhs_norm(Model,n,seed):
    no_parameters = Model.no_parameters
    priorMean = Model.priorMean
    LHS_final = np.zeros([n,n])
    maxmin = 0
    RS = np.random.RandomState(seed)
    for i in range(5):
        R=RS.uniform(size=[n,no_parameters])
        P=np.zeros([n,no_parameters])
        
        for i in range(no_parameters):
            P[:,i] = RS.permutation(n)
        
        LHS_uni = (P+R)/n
        
        distances = np.zeros([n,n])
        for j in range(no_parameters):
            temp = np.matlib.repmat(np.reshape(LHS_uni[:,j],(1,n)),n,1)-np.matlib.repmat(np.reshape(LHS_uni[:,j],(n,1)),1,n)
            distances = distances + np.multiply(temp,temp)

        quality = np.min(distances+np.eye(n)*1000)
        if quality > maxmin:
            LHS_final = LHS_uni

    priorStd = Model.priorStd
    if priorStd.size > no_parameters:
        priorStd = np.diag(priorStd)
    LHS_norm = norm.ppf(LHS_final, loc=priorMean, scale=priorStd)
    
    return LHS_norm
