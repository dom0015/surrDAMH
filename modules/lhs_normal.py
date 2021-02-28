#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:55:07 2018

@author: dom0015
"""

import numpy as np
import numpy.random
import numpy.matlib
from scipy.stats import norm

def lhs_normal(no_parameters,prior_mean,prior_std,n,seed):
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
            temp = numpy.matlib.repmat(np.reshape(LHS_uni[:,j],(1,n)),n,1)-numpy.matlib.repmat(np.reshape(LHS_uni[:,j],(n,1)),1,n)
            distances = distances + np.multiply(temp,temp)

        quality = np.min(distances+np.eye(n)*1000)
        if quality > maxmin:
            LHS_final = LHS_uni
    
    # TO DO: target covariance matrix
    if np.array(prior_std).ndim == 2:
        prior_std = np.sqrt(np.diag(prior_std))
    LHS_norm = norm.ppf(LHS_final, loc=prior_mean, scale=prior_std)
    
    return LHS_norm
