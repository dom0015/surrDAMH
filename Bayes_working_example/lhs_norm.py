#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:55:07 2018

@author: dom0015
"""

import numpy as np
import numpy.random
from scipy.stats import norm

def lhs_norm(d,n,priorMean,priorStd):
    LHS_final = np.zeros([n,n])
    maxmin = 0
    for i in range(5):
        R=numpy.random.uniform(size=[n,d])
        P=np.zeros([n,d])
        
        for i in range(d):
            P[:,i] = numpy.random.permutation(n)
        
        LHS_uni = (P+R)/n
        
        distances = np.zeros([n,n])
        for j in range(d):
            temp = np.matlib.repmat(np.reshape(LHS_uni[:,j],(1,n)),n,1)-np.matlib.repmat(np.reshape(LHS_uni[:,j],(n,1)),1,n)
            distances = distances + np.multiply(temp,temp)

        quality = np.min(distances+np.eye(n)*1000)
        if quality > maxmin:
            LHS_final = LHS_uni
        
    LHS_norm = norm.ppf(LHS_final, loc=priorMean, scale=priorStd)
    
    return LHS_norm
