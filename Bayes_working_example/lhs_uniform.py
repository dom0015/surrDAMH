#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:55:07 2018

@author: dom0015
"""

import numpy as np
import numpy.random
import scipy.stats

def lhs_norm(d,n):
    R=numpy.random.uniform(size=[n,d])
    P=np.zeros([n,d])
    
    for i in range(d):
        P[:,i] = numpy.random.permutation(n)
        
    return (P+R)/n


print(lhs_uniform(2,5))