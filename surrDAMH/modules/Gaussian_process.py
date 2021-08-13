#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:07:23 2021

@author: domesova
"""

import numpy as np
import scipy.linalg

# Gaussian random process
# zero mean expected

def autocorr_function_default(distance, corr_length):
    # Ornstein-Uhlenbeck covariance function
    return np.exp(-distance/corr_length)

def autocorr_function_sqexp(distance, corr_length):
    # squared exponential covariance function
    return np.exp(-(distance**2)/(2*corr_length**2))

def assemble_covariance_matrix(grid, parameters, cov_type):
    grid = np.array(grid)
    distances = np.abs(grid - grid.transpose())
    nP = len(parameters)
    blocks = [None]*nP
    for i in range(nP):
        P = parameters[i]
        corr_length = P[0]
        variance = P[1]**2
        if cov_type=="squared_exponential":
            blocks[i] = variance * autocorr_function_sqexp(distances,corr_length)
        else:
            blocks[i] = variance * autocorr_function_default(distances,corr_length)
    return scipy.linalg.block_diag(*blocks)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 100
    grid = np.linspace(0,365,N).reshape((N,1))
    parameters = [[30,166],[50,166],[50,500]]
    cov_type = "default"
    # cov_type = "squared_exponential"
    
    C = assemble_covariance_matrix(grid, parameters, cov_type)
    
    D,V = np.linalg.eig(C)
    D = np.real(D)
    D[D<0] = 0
    L = V*np.sqrt(D)
    
    plt.figure()
    np.random.seed(1)
    n = C.shape[0]
    for i in range(10):
        eta = np.random.randn(n).reshape((n,1))
        plt.plot(np.matmul(L,eta))
    plt.show()