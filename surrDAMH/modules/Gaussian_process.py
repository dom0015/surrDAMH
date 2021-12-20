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
    grid = np.array(grid).reshape((1,-1))
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
    # N = 100
    # grid = np.linspace(0,365,N).reshape((N,1))
    # parameters = [[30,50],[30,166],[50,166]]
    N=141
    grid=np.linspace(0,140,141).reshape((N,1))
    parameters = [[30,50]]
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
    for i in range(4):
        eta = np.random.randn(n).reshape((n,1))
        plt.plot(grid,np.matmul(L,eta))
    plt.show()
    plt.grid()
    
    grid2=([0, 10, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117, 127, 137, 140])
    plt.figure()
    np.random.seed(1)
    n = C.shape[0]
    for i in range(4):
        eta = np.random.randn(n).reshape((n,1))
        y=np.matmul(L,eta)
        plt.plot(grid2,y[grid2])
    plt.show()
    plt.grid()