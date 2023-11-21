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
    if corr_length == 0:
        return np.eye(np.shape(distance)[0])
    # Ornstein-Uhlenbeck covariance function
    return np.exp(-distance/corr_length)


def autocorr_function_sqexp(distance, corr_length):
    if corr_length == 0:
        return np.eye(np.shape(distance)[0])
    # squared exponential covariance function
    return np.exp(-(distance**2)/(2*corr_length**2))


def assemble_covariance_matrix(list_block_params):
    blocks = []
    for b in list_block_params:
        grid = np.array(b["time_grid"]).reshape((1, -1))
        distances = np.abs(grid - grid.transpose())
        corr_length = b["corr_length"]
        std_list = b["std"]
        if type(std_list) is list:
            std = np.array(b["std"]).reshape((1, -1))
        else:
            std = std_list
        variance = std**2
        if "cov_type" not in b.keys():
            cov_type = None
        else:
            cov_type = b["cov_type"]

        if cov_type == "squared_exponential":
            block = variance * autocorr_function_sqexp(distances, corr_length)
        else:
            block = variance * autocorr_function_default(distances, corr_length)
        blocks.append(block)
    return scipy.linalg.block_diag(*blocks)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # N = 100
    # grid = np.linspace(0,365,N).reshape((N,1))
    # parameters = [[30,50],[30,166],[50,166]]
    N = 141
    grid = np.linspace(0, 140, 141).reshape((N, 1))

    conf_list = []
    # blok 1
    dict_01 = dict()
    dict_01["time_grid"] = grid
    dict_01["corr_length"] = 30
    dict_01["std"] = 50
    dict_01["cov_type"] = "default"
    # dict_01["cov_type"] = "squared_exponential"
    conf_list.append(dict_01)

    # blok 2
    dict_02 = dict_01.copy()
    dict_02["std"] = [50] * N
    conf_list.append(dict_02)
    # blok 3
    dict_03 = dict_01.copy()
    dict_03["corr_length"] = 0
    dict_03["std"] = 60
    conf_list.append(dict_03)
    # blok 4
    dict_04 = dict_01.copy()
    dict_04["time_grid"] = [141]
    dict_04["corr_length"] = 30
    dict_04["std"] = 70
    conf_list.append(dict_04)

    C2 = assemble_covariance_matrix(conf_list)
    print(np.shape(C2))

    C = C2[:N, :N]
    print(np.shape(C))
    D, V = np.linalg.eig(C)
    D = np.real(D)
    D[D < 0] = 0
    L = V*np.sqrt(D)

    plt.figure()
    np.random.seed(1)
    n = C.shape[0]
    for i in range(4):
        eta = np.random.randn(n).reshape((n, 1))
        plt.plot(grid, np.matmul(L, eta))
    plt.grid()
    plt.show()

    grid2 = ([0, 10, 17, 27, 37, 47, 57, 67, 77, 87, 97, 107, 117, 127, 137, 140])
    plt.figure()
    np.random.seed(1)
    n = C.shape[0]
    for i in range(4):
        eta = np.random.randn(n).reshape((n, 1))
        y = np.matmul(L, eta)
        plt.plot(grid2, y[grid2])
    plt.grid()
    plt.show()
