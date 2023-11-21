#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.matlib
import scipy.sparse.linalg as splin


class Surrogate_apply:  # initiated by all SAMPLERs
    def __init__(self, no_parameters, no_observations, kernel_type="polyharmonic", beta=3):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.kernel_type = kernel_type
        self.beta = beta
        self.alldata_par = None

    def apply(self, SOL, datapoints):
        COEFS, self.alldata_par = SOL
        no_snapshots = self.alldata_par.shape[0]
        datapoints = datapoints.reshape(-1, self.no_parameters)
        no_datapoints = datapoints.shape[0]

        TEMP = np.zeros([no_datapoints, no_snapshots])
        for i in range(self.no_parameters):
            v = self.alldata_par[:, i]
            M = np.matlib.repmat(v, no_datapoints, 1)
            v_new = datapoints[:, i]
            M_new = np.matlib.repmat(v_new, no_snapshots, 1)
            T = np.transpose(M_new)
            TEMP = TEMP + np.power(M-T, 2)
        np.sqrt(TEMP, out=TEMP)
        kernel(TEMP, self.kernel_type, self.beta)
        no_polynomials = 1 + self.no_parameters  # plus linear polynomials
        GS_datapoints = np.matmul(TEMP, COEFS[:-no_polynomials])
        pp = COEFS[-1] + np.matmul(datapoints, COEFS[-self.no_parameters-1:-1])
        GS_datapoints = np.reshape(GS_datapoints, (no_datapoints, self.no_observations)
                                   ) + np.reshape(pp, (no_datapoints, self.no_observations))
        return GS_datapoints


class Surrogate_update:  # initiated by COLLECTOR
    def __init__(self, no_parameters, no_observations, initial_iteration=None, max_centers=None, expensive=False,
                 kernel_type="polyharmonic", beta=3, solver_tol_exp=-6, solver_type='minres'):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.initial_iteration = initial_iteration
        self.max_centers = max_centers
        self.expensive = expensive
        self.kernel_type = kernel_type
        self.beta = beta
        self.solver_tol_exp = solver_tol_exp
        self.solver_type = solver_type
        # all processed data (used for surrogate construction):
        self.processed_par = np.empty((0, self.no_parameters))
        self.processed_obs = np.empty((0, self.no_observations))
        self.processed_wei = np.empty((0, 1))
        self.no_processed = self.processed_par.shape[0]
        # all data not yet used for surrogate construction:
        self.non_processed_par = np.empty((0, self.no_parameters))
        self.non_processed_obs = np.empty((0, self.no_observations))
        self.non_processed_wei = np.empty((0, 1))

    def add_data(self, snapshots):
        # add new data to a matrix of non-processed data
        L = len(snapshots)
        new_par = np.empty((L, self.no_parameters))
        new_obs = np.empty((L, self.no_observations))
        new_wei = np.empty((L, 1))
        for i in range(L):
            new_par[i, :] = snapshots[i].sample
            new_obs[i, :] = snapshots[i].G_sample
            new_wei[i, :] = 1
        self.non_processed_par = np.vstack((self.non_processed_par, new_par))
        self.non_processed_obs = np.vstack((self.non_processed_obs, new_obs))
        self.non_processed_wei = np.vstack((self.non_processed_wei, new_wei))

    def update(self):
        # print("SURROGATE UPDATE")
        no_non_processed = self.non_processed_par.shape[0]
        # both processed and non-processed
        no_snapshots = self.no_processed + no_non_processed
        TEMP = np.zeros((no_snapshots, no_snapshots))

        self.processed_par = np.vstack(
            (self.processed_par, self.non_processed_par))
        self.processed_obs = np.vstack(
            (self.processed_obs, self.non_processed_obs))
        self.processed_wei = np.vstack(
            (self.processed_wei, self.non_processed_wei))
        self.no_processed += no_non_processed

        self.non_processed_par = np.empty((0, self.no_parameters))
        self.non_processed_obs = np.empty((0, self.no_observations))
        self.non_processed_wei = np.empty((0, 1))

        for i in range(self.no_parameters):
            v = self.processed_par[:, i]
            Q = np.matlib.repmat(v, no_snapshots, 1)
            T = np.transpose(Q)
            TEMP = TEMP + np.power(Q-T, 2)
        np.sqrt(TEMP, out=TEMP)  # distances between all points
        if self.max_centers is not None:
            MAX = 2*np.max(TEMP)
            # add MAX to zero distances on diagonal
            M = TEMP+MAX*np.eye(no_snapshots)
            # bool - centers to keep
            to_keep = np.ones(no_snapshots, dtype=bool)
            # (no_snapshots - max_centers) have to be removed
            for i in range(no_snapshots - self.max_centers):
                argmin = np.argmin(M)
                xx = argmin // no_snapshots
                if self.expensive is True:
                    S = sum(M)
                    yy = argmin % no_snapshots
                    M[xx, yy] = MAX
                    M[yy, xx] = MAX
                    if S[yy] < S[xx]:
                        yy = xx
                M[xx, :] = MAX
                M[:, xx] = MAX
                to_keep[xx] = False
            TEMP = TEMP[to_keep, :]
            TEMP = TEMP[:, to_keep]
            self.processed_par = self.processed_par[to_keep, :]
            self.processed_obs = self.processed_obs[to_keep, :]
            self.processed_wei = self.processed_wei[to_keep, :]
            self.no_processed = self.processed_par.shape[0]
            no_snapshots = self.no_processed
        kernel(TEMP, self.kernel_type, self.beta)
        P = np.ones((no_snapshots, 1))  # only constant polynomials
        P = np.append(self.processed_par, P, axis=1)  # plus linear polynomials
        no_polynomials = P.shape[1]
        TEMP = np.append(TEMP, P, axis=1)
        TEMP2 = np.append(np.transpose(P), np.zeros(
            [no_polynomials, no_polynomials]), axis=1)
        TEMP2 = np.vstack([TEMP, TEMP2])
        RHS = np.vstack([self.processed_obs, np.zeros(
            (no_polynomials, self.no_observations))])
        if self.solver_type == 'direct':
            COEFS = np.linalg.solve(TEMP2, RHS)
        else:
            COEFS = np.empty(
                (no_snapshots+no_polynomials, self.no_observations))
            for i in range(self.no_observations):
                c_ = splin.minres(TEMP2, RHS[:, i], x0=self.initial_iteration, tol=pow(
                    10, self.solver_tol_exp), show=False)
                COEFS[:, i] = c_[0]
        SOL = [COEFS.copy(), self.processed_par.copy()]
        return SOL, no_snapshots


def analyze(SOL, TEMP2, RHS):
    cond_num = 0  # np.linalg.cond(TEMP2)
    RES = RHS-np.reshape(np.matmul(TEMP2, SOL), (RHS.shape[0], 1))
    norm_RES = np.linalg.norm(RES)
    norm_RHS = np.linalg.norm(RHS)
    return cond_num, norm_RES, norm_RHS


def kernel(arr, kernel_type, beta):
    #    arr=arr
    if kernel_type == "polyharmonic":
        if beta % 2 == 1:
            np.power(arr, beta, out=arr)
        else:
            tmp = np.log(arr)
            np.power(arr, beta, out=arr)
            arr = arr*tmp
        return
    if kernel_type == "multiquadric":
        np.power(arr, 2, out=arr)
        arr = np.sqrt(arr+beta**2)
        return
    if kernel_type == "inverse multiquadric":
        np.power(arr, 2, out=arr)
        arr = np.sqrt(arr+beta**2)
        arr = 1/arr
        return
    if kernel_type == "Gaussian":
        np.power(arr, beta, out=arr)
        arr = np.exp(-beta*arr)
        return
    else:
        print("polyharmonic RBF with beta=1")
        np.power(arr, 1, out=arr)
        return
