#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:07:02 2018

@author: dom0015
"""

import numpy as np
import numpy.matlib as npm
import scipy.sparse.linalg as splin
import time

import kernel
import Surrogate_parent

class rbf(Surrogate_parent.Surrogate_parent):
    def __init__(self):
        print('RBF surrogate model will be constructed.')
        
    def calculate(self,alldata_par, alldata_obs, alldata_wei, no_parameters, rbf_parameters):
        initial_iteration, no_keep, expensive, kernel_type, solver_type = rbf_parameters
        no_evaluations = alldata_par.shape[0]
        TEMP = np.zeros([no_evaluations,no_evaluations])
        for i in range(no_parameters):
            v = alldata_par[:,i]
            Q = npm.repmat(v,no_evaluations,1)
            T = np.transpose(Q)
            TEMP = TEMP + np.power(Q-T,2)
        np.sqrt(TEMP,out=TEMP) # distances between all points
        if no_keep>0:
            MAX = 2*np.max(TEMP)
            M = TEMP+MAX*np.eye(no_evaluations)
            to_keep = np.ones(no_evaluations,dtype=bool)
            for i in range(no_evaluations - no_keep):
                argmin = np.argmin(M)
                xx = argmin // no_evaluations
                if expensive>0:
                    S=sum(M)
                    yy = argmin % no_evaluations
                    M[xx,yy]=MAX
                    M[yy,xx]=MAX
                    if S[yy]<S[xx]:
                        yy=xx
                M[xx,:]=MAX
                M[:,xx]=MAX
                to_keep[xx]=False
            TEMP = TEMP[to_keep,:]
            TEMP = TEMP[:,to_keep]
            alldata_par = alldata_par[to_keep,:]
            alldata_obs = alldata_obs[to_keep,:]
            no_evaluations = alldata_par.shape[0]
            try:
                to_keep=np.append(to_keep,np.ones(no_parameters+1,dtype=bool))
                initial_iteration = initial_iteration[to_keep]
            except:
                print("Exception!")    
        kernel.kernel(TEMP,kernel_type)
        P = np.ones([no_evaluations,1]) # only constant polynomials
    #    print(P.shape, no_evaluations, alldata_par.shape, TEMP.shape)
        P = np.append(alldata_par,P,axis=1) # plus linear polynomials
        no_polynomials = P.shape[1]
        TEMP = np.append(TEMP,P,axis=1)
        TEMP2 = np.append(np.transpose(P),np.zeros([no_polynomials,no_polynomials]),axis=1)
        TEMP2 = np.vstack([TEMP,TEMP2])
        RHS = np.vstack([ alldata_obs, np.zeros([no_polynomials,1]) ])
    #    print("Condition_number:",np.linalg.cond(TEMP2))
#        print("DEBUG2:", TEMP2.shape, np.linalg.matrix_rank(TEMP2), RHS.shape)
        if solver_type == 0:
            SOL=np.linalg.solve(TEMP2,RHS)
        else:
            SOL_=splin.minres(TEMP2,RHS,x0=initial_iteration,tol=pow(10,-solver_type),show=False)
            SOL=SOL_[0]
    #    RES=RHS-np.reshape(np.matmul(TEMP2,SOL),(no_evaluations+no_polynomials,1))
    #    print("Residual norm:",np.linalg.norm(RES), "RHSnorm:", np.linalg.norm(RHS))
        return SOL, no_evaluations, alldata_par, alldata_obs, alldata_wei, TEMP2, RHS
        
    #minres(A, b, x0=None, shift=0.0, tol=1e-05, maxiter=None, xtype=None, M=None, callback=None, show=False, check=False)
    #gmres(A, b, x0=None, tol=1e-05, restart=None, maxiter=None, M=None, callback=None, restrt=None, atol=None)
        
    def apply(self, SOL, newdata_par, alldata_par, no_parameters, kernel_type):
    #    print(SOL.shape, alldata_par.shape, no_parameters)
        no_new = newdata_par.shape[0]
        no_evaluations = alldata_par.shape[0]
        TEMP = np.zeros([no_new,no_evaluations])
        for i in range(no_parameters):
            v = alldata_par[:,i]
            M = npm.repmat(v,no_new,1)
            v_new = newdata_par[:,i]
            M_new = npm.repmat(v_new,no_evaluations,1)
            T = np.transpose(M_new)
            TEMP = TEMP + np.power(M-T,2)
        np.sqrt(TEMP,out=TEMP)
        kernel.kernel(TEMP,kernel_type)
        no_polynomials = 1 + no_parameters # plus linear polynomials
        newdata_surrogate=np.matmul(TEMP,SOL[:-no_polynomials])
        pp = SOL[-1] + np.matmul(newdata_par,SOL[-no_parameters-1:-1])
    #    print(pp.shape)
        newdata_surrogate = np.reshape(newdata_surrogate,(no_new,1)) + np.reshape(pp,(no_new,1))
#        print(newdata_par)
#        print(newdata_surrogate)
        return newdata_surrogate
    
    def analyze(self, SOL, TEMP2, RHS):
        cond_num = 0 # np.linalg.cond(TEMP2)
        RES=RHS-np.reshape(np.matmul(TEMP2,SOL),(RHS.shape[0],1))
        norm_RES = np.linalg.norm(RES)
        norm_RHS = np.linalg.norm(RHS)
        return cond_num, norm_RES, norm_RHS