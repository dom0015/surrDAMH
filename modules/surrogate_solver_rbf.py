#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.matlib
import scipy.sparse.linalg as splin

class Surrogate_apply: # initiated by all SAMPLERs
    def __init__(self, no_parameters, no_observations, kernel_type=0):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.kernel_type = kernel_type
        self.alldata_par = None

    def apply(self, SOL, datapoints):
        coefs, alldata_par = SOL
#        if self.alldata_par == None:
        self.alldata_par = alldata_par
#        else:
#        self.alldata_par = np.vstack((self.alldata_par,alldata_par))
        if len(datapoints.shape)==1:
            datapoints.shape = (1,self.no_parameters)
        no_datapoints = datapoints.shape[0]
        no_snapshots = self.alldata_par.shape[0]
        TEMP = np.zeros([no_datapoints,no_snapshots])
        for i in range(self.no_parameters):
            v = self.alldata_par[:,i]
            M = np.matlib.repmat(v,no_datapoints,1)
            v_new = datapoints[:,i]
            M_new = np.matlib.repmat(v_new,no_snapshots,1)
            T = np.transpose(M_new)
            TEMP = TEMP + np.power(M-T,2)
        np.sqrt(TEMP,out=TEMP)
        kernel(TEMP,self.kernel_type)
        no_polynomials = 1 + self.no_parameters # plus linear polynomials
        GS_datapoints=np.matmul(TEMP,coefs[:-no_polynomials])
        pp = coefs[-1] + np.matmul(datapoints,coefs[-self.no_parameters-1:-1])
        GS_datapoints = np.reshape(GS_datapoints,(no_datapoints,self.no_observations)) + np.reshape(pp,(no_datapoints,self.no_observations))
        return GS_datapoints

class Surrogate_update: # initiated by COLLECTOR # no_keep=50, expensive=True
    def __init__(self, no_parameters, no_observations, initial_iteration=None, no_keep=None, expensive=False, kernel_type=0, solver_tol_exp=-6, solver_type='minres'):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.initial_iteration = initial_iteration           
        self.no_keep = no_keep
        self.expensive = expensive
        self.kernel_type = kernel_type
        self.solver_tol_exp = solver_tol_exp
        self.solver_type = solver_type
        # all processed data (used for surrogate construction):
        self.processed_par = np.empty((0,self.no_parameters))
        self.processed_obs = np.empty((0,self.no_observations))
        self.processed_wei = np.empty((0,1))
        self.no_processed = self.processed_par.shape[0]
        # all data not yet used for surrogate construction:
        self.non_processed_par = np.empty((0,self.no_parameters))
        self.non_processed_obs = np.empty((0,self.no_observations))
        self.non_processed_wei = np.empty((0,1))

    def add_data(self,snapshots):
        # add new data to a matrix of non-processed data
        L = len(snapshots)
        print("new snapshots:",L)
        new_par = np.empty((L,self.no_parameters))
        new_obs = np.empty((L,self.no_observations))
        new_wei = np.empty((L,1))
        for i in range(L):
            new_par[i,:] = snapshots[i].sample
            new_obs[i,:] = snapshots[i].G_sample
            new_wei[i,:] = 1 #snapshots[i].weight # TO DO! TEMP!
        self.non_processed_par = np.vstack((self.non_processed_par, new_par))
        self.non_processed_obs = np.vstack((self.non_processed_obs, new_obs))
        self.non_processed_wei = np.vstack((self.non_processed_wei, new_wei))

    def update(self):
        print("SURROGATE UPDATE")
        no_non_processed = self.non_processed_par.shape[0]
        no_snapshots = self.no_processed + no_non_processed # both processed and non-processed
        TEMP = np.zeros((no_snapshots,no_snapshots))
        ## TO DO: temporary
        self.processed_par = np.vstack((self.processed_par, self.non_processed_par))
        self.processed_obs = np.vstack((self.processed_obs, self.non_processed_obs))
        self.processed_wei = np.vstack((self.processed_wei, self.non_processed_wei))
        self.no_processed += no_non_processed
        
        self.non_processed_par = np.empty((0,self.no_parameters))
        self.non_processed_obs = np.empty((0,self.no_observations))
        self.non_processed_wei = np.empty((0,1))
        
        for i in range(self.no_parameters):
            v = self.processed_par[:,i]
            Q = np.matlib.repmat(v,no_snapshots,1)
            T = np.transpose(Q)
            TEMP = TEMP + np.power(Q-T,2)
        np.sqrt(TEMP,out=TEMP) # distances between all points
        # print("surr TEMP shape:",TEMP.shape)
        if self.no_keep is not None:
            MAX = 2*np.max(TEMP)
            M = TEMP+MAX*np.eye(no_snapshots) # add MAX to zero distances on diagonal
            to_keep = np.ones(no_snapshots,dtype=bool) # bool - centers to keep
            for i in range(no_snapshots - self.no_keep): # (no_snapshots - no_keep) have to ne removed 
                argmin = np.argmin(M)
                xx = argmin // no_snapshots
                if self.expensive == True: # TO DO: check - yy is unused
                    S=sum(M)
                    yy = argmin % no_snapshots
                    M[xx,yy]=MAX
                    M[yy,xx]=MAX
                    if S[yy]<S[xx]:
                        yy=xx
                M[xx,:]=MAX
                M[:,xx]=MAX
                to_keep[xx]=False
            aaa = np.arange(no_snapshots)
            print(aaa[to_keep==False])
            TEMP = TEMP[to_keep,:]
            TEMP = TEMP[:,to_keep]
            self.processed_par = self.processed_par[to_keep,:]
            self.processed_obs = self.processed_obs[to_keep,:]
            self.processed_wei = self.processed_wei[to_keep,:]
            self.no_processed = self.processed_par.shape[0]
            no_snapshots = self.no_processed
#            try:
#                print(to_keep)
#                to_keep=np.append(to_keep,np.ones(self.no_parameters+1,dtype=bool))
#                print(to_keep)
#                self.initial_iteration = self.initial_iteration[to_keep]
#            except:
#                print("Exception!")    
        kernel(TEMP,self.kernel_type)
        P = np.ones((no_snapshots,1)) # only constant polynomials
        P = np.append(self.processed_par,P,axis=1) # plus linear polynomials
        no_polynomials = P.shape[1]
        TEMP = np.append(TEMP,P,axis=1)
        TEMP2 = np.append(np.transpose(P),np.zeros([no_polynomials,no_polynomials]),axis=1)
        TEMP2 = np.vstack([TEMP,TEMP2])
        RHS = np.vstack([ self.processed_obs, np.zeros((no_polynomials,self.no_observations)) ])
#        print("Condition_number:",np.linalg.cond(TEMP2))
#        print("DEBUG2:", TEMP2.shape, np.linalg.matrix_rank(TEMP2), RHS.shape)
        if self.solver_type == 'direct':
            c=np.linalg.solve(TEMP2,RHS)
        else:
            c=np.empty((no_snapshots+no_polynomials,self.no_observations))
            for i in range(self.no_observations):
                c_=splin.minres(TEMP2,RHS[:,i],x0=self.initial_iteration,tol=pow(10,self.solver_tol_exp),show=False)
                c[:,i]=c_[0]
    #    RES=RHS-np.reshape(np.matmul(TEMP2,SOL),(no_evaluations+no_polynomials,1))
    #    print("Residual norm:",np.linalg.norm(RES), "RHSnorm:", np.linalg.norm(RHS))
        SOL = [None] * 2
        SOL[0] = c.copy()
        SOL[1] = self.processed_par.copy()
        print("Surrogate updated:",SOL[0].shape,SOL[1].shape)
        return SOL, no_snapshots
        
    #minres(A, b, x0=None, shift=0.0, tol=1e-05, maxiter=None, xtype=None, M=None, callback=None, show=False, check=False)
    #gmres(A, b, x0=None, tol=1e-05, restart=None, maxiter=None, M=None, callback=None, restrt=None, atol=None)
        


def analyze(SOL, TEMP2, RHS):
    cond_num = 0 # np.linalg.cond(TEMP2)
    RES=RHS-np.reshape(np.matmul(TEMP2,SOL),(RHS.shape[0],1))
    norm_RES = np.linalg.norm(RES)
    norm_RHS = np.linalg.norm(RHS)
    return cond_num, norm_RES, norm_RHS

def kernel(arr,kernel_type):
#    arr=arr
    if kernel_type==0:
        return
    if kernel_type==1:
        np.power(arr,3,out=arr)
        return
    if kernel_type==2: # best for 2 parameters
        np.power(arr,5,out=arr)
        return
    if kernel_type==3:
        np.power(arr,7,out=arr)
        return
    if kernel_type==4:
        temp = -np.power(arr,2)
        np.exp(temp,out=arr)
        return
    if kernel_type==5:
        temp = np.power(arr,2)+1
        np.power(temp,-0.5,out=arr)
        return
    if kernel_type==6:
        arr=np.power(arr,5)
        arr=-arr
        return
    if kernel_type==7:
        np.power(arr,9,out=arr)
        return