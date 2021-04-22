#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
            
class Surrogate_apply: # initiated by all SAMPLERs
    def __init__(self, no_parameters, no_observations):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.current_degree = 1;
        self.on_degree_change()
        
    def apply(self, SOL, datapoints):
        MAT, degree = SOL # current surr. model to be evaluated in datapoints
        datapoints = datapoints.reshape(-1,self.no_parameters)
        no_datapoints = datapoints.shape[0]  
        if degree > self.current_degree:
            self.current_degree = degree
            self.on_degree_change()
        hermite_eval = poly_eval_multi(self.hermite,datapoints)
        PHI = np.ones((no_datapoints,self.no_poly))
        for j in range(self.no_parameters):
            PHI *= hermite_eval[:,j,self.poly[:,j]]
        GS_datapoints = np.matmul(PHI,MAT)
        return GS_datapoints

    def on_degree_change(self):
        self.poly = generate_polynomials_degree(self.no_parameters, self.current_degree)
        self.no_poly = self.poly.shape[0]
        self.hermite = hermite_poly_normalized(self.current_degree)
        
class Surrogate_update: # initiated by COLLECTOR
    def __init__(self, no_parameters, no_observations, max_degree=5):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_degree = max_degree
        # all processed data (used for surrogate construction):
        self.processed_par = np.empty((0,self.no_parameters))
        self.processed_obs = np.empty((0,self.no_observations))
        self.processed_wei = np.empty((0,1))
        self.no_processed = self.processed_par.shape[0]
        # all data not yet used for surrogate construction:
        self.non_processed_par = np.empty((0,self.no_parameters))
        self.non_processed_obs = np.empty((0,self.no_observations))
        self.non_processed_wei = np.empty((0,1))
        self.current_degree = 1
        self.on_degree_change()

    def add_data(self,snapshots):
        # add new data to a matrix of non-processed data
        L = len(snapshots)
        new_par = np.empty((L,self.no_parameters))
        new_obs = np.empty((L,self.no_observations))
        new_wei = np.empty((L,1))
        for i in range(L):
            new_par[i,:] = snapshots[i].sample
            new_obs[i,:] = snapshots[i].G_sample
            if snapshots[i].weight==0:
                new_wei[i,:] = 0
            else:
                new_wei[i,:] = snapshots[i].weight #1
        self.non_processed_par = np.vstack((self.non_processed_par, new_par))
        self.non_processed_obs = np.vstack((self.non_processed_obs, new_obs))
        self.non_processed_wei = np.vstack((self.non_processed_wei, new_wei))

    def update(self):
        no_non_processed = self.non_processed_par.shape[0]
        no_snapshots = self.no_processed + no_non_processed # both processed and non-processed
        degree = int(np.floor(np.log(no_snapshots)/np.log(self.no_parameters)))
        degree = min(degree,self.max_degree)
        if degree > self.current_degree:
            self.current_degree = degree
            self.on_degree_change()
        PHI_non_processed = np.ones((no_non_processed,self.no_poly))
        hermite_eval = poly_eval_multi(self.hermite,self.non_processed_par)
        for j in range(self.no_parameters):
            PHI_non_processed *= hermite_eval[:,j,self.poly[:,j]]
        PHI_non_processed_wei = PHI_non_processed * self.non_processed_wei
        
        self.processed_par = np.vstack((self.processed_par, self.non_processed_par))
        self.processed_obs = np.vstack((self.processed_obs, self.non_processed_obs))
        self.processed_wei = np.vstack((self.processed_wei, self.non_processed_wei))
        self.PHI_processed = np.vstack((self.PHI_processed, PHI_non_processed))
        self.PHI_processed_wei = np.vstack((self.PHI_processed_wei, PHI_non_processed_wei))
        self.no_processed += no_non_processed
        
        self.non_processed_par = np.empty((0,self.no_parameters))
        self.non_processed_obs = np.empty((0,self.no_observations))
        self.non_processed_wei = np.empty((0,1))
        
        A = np.matmul(self.PHI_processed_wei.transpose(),self.PHI_processed)
        RHS = np.matmul(self.PHI_processed_wei.transpose(),self.processed_obs)
        MAT = np.matmul(np.linalg.pinv(A),RHS) # shape (no_poly,no_observations)
        SOL = [MAT, self.current_degree]
        return SOL, no_snapshots

    def on_degree_change(self):
        self.poly = generate_polynomials_degree(self.no_parameters, self.current_degree)
        self.no_poly = self.poly.shape[0]
        self.hermite = hermite_poly_normalized(self.current_degree)

        self.PHI_processed = np.ones((self.no_processed,self.no_poly))
        hermite_eval = poly_eval_multi(self.hermite,self.processed_par)
        for j in range(self.no_parameters):
            self.PHI_processed *= hermite_eval[:,j,self.poly[:,j]]
        
        self.PHI_processed_wei = (self.PHI_processed * self.processed_wei)
        print("SURROGATE polynomial degree increased to:", self.current_degree, "- no poly:", self.no_poly)

def generate_polynomials_degree(dim,degree):
    poly = np.zeros([1,dim],dtype=int)
    
    if degree==0:
        return poly
    
    if degree>0:
        poly = np.vstack((poly,np.eye(dim,dtype=int)))
        
    if degree>1:
        tmp0 = np.eye(dim,dtype=int)
        tmp1 = np.eye(dim,dtype=int)
        for i in range(degree-1):
            polynew = np.zeros((tmp0.shape[0]*tmp1.shape[0],dim),dtype=int)
            idx = 0
            for j in range(tmp0.shape[0]):
                for k in range(tmp1.shape[0]):
                    polynew[idx] = tmp0[j,:]+tmp1[k,:]
                    idx += 1
            tmp1 = np.unique(polynew,axis=0)
            poly = np.vstack((poly,tmp1))
    return poly
    
def hermite_poly_normalized(degree):
    n = degree + 1
    H = np.zeros((n,n))
    H[0,0] = 1
    if degree==0:
        return H
    H[1,1] = 1
    diff = np.arange(1,n)
    for i in range(2,n):
        H[i,1:] += H[i-1,:-1]
        H[i,:-1] -= diff*H[i-1,1:]
    for i in range(n):
        H[i,:] = np.divide(H[i,:],np.sqrt(np.math.factorial(i)))
    return H
    
def poly_eval(p, points):
    # p ... coefficients of univariate polynomial
    # points ... points of evaluation, shape (n,)
    n = p.size
    values = np.zeros(points.size)
    temp = np.ones(points.shape)
    values += p[0]
    for i in range(1,n):
        temp = temp*points
        values += temp*p[i]
    return values

def poly_eval_multi(p, points):
    # p ... each row = coefficients of univariate polynomial
    # points ... points of evaluation, shape (n1,n2)
    no_polynomials, n = p.shape
    n1, n2 = points.shape
    values = np.zeros((n1,n2,no_polynomials))
    for j in range(no_polynomials): # loop over polynomials
        values[:,:,j]=p[j,0]; # constant term
    points_pow = np.ones((n1,n2))
    for i in range(1,n): # loop over degree+1
        points_pow *= points 
        for j in range(no_polynomials): # loop over polynomials
            values[:,:,j] += p[j,i]*points_pow
    return values # shape (n1, n2, no_polynomials)