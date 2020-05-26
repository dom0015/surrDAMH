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
#        self.is_updated = is_updated
#        self.max_degree = max_degree
#        self.alldata_par = np.empty((0,self.no_parameters))
#        self.alldata_obs = np.empty((0,self.no_observations))
#        self.alldata_wei = np.empty((0,1))
        self.current_degree = 1;
        self.on_degree_change()
        
    def apply(self, SOL, datapoints):
        c, degree = SOL
        if degree > self.current_degree:
            self.current_degree = degree
            self.on_degree_change()
        if len(datapoints.shape)==1:
            datapoints.shape = (1,self.no_parameters)
        no_datapoints = datapoints.shape[0]  
        GS_datapoints = np.zeros((no_datapoints,self.no_parameters))
        phi = np.ones((no_datapoints,self.no_poly))
        for i in range(self.no_poly):
            for j in range(self.no_parameters):
                H_row = self.hermite[int(self.poly[i,j]),:]#.reshape((1,degree+1))
                par_col = datapoints[:,j]#.reshape((1,1))
                phi[:,i] *= poly_eval(H_row,par_col)#.reshape((1,))
        GS_datapoints = np.matmul(phi,c)
        return GS_datapoints

    def on_degree_change(self):
        self.poly = generate_polynomials_degree(self.no_parameters, self.current_degree)
        self.no_poly = self.poly.shape[0]
        self.hermite = hermite_poly_normalized(self.current_degree)
        
class Surrogate_update: # initiated by COLLECTOR
    def __init__(self, no_parameters, no_observations, max_degree=5):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
#        self.is_updated = is_updated
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
            new_wei[i,:] = 1 #snapshots[i].weight # TO DO! TEMP!
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
        poly_non_processed = np.ones((no_non_processed,self.no_poly))
        for i in range(self.no_poly):
            for j in range(self.no_parameters):
                H_row = self.hermite[int(self.poly[i,j]),:]#.reshape((1,degree+1))
                par_col = self.non_processed_par[:,j]#.reshape((no_snapshots,1))
                poly_non_processed[:,i] *= poly_eval(H_row,par_col)
        poly_non_processed_wei = (poly_non_processed * self.non_processed_wei)#.transpose()
        
        self.processed_par = np.vstack((self.processed_par, self.non_processed_par))
        self.processed_obs = np.vstack((self.processed_obs, self.non_processed_obs))
        self.processed_wei = np.vstack((self.processed_wei, self.non_processed_wei))
        self.poly_processed = np.vstack((self.poly_processed, poly_non_processed))
        self.poly_processed_wei = np.vstack((self.poly_processed_wei, poly_non_processed_wei))
        self.no_processed += no_non_processed
        
        self.non_processed_par = np.empty((0,self.no_parameters))
        self.non_processed_obs = np.empty((0,self.no_observations))
        self.non_processed_wei = np.empty((0,1))
        
        A = np.matmul(self.poly_processed_wei.transpose(),self.poly_processed)
        RHS = np.matmul(self.poly_processed_wei.transpose(),self.processed_obs)
        c = np.matmul(np.linalg.pinv(A),RHS)
        SOL = [c, self.current_degree]
        return SOL, no_snapshots

    def on_degree_change(self): # TO DO: check for repetitive calculations
        self.poly = generate_polynomials_degree(self.no_parameters, self.current_degree)
        self.no_poly = self.poly.shape[0]
        self.hermite = hermite_poly_normalized(self.current_degree)
        self.poly_processed = np.ones((self.no_processed,self.no_poly))
        for i in range(self.no_poly):
            for j in range(self.no_parameters):
                H_row = self.hermite[int(self.poly[i,j]),:]#.reshape((1,degree+1))
                par_col = self.processed_par[:,j]#.reshape((no_snapshots,1))
                self.poly_processed[:,i] *= poly_eval(H_row,par_col)
        self.poly_processed_wei = (self.poly_processed * self.processed_wei)#.transpose()

def generate_polynomials_degree(dim,degree):
    poly = np.zeros([1,dim])
    
    if degree==0:
        return poly
    
    if degree>0:
        poly = np.vstack((poly,np.eye(dim)))
        
    if degree>1:
        temp1 = np.eye(dim) # const
        temp = np.eye(dim)
        for i in range(degree-1):
            polynew = np.zeros([temp1.shape[0]*temp.shape[0],dim])
            idx = 0
            for j in range(temp1.shape[0]):
                for k in range(temp.shape[0]):
                    polynew[idx] = temp1[j,:]+temp[k,:]
                    idx += 1
            temp = np.unique(polynew,axis=0)
            poly = np.vstack((poly,temp))
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
    
def poly_eval(p, grid):
#        print("p shape",p.shape)
#        print("g shape",grid.shape, grid.size)
    n = p.size
    values = np.zeros(grid.size)
    temp = np.ones(grid.shape)
    values += p[0]
    for i in range(1,n):
        temp = temp*grid
        values += temp*p[i]
#        print("values:",values)
    return values