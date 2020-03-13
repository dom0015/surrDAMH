#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
            
class Surrogate_col: # initiated by SAMPLERs
    def __init__(self, no_parameters, no_observations, is_updated=True, max_degree=5):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.is_updated = is_updated
        self.max_degree = max_degree
        self.alldata_par = np.empty((0,self.no_parameters))
        self.alldata_obs = np.empty((0,self.no_observations))
        self.alldata_wei = np.empty((0,1))
        
    def add_data(self,snapshots):
        L = len(snapshots)
        newdata_par = np.empty((L,self.no_parameters))
        newdata_obs = np.empty((L,self.no_observations))
        newdata_wei = np.empty((L,1))
        for i in range(L):
            newdata_par[i,:] = snapshots[i].sample
            newdata_obs[i,:] = snapshots[i].G_sample
            newdata_wei[i,:] = 1 #snapshots[i].weight TEMP!
        self.alldata_par = np.vstack((self.alldata_par, newdata_par))
        self.alldata_obs = np.vstack((self.alldata_obs, newdata_obs))
        self.alldata_wei = np.vstack((self.alldata_wei, newdata_wei))
    
    def update(self):
        no_snapshots, no_parameters = self.alldata_par.shape
        degree = int(np.floor(np.log(no_snapshots)/np.log(no_parameters)))
#        print("degree",degree)
        if degree == 0:
            degree=1
        if degree>self.max_degree:
            degree=self.max_degree
        poly = self.generate_polynomials_degree(no_parameters,degree)
        N = poly.shape[0]
        H = self.hermite_poly_normalized(degree)
        coefs = np.ones((no_snapshots,N))
        for i in range(N):
            for j in range(no_parameters):
                H_row = H[int(poly[i,j]),:]#.reshape((1,degree+1))
                par_col = self.alldata_par[:,j]#.reshape((no_snapshots,1))
                coefs[:,i] *= self.poly_eval(H_row,par_col)
        coefs_wei = (coefs * self.alldata_wei).transpose()
        A = np.matmul(coefs_wei,coefs)
        RHS = np.matmul(coefs_wei,self.alldata_obs)
        c = np.matmul(np.linalg.pinv(A),RHS)
        SOL = [c, H, poly, degree]
#        print("C",c,"H",H,"poly",poly,"degree",degree,"wei",alldata_wei)
        return SOL, no_snapshots

    def apply(self, SOL, newdata_par):
        c, H, poly, degree = SOL
        N = poly.shape[0]
        no_observations = c.shape[1]
        if len(newdata_par.shape)==1:
            newdata_par.shape = (1,no_observations)
        no_newdata = newdata_par.shape[0]  
        newdata_surrogate = np.zeros((no_newdata,no_observations))
        phi = np.ones((no_newdata,N))
        for i in range(N):
            for j in range(self.no_parameters):
                H_row = H[int(poly[i,j]),:]#.reshape((1,degree+1))
                par_col = newdata_par[:,j]#.reshape((1,1))
                phi[:,i] *= self.poly_eval(H_row,par_col)#.reshape((1,))
#        for k in range(no_newdata):
#            newdata_surrogate[k,:] = np.matmul(phi[k,:],c)
        newdata_surrogate = np.matmul(phi,c)
        return newdata_surrogate
    
    def generate_polynomials_degree(self,dim,degree):
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
    
    def hermite_poly_normalized(self,degree):
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
    
    def poly_eval(self, p, grid):
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