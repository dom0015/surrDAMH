#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:07:02 2018

@author: dom0015
"""

import numpy as np
import time

from Surrogate_parent import Surrogate_parent

class scm(Surrogate_parent):
    def __init__(self, max_degree):
        self.max_degree = max_degree
        print('SCM surrogate model will be constructed.')
        
    def calculate(self,alldata_par, alldata_obs, alldata_wei):
        alldata_wei = alldata_wei*0 + 1 # TEMP !!!
        no_snapshots, no_parameters = alldata_par.shape
        degree = int(np.floor(np.log(no_snapshots)/np.log(no_parameters)))
        print("degree",degree)
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
                par_col = alldata_par[:,j]#.reshape((no_snapshots,1))
                coefs[:,i] *= self.poly_eval(H_row,par_col)
        coefs_wei = (coefs * alldata_wei).transpose()
        A = np.matmul(coefs_wei,coefs)
        RHS = np.matmul(coefs_wei,alldata_obs)
        c = np.matmul(np.linalg.pinv(A),RHS)
        SOL = [c, H, poly, degree]
#        print("C",c,"H",H,"poly",poly,"degree",degree,"wei",alldata_wei)
        return SOL, no_snapshots, alldata_par, alldata_obs, alldata_wei, 0, 0

    def apply(self, SOL, newdata_par, alldata_par):
        c, H, poly, degree = SOL
        N = poly.shape[0]
        no_newdata = newdata_par.shape[0]
        no_observations = c.shape[1]
        newdata_surrogate = np.zeros((no_newdata,no_observations))
        phi = np.ones((no_newdata,N))
        for i in range(N):
            for j in range(no_parameters):
                H_row = H[int(poly[i,j]),:]#.reshape((1,degree+1))
                par_col = newdata_par[:,j]#.reshape((1,1))
                phi[:,i] *= self.poly_eval(H_row,par_col)#.reshape((1,))
#        for k in range(no_newdata):
#            newdata_surrogate[k,:] = np.matmul(phi[k,:],c)
        newdata_surrogate = np.matmul(phi,c)
        return newdata_surrogate
        
#        c, H, poly, degree = SOL
#        N = poly.shape[0]
##        print("poly shape", poly.shape)
#        no_newdata = newdata_par.shape[0]
##        print("no_observations from c:",c.shape)
#        no_observations = c.shape[1]
#        newdata_surrogate = np.zeros((no_newdata,no_observations))
#        for k in range(no_newdata):
#            phi = np.ones((N))
#            for i in range(N):
#                for j in range(dim):
#                    H_row = H[int(poly[i,j]),:]#.reshape((1,degree+1))
#                    par = newdata_par[k,j]#.reshape((1,1))
#    #                    print("DEB:", newdata_par.shape, phi.shape, k, j)
#                    phi[i] *= self.poly_eval(H_row,par)#.reshape((1,))
#            newdata_surrogate[k,:] = np.matmul(c.transpose(),phi)
##        print(newdata_par)
##        print(newdata_surrogate)
#        return newdata_surrogate
    
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