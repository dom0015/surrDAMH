#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:02:32 2019

@author: simona
"""

import numpy as np

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
    n = p.shape[0]
    print("dim",n+1)
    values = np.zeros([grid.shape[0],1])
    temp = np.ones([grid.shape[0],1])
    values += p[0]
    for i in range(1,n):
        temp = temp*grid
        values = values + temp*p[i]
    return values
    

dim = 4
degree = 4
p=generate_polynomials_degree(dim,degree)
print(p)

H=hermite_poly_normalized(4)
print(H)

v=poly_eval(np.array([11,10,100]),np.array([0,1,2]).reshape((3,1)))
print(v)