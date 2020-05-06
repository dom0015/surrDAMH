#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:19:41 2020

@author: simona
"""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import time

# rectangular domain
# points are uniformly spaced on a rectangular grid

nx = 120 # number of grid points in x direction
ny = 120 # number of grid points in y direction
lx = 10 # length of the domain in x direction
ly = 10 # length of the domain in y direction

sigma = 1 # std
lam = 1 # autocorrelation length

def cov_function(r,sigma,lam):
    return np.power(sigma,2)*np.exp(-r/lam)

def calculate_dist(nx, ny, lx, ly):
    x = np.linspace(0,lx,nx)
    y = np.linspace(0,ly,ny)
    X,Y = np.meshgrid(x,y)
    return np.sqrt(np.power(X,2)+np.power(Y,2))

def calculate_Cov(nx, ny, lx, ly, sigma, lam,show=False):
    distances = calculate_dist(nx, ny, lx, ly)
    cov_dist = cov_function(distances, sigma, lam)
    blocks = list(scipy.linalg.toeplitz(cov_dist[i,:]) for i in range(ny))
    structure = scipy.linalg.toeplitz(range(ny))
    blocks_list = [None] * ny
    for i in range(ny):
        blocks_list[i] = list(blocks[j] for j in structure[i,:])
    Cov = np.block(blocks_list)
    if show:
        plt.imshow(Cov)
        plt.show()
    return Cov

def calculate_Cholesky_factor(Cov):
    t = time.time()
    L = scipy.linalg.cholesky(Cov)
    print('Chol factorization time:',time.time() - t)
    return L

def sample_using_Cholesky(L,seed=None,show=True,nx=None):
    if not seed is None: 
        np.random.seed(seed)
    nxny = L.shape[0]
    grf = np.matmul(np.transpose(L),np.random.randn(nxny))
    if show:
        if nx is None:
            nx = int(np.sqrt(nxny))
            ny = nx
        else:
            ny = int(nxny/nx)
        plt.imshow(grf.reshape((ny,nx)))
        plt.show()
    return grf

def calculate_eig(Cov):
    t = time.time()
    D,V = np.linalg.eigh(Cov)
    print('eigh factorization time:',time.time() - t)
    # sort eigenvalues and eigenvectors
    indices = np.flip(np.argsort(D))
    Dsorted = D[indices]
    Vsorted = V[:,indices]
    return Dsorted,Vsorted
    
def eigenfunctions(V,indices,nx,ny,lx,ly):
    # TO DO: lambda
    x = np.linspace(0,lx,nx)
    y = np.linspace(0,ly,ny)
    f = [None] * len(indices)
    for i,index in enumerate(indices):
        f[i] = scipy.interpolate.interp2d(x,y,V[:,index],kind='cubic')
    return f

def evaluate_eigenfunctions(f,x,y,show=True):
    n = len(f)
    z = [None] * n
    for i in range(n):
        z[i] = f[i](x,y)
    if show:
        fig, axes = plt.subplots(1, n, figsize=(12, 3))
        for i in range(n):
            axes[i].imshow(z[i])
    return z
    
def plot_eigenvectors(V,indices,nx=None):
    # indices ... columns of V to be visualized on the grid
    if nx is None:
        nx = int(np.sqrt(V.shape[0]))
        ny = nx
    else:
        ny = int(V.shape[0]/nx)
    fig, axes = plt.subplots(1, len(indices), figsize=(12, 3))
    for i,index in enumerate(indices):
        axes[i].imshow(V[:,index].reshape((ny,nx)))
        axes[i].set_xlabel("$index:  {0}$".format(index))
    plt.show()

def save_eig(filename, D, V, Cov, nx, ny, lx, ly, sigma, lam):
    import pickle
    f = open(filename, 'wb')
    pickle.dump([D, V, Cov, nx, ny, lx, ly, sigma, lam],f)
    f.close()

def load_eig(filename):
    import pickle
    f = open(filename,'rb')
    obj = pickle.load(f)
    f.close()
    return obj


#Cov = calculate_Cov(nx,ny,lx,ly,sigma,lam,show=True)
#D,V = calculate_eig(Cov)
#save_eig(D, V, Cov, nx, ny, lx, ly, sigma, lam)
#plot_eigenvectors(V,[0,1,2,100,1000],nx)
D, V, Cov, nx, ny, lx, ly, sigma, lam = load_eig('eig.pckl')
indices = [0,1,5,10,20,5000]
plot_eigenvectors(V,indices,nx)
f = eigenfunctions(V,indices,nx,ny,lx,ly)
for i in [12,120,1200]:
    x = np.linspace(4,5,i)
    y = np.linspace(3,4,i)
    z = evaluate_eigenfunctions(f,x,y)