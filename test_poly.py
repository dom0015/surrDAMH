#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:19:05 2021

@author: domesova
"""

import matplotlib.pyplot as plt
import numpy as np

def G(x,y):
    #return np.sin(x) + np.cos(2*y)
    return -1/80*(3/np.exp(x)+1/np.exp(y))

extent = [0,10,-2,8]
NX = 200
NY = 200
xx = np.linspace(extent[0],extent[1],NX)
yy = np.linspace(extent[2],extent[3],NY)
zz = np.zeros((NY,NX))
datapoints = np.zeros((NY*NX,2))
k=0
for ix in range(NX):
    for iy in range(NY):
        zz[iy,ix] = G(xx[ix],yy[iy])
        datapoints[k,:] = [xx[ix],yy[iy]]
        k=k+1

K=10000
#data = np.random.randn(K,2)
#data = np.random.multivariate_normal(mean=[5,3], cov=[[4, -2],[-2, 4]], size=(K,))

## Load data from illustrative_updates
from surrDAMH.modules import visualization_and_analysis as va
folder_samples = "/home/domesova/GIT/MCMC-Bayes-python/saved_samples/illustrative_updates"
no_parameters = 2
S = va.Samples()
S.load_MH(folder_samples, no_parameters)
S.load_rejected(folder_samples=folder_samples)
S.load_accepted(folder_samples=folder_samples)
#data = S.x[40][:int(1e5):int(1e5/K),:] # from illustrative_updates.py
data = S.x[40][1:K+1,:] # from illustrative_updates.py
#data=np.unique(data, axis=0)

raw_data = S.get_raw_data(folder_samples, 2)
indices = S.accepted[0][:,0]
data_accepted = [raw_data[40][int(i),:] for i in list(indices) if i<K]
indices = S.rejected[0][:,0]
data_rejected = [raw_data[40][int(i),:] for i in list(indices) if i<K]
data_list = data_accepted #+ data_rejected
data = np.array(data_list)
K = data.shape[0]

class Snapshot():
    def __init__(self, sample, G_sample):
        self.sample = sample
        self.G_sample = G_sample
        self.weight = 1.0

G_data = np.zeros((K))
snapshots = []
for i in range(K):
    G_data[i] = G(data[i,0],data[i,1])
    snapshots.append(Snapshot(data[i,:],G_data[i]))
    
## POLY:
rbf = False
from surrDAMH.modules import surrogate_poly as surr
Update = surr.Surrogate_update(no_parameters=2, no_observations=1, max_degree=8)
Apply = surr.Surrogate_apply(no_parameters=2, no_observations=1)

## RBF:
# rbf = True
# from surrDAMH.modules import surrogate_rbf as surr
# kernel_type = 1
# Update = surr.Surrogate_update(no_parameters=2, no_observations=1, no_keep=500, expensive=False, kernel_type=kernel_type)
# Apply = surr.Surrogate_apply(no_parameters=2, no_observations=1, kernel_type=kernel_type)

Update.add_data(snapshots)
SOL,x = Update.update()

zz_poly = Apply.apply(SOL, datapoints)
zz_poly = zz_poly.reshape((NY,NX), order="F")

# plt.figure()
# plt.imshow(zz, origin="lower", extent = extent, cmap="inferno")
# plt.colorbar()

# plt.figure()
# plt.imshow(zz_poly, vmin = np.min(zz), vmax = np.max(zz), origin="lower", extent = extent, cmap="inferno")
# plt.colorbar()

abs_error = np.abs(zz-zz_poly)

plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3

import matplotlib.colors as colors
plt.figure()
vmin = 1e-11 # min(abs_error.min(), 1e-11)
vmax = 1e1 # max(abs_error.max(), 1e1)
plt.imshow(abs_error, origin="lower", extent = extent, cmap="bwr", norm=colors.LogNorm(vmin=vmin, vmax=vmax))
plt.colorbar()


markersize = 1
if rbf:
    #plt.plot(data[:,0],data[:,1],'.', color="tab:orange", markersize=1)
    plt.plot(SOL[1][:,0],SOL[1][:,1],'.', color="tab:red", markersize=markersize)
else:
    acc = np.array(data_accepted)
    plt.plot(acc[:,0],acc[:,1],'.',color="tab:orange", markersize=markersize)
    rej = np.array(data_rejected)
    plt.plot(rej[:,0],rej[:,1],'.',color="tab:green", markersize=markersize)
    
#plt.title(str(K))
plt.xlim([extent[0],extent[1]])
plt.ylim([extent[2],extent[3]])
plt.show()

fig = plt.gcf()
fig.set_size_inches(4.4, 3.4, forward=True)
plt.tight_layout()

print("K =",K)
