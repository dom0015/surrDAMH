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
offset = 10
extent = [5-offset, 5+offset, 3-offset, 3+offset]
NX = 600
NY = 600
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
# rbf = False
# from surrDAMH.modules import surrogate_poly as surr
# Update = surr.Surrogate_update(no_parameters=2, no_observations=1, max_degree=8)
# Apply = surr.Surrogate_apply(no_parameters=2, no_observations=1)

## RBF:
rbf = True
from surrDAMH.modules import surrogate_rbf as surr
kernel_type = 1
Update = surr.Surrogate_update(no_parameters=2, no_observations=1, no_keep=100, expensive=False, kernel_type=kernel_type)
Apply = surr.Surrogate_apply(no_parameters=2, no_observations=1, kernel_type=kernel_type)

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




### QUANTIFICATION OF THE SURROGATE MODEL ERROR:
import importlib.util as iu
import os
import sys
import json
import numpy as np
import scipy
import matplotlib.pyplot as plt

# wdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
wdir = os.getcwd() 
sys.path.append(wdir)
from surrDAMH.modules import visualization_and_analysis as va

saved_samples_name = "illustrative"
conf_name = "illustrative"
conf_path = wdir + "/examples/" + conf_name + ".json"

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 4
    
plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3

### OBSERVATION OPERATOR
spec = iu.spec_from_file_location(conf["solver_module_name"], wdir + "/" + conf["solver_module_path"])
solver_module = iu.module_from_spec(spec)
spec.loader.exec_module(solver_module)
solver_init = getattr(solver_module, conf["solver_init_name"])     
solver_instance = solver_init(**conf["solver_parameters"])


prior_mean = conf["problem_parameters"]["prior_mean"]
prior_cov = conf["problem_parameters"]["prior_std"]

N = NX
res = np.zeros((N,N))
likelihood = np.zeros((N,N))
prior = np.zeros((N,N))
y = conf["problem_parameters"]["observations"][0]
f_eta = scipy.stats.norm(loc = 0, scale = conf["problem_parameters"]["noise_std"])
f_prior = scipy.stats.multivariate_normal(mean = prior_mean, cov = prior_cov)
x1 = np.linspace(prior_mean[0]-offset,prior_mean[0]+offset,N)
x2 = np.linspace(prior_mean[1]-offset,prior_mean[1]+offset,N)
for idi,i in enumerate(x1):
    for idj,j in enumerate(x2):
        Gij = G(i,j)
        res[idj,idi] = Gij
        likelihood[idj,idi] = f_eta.pdf(y-Gij)
        prior[idj,idi] = f_prior.pdf([i,j])
posterior = prior*likelihood/292.70
plt.figure()
plt.imshow(posterior, origin="lower", extent = extent, cmap="jet")
plt.colorbar()
plt.figure()
plt.imshow(posterior*abs_error, origin="lower", extent = extent, cmap="jet")
plt.colorbar()
print("RESULT POSTERIOR:",np.sum(posterior*abs_error)*(4*offset*offset/NY/NX))
plt.figure()
plt.imshow(prior, origin="lower", extent = extent, cmap="jet")
plt.colorbar()
plt.figure()
plt.imshow(prior*abs_error, origin="lower", extent = extent, cmap="jet")
plt.colorbar()
plt.show()
print("RESULT PRIOR:",np.sum(prior*abs_error)*(4*offset*offset/NY/NX))