#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:34:58 2018

@author: dom0015
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import time

import rbf

no_parameters = 5
no_observations = 1

# IMPORT DATA
file=pd.read_csv('G_data_5par.csv')
G_data=file.values
obs = G_data[:,1]
par_x = G_data[:,2]
par_y = G_data[:,3] 
par = G_data[:,2:]
plt.figure()  
konec = G_data.shape[0]
plt.scatter(par_x[:konec],par_y[:konec])

# DIVIDE DATA INTO GROUPS FOR UPDATES
#bounds = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000]
bounds = range(100, 2001, 100)
#bounds = [10, 100, 1000]
#bounds = range(100,1000)

# PREPARE GRID
nx, ny = (200,200)
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
xv, yv = np.meshgrid(x, y)
xv=np.reshape(xv,(nx*ny,1))
yv=np.reshape(yv,(nx*ny,1))
grid_par_ = np.append(xv,yv,axis=1)
#grid_obs = pow(pow(xv,2)+yv-11,2)+pow(xv+pow(yv,2)-7,2)
grid_obs = pow(pow(xv,2)+yv-1,2)+pow(yv+pow(1.5,2)-1,2)+pow(1.5+pow(1.5,2)-1,2)+pow(1.5+pow(1.5,2)-1,2)+pow(1.5+pow(xv,2)-1,2)
grid_prior = np.multiply(ss.norm.pdf(xv),ss.norm.pdf(yv))
#grid_likelihood = np.exp(-np.multiply(66.37369073 - grid_obs,66.37369073 - grid_obs)/8)
grid_likelihood = np.exp(-np.multiply(41.06119073 - grid_obs,41.06119073 - grid_obs)/8)
grid_posterior = np.multiply(grid_prior, grid_likelihood)

# CONSTRUCT ALL SURROGATE MODELS
no_all = par.shape[0]
obs = np.reshape(obs,(no_all,1))

alldata_par = par[:bounds[0],:]
alldata_obs = obs[:bounds[0],:]

all_time = 0

# calculate first surrogate model
SOL, no_evaluations, aa, aaa = rbf.calculate(alldata_par,alldata_obs,no_parameters,None)

grid_par = np.ones([nx*ny,no_parameters])*1.5
grid_par[:,0:2] = grid_par_

griddata_surrogate = rbf.apply(grid_par, alldata_par, no_parameters, SOL)
#grid_likelihood_surrogate = np.exp(-np.multiply(66.37369073 - griddata_surrogate,66.37369073 - griddata_surrogate)/8)
grid_likelihood_surrogate = np.exp(-np.multiply(41.06119073 - griddata_surrogate,41.06119073 - griddata_surrogate)/8)
grid_posterior_surrogate = np.multiply(grid_prior, grid_likelihood_surrogate)
grid_norm = np.linalg.norm(grid_posterior_surrogate-grid_posterior)
print(grid_norm)
plt.figure()
plt.imshow(np.reshape(np.abs(grid_posterior_surrogate-grid_posterior),(200,200)))
plt.colorbar()

for b in bounds[1:]:
    print(b)
    t=time.time()
    no_evaluations_old = no_evaluations
    newdata_par = par[no_evaluations_old:b,:]
    newdata_obs = obs[no_evaluations_old:b,:]
    alldata_par = par[:b,:]
    alldata_obs = obs[:b,:]
    # calculate updated model and send to surrogate solver
    no_evaluations = alldata_par.shape[0]
    initial_iteration=np.append(SOL[:-3],np.zeros([no_evaluations-no_evaluations_old,1]))
    initial_iteration=np.append(initial_iteration,SOL[-3:])
    SOL, no_evaluations,aa, aaa = rbf.calculate(alldata_par,alldata_obs,no_parameters,initial_iteration)
#    plt.figure()
#    plt.plot(initial_iteration,'r')
#    plt.plot(SOL,'b')

#    print(alldata_par.shape,alldata_obs.shape,SOL.shape, no_evaluations, b)
    time_new = time.time() - t
    all_time = all_time + time_new
    # CALCULATE NORM (pripadne prubezne)
    griddata_surrogate = rbf.apply(grid_par, alldata_par, no_parameters, SOL)
    grid_likelihood_surrogate = np.exp(-np.multiply(66.37369073 - griddata_surrogate,66.37369073 - griddata_surrogate)/8)
    grid_posterior_surrogate = np.multiply(grid_prior, grid_likelihood_surrogate)
    grid_norm = np.linalg.norm(grid_posterior_surrogate-grid_posterior)
    print(grid_norm)
    plt.figure()
    plt.imshow(np.reshape(np.abs(grid_posterior_surrogate-grid_posterior),(200,200)))
    plt.colorbar()

#plt.figure()
#plt.imshow(np.reshape(griddata_surrogate,(200,200)))
#plt.figure()
#plt.imshow(np.reshape(grid_obs,(200,200)))
#plt.figure()
#plt.imshow(np.reshape(np.abs(grid_posterior_surrogate-grid_posterior),(200,200)))
#plt.figure()
#plt.imshow(np.reshape(np.abs(grid_posterior),(200,200)))
print(all_time, grid_norm)

# 1.3529999256134033 0.002435892309332884 None minres
# 1.483860731124878 0.002435892309332884 None minres
# 1.346437931060791 0.002435892309332884 initial_iteration minres
# 1.4688384532928467 0.002435892309332884 initial_iteration minres
# 1.5245673656463623 5.4322189413504574e-05 solve