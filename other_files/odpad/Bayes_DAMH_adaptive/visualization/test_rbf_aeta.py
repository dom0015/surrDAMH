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
import csv

import rbf

# 2 PARAMETERS
#no_parameters = 2
#y = 66.37369073
#file=pd.read_csv('G_data_2par.csv')

# 3 PARAMETERS
#no_parameters = 3
#y = 25.93619073
#file=pd.read_csv('G_data_3par.csv')

# 4 PARAMETERS
#no_parameters = 4
#y = 33.49869073
#file=pd.read_csv('G_data_4par.csv')

# 5 PARAMETERS
no_parameters = 2
y=41.06119073
file=pd.read_csv('G_data_linela_MH.csv')

# 5 PARAMETERS more complicated
#no_parameters = 5
#y=151.06119073
#file=pd.read_csv('G_data2_5par.csv')

# 10 PARAMETERS more complicated
#no_parameters = 10
#y=203.87369073
#file=pd.read_csv('G_data2_10par.csv')

kernel_type = 7
solver_type = 6

# DIVIDE DATA INTO GROUPS FOR UPDATES
no_testing = 4000
bound = 8000
#bounds = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
#bounds = range(100, bound+1, 100)
bounds = range(1000,8001,1000)
#bounds = [10, 100]
#bounds = range(100,1000)
#bounds = range(10,1000,10)
#bounds = range(200,4001,20)

filename = 'results1000_xx' + str(no_parameters) + '_' + str(kernel_type) + '_' + str(solver_type) + '.csv'
file_out = open(filename, 'w')
writer = csv.writer(file_out)
writer.writerow(['Observation','no_for_surrogate','no_testing'])
writer.writerow([y,bound,no_testing])
#writer.writerow(bounds)

no_observations = 1
priorMean = np.zeros(no_parameters)
priorStd = np.ones(no_parameters)*1.5
noiseStd = np.ones(no_observations)*2
calculate_error = True
use_initial_it = False
no_keep = 0
expensive=0

# IMPORT DATA AND DIVIDE INTO SOURCE AND TESTING
G_data=file.values
G_data=G_data[:(bound+no_testing),:]
no_all = G_data.shape[0]
obs = np.reshape(G_data[:,1],(no_all,1))
#par_x = G_data[:,2]
#par_y = G_data[:,3] 
par = G_data[:,2:]
par_surr = par[:bound,:]
obs_surr = obs[:bound,:]
par_test = par[bound:,:]
obs_test = obs[bound:,:]

#plt.figure()  
#plt.scatter(par_surr[:,0],par_surr[:,1]) 
#plt.title('Data pro surrogate:'+str(bound))
#plt.figure() 
#plt.scatter(par_test[:,0],par_test[:,1]) 
#plt.title('Testovaci data:'+str(no_all-bound))

# CALCULATE POSTERIOR FOR TESTING DATA
prior_test = np.sum(np.power(par_test,2),axis=1)/(2*priorStd[0]) # bez exponencialy, konstantni std
prior_test = np.reshape(prior_test,(prior_test.shape[0],1))
likelihood_test = np.multiply(y - obs_test,y - obs_test)/(2*noiseStd[0])
posterior_test = np.exp(-prior_test-likelihood_test)

# CONSTRUCT ALL SURROGATE MODELS
alldata_par = par_surr[:bounds[0],:]
alldata_obs = obs_surr[:bounds[0],:]

all_time = []
all_mean = []
all_perc = []
all_max = []
all_cond_num = []
all_norm_RES = []
all_norm_RHS = []
t=time.time()
# calculate first surrogate model
SOL, no_evaluations, alldata_par, alldata_obs, TEMP2, RHS = rbf.calculate(alldata_par,alldata_obs,no_parameters,None,no_keep,expensive,kernel_type,solver_type)
all_time.append(time.time()-t)
cond_num, norm_RES, norm_RHS = rbf.analyze(SOL, TEMP2, RHS)
all_cond_num.append(cond_num)
all_norm_RES.append(norm_RES)
all_norm_RHS.append(norm_RHS)

# CALCULATE APPROXIMATED POSTERIOR FOR TESTING DATA AND ERROR
if calculate_error:
    testdata_surrogate = rbf.apply(par_test, alldata_par, no_parameters, SOL, kernel_type)
    likelihood_test_surrogate = np.multiply(y - testdata_surrogate,y - testdata_surrogate)/(2*noiseStd[0])
    posterior_test_surrogate = np.exp(-prior_test-likelihood_test_surrogate)
    avg_error_test = np.mean(np.abs(posterior_test_surrogate-posterior_test)) 
    med_error_test = np.median(np.abs(posterior_test_surrogate-posterior_test)) 
    min_error_test = np.min(np.abs(posterior_test_surrogate-posterior_test)) 
    max_error_test = np.max(np.abs(posterior_test_surrogate-posterior_test)) 
    perc_error_test = np.percentile(np.abs(posterior_test_surrogate-posterior_test),95)
    all_mean.append(avg_error_test)
    all_perc.append(perc_error_test)
    all_max.append(max_error_test)
    print(avg_error_test, min_error_test, med_error_test, perc_error_test, max_error_test)

for b in bounds[1:]:
    print(b, no_evaluations)
    t=time.time()
    no_evaluations_old = no_evaluations
    newdata_par = par_surr[no_evaluations_old:b,:]
    newdata_obs = obs_surr[no_evaluations_old:b,:]
    alldata_par = np.vstack([alldata_par,newdata_par])
    alldata_obs = np.vstack([alldata_obs,newdata_obs])
    # choose only random subset
#    choice = np.random.choice(b,300,replace=False)
#    alldata_par = par_surr[choice,:]
#    alldata_obs = obs_surr[choice,:]
    # calculate updated model and send to surrogate solver
    no_evaluations = alldata_par.shape[0]
    if use_initial_it:
        initial_iteration=np.append(SOL[:-(1+no_parameters)],np.zeros([no_evaluations-no_evaluations_old,1]))
        initial_iteration=np.append(initial_iteration,SOL[-(1+no_parameters):])
    else:
        initial_iteration = None
    SOL, no_evaluations, alldata_par, alldata_obs, TEMP2, RHS = rbf.calculate(alldata_par,alldata_obs,no_parameters,initial_iteration,no_keep,expensive,kernel_type,solver_type)
    all_time.append(time.time()-t)
    cond_num, norm_RES, norm_RHS = rbf.analyze(SOL, TEMP2, RHS)
    all_cond_num.append(cond_num)
    all_norm_RES.append(norm_RES)
    all_norm_RHS.append(norm_RHS)

    if calculate_error:
        testdata_surrogate = rbf.apply(par_test, alldata_par, no_parameters, SOL, kernel_type)
        likelihood_test_surrogate = np.multiply(y - testdata_surrogate,y - testdata_surrogate)/(2*noiseStd[0])
        posterior_test_surrogate = np.exp(-prior_test-likelihood_test_surrogate)
        avg_error_test = np.mean(np.abs(posterior_test_surrogate-posterior_test)) 
        med_error_test = np.median(np.abs(posterior_test_surrogate-posterior_test)) 
        min_error_test = np.min(np.abs(posterior_test_surrogate-posterior_test)) 
        perc_error_test = np.percentile(np.abs(posterior_test_surrogate-posterior_test),95)
        all_mean.append(avg_error_test)
        all_perc.append(perc_error_test)
        all_max.append(max_error_test)
        print(avg_error_test, min_error_test, med_error_test, perc_error_test, max_error_test)

plt.figure(11)
plt.plot(bounds,all_time)
plt.title('time')

plt.figure(12)
plt.semilogy(bounds,all_mean)
plt.title('mean')

plt.figure(13)
plt.semilogy(bounds,all_perc)
plt.title('perc')

plt.figure(14)
plt.plot(np.sort(np.abs(obs_test-testdata_surrogate),axis=0))
plt.title('G error')

plt.figure(15)
plt.semilogy(bounds,all_cond_num)
plt.title('cond')

plt.figure(16)
plt.semilogy(bounds,all_norm_RES)
plt.title('res')

plt.figure(17)
plt.semilogy(bounds,all_norm_RHS)
plt.title('rhs')

writer.writerow(all_time)
writer.writerow(['max-mean-perc-time-cond-res-rhs:'])
writer.writerow(all_max)
writer.writerow(all_mean)
writer.writerow(all_perc)
writer.writerow(all_time)
writer.writerow(all_cond_num)
writer.writerow(all_norm_RES)
writer.writerow(all_norm_RHS)
writer.writerow(['File closing'])
file_out.close()