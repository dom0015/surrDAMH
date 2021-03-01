#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

from modules import autocorr_analysis as aa
import numpy as np
import matplotlib.pyplot as plt

from configuration import Configuration
C = Configuration()
# problem_name = C.problem_name
problem_name = "linela2"

S = aa.Samples()
N = C.no_samplers # number of MH/DAMH chains
P = C.no_parameters

S.load_notes('saved_samples/' + problem_name,N)
S.load_MH('saved_samples/' + problem_name)
# Which part of the sampling process is analyzed? MH/DAMH-SMU/DAMH
# Keep only the corresponding set of chains.
setnumber = 1;
S.extract_chains(range(setnumber*N,(setnumber+1)*N))
S.calculate_properties()

# S.load_rejected('saved_samples/' + problem_name)
# S.load_accepted('saved_samples/' + problem_name)
# S.merge_evaluated()
# S.plot_evaluated_sliding(range(N), no_sw=10, L = S.evaluated)

# S.calculate_autocorr_time()
# print("autocorr time:",np.mean(S.autocorr_time,axis=1))
# S.calculate_autocorr_function()
# S.calculate_autocorr_function_mean()
# S.calculate_autocorr_time_mean()
# tau = int(max(S.autocorr_time_mean))
# burn_in = [2*tau]*N

# S.remove_burn_in(burn_in)
# S.calculate_properties()
# S.calculate_autocorr_function()
# S.calculate_autocorr_function_mean()
# S.calculate_autocorr_time_mean()
# tau2 = int(max(S.autocorr_time_mean))
# print('tau (before, after):', tau, tau2)

S.plot_average(parameters_disp = [0, 1, 2, 3, 40], show_legend = True)
S.plot_average_reverse()
S.plot_hist()
plt.figure()
S.plot_hist_1d()
S.plot_hist_grid(bins1d=20, bins2d=30)

solver_instance = C.child_solver_init(**C.child_solver_parameters)
sol = np.zeros((60,60))
for nj,j in enumerate(np.linspace(2,8,60)):
    for ni,i in enumerate(np.linspace(2,8,60)):
        solver_instance.set_parameters([i,j])
        #sol[ni,nj]=solver_instance.get_observations()
        sol[ni,nj]=np.abs(solver_instance.get_observations()-C.problem_parameters['observations'])

plt.figure()
plt.imshow(sol,extent=[2,8,2,8],origin='lower')
plt.colorbar()
plt.show()

plt.figure()
plt.scatter(S.x[0][:,0],S.x[0][:,1],s=30,alpha=0.1,marker=".")
plt.grid()
plt.show()