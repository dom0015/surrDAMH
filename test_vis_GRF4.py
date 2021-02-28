#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""
from modules import FEM_grf as FEM_wrapper
from modules import visualization_and_analysis as va
from modules import grf_eigenfunctions as grf
import matplotlib.pyplot as plt
import numpy as np

### REFERENCE SOLUTION for GRF with 50*50 parameters (or truncated to 40)
solver_init = FEM_wrapper.FEM
no_configurations = 4
no_parameters = 40
no_observations = 10 * no_configurations
solver_parameters = {'no_parameters': no_parameters,
                        'no_observations': no_observations, 
                        'no_configurations': no_configurations,
                        'n': 50,
                        'quiet': True,
                        'tolerance': 1e-8,
                        'PC': "icc",
                        'use_deflation': False, # True,
                        'deflation_imp': 1e-2}
solver_instance = solver_init(**solver_parameters)
reference_parameters = np.ones((no_parameters,)) # ((50*50,)) # ((40,))
solver_instance.set_parameters(reference_parameters)
reference_observations = solver_instance.get_observations()
print(reference_observations)
grf.plot_grf(reference_parameters, cmap="bwr")
solver_instance.all_solvers[0].plot_solution_image()

# from configuration import Configuration
# C = Configuration()
# problem_name = C.problem_name
problem_name = "GRF_rbftest"

S = va.Samples()
N = 8 # C.no_samplers # number of MH/DAMH chains
P = no_parameters # C.no_parameters

S.load_notes('saved_samples/' + problem_name,N)
S.load_MH('saved_samples/' + problem_name)
# Which part of the sampling process is analyzed? MH/DAMH-SMU/DAMH
# Keep only the corresponding set of chains.
setnumber = 2;
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

parameters_disp = [0,1,2,3,39]
S.plot_average(parameters_disp=parameters_disp, show_legend = True)
S.plot_average_reverse(parameters_disp=parameters_disp, show_legend = True)
S.plot_hist(parameters_disp=parameters_disp)
plt.figure()
S.plot_hist_1d(dimension=0)
S.plot_hist_grid(parameters_disp=parameters_disp, bins1d=20, bins2d=30)

plt.figure()
plt.scatter(S.x[0][:,0],S.x[0][:,1],s=30,alpha=0.1,marker=".")
plt.grid()
plt.show()

plt.figure()
plt.plot(S.x[0][:,0],S.x[0][:,1],alpha=0.5,marker=".",)
plt.grid()
plt.show()

### GRF
# S.plot_mean_and_std_grf_rotated()#(chains_disp = [0])
S.plot_mean_and_std_grf_rotated_merged()