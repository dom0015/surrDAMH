#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:13:07 2020

@author: domesova
"""

from modules import autocorr_analysis as aa
import numpy as np
import matplotlib.pyplot as plt

from configuration import Configuration
C = Configuration()
# problem_name = "GRF_linela2"
problem_name = "GRF_24to10_02_unlimited"
# problem_name = "GRF_double24to10_start1"
# problem_name = C.problem_name

S = aa.Samples()
N = C.no_samplers # number of MH/DAMH chains
P = C.no_parameters

S.load_notes('saved_samples/' + problem_name,N)
S.load_MH('saved_samples/' + problem_name)
# S.remove_chains(range(N)) # keep only MH chains
S.remove_chains(range(N,2*N)) # keep only DAMH chains
S.calculate_properties()

S.load_rejected('saved_samples/' + problem_name)
S.load_accepted('saved_samples/' + problem_name)
S.merge_evaluated()
S.plot_evaluated_sliding(range(N), no_sw=10, L = S.evaluated)

# S.calculate_autocorr_time_sliding(no_sw=10, chains_range=range(N))

# plt.plot(S.tau_sliding_min)
# plt.plot(S.tau_sliding_mean)
# plt.plot(S.tau_sliding_max)
# plt.show()

# S.load_MH_with_posterior('saved_samples/' + problem_name, P)
# S.plot_grf_minmax()

S.calculate_autocorr_time()
print("autocorr time:",np.mean(S.autocorr_time,axis=1))
S.calculate_autocorr_function()
S.calculate_autocorr_function_mean()
S.calculate_autocorr_time_mean()
tau = int(max(S.autocorr_time_mean))
burn_in = [2*tau]*N

S.remove_burn_in(burn_in)
S.calculate_properties()
S.calculate_autocorr_function()
S.calculate_autocorr_function_mean()
S.calculate_autocorr_time_mean()
tau2 = int(max(S.autocorr_time_mean))
print('tau (before, after):', tau, tau2)
# S.plot_segment(end_disp = [5001]*P, parameters_disp = [0,1,2,17,18,19])
# S.plot_hist_grid(parameters_disp = [0,1,2,17,18,19])
# tmp = int(min(S.length)*0.9)
# Phalf = int(P/2)
# S.plot_average(begin_disp = [tmp]*P, parameters_disp = range(Phalf))
# S.plot_average(begin_disp = [tmp]*P, parameters_disp = range(Phalf,P))
# S.plot_average(begin_disp = [tmp]*P, parameters_disp = [0,1,2,17,18,19])
# S.plot_autocorr_function(length_disp = [10001]*P, parameters_disp = range(Phalf))
# S.plot_autocorr_function(length_disp = [10001]*P, parameters_disp = range(Phalf,P))
# S.plot_autocorr_function(length_disp = [10001]*P, parameters_disp = [0,1,2,17,18,19])

# # common color scale
# S.plot_mean_and_std_grf()



# mean posterior
from modules import classes_SAMPLER as cS
my_Prob = cS.Problem_Gauss(no_parameters=C.no_parameters,
                            noise_std=C.problem_parameters['noise_std'],
                            prior_mean=C.problem_parameters['prior_mean'], 
                            prior_std=C.problem_parameters['prior_std'],
                            no_observations=C.no_observations,
                            observations=C.problem_parameters['observations'],
                            seed=0,
                            name=C.problem_name)
sample = S.mean[0]
from modules import FEM_wrapper as FEM_wrapper
FEM = FEM_wrapper.FEM(**C.child_solver_parameters)
FEM.pass_parameters(sample)
G_sample = FEM.get_observations()
log_posterior = my_Prob.get_log_posterior(sample, G_sample)
print("mean log_posterior:", log_posterior)