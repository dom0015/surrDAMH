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
problem_name = "GRF_double20to20"

S = aa.Samples()

N = C.no_samplers # number of MH/DAMH chains
P = C.no_parameters

S.load_MH('saved_samples/' + problem_name)
S.remove_chains(range(N,2*N)) # keep only DAMH chains
S.calculate_properties()
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
S.plot_segment(end_disp = [5001]*P, parameters_disp = [0,1,2,17,18,19])
S.plot_hist_grid(parameters_disp = [0,1,2,17,18,19])
tmp = int(min(S.length)*0.9)
Phalf = int(P/2)
S.plot_average(begin_disp = [tmp]*P, parameters_disp = range(Phalf))
S.plot_average(begin_disp = [tmp]*P, parameters_disp = range(Phalf,P))
S.plot_average(begin_disp = [tmp]*P, parameters_disp = [0,1,2,17,18,19])
S.plot_autocorr_function(length_disp = [10001]*P, parameters_disp = range(Phalf))
S.plot_autocorr_function(length_disp = [10001]*P, parameters_disp = range(Phalf,P))
S.plot_autocorr_function(length_disp = [10001]*P, parameters_disp = [0,1,2,17,18,19])

# common color scale
S.plot_mean_and_std_grf()

# begin_disp = [0] * S.no_parameters
# # end_disp = [25001] * S.no_parameters
# end_disp = None
# parameters_disp = [0,1,2,9]
# chains_disp = [0,1,8,9]
# #chains_disp = range(8,16)
# burn_in = [0, 0, 500, 500]
# #burn_in = [500] * 8

# # TO DO: convergence to mean H(X) (also for comparison of parallel chains)
# S.calculate_properties()
# #S.print_properties()
# S.plot_segment(begin_disp,end_disp,parameters_disp,chains_disp)
# S.plot_hist(burn_in,parameters_disp,chains_disp)
# S.plot_segment(begin_disp,end_disp,parameters_disp,[0,1])
# S.plot_hist(burn_in,parameters_disp,[0,1])
# #S.plot_average([0, 0, 0, 500, 500, 500],begin_disp,end_disp=None,parameters_disp=None,chains_disp=None)
# S.calculate_autocorr_function()
# S.calculate_autocorr_function_mean()
# S.plot_autocorr_function([1001] * S.no_parameters,plot_mean=True)
# S.calculate_autocorr_time()
# S.calculate_autocorr_time_mean()
# print(S.autocorr_time[3:])
# #print(S.autocorr_time_mean)
# #print(S.autocorr_time_mean_beta)

# #from modules import FEM_wrapper
# #G = FEM_wrapper.FEM(no_parameters = S.no_parameters, no_observations = 6, n = 50)
# #G.pass_parameters(S.mean[3])
# #print("observation:",G.get_observations())

# S.plot_mean_as_grf()
# S.plot_mean_and_std_grf(chains_disp=[0,1,8,9])

# S.plot_hist_2d(dimensions = [0,1], burn_in = [0,0], chains_disp = [0], bins = 20, show = True)
# S.plot_hist_grid([0,0,0,0], [0,1,2], range(4))

# # generate material shample and calculate observation:
# #no_parame = 5
# #grf_instance = grf.GRF('modules/unit50.pckl', truncate=no_parame)
# #eta = np.random.randn(no_parame)*0.1
# #z = grf_instance.realization_grid_new(eta,np.linspace(0,1,50),np.linspace(0,1,50))
# #plt.imshow(z)
# #plt.show()
# #G = FEM_wrapper.FEM(no_parameters = no_parame, no_observations = 6, n = 50)
# #G.pass_parameters(eta)
# #print("observation:",G.get_solution())

# #error for higher max_samples:
# #  File "/home/ber0061/Repositories_dom0015/MCMC-Bayes-python/modules/classes_communication.py", line 162, in receive_update_if_ready
# #    r = self.request_recv.wait()
# #  File "mpi4py/MPI/Request.pyx", line 235, in mpi4py.MPI.Request.wait
# #  File "mpi4py/MPI/msgpickle.pxi", line 411, in mpi4py.MPI.PyMPI_wait
# #mpi4py.MPI.Exception: MPI_ERR_TRUNCATE: message truncated
# #double free or corruption (!prev)
