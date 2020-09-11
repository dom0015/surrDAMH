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
problem_name = "GRF_24to10_05"
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

S.plot_segment(begin_disp = [1000]*P, end_disp = [9001]*P, parameters_disp = [0,1,2,23], chains_disp = [0])
tmp = int(min(S.length)*0.1)
S.plot_average_reverse(parameters_disp = [0,1,2,23], chains_disp = [0])

S.plot_segment(begin_disp = [1000]*P, end_disp = [9001]*P, parameters_disp = [0,1,2,23])#, chains_disp = [0])
S.plot_hist_grid(parameters_disp = [0,1,2,23])
tmp = int(min(S.length)*0.1)
# S.plot_average(begin_disp = [tmp]*P, parameters_disp = [0,1,2,23])
S.plot_average_reverse(parameters_disp = [0,1,2,23])

# common color scale
S.plot_mean_and_std_grf_rotated(chains_disp = [0])

# 32 parameters:
C = Configuration()
problem_name = "GRF_24to10_06_unlimited_32par"
S = aa.Samples()
N = 8
S.load_notes('saved_samples/' + problem_name,N)
S.load_MH('saved_samples/' + problem_name)
S.remove_chains(range(N,2*N)) # keep only DAMH chains
S.calculate_properties()
# common color scale
S.plot_mean_and_std_grf_rotated(chains_disp = [0])


# 40 parameters:
C = Configuration()
problem_name = "GRF_24to10_06_unlimited_40par"
S = aa.Samples()
N = 8
S.load_notes('saved_samples/' + problem_name,N)
S.load_MH('saved_samples/' + problem_name)
S.remove_chains(range(N,2*N)) # keep only DAMH chains
S.calculate_properties()
# common color scale
S.plot_mean_and_std_grf_rotated(chains_disp = [0])


# problem_name = "GRF_24to10_onlyMH"
# S = aa.Samples()
# N = 10
# S.load_notes('saved_samples/' + problem_name,N)
# S.load_MH('saved_samples/' + problem_name)
# S.remove_chains(range(N)) # keep only MH chains
# S.calculate_properties()
# S.calculate_autocorr_time()
# print("autocorr time:",np.mean(S.autocorr_time,axis=1))
