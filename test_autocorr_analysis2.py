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
problem_name = "GRF_double24to10_start1"

S = aa.Samples()

N = C.no_samplers # number of MH/DAMH chains
P = C.no_parameters

S.load_MH('saved_samples/' + problem_name)
S.calculate_properties()
S.load_MH_with_posterior('saved_samples/' + problem_name)
S.plot_grf_minmax()

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
