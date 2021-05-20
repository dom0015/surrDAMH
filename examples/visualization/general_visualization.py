#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import os
import sys
import json

wdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
# wdir = os.getcwd() 
sys.path.append(wdir)
from surrDAMH.modules import visualization_and_analysis as va

### DEFAULT PARAMETERS:
conf_name = "simple_scaling" # requires configuration file "conf/" + conf_name + ".json"
no_samplers = 10 # number of sampling processes

### PARSE COMMAND LINE ARGUMENTS: 
len_argv = len(sys.argv)
if len_argv>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
if len_argv>2:
    conf_name = sys.argv[2]

""" Visualization and autocorrelation analysis """
### LOAD CONFIGURATION:
conf_path = wdir + "/examples/" + conf_name + ".json"
with open(conf_path) as f:
    conf = json.load(f)
if "saved_samples_name" in conf:
    saved_samples_name = conf["saved_samples_name"]
else:
    saved_samples_name = conf_name
no_parameters = conf["no_parameters"]
list_alg = conf["samplers_list"]

### LOAD SAMPLES:
folder_samples = wdir + "/saved_samples/" + saved_samples_name
S = va.Samples()
S.load_notes(folder_samples, no_samplers)
no_alg = len(list_alg)
for i in range(no_alg):
    print("Notes - ALGORTIHM " + str(i) + ":")
    print(S.notes[i])
S.load_MH(folder_samples, no_parameters)

print("\nBEFORE BURN-IN PERIOD REMOVAL:")
S.print_properties()
# burn_in = S.calculate_burn_in(no_samplers, multiplier = 2)
# S.remove_burn_in(burn_in)

# print("\nAFTER BURN-IN PERIOD REMOVAL:")
# S.print_properties()
# S.calculate_tau(no_samplers)
S.calculate_autocorr_time(tol=1)

# print("\nSAMPLING EFFICIENCY:")
# S.calculate_CpUS(no_samplers, surr_cost_ratio = 0.0)

### SAMPLES VISUALIZATION: 
parameters_disp = range(min(no_parameters,5))
S.plot_hist_grid(parameters_disp = parameters_disp, bins1d=30, bins2d=30, show_title = True, sharex=False, sharey=False)
S.plot_hist_grid(chains_disp = range(no_samplers*(no_alg-1),no_samplers*no_alg), parameters_disp = parameters_disp, bins1d=40, bins2d=40, show_title = True, sharex=False, sharey=False)

# plot convergence of averages for all parts of the sampling process
for i in range(no_alg):
    S.plot_average(parameters_disp = parameters_disp, chains_disp = range(i*no_samplers,(i+1)*no_samplers), show_legend = True, sharey=False)

# plot all chains
# for i in range(no_alg):
#     S.plot_segment(parameters_disp = parameters_disp, chains_disp = range(i*no_samplers,(i+1)*no_samplers), show_legend = True)

# calculate autocorrelation time mean (after burn in removal) and plot autocorrelation function
# for i in range(no_alg):
#     chains = range(i*no_samplers,(i+1)*no_samplers)
#     length_disp = [1001] * len(parameters_disp)
#     S.plot_autocorr_function(parameters_disp = parameters_disp, chains_disp = chains, length_disp = length_disp, plot_mean = True, show_legend = True)

