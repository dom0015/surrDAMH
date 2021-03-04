#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import os
import sys
import json
sys.path.append(os.getcwd())
from surrDAMH.modules import visualization_and_analysis as va

### DEFAULT PARAMETERS:
conf_name = "simple" # requires configuration file "conf/" + conf_name + ".json"
N = 4 # number of sampling processes

### PARSE COMMAND LINE ARGUMENTS: 
len_argv = len(sys.argv)
if len_argv>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
if len_argv>2:
    conf_name = sys.argv[2]

### LOAD CONFIGURATION:
conf_path = "examples/" + conf_name + ".json"
with open(conf_path) as f:
    conf = json.load(f)
saved_samples_name = conf["saved_samples_name"]
no_parameters = conf["no_parameters"]
list_alg = conf["samplers_list"]

### LOAD SAMPLES:
folder_samples = "saved_samples/" + saved_samples_name
S = va.Samples()
S.load_notes(folder_samples, no_samplers)
no_alg = len(list_alg)
for i in range(no_alg):
    print("Notes - ALGORTIHM " + str(i) + ":")
    print(S.notes[i])
S.load_MH(folder_samples, no_parameters)
S.calculate_properties()
S.print_properties()

### SAMPLES VISUALIZATION:
S.plot_hist_grid(bins1d=30, bins2d=30, show_title = True)
# plot convergence of averages for all parts of the sampling process
for i in range(no_alg):
    parameters_disp = range(min(no_parameters,5))
    S.plot_average(parameters_disp = parameters_disp, chains_disp = range(i*no_samplers,(i+1)*no_samplers), show_legend = True)

# plot all chains
for i in range(no_alg):
    parameters_disp = range(min(no_parameters,5))
    S.plot_segment(parameters_disp = parameters_disp, chains_disp = range(i*no_samplers,(i+1)*no_samplers), show_legend = True)
