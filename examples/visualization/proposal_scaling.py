#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

wdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
# wdir = os.getcwd() 
sys.path.append(wdir)
from surrDAMH.modules import visualization_and_analysis as va

### DEFAULT PARAMETERS:
conf_name = "uninformative3" # requires configuration file "conf/" + conf_name + ".json"
no_samplers = 8 # number of sampling processes

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
saved_samples_name = conf["saved_samples_name"]
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

proposal_scale_all = [None] * no_alg *no_samplers
for i in range(no_alg):
    proposal_scale_all[i*no_samplers:(i+1)*no_samplers] = np.array(S.notes[i]["proposal_scale"])
deg = 2
xx = np.linspace(min(proposal_scale_all),max(proposal_scale_all),100)

""" Autocorrelation time calculated separately """
S.calculate_autocorr_time(tol=1)
autocorr_time_mean = np.array([np.mean(i) for i in S.autocorr_time]) # working autororrelation time
plt.figure()
for i in range(no_alg):
    chains = range(i*no_samplers,(i+1)*no_samplers)
    x = np.array(S.notes[i]["proposal_scale"])
    plt.plot(x,autocorr_time_mean[chains],'.')
F = va.fit(proposal_scale_all,autocorr_time_mean,deg=deg)
yy = F(xx)
plt.plot(xx, yy)
plt.title("Autocorrelation time calculated separately (before burn-in removal)")
plt.grid()
plt.show()

multiplier = 2
burn_in = [int(np.ceil(i*multiplier)) for i in autocorr_time_mean] # burn in to be removed
S.remove_burn_in(burn_in = burn_in)

S.calculate_autocorr_time(tol=1)
autocorr_time_mean = np.array([np.mean(i) for i in S.autocorr_time]) # working autororrelation time
plt.figure()
for i in range(no_alg):
    chains = range(i*no_samplers,(i+1)*no_samplers)
    x = np.array(S.notes[i]["proposal_scale"])
    plt.plot(x,autocorr_time_mean[chains],'.')
F = va.fit(proposal_scale_all,autocorr_time_mean,deg=deg)
yy = F(xx)
plt.plot(xx, yy)
plt.title("Autocorrelation time calculated separately (after burn-in removal)")
plt.grid()
plt.show()

print("\nSAMPLING EFFICIENCY:")
surr_cost_ratio = 0.0
print("calculated for surrogate evaluation cost ratio", surr_cost_ratio)
plt.figure()
CpUS_all = [None] * no_alg *no_samplers
for i in range(no_alg):
    no_full_evaluations = np.array(S.notes[i]["no_accepted"] + S.notes[i]["no_rejected"])
    no_all = np.array(S.notes[i]["no_all"])
    chains = range(i*no_samplers,(i+1)*no_samplers)
    autocorr_time = autocorr_time_mean[chains]
    CpUS = (no_full_evaluations/no_all + surr_cost_ratio) * autocorr_time
    CpUS_all[i*no_samplers:(i+1)*no_samplers] = CpUS
    proposal_scale = np.array(S.notes[i]["proposal_scale"])
    plt.plot(proposal_scale,CpUS,'.')
F = va.fit(proposal_scale_all,CpUS_all,deg=deg)
yy = F(xx)
plt.plot(xx, yy)
plt.title("CpUS calculated separately (after burn-in removal)")
plt.grid()
plt.show()

### SAMPLES VISUALIZATION: 
parameters_disp = range(min(no_parameters,5))
S.plot_hist_grid(parameters_disp = parameters_disp, bins1d=30, bins2d=30, show_title = True)
