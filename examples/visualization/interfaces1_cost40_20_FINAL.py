#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import importlib.util as iu
import os
import sys
import json
import numpy as np
import scipy
import matplotlib.pyplot as plt

wdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
# wdir = os.getcwd() 
sys.path.append(wdir)
from surrDAMH.modules import visualization_and_analysis as va

### DEFAULT PARAMETERS:
conf_name = "interfaces1_cost40_20"
no_samplers = 40 # number of sampling processes

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
saved_samples_name = conf_name
no_parameters = conf["no_parameters"]
list_alg = conf["samplers_list"]

### LOAD SAMPLES:
folder_samples = wdir + "/saved_samples/" + saved_samples_name
S = va.Samples()
S.load_notes(folder_samples, no_samplers)
no_alg = len(list_alg) #47
alg_range = range(1,no_alg)
no_alg = no_alg-1
for i in range(no_alg):
    print("Notes - ALGORTIHM " + str(i) + ":")
    print(S.notes[i])
S.load_MH(folder_samples, no_parameters)

proposal_scale = [None] * no_alg
proposal_scale_all = [None] * no_alg * no_samplers
#representative_chains = [None] * no_alg
acceptance_rate = [None] * no_alg
for idi,i in enumerate(alg_range):
    proposal_scale[idi] = np.array(S.notes[i]["proposal_scale"])[0]
    proposal_scale_all[idi*no_samplers:(idi+1)*no_samplers] = np.array(S.notes[i]["proposal_scale"])
    acceptance_rate[idi] = np.mean(S.notes[i]["acceptance_rate"])

plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3

plt.rcParams['font.size'] = '14'
fontsize = 16
markersize = 9
linewidth = 2

""" surrogate time ratio """
time_G_all = [None] * (no_alg+1)
time_GS_all = [None] * (no_alg+1)
for i in range(no_alg+1):
    chains = range(i*no_samplers,(i+1)*no_samplers)
    [time_G, time_GS] = S.plot_evaluation_time(folder_samples, no_parameters, chains_range=chains, plot=False) 
    time_G_all[i] = np.mean(time_G)
    time_GS_all[i] = np.mean(time_GS)
plt.figure()
plt.plot(acceptance_rate,time_G_all[1:],'.-')
plt.xlabel("acceptace rate")
plt.ylabel("average $G$ evaluation time")
plt.tight_layout()
plt.figure()
plt.plot(proposal_scale,time_G_all[1:],'.-')
plt.xlabel("proposal SD $\sigma$")
plt.ylabel("average $G$ evaluation time")
plt.tight_layout()
plt.figure()
plt.plot(proposal_scale,time_GS_all[1:],'.-')
plt.xlabel("proposal SD $\sigma$")
plt.ylabel("average $\widetilde{G}$ evaluation time")
plt.tight_layout()
plt.figure()
plt.plot(proposal_scale,np.array(time_GS_all[1:])/np.array(time_G_all[1:]),'.-')
plt.xlabel("proposal SD $\sigma$")
plt.ylabel("$\widetilde{G}$ evaluation cost")
plt.tight_layout()
# surrogate time taken as constant (average)
plt.figure()
plt.plot(proposal_scale,np.mean(time_GS_all[1:])/np.array(time_G_all[1:]),'.-')
plt.xlabel("proposal SD $\sigma$")
plt.ylabel("$\widetilde{G}$ evaluation cost")
plt.tight_layout()
plt.figure()
plt.plot(proposal_scale,acceptance_rate,'.-')
plt.xlabel("proposal SD $\sigma$")
plt.ylabel("acceptance rate")
plt.tight_layout()
plt.show()
