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
conf_name = "illustrative_scaling_DAMH" # requires configuration file "conf/" + conf_name + ".json"
#conf_name = "interfaces_scaling"
no_samplers = 4 # number of sampling processes

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
saved_samples_name = "interfaces_scaling"
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
proposal_scale_all = [None] * no_alg *no_samplers
representative_chains = [None] * no_alg
acceptance_rate = [None] * no_alg
for idi,i in enumerate(alg_range):
    proposal_scale[idi] = np.array(S.notes[i]["proposal_scale"])[0]
    proposal_scale_all[idi*no_samplers:(idi+1)*no_samplers] = np.array(S.notes[i]["proposal_scale"])
    representative_chains[idi] = range(i*no_samplers,(i+1)*no_samplers)[0]
    acceptance_rate[idi] = np.mean(S.notes[i]["acceptance_rate"])
deg = 2
xx = np.linspace(min(proposal_scale),max(proposal_scale),100)

""" Autocorrelation time calculated separately """
S.calculate_autocorr_time(tol=1)
autocorr_time_mean = np.array([np.mean(i) for i in S.autocorr_time]) # working autororrelation time
plt.figure()
for i in alg_range:
    chains = range(i*no_samplers,(i+1)*no_samplers)
    x = np.array(S.notes[i]["proposal_scale"])
    plt.plot(x,autocorr_time_mean[chains],'.')
F = va.fit(proposal_scale_all,autocorr_time_mean[no_samplers:],deg=deg)
yy = F(xx)
plt.plot(xx, yy)
plt.title("Autocorrelation time calculated separately (before burn-in removal)")
plt.grid()
plt.show()

"""lazy FIX stagnating chains"""
# for idi,i in enumerate(S.autocorr_time):
#     if i[0]>350 or i[1]>350:
#         print(i,idi)
#         S.x[idi] = S.x[idi-1]

plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3

""" Autocorrelation time calculated from all chains """
S.calculate_tau(no_samplers, c=5, smooth=2)
plt.figure()
# for idi,i in enumerate(alg_range):
#     chain = representative_chains[idi]
#     #plt.plot(proposal_scale[i],S.tau[chain],'.-r')
F = va.fit(proposal_scale,S.tau_aggregate[alg_range],deg=deg)
yy_tau = F(xx)
plt.plot(xx, yy_tau, linewidth=linewidth)
plt.plot(proposal_scale, S.tau_aggregate[alg_range], '.', markersize=markersize)
plt.ylabel("estimation of $\\tau$", fontsize=fontsize)
plt.xlabel("proposal std $\sigma$", fontsize=fontsize)
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig('img/illustrative_DAMH_proposal_tau.pdf')  

plt.figure()
F = va.fit(proposal_scale,acceptance_rate,deg=deg)
yy_rate = F(xx)
plt.plot(xx, yy_rate, linewidth=linewidth)
plt.plot(proposal_scale,acceptance_rate,'.', markersize=markersize)
plt.ylabel("average acceptance rate", fontsize=fontsize)
plt.xlabel("proposal std $\sigma$", fontsize=fontsize)
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig('img/illustrative_DAMH_proposal_rate.pdf')  

""" Autocorelation depending on acceptance rate """
plt.figure()
F = va.fit(acceptance_rate,S.tau_aggregate[alg_range],deg=deg)
xxinv = np.linspace(min(acceptance_rate),max(acceptance_rate),100)
yy = F(xxinv)
plt.plot(xxinv, yy, linewidth=linewidth)
plt.plot(acceptance_rate,S.tau_aggregate[alg_range],'.', markersize=markersize)
plt.xlabel("average acceptance rate", fontsize=fontsize)
plt.ylabel("estimation of $\\tau$", fontsize=fontsize)
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig('img/illustrative_DAMH_rate_tau.pdf')  

""" CpUS as a function of proposal scale """
surr_cost_ratio = 0.01
CpUS_aggregate = S.calculate_CpUS(no_samplers, surr_cost_ratio = surr_cost_ratio)
plt.figure()
# F = va.fit(proposal_scale,CpUS_aggregate[alg_range],deg=deg+1)
# yy = F(xx)
# plt.plot(xx, yy, linewidth=linewidth)
plt.plot(xx, (yy_rate+surr_cost_ratio)*yy_tau, linewidth=linewidth)
plt.plot(proposal_scale,CpUS_aggregate[alg_range],'.', markersize=markersize)
plt.title("CpUS calculated from all chains")
plt.grid()
plt.show()

plt.rcParams['font.size'] = '14'
fontsize = 16
markersize = 9
linewidth = 2

plt.figure()
values = [0.2, 0.1, 0.01, 0]
colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
for i,surr_cost_ratio in enumerate(values):
    yy = (yy_rate+surr_cost_ratio)*yy_tau
    plt.plot(xx, yy, linewidth=linewidth, color=colors[i])
legend = ["cost = " + str(ii) for ii in values]
plt.legend(legend)
for i,surr_cost_ratio in enumerate(values):
    yy = (yy_rate+surr_cost_ratio)*yy_tau
    idx = np.argmin(yy)
    plt.plot(xx[idx],yy[idx],'.',markersize=markersize, color=colors[i])
plt.xlabel("proposal std $\sigma$", fontsize=fontsize)
plt.ylabel("CpUS", fontsize=fontsize)
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig('img/illustrative_DAMH_proposal_cpus.pdf')  

# chain illustration 
plt.figure()
S.plot_hist_2d(bins=50)
plt.xlabel("$u_1$")
plt.ylabel("$u_2$")
#plt.colorbar()
plt.axis("equal")
plt.xlim([0,10])
plt.ylim([0,10])
plt.show()

plt.figure()
S.plot_raw_data(folder_samples, no_parameters, chains_range=[63], end_disp = 200) 
plt.axis("equal")
plt.xlim([0,10])
plt.ylim([0,10])
plt.tight_layout()
plt.show()
plt.savefig('img/illustrative_DAMH200.pdf')  
