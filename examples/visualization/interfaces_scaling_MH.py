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
conf_name = "illustrative_scaling2" # requires configuration file "conf/" + conf_name + ".json"
conf_name = "interfaces1_scaling_MH_long"
no_samplers = 30 # number of sampling processes
savefig_name = "interfaces1_scaling_MH"

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
# saved_samples_name = "illustrative_scaling2"
# saved_samples_name = "interfaces_scaling_MH"
no_parameters = conf["no_parameters"]
list_alg = conf["samplers_list"]

### LOAD SAMPLES:
folder_samples = wdir + "/saved_samples/" + conf_name
S = va.Samples()
S.load_notes(folder_samples, no_samplers)
no_alg = len(list_alg) #12
for i in range(no_alg):
    print("Notes - ALGORTIHM " + str(i) + ":")
    print(S.notes[i])
S.load_MH(folder_samples, no_parameters)

proposal_scale = [None] * no_alg
proposal_scale_all = [None] * no_alg *no_samplers
representative_chains = [None] * no_alg
acceptance_rate = [None] * no_alg
for i in range(no_alg):
    proposal_scale[i] = np.array(S.notes[i]["proposal_scale"])[0]
    proposal_scale_all[i*no_samplers:(i+1)*no_samplers] = np.array(S.notes[i]["proposal_scale"])
    representative_chains[i] = range(i*no_samplers,(i+1)*no_samplers)[0]
    acceptance_rate[i] = np.mean(S.notes[i]["acceptance_rate"])
deg = 2
xx = np.linspace(min(proposal_scale),max(proposal_scale),100)

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
plt.tight_layout()
# plt.show()

plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3

""" Autocorrelation time calculated from all chains """
#S.calculate_burn_in(no_samplers, multiplier=2, c=5)
S.calculate_tau(no_samplers, c=5, smooth=2)
plt.figure()
for i in range(no_alg):
    chain = representative_chains[i]
    #plt.plot(proposal_scale[i],S.tau[chain],'.-r')
F = va.fit(proposal_scale,S.tau_aggregate,deg=deg)
yy = F(xx)
idx = np.argmin(yy)
print("OPTIMAL:",xx[idx],yy[idx])
plt.plot(xx, yy, linewidth=linewidth)
plt.plot(proposal_scale, S.tau_aggregate, '.', markersize=markersize)
plt.ylabel("estimation of $\\tau$", fontsize=fontsize)
plt.xlabel("proposal SD", fontsize=fontsize)
plt.grid()
plt.tight_layout()
#plt.show()
plt.savefig('img/' + savefig_name + '_proposal_tau.pdf')  

# S.remove_burn_in()
# S.calculate_tau(no_samplers)
# plt.figure()
# for i in range(no_alg):
#     chain = representative_chains[i]
#     plt.plot(proposal_scale[i],S.tau[chain],'.-g')
# nan_idx = np.isnan(S.tau[representative_chains])
# valid = nan_idx == False
# x = np.array(proposal_scale)[valid]
# y = S.tau[np.array(representative_chains)[valid]]
# #F = np.polynomial.Polynomial.fit(x = x, y = y, deg = 4)
# F = va.fit(x,y,deg=deg)
# yy = F(xx)
# plt.plot(xx, yy)
# #argmin = F.deriv().roots()
# #plt.plot(argmin,F(argmin),'.')
# idx = np.argmin(yy)
# plt.plot(xx[idx],yy[idx],'.')
# plt.title("Autocorrelation time calculated from all chains (after burn-in removal)")
# plt.grid()
# plt.show()

plt.figure()
# F = va.fit(proposal_scale,acceptance_rate,deg=deg)
# yy = F(xx)
# plt.plot(xx, yy, linewidth=linewidth)
plt.plot(proposal_scale,acceptance_rate,'.', markersize=markersize)
plt.ylabel("average acceptance rate", fontsize=fontsize)
plt.xlabel("proposal SD", fontsize=fontsize)
plt.grid()
plt.tight_layout()
# plt.show() 
plt.savefig('img/' + savefig_name + '_proposal_rate.pdf')  

""" Autocorelation depending on acceptance rate """
plt.figure()
F = va.fit(acceptance_rate,S.tau_aggregate,deg=deg)
xxinv = np.linspace(min(acceptance_rate),max(acceptance_rate),100)
yy = F(xxinv)
plt.plot(xxinv, yy, linewidth=linewidth)
plt.plot(acceptance_rate,S.tau_aggregate,'.', markersize=markersize)
plt.xlabel("average acceptance rate", fontsize=fontsize)
plt.ylabel("estimation of $\\tau$", fontsize=fontsize)
plt.grid()
plt.tight_layout()
# plt.show()
# plt.savefig('img/illustrative_rate_tau.pdf')  

# CpUS_aggregate = S.calculate_CpUS(no_samplers, surr_cost_ratio = 0.0)
# plt.figure()
# plt.plot(proposal_scale,CpUS_aggregate,'.')
# F = va.fit(proposal_scale,CpUS_aggregate,deg=deg)
# yy = F(xx)
# plt.plot(xx, yy)
# plt.title("CpUS calculated from all chains (after burn-in removal)")
# plt.grid()
# plt.show()

# plt.figure()
# for i in range(no_alg):
#     chain = representative_chains[i]
#     tmp = S.autocorr_function_mean[chain][:round(3*max(S.tau)),0]
#     plt.plot(tmp, label=i)
# plt.title("Autocorrelation function mean [0] after burn-in removal")
# plt.legend()
# plt.grid()
# plt.show()

# plt.rcParams['font.size'] = '14'
# fontsize = 16
# markersize = 9
# linewidth = 2

# # chain illustration 
# plt.figure()
# S.plot_hist_2d(bins=50)
# plt.xlabel("$u_1$")
# plt.ylabel("$u_2$")
# plt.colorbar()
# plt.axis("equal")
# plt.xlim([0,10])
# plt.ylim([0,10])
# plt.tight_layout()
# # plt.show()
# # plt.savefig('img/illustrative_hist2d.pdf')  

# plt.figure()
# # S.plot_raw_data(folder_samples, no_parameters, chains_range=[0], end_disp = 100) # 0.5
# S.plot_raw_data(folder_samples, no_parameters, chains_range=[26*10+4], end_disp = 200) # 1.0
# # S.plot_raw_data(folder_samples, no_parameters, chains_range=[8*10], end_disp = 100) # 2.2
# # S.plot_raw_data(folder_samples, no_parameters, chains_range=[23*10], end_disp = 100) # 3.5
# plt.axis("equal")
# plt.xlim([0,10])
# plt.ylim([0,10])
# plt.tight_layout()
# # plt.show()
# # plt.savefig('img/illustrative_MH200.pdf')  
