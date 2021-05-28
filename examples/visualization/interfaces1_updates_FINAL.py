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
sys.path.append(wdir)
from surrDAMH.modules import visualization_and_analysis as va

plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3

### PARAMETERS:
no_samplers = 30 # number of sampling processes
savefig_name = "interfaces1_updates"
#conf_names = ["interfaces1_updates_rbf500p", "interfaces1_updates_rbf500_0", "interfaces1_updates_rbf501", "interfaces1_updates_rbf500", "interfaces1_updates_rbf250", "interfaces1_updates_rbf100", "interfaces1_updates_rbf500_2", "interfaces1_updates_poly8", "interfaces1_updates_poly6"]
#conf_names = ["interfaces1_updates_rbf500", "interfaces1_updates_rbf501", "interfaces1_updates_rbf500b", "interfaces1_updates_rbf500c", "interfaces1_updates_poly8", "interfaces1_updates_poly8b", "interfaces1_updates_poly8c", "interfaces1_updates_poly8d"]
#conf_names = ["interfaces1_updates_rbf500_test1", "interfaces1_updates_rbf500_test2", "interfaces1_updates_rbf500_test3", "interfaces1_updates_rbf500_test4", "interfaces1_updates_rbf500_test5","interfaces1_updates_rbf250_test1", "interfaces1_updates_rbf250_test2", "interfaces1_updates_rbf250_test3", "interfaces1_updates_rbf250_test4", "interfaces1_updates_rbf250_test5", "interfaces1_updates_poly8_test1", "interfaces1_updates_poly8_test2", "interfaces1_updates_poly8_test3", "interfaces1_updates_poly8_test4", "interfaces1_updates_poly8_test5"]
#conf_names = ["grf1_updates_rbf500_test1", "grf1_updates_rbf500_test2", "grf1_updates_rbf500_test3", "grf1_updates_rbf500_test4", "grf1_updates_rbf500_test5", "grf1_updates_poly8_test1", "grf1_updates_poly8_test2", "grf1_updates_poly8_test3", "grf1_updates_poly8_test4", "grf1_updates_poly8_test5"]
#conf_names = ["interfaces1_updates_poly8_stopped_test1", "interfaces1_updates_poly8_stopped_test2", "interfaces1_updates_poly8_stopped_test3", "interfaces1_updates_poly8_stopped_test4", "interfaces1_updates_poly8_stopped_test5", "grf1_updates_poly8_stopped_test1", "grf1_updates_poly8_stopped_test2", "grf1_updates_poly8_stopped_test3", "grf1_updates_poly8_stopped_test4", "grf1_updates_poly8_stopped_test5"]
#conf_names = ["interfaces1_updates_poly8_stopped0_test1", "interfaces1_updates_poly8_stopped0_test2", "interfaces1_updates_poly8_stopped0_test3", "interfaces1_updates_poly8_stopped0_test4", "interfaces1_updates_poly8_stopped0_test5", "grf1_updates_poly8_stopped0_test1", "grf1_updates_poly8_stopped0_test2", "grf1_updates_poly8_stopped0_test3", "grf1_updates_poly8_stopped0_test4", "grf1_updates_poly8_stopped0_test5"]
#conf_names = ["interfaces1_updates_rbf500_test1"]#,"grf1_updates_rbf500_test1"]

# GRF poly8 surrogate non-interrupted, interrupted at 1e4, interrupted at 5e4
#conf_names = ["grf1_updates_poly8_test1", "grf1_updates_poly8_test2", "grf1_updates_poly8_test3", "grf1_updates_poly8_test4", "grf1_updates_poly8_test5"]
#conf_names = ["grf1_updates_poly8_stopped_test1", "grf1_updates_poly8_stopped_test2", "grf1_updates_poly8_stopped_test3", "grf1_updates_poly8_stopped_test4", "grf1_updates_poly8_stopped_test5"]
conf_names = ["grf1_updates_poly8_stopped0_test1", "grf1_updates_poly8_stopped0_test2", "grf1_updates_poly8_stopped0_test3", "grf1_updates_poly8_stopped0_test4", "grf1_updates_poly8_stopped0_test5"]
# GRF rbf500:
#conf_names = ["grf1_updates_rbf500_test1", "grf1_updates_rbf500_test2", "grf1_updates_rbf500_test3", "grf1_updates_rbf500_test4", "grf1_updates_rbf500_test5"]

# INTERFACES poly8 surrogate non-interrupted, interrupted at 1e4, interrupted at 5e4
#conf_names = ["interfaces1_updates_poly8_test1", "interfaces1_updates_poly8_test2", "interfaces1_updates_poly8_test3", "interfaces1_updates_poly8_test4", "interfaces1_updates_poly8_test5"]
#conf_names = ["interfaces1_updates_poly8_stopped_test1", "interfaces1_updates_poly8_stopped_test2", "interfaces1_updates_poly8_stopped_test3", "interfaces1_updates_poly8_stopped_test4", "interfaces1_updates_poly8_stopped_test5"]
#conf_names = ["interfaces1_updates_poly8_stopped0_test1", "interfaces1_updates_poly8_stopped0_test2", "interfaces1_updates_poly8_stopped0_test3", "interfaces1_updates_poly8_stopped0_test4", "interfaces1_updates_poly8_stopped0_test5"]
# GRF rbf500:
#conf_names = ["interfaces1_updates_rbf500_test1", "interfaces1_updates_rbf500_test2", "interfaces1_updates_rbf500_test3", "interfaces1_updates_rbf500_test4", "interfaces1_updates_rbf500_test5"]


colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
for conf_name in conf_names:
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
    # for i in range(no_alg):
    #     print("Notes - ALGORTIHM " + str(i) + ":")
    #     print(S.notes[i])
    S.load_MH(folder_samples, no_parameters)
    # S.calculate_autocorr_time(tol=1)
    S.print_properties()
    
    ### SAMPLES VISUALIZATION: 
    parameters_disp = range(min(no_parameters,5))
    rejected_sum = 0
    accepted_sum = 0
    begin = 1
    end_disp = 125000 # 100000 # None
    for i in [1,2]:
        chains_range=range(i*no_samplers,(i+1)*no_samplers)
        #S.plot_average(parameters_disp = parameters_disp, chains_disp = range(i*no_samplers,(i+1)*no_samplers), show_legend = False, sharey=False)
        rejected, accepted = S.plot_rejected_cumsum(folder_samples, no_parameters, chains_range=chains_range, end_disp = end_disp, plot=False)
        length = len(rejected)
        xx = range(begin,begin+length)
        rejected += rejected_sum
        accepted += accepted_sum
        color = colors[0]
        if 'poly' in conf_name:
            color = colors[1]
        if 'stopped_' in conf_name:
            color = colors[2]
        if 'stopped0' in conf_name:
            color = colors[3]
        plt.figure(11)
        plt.plot(xx,rejected,color=color)
        plt.figure(12)
        plt.plot(xx,accepted,color=color)
        rejected_sum = rejected[-1]
        accepted_sum = accepted[-1]
        begin = length
        time_G, time_GS = S.plot_evaluation_time(folder_samples, no_parameters, chains_range=chains_range, plot=False)
        print("Observation operator evaluation time:", np.mean(time_G))
        print("Surrogate model evaluation time:", np.mean(time_GS))
        print("ratio:", np.mean(time_GS)/np.mean(time_G))
#legend = ['rbf(500)','_nolegend_','_nolegend_','_nolegend_','poly(8)','_nolegend_','_nolegend_','_nolegend_']
#legend = ['rbf(500)','_nolegend_','_nolegend_','_nolegend_','_nolegend_','rbf(250)','_nolegend_','_nolegend_','_nolegend_','_nolegend_','poly(8)','_nolegend_','_nolegend_','_nolegend_','_nolegend_']
#legend = ['rbf(500)','_nolegend_','_nolegend_','_nolegend_','_nolegend_','poly(8)','_nolegend_','_nolegend_','_nolegend_','_nolegend_']
legend = ['model problem 1'] + 9*['_nolegend_'] + ['model problem 2'] + 9*['_nolegend_']
#legend = ['with updates'] + 4*['_nolegend_'] + ['stopped after 5e5 samples'] + 9*['_nolegend_']+ ['stopped after 1e5 samples'] + 9*['_nolegend_']
#legend = ["interfaces"]#, "grf"]
plt.figure(11)
plt.xlabel("number of samples")
plt.ylabel("number of rejected samples (mean)")
plt.legend(legend)
plt.grid()
plt.tight_layout()
#plt.savefig('img/' + savefig_name + '_rejected.pdf')  
plt.figure(12)
plt.xlabel("number of samples")
plt.ylabel("number of accepted samples (mean)")
plt.legend(legend)
plt.grid()
plt.tight_layout()
#plt.savefig('img/' + savefig_name + '_accepted.pdf')  
plt.show()

# plt.figure(11)
# plt.plot([50000,50000],[0,120],'--',color=colors[2])
# plt.plot([10000,10000],[0,120],'--',color=colors[3])
# plt.figure(12)
# plt.plot([50000,50000],[0,120],'--',color=colors[2])
# plt.plot([10000,10000],[0,120],'--',color=colors[3])