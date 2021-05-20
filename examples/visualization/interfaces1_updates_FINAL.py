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

### PARAMETERS:
no_samplers = 30 # number of sampling processes
savefig_name = "interfaces1_updates"
#conf_names = ["interfaces1_updates_rbf500p", "interfaces1_updates_rbf500_0", "interfaces1_updates_rbf501", "interfaces1_updates_rbf500", "interfaces1_updates_rbf250", "interfaces1_updates_rbf100", "interfaces1_updates_rbf500_2", "interfaces1_updates_poly8", "interfaces1_updates_poly6"]
conf_names = ["interfaces1_updates_rbf500", "interfaces1_updates_rbf501", "interfaces1_updates_rbf500b", "interfaces1_updates_rbf500c", "interfaces1_updates_poly8", "interfaces1_updates_poly8b", "interfaces1_updates_poly8c", "interfaces1_updates_poly8d"]
#legend = conf_names
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
    S.calculate_autocorr_time(tol=1)
    S.print_properties()
    
    ### SAMPLES VISUALIZATION: 
    parameters_disp = range(min(no_parameters,5))
    for i in [1]:
        #S.plot_average(parameters_disp = parameters_disp, chains_disp = range(i*no_samplers,(i+1)*no_samplers), show_legend = False, sharey=False)
        rejected, accepted = S.plot_rejected_cumsum(folder_samples, no_parameters, chains_range=range(i*no_samplers,(i+1)*no_samplers), end_disp = None, plot=False)
        if 'rbf' in conf_name:
            plt.figure(11)
            plt.plot(rejected,color=colors[0])
            plt.figure(12)
            plt.plot(accepted,color=colors[0])
        else:
            plt.figure(11)
            plt.plot(rejected,color=colors[1])
            plt.figure(12)
            plt.plot(accepted,color=colors[1])
        time_G, time_GS = S.plot_evaluation_time(folder_samples, no_parameters, chains_range=range(i*no_samplers,(i+1)*no_samplers), plot=False)
        print("Observation operator evaluation time:", np.mean(time_G))
        print("Surrogate model evaluation time:", np.mean(time_GS))
        print("ratio:", np.mean(time_GS)/np.mean(time_G))
legend = ['rbf(500)','_nolegend_','_nolegend_','_nolegend_','poly(8)','_nolegend_','_nolegend_','_nolegend_']
plt.figure(11)
plt.xlabel("number of samples")
plt.ylabel("number of rejected samples (mean)")
plt.legend(legend)
plt.grid()
plt.tight_layout()
plt.savefig('img/' + savefig_name + '_rejected.pdf')  
plt.figure(12)
plt.xlabel("number of samples")
plt.ylabel("number of accepted samples (mean)")
plt.legend(legend)
plt.grid()
plt.tight_layout()
plt.savefig('img/' + savefig_name + '_accepted.pdf')  
plt.show()