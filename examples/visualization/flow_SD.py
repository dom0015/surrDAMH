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

problem_name = "flow_SD"
problem_name = "flow_SD_lhs_lower_var"
conf_path = "/home/domesova/GIT/Endorse-2Dtest-Bayes/flow_SD.json"
folder_samples = 'saved_samples/' + problem_name

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 30
print("no samplers:", no_samplers)

### SAMPLES VISUALIZATION:
S = va.Samples()
no_parameters = conf["no_parameters"]
no_observations = conf["no_observations"]
S.load_notes(folder_samples,no_samplers)
S.load_MH(folder_samples,no_parameters)

# Which part of the sampling process is analyzed? 0/1/2 = MH/DAMH-SMU/DAMH
for setnumber in range(1):
    #try:
    print("set number:", setnumber)
    #S.extract_chains(range(setnumber*no_samplers,(setnumber+1)*no_samplers)) # keep only the corresponding chains
    S.calculate_properties()
    S.print_properties()
    chains_disp = range(setnumber*no_samplers,(setnumber+1)*no_samplers)
    S.plot_hist_grid(chains_disp = chains_disp, bins1d=30, bins2d=30)
    S.plot_average(chains_disp = chains_disp, show_legend = True)
    # except:
    #     print("Set " + str(setnumber) + " does not exist.")

import matplotlib.pyplot as plt
sys.path.append("/home/domesova/GIT/Endorse-2Dtest-Bayes")
import transformations
def transform(param):
    L = len(param)
    for i in range(L):
        param[i][:,0] = transformations.normal_to_lognormal(param[i][:,0])
        param[i][:,1] = transformations.normal_to_beta(param[i][:,1],alfa=5,beta=5)

plt.figure()
S.analyze_raw_data(folder_samples, no_parameters, no_observations, chains_range = list(range(0,30)), begin_disp = 0, end_disp = None)
# sample_type, parameters, tag, observations = S.get_raw_data(folder_samples, no_parameters, no_observations, chains_range = list(range(0,30)))
# transform(parameters)
# #plt.figure()
# for i in range(len(tag)):
#     print(i, len(tag[i]), sum(tag[i]>0), sum(tag[i]<0))
#     color = tag[i].copy()
#     color[sample_type[i]=="accepted"] = 1
#     color[sample_type[i]=="rejected"] = 2
#     color[sample_type[i]=="prerejected"] = 3
#     color[tag[i]<0] = 4
#     print(i,sum(color==4))
#     plt.scatter(parameters[i][:,0],parameters[i][:,1], s=10, c=color)
# sample_type, parameters, tag, observations = S.get_raw_data(folder_samples, no_parameters, no_observations, chains_range = list(range(30,60)))
# transform(parameters)
# plt.figure()
# for i in range(len(tag)):
#     print(i, len(tag[i]), sum(tag[i]>0), sum(tag[i]<0))
#     color = tag[i].copy()
#     color[sample_type[i]=="accepted"] = 1
#     color[sample_type[i]=="rejected"] = 2
#     color[sample_type[i]=="prerejected"] = 3
#     color[tag[i]<0] = 4
#     print(i,sum(color==4))
#     plt.scatter(parameters[i][:,0],parameters[i][:,1], s=5, c=color)
# sample_type, parameters, tag, observations = S.get_raw_data(folder_samples, no_parameters, no_observations, chains_range = list(range(60,90)))
# transform(parameters)
# plt.figure()
# for i in range(len(tag)):
#     print(i, len(tag[i]), sum(tag[i]>0), sum(tag[i]<0))
#     color = tag[i].copy()
#     color[sample_type[i]=="accepted"] = 1
#     color[sample_type[i]=="rejected"] = 2
#     color[sample_type[i]=="prerejected"] = 3
#     color[tag[i]<0] = 4
#     print(i,sum(color==4))
#     plt.scatter(parameters[i][:,0],parameters[i][:,1], s=2, c=color)


chains_range = list(range(0,90))
sample_type, parameters, tag, observations = S.get_raw_data(folder_samples, no_parameters, no_observations, chains_range = chains_range)
transform(parameters)
# only non converging:
import numpy as np
plt.figure()
all_non_converging = np.zeros((0,2))
for i in range(len(tag)):
    choice = tag[i]<0
    plt.scatter(parameters[i][choice,0],parameters[i][choice,1], s=5, c="tab:red")
    all_non_converging = np.vstack((all_non_converging,parameters[i][choice,:]))
plt.title("non converging")
    
# only accepted:
plt.figure()
for i in range(len(tag)):
    choice = sample_type[i]=="accepted"
    plt.scatter(parameters[i][choice,0],parameters[i][choice,1], s=5, c="tab:green")
plt.title("accepted")
    
# only rejected:
plt.figure()
for i in range(len(tag)):
    choice = sample_type[i]=="rejected"
    plt.scatter(parameters[i][choice,0],parameters[i][choice,1], s=5, c="tab:blue")
plt.title("rejected")