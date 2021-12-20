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
import matplotlib.pyplot as plt
import numpy as np

conf_path = sys.argv[2]
basename = os.path.basename(conf_path)
problem_name, fext = os.path.splitext(basename)

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 1

### PREPARATION:
S = va.Samples()
no_parameters = conf["no_parameters"]
scale = ["linear"]*no_parameters
if "transformations" in conf.keys():
    transformations = conf["transformations"]
    for i in range(no_parameters):
        if transformations[i][0] == "normal_to_lognormal":
            scale[i] = "log"
S.load_notes('saved_samples/' + problem_name,no_samplers)
S.load_MH('saved_samples/' + problem_name,no_parameters)
S.calculate_properties()
S.print_properties()
S.load_MH_with_posterior('saved_samples/' + problem_name,no_parameters)
print("modus: ", S.find_modus())
S.find_max_likelihood('saved_samples/' + problem_name,no_parameters,conf["problem_parameters"]["observations"])

### SAMPLES VISUALIZATION:
no_stages = int(S.no_chains/no_samplers)
for i in range(no_stages):
    chains_disp=range(i*no_samplers,(i+1)*no_samplers)
    S.plot_hist_grid(chains_disp=chains_disp, bins1d=30, bins2d=30, scale=scale)
    S.plot_hist_grid_add(chains_disp=chains_disp, scale=scale)
    plt.savefig('saved_samples/' + problem_name + "/histograms" +str(i)+ ".pdf",bbox_inches="tight")
    S.plot_segment(chains_disp=chains_disp,scale=scale)
    plt.savefig('saved_samples/' + problem_name + "/chains" +str(i)+ ".pdf",bbox_inches="tight")
    S.plot_average(chains_disp=chains_disp,scale=scale)
    plt.savefig('saved_samples/' + problem_name + "/average" +str(i)+ ".pdf",bbox_inches="tight")