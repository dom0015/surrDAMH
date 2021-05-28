#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib.util as iu
import os
import sys
import json
sys.path.append(os.getcwd())
from surrDAMH.modules import visualization_and_analysis as va

plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3
    
no_samplers = 30

problems = ["interfaces_test1","interfaces_test2","interfaces_test3","interfaces_test4"]

fig, axes = plt.subplots(1, 4, sharex=False, sharey=False, figsize = [6.4*2, 2.35])

for problem_name in problems:
    saved_samples_name = problem_name
    conf_path = "examples/" + problem_name +".json" 
    savefig_name = problem_name
    with open(conf_path) as f:
        conf = json.load(f)
    
    ### SAMPLES VISUALIZATION:
    S = va.Samples()
    no_parameters = conf["no_parameters"]
    S.load_notes('saved_samples/' + saved_samples_name,no_samplers)
    S.load_MH('saved_samples/' + saved_samples_name,no_parameters)

    setnumber = 2;
    S.calculate_properties()
    S.print_properties()
    chains_disp = range(setnumber*no_samplers,(setnumber+1)*no_samplers)
    
    S.plot_hist_1d_multi(chains_disp = chains_disp, bins=29, show_title=False, sharex = False, sharey = False, figure=False, show=False, hist=False, density=True)
    fig = plt.gcf()
    fig.set_size_inches(12.8, 2.35, forward=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, left=0.06, bottom=0.15, top=0.8)

plt.show()