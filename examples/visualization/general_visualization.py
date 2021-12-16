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

conf_path = sys.argv[2]
basename = os.path.basename(conf_path)
problem_name, fext = os.path.splitext(basename)

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 1

### SAMPLES VISUALIZATION:
S = va.Samples()
no_parameters = conf["no_parameters"]
S.load_notes('saved_samples/' + problem_name,no_samplers)
S.load_MH('saved_samples/' + problem_name,no_parameters)

S.calculate_properties()
S.print_properties()
S.plot_hist_grid(bins1d=30, bins2d=30)
plt.savefig('saved_samples/' + problem_name + "/histograms.pdf",bbox_inches="tight")
S.plot_segment()
plt.savefig('saved_samples/' + problem_name + "/chains.pdf",bbox_inches="tight")