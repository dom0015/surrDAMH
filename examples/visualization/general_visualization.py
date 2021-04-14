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

# path = os.path.abspath(os.path.dirname(__file__)) # file directory 
# path_up = os.path.dirname(os.path.dirname(path))
# conf_path = path_up + "/conf/simple.json" 

if len(sys.argv)>2:
    problem_name = sys.argv[2]
    conf_path = "examples/" + problem_name + ".json"
else:
    conf_path = "examples/simple.json"

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 4

### SAMPLES VISUALIZATION:
if "problem_name" in conf.keys():
    problem_name = conf["problem_name"]
S = va.Samples()
no_parameters = conf["no_parameters"]
S.load_notes('saved_samples/' + problem_name,no_samplers)
S.load_MH('saved_samples/' + problem_name,no_parameters)

# Which part of the sampling process is analyzed? 0/1/2 = MH/DAMH-SMU/DAMH
setnumber = 0;
S.extract_chains(range(setnumber*no_samplers,(setnumber+1)*no_samplers)) # keep only the corresponding chains
S.calculate_properties()
S.print_properties()
S.plot_hist_grid(bins1d=30, bins2d=30)
S.plot_average(show_legend = True)
