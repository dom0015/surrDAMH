#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import sys
from modules import visualization_and_analysis as va
from configuration import Configuration

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 4
C = Configuration(no_samplers,"simple")

### SAMPLES VISUALIZATION:
problem_name = C.problem_name
S = va.Samples()
no_parameters = C.no_parameters
S.load_notes('saved_samples/' + problem_name,no_samplers)
S.load_MH('saved_samples/' + problem_name,no_parameters)

# Which part of the sampling process is analyzed? 0/1/2 = MH/DAMH-SMU/DAMH
setnumber = 2;
S.extract_chains(range(setnumber*no_samplers,(setnumber+1)*no_samplers)) # keep only the corresponding chains
S.calculate_properties()
S.print_properties()
S.plot_hist_grid(bins1d=30, bins2d=30)
S.plot_average(show_legend = True)
