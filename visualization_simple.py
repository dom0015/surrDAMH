#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import numpy as np
import matplotlib.pyplot as plt
from modules import visualization_and_analysis as va
from modules import grf_eigenfunctions as grf
#from configuration import Configuration
import configuration

C = Configuration()

### SAMPLES VISUALIZATION:
problem_name = C.problem_name
S = va.Samples()
no_samplers = C.no_samplers # number of MH/DAMH chains
no_parameters = C.no_parameters
S.load_notes('saved_samples/' + problem_name,no_samplers)
S.load_MH('saved_samples/' + problem_name,no_parameters)

# components 0,1 of a MH + DAMH-SMU + DAMH chain:
no_chain=1;
plt.figure()
plt.plot(S.x[no_chain][:,0],S.x[no_chain][:,1],alpha=0.5,marker=".",color='k')
plt.plot(S.x[no_chain+no_samplers][:,0],S.x[no_chain+no_samplers][:,1],alpha=0.5,marker=".",color="r")
plt.plot(S.x[no_chain+2*no_samplers][:,0],S.x[no_chain+2*no_samplers][:,1],alpha=0.5,marker=".",color="b")
plt.grid()
plt.show()

# Which part of the sampling process is analyzed? 0/1/2 = MH/DAMH-SMU/DAMH
setnumber = 2;
S.extract_chains(range(setnumber*no_samplers,(setnumber+1)*no_samplers)) # keep only the corresponding chains
S.calculate_properties()
S.print_properties()
S.plot_average(show_legend = True)
S.plot_hist_grid(bins1d=40, bins2d=40)