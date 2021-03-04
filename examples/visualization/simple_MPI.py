#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import matplotlib.pyplot as plt
import os
import sys
import json
sys.path.append(os.getcwd())
from surrDAMH.modules import visualization_and_analysis as va

saved_samples_name = "simple_MPI"
conf_path = "examples/simple_MPI.json" 

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 4

### SAMPLES VISUALIZATION:
S = va.Samples()
no_parameters = conf["no_parameters"]
S.load_notes('saved_samples/' + saved_samples_name,no_samplers)
S.load_MH('saved_samples/' + saved_samples_name,no_parameters)

# components 0,1 of a MH + DAMH-SMU + DAMH chain:
no_chain=0;
plt.figure()
plt.plot(S.x[no_chain][:,0],S.x[no_chain][:,1],alpha=0.5,marker=".",color='k')
plt.plot(S.x[no_chain+no_samplers][:,0],S.x[no_chain+no_samplers][:,1],alpha=0.5,marker=".",color="r")
plt.plot(S.x[no_chain+2*no_samplers][:,0],S.x[no_chain+2*no_samplers][:,1],alpha=0.5,marker=".",color="b")
plt.legend(["MH","DAMH-SMU","DAMH"])
plt.title("sampling process" + str(no_chain))
plt.grid()
plt.show()

# Which part of the sampling process is analyzed? 0/1/2 = MH/DAMH-SMU/DAMH
setnumber = 2;
S.calculate_properties()
S.print_properties()
chains_disp = range(setnumber*no_samplers,(setnumber+1)*no_samplers)
S.plot_hist_grid(chains_disp = chains_disp, bins1d=30, bins2d=30)
S.plot_average(chains_disp = chains_disp, show_legend = True)
