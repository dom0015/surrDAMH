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

# import os
# path = os.path.abspath(os.path.dirname(__file__)) # file directory 
# path_up = os.path.dirname(os.path.dirname(path))
# conf_path = path_up + "/conf/Darcy.json" 
conf_path = "examples/Darcy.json" 

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 4

### REFERENCE SOLUTION:
# module_path = path_up + "/" + conf["solver_module_path"] 
if "paths_to_append" in conf.keys():
    for path in conf["paths_to_append"]:
        sys.path.append(path)
spec = iu.spec_from_file_location(conf["solver_module_name"], conf["solver_module_path"])
solver_module = iu.module_from_spec(spec)
spec.loader.exec_module(solver_module)
solver_init = getattr(solver_module, conf["solver_init_name"])     
solver_instance = solver_init(**conf["solver_parameters"])
reference_parameters = np.array([-1, -0.5, 0.5, 1])
solver_instance.set_parameters(reference_parameters)
reference_observations = solver_instance.get_observations()
print('ref. obs.:',reference_observations)
G = va.grf_eigenfunctions.GRF("surrDAMH/modules/unit30.pckl", truncate=100)
n = conf["solver_parameters"]['n']
G.plot_realization_interfaces(quantiles=[0.25, 0.5, 0.75, 1.0], nx_new=n, ny_new=n)
solver_instance.all_solvers[0].plot_solution_image()

### SAMPLES VISUALIZATION:
problem_name = conf["problem_name"]
S = va.Samples()
no_parameters = conf["no_parameters"]
S.load_notes('saved_samples/' + problem_name,no_samplers)
S.load_MH('saved_samples/' + problem_name,no_parameters)

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
S.extract_chains(range(setnumber*no_samplers,(setnumber+1)*no_samplers)) # keep only the corresponding chains
S.calculate_properties()
S.print_properties()
S.plot_average(show_legend = True)
S.plot_hist_grid(bins1d=20, bins2d=20)
