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

os.chdir('/home/domesova/GIT/MCMC-Bayes-python')

#problem_name = "grf_test1"
#problem_name = "grf_test2"
#problem_name = "grf_test3"
#problem_name = "grf_test4"
#problem_name = "grf_test4_20"
#problem_name = "grf_test4_40"
problem_name = "grf_test4_40_rbf2000"
#problem_name = "grf_test4_80_rbf2000"

saved_samples_name = problem_name
conf_path = "examples/" + problem_name + ".json" 

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 30
    
    
plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3

### REFERENCE SOLUTION:
if "paths_to_append" in conf.keys():
    for path in conf["paths_to_append"]:
        sys.path.append(path)
spec = iu.spec_from_file_location(conf["solver_module_name"], conf["solver_module_path"])
solver_module = iu.module_from_spec(spec)
spec.loader.exec_module(solver_module)
solver_init = getattr(solver_module, conf["solver_init_name"])     
solver_instance = solver_init(**conf["solver_parameters"])
n = conf["solver_parameters"]['n'] # (60)
n = 200
no_ref_parameters = 900
#solver_instance = solver_init(no_ref_parameters, 20, 4, n, grf_filename = None, quiet = True, tolerance = 1e-8, PC = "icc", use_deflation = False, deflation_imp = 1e-2) 
solver_instance = solver_init(no_parameters=no_ref_parameters,no_observations=20,no_configurations=4, n=n, grf_filename = None, quiet = True, tolerance = 1e-8, PC = "icc", use_deflation = False, deflation_imp = 1e-2) 
reference_parameters = np.ones((no_ref_parameters,))
solver_instance.set_parameters(reference_parameters)
reference_observations = solver_instance.get_observations()
print('ref. obs.:',reference_observations)
from surrDAMH.modules import visualization_and_analysis as va
G = va.grf_eigenfunctions.GRF("surrDAMH/modules/unit30.pckl", truncate=no_ref_parameters)

G.plot_grf(reference_parameters,nx_new=200,ny_new=200)
#G.plot_realization_interfaces(quantiles=[0.25, 0.5, 0.75, 1.0], nx_new=n, ny_new=n)
# for i in range(1):
#     solver_instance.all_solvers[i].plot_solution_image(flow=True)

solver_instance.plot_problem_left(flow=True)

""" VISUALIZATION OF WINDOWS """
no_windows = 5
color1 = "orange"
color2 = "red"
for i in range(no_windows - 1):
    color = color1
    if np.mod(i,2)==0:
        color = color2
    length = 1.0/(no_windows-1)
    plt.plot([1,1],[i*length,(i+1)*length],linewidth=4,color=color)
    label = "$S_{"+ str(i+1) +"}$"
    plt.text(1-0.1,(i+0.4)*length,label,color=color)
plt.plot([0,0],[0,1],linewidth=6,color=color2)
label = "$S_{"+ str(no_windows) +"}$"
plt.text(0.03,0.48,label,color=color2)
plt.show()

