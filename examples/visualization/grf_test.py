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

saved_samples_name = "grf_test"
conf_path = "examples/grf_test.json" 

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 4

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
no_ref_parameters = 900
solver_instance = solver_init(no_ref_parameters, 10, 2, n, grf_filename = None, quiet = True, tolerance = 1e-8, PC = "icc", use_deflation = False, deflation_imp = 1e-2) 
reference_parameters = np.ones((no_ref_parameters,))
solver_instance.set_parameters(reference_parameters)
reference_observations = solver_instance.get_observations()
print('ref. obs.:',reference_observations)
from surrDAMH.modules import visualization_and_analysis as va
G = va.grf_eigenfunctions.GRF("surrDAMH/modules/unit30.pckl", truncate=no_ref_parameters)

G.plot_grf(reference_parameters)
#G.plot_realization_interfaces(quantiles=[0.25, 0.5, 0.75, 1.0], nx_new=n, ny_new=n)
solver_instance.all_solvers[0].plot_solution_image()
solver_instance.all_solvers[1].plot_solution_image()

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
S.plot_hist_grid(chains_disp = chains_disp, bins1d=20, bins2d=20)
S.plot_average(chains_disp = chains_disp, show_legend = True)