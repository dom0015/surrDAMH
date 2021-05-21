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

problem_name = "interfaces_test1"
saved_samples_name = problem_name
conf_path = "examples/" + problem_name +".json" 
savefig_name = problem_name

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
solver_parameters = conf["solver_parameters"]
solver_parameters["no_observations"] = 20
solver_parameters["no_configurations"] = 4
solver_instance = solver_init(**solver_parameters)
reference_parameters = np.array([-1, -0.5, 0.5, 1])
solver_instance.set_parameters(reference_parameters)
reference_observations = solver_instance.get_observations()
print('ref. obs.:',reference_observations)

from surrDAMH.modules import visualization_and_analysis as va
G = va.grf_eigenfunctions.GRF("surrDAMH/modules/unit30.pckl", truncate=100)
n = conf["solver_parameters"]['n']
G.plot_realization_interfaces(quantiles=[0.25, 0.5, 0.75, 1.0], nx_new=n, ny_new=n)
solver_instance.all_solvers[0].plot_solution_image()
n = 200
G.plot_realization_interfaces(quantiles=[0.25, 0.5, 0.75, 1.0], nx_new=n, ny_new=n)
plt.savefig('examples/visualization/img/' + savefig_name + '_subdomains.pdf')  
solver_instance.all_solvers[0].plot_solution_image()

""" VISUALIZATION OF WINDOWS """
no_windows = 5
for i in range(no_windows - 1):
    color = "yellow"
    if np.mod(i,2)==0:
        color = "red"
    length = 1.0/(no_windows-1)
    plt.plot([1,1],[i*length,(i+1)*length],linewidth=4,color=color)
    label = "$S_{"+ str(i+1) +"}$"
    plt.text(1-0.1,(i+0.4)*length,label,color=color)
plt.plot([0,0],[0,1],linewidth=6,color='red')
label = "$S_{"+ str(no_windows) +"}$"
plt.text(0.03,0.48,label,color='red')
plt.show()
plt.savefig('examples/visualization/img/' + savefig_name + '_p.pdf')  

"""

### SAMPLES VISUALIZATION:
S = va.Samples()
no_parameters = conf["no_parameters"]
S.load_notes('saved_samples/' + saved_samples_name,no_samplers)
S.load_MH('saved_samples/' + saved_samples_name,no_parameters)

# components 0,1 of a MH + DAMH-SMU + DAMH chain:
# no_chain=0;
# plt.figure()
# plt.plot(S.x[no_chain][:,0],S.x[no_chain][:,1],alpha=0.5,marker=".",color='k')
# plt.plot(S.x[no_chain+no_samplers][:,0],S.x[no_chain+no_samplers][:,1],alpha=0.5,marker=".",color="r")
# plt.plot(S.x[no_chain+2*no_samplers][:,0],S.x[no_chain+2*no_samplers][:,1],alpha=0.5,marker=".",color="b")
# plt.legend(["MH","DAMH-SMU","DAMH"])
# plt.title("sampling process" + str(no_chain))
# plt.grid()
#plt.show()

# Which part of the sampling process is analyzed? 0/1/2 = MH/DAMH-SMU/DAMH
setnumber = 2;
S.calculate_properties()
S.print_properties()
chains_disp = range(setnumber*no_samplers,(setnumber+1)*no_samplers)
S.plot_hist_grid(chains_disp = chains_disp, bins1d=40, bins2d=40, show_title=False, sharex = False, sharey = False)
S.plot_average(chains_disp = chains_disp, show_legend = False, show_title = False, sharey=True)
plt.tight_layout()
plt.show()

"""