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
from configuration import Configuration

C = Configuration()

### REFERENCE SOLUTION:
solver_instance = C.child_solver_init(**C.child_solver_parameters)
reference_parameters = np.array([-1, -0.5, 0.5, 1])
solver_instance.set_parameters(reference_parameters)
reference_observations = solver_instance.get_observations()
print('ref. obs.:',reference_observations)
G = grf.GRF("modules/unit50.pckl", truncate=100)
n = C.child_solver_parameters['n']
G.plot_realization_interfaces(quantiles=[0.25, 0.5, 0.75, 1.0], nx_new=n, ny_new=n)
solver_instance.all_solvers[0].plot_solution_image()

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
S.plot_hist_grid(bins1d=20, bins2d=20)