#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

""" Preparation of reference observation operators.
    Differ in number of parameters, observations, etc.
    By specifying prior and noise, Bayesian problems are defined.
    One of them will be chosen for experiments with MCMC,
    others just solved to illustrate the impact of adding measurements, etc."""

import numpy as np
import matplotlib.pyplot as plt
import importlib.util as iu
import os
import sys
sys.path.append(os.getcwd())

### REFERENCE SOLUTION:
sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM")
sys.path.append("/home/simona/GIT/MCMC-Bayes-python")
spec = iu.spec_from_file_location("MyFEM_wrapper","solvers/MyFEM_wrapper.py")
solver_module = iu.module_from_spec(spec)
spec.loader.exec_module(solver_module)
solver_init = getattr(solver_module, "FEM")
filename_unit30 = "/home/simona/GIT/MCMC-Bayes-python/surrDAMH/modules/unit30.pckl"

n=60
no_parameters = 4
no_configurations = 2
no_windows = 5 # 1+oposite
reference_parameters = np.array([-1, -0.5, 0.5, 1])
solver_parameters = {
        "no_parameters": no_parameters,
        "no_observations": no_windows*no_configurations,
        "no_configurations": no_configurations,
        "n": n,
        "filename": filename_unit30,
        "quiet": True,
        "tolerance": 1e-8,
        "PC": "icc",
        "use_deflation": False,
        "deflation_imp": 1e-2}
solver_instance = solver_init(**solver_parameters)
solver_instance.set_parameters(reference_parameters)
reference_observations = solver_instance.get_observations()
for i in range(no_configurations):
    solver_instance.all_solvers[i].plot_solution_image()
print('ref. obs.:',reference_observations)
noise_std = [0.2/(no_windows-1)] * no_windows
noise_std[0] = 0.2
noise_std = noise_std*no_configurations
print(noise_std)


from surrDAMH.modules import visualization_and_analysis as va
n_viz=200
G = va.grf_eigenfunctions.GRF(filename_unit30, truncate=100)
quantiles = list(np.linspace(0,1,no_parameters+1)[1:])
G.plot_realization_interfaces(quantiles=quantiles, nx_new=n_viz, ny_new=n_viz)