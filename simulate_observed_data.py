#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:08:18 2020

@author: domesova
"""

import numpy as np
import matplotlib.pyplot as plt
from configuration import Configuration
C = Configuration()

from modules import FEM_wrapper

# generate parameters from prior
no_parameters = C.no_parameters
prior_mean = C.problem_parameters["prior_mean"]
prior_std = C.problem_parameters["prior_std"]
np.random.seed(2)
# parameters = np.random.randn(no_parameters)*prior_std + prior_mean
parameters = np.ones((no_parameters,))
print(parameters)

FEM = FEM_wrapper.FEM(**C.child_solver_parameters)
FEM.pass_parameters(parameters)
observations = FEM.get_observations()
print(observations)

material = FEM.grf_instance.realization_grid_orig(parameters)
plt.imshow(material)

FEM.solver.plot_solution_image()

sigma_num = 0.1
sigma_meas = sigma_num/10
no_observations = C.no_observations
window_lengths = np.ones((no_observations,))
window_lengths[1:] = 1/(no_observations-1)
noise_var = np.power(sigma_meas,2) + window_lengths*np.power(sigma_num,2)
noise_std = np.sqrt(noise_var)
print(noise_std)
