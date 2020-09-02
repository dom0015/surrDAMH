#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:08:18 2020

@author: domesova
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from configuration import Configuration
C = Configuration()

from modules import FEM_wrapper2 as FEM_wrapper

# generate parameters from prior
no_parameters = C.no_parameters
no_parameters = 24
prior_mean = C.problem_parameters["prior_mean"]
prior_std = C.problem_parameters["prior_std"]
np.random.seed(2)
# parameters = np.random.randn(no_parameters)*prior_std + prior_mean
parameters = np.ones((no_parameters,))

print(parameters)

FEM = FEM_wrapper.FEM(**C.child_solver_parameters)
FEM.pass_parameters(parameters)
observations = FEM.get_observations()
print("observations:")
print(observations)

material = FEM.grf_instance.realization_grid_orig(parameters)
fig = plt.figure(figsize=(7,5))
plt.imshow(material.transpose(), extent=[0,1,1,0])
plt.gca().invert_yaxis()
plt.colorbar()

Solver = FEM.solver_left #_top
Solver.plot_solution_image()

# tmp = FEM.solver_up.solution[:]
# tmp = tmp.reshape((FEM.solver_up.assembled_matrices.problem_setting.geometry.h_elem + 1,
#                    FEM.solver_up.assembled_matrices.problem_setting.geometry.w_elem + 1), order='F')
# fig, axes = plt.subplots(1, 2, figsize=(12,3))#, sharey=False)
# m0 = axes[0].imshow(tmp, extent = [0,1,1,0])
# plt.imshow(tmp, extent = [0,1,1,0])
# plt.gca().invert_yaxis()
# plt.clim(0,15)
# plt.colorbar()
# fig.colorbar(m0, ax=axes[0])

sigma_num = 0.1
sigma_meas = sigma_num/10
no_observations = C.no_observations
no_observations = 10
window_lengths = np.ones((no_observations,))
window_lengths[1:] = 1/(no_observations-1)
noise_var = np.power(sigma_meas,2) + window_lengths*np.power(sigma_num,2)
noise_std = np.sqrt(noise_var)
print(noise_std)

# plot material and solution (corresponding orientation):
# material:
Problem = FEM.my_problem_left #_bottom
M = Problem.material
M = M["permeability"]
r = range(0,5000,2)
M = M[r].reshape(50,50).transpose()
fig = plt.figure(figsize=(7,5))
plt.imshow(np.log(M), extent=[0,1,1,0])
plt.gca().invert_yaxis()
plt.colorbar()
# solution:
Solver = FEM.solver_left #_bottom
S = Solver.solution[:]
S = S.reshape((Solver.assembled_matrices.problem_setting.geometry.h_elem + 1,
               Solver.assembled_matrices.problem_setting.geometry.w_elem + 1), order='F')
fig = plt.figure(figsize=(7,5))
plt.imshow(S, extent=[0,1,1,0])
plt.gca().invert_yaxis()
plt.clim(0,10)
plt.colorbar()

# wofex2020:
Solver = FEM.solver_left #_top
mpl.rcParams.update({'font.size': 16})
tmp = Solver.solution[:]
tmp = tmp.reshape((Solver.assembled_matrices.problem_setting.geometry.h_elem + 1,
                   Solver.assembled_matrices.problem_setting.geometry.w_elem + 1), order='F')
fig = plt.figure(figsize=(7,5))
plt.imshow(tmp, extent = [0,1,1,0])
plt.gca().invert_yaxis()
plt.clim(0,10)
plt.colorbar()
plt.title('solution sample: function $p$')
plt.plot([0*1.0/9,1*1.0/9],[0,0],linewidth=4,color='red')
plt.text(0.2*1.0/9,0.04,'$S_1$',color='red')
plt.plot([1*1.0/9,2*1.0/9],[0,0],linewidth=4,color='yellow')
plt.text(1.2*1.0/9,0.04,'$S_2$',color='yellow')
plt.plot([2*1.0/9,3*1.0/9],[0,0],linewidth=4,color='red')
plt.text(2.2*1.0/9,0.04,'$S_3$',color='red')
plt.plot([3*1.0/9,4*1.0/9],[0,0],linewidth=4,color='yellow')
plt.text(3.2*1.0/9,0.04,'$S_4$',color='yellow')
plt.plot([4*1.0/9,5*1.0/9],[0,0],linewidth=4,color='red')
plt.text(4.2*1.0/9,0.04,'$S_5$',color='red')
plt.plot([5*1.0/9,6*1.0/9],[0,0],linewidth=4,color='yellow')
plt.text(5.2*1.0/9,0.04,'$S_6$',color='yellow')
plt.plot([6*1.0/9,7*1.0/9],[0,0],linewidth=4,color='red')
plt.text(6.2*1.0/9,0.04,'$S_7$',color='red')
plt.plot([7*1.0/9,8*1.0/9],[0,0],linewidth=4,color='yellow')
plt.text(7.2*1.0/9,0.04,'$S_8$',color='yellow')
plt.plot([8*1.0/9,9*1.0/9],[0,0],linewidth=4,color='red')
plt.text(8.2*1.0/9,0.04,'$S_9$',color='red')
plt.plot([0,1],[1,1],linewidth=6,color='red')
plt.text(0.47,0.92,'$S_{10}$',color='red')
plt.show()

# fig = plt.figure(figsize=(7,5))
# plt.title('material sample: function $u$')
# plt.imshow(material, extent = [0,1,1,0])
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.show()