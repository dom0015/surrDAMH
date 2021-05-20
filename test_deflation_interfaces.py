#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:23:44 2021

@author: domesova
"""

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
import petsc4py
import csv
sys.path.append(os.getcwd())

if len(sys.argv)>1:
    seed = int(sys.argv[1]) # number of MH/DAMH chains
else:
    seed = 99

plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3

### REFERENCE SOLUTION:
path = "/home/simona/GIT/Simple_Python_PETSc_FEM"
sys.path.append(path)

# no_parameters = 4
# spec = iu.spec_from_file_location("MyFEM_wrapper", "examples/solvers/MyFEM_wrapper.py")

no_parameters = 10
spec = iu.spec_from_file_location("MyFEM_wrapper", "examples/solvers/MyFEM_wrapper_grf.py")

solver_module = iu.module_from_spec(spec)
spec.loader.exec_module(solver_module)
solver_init = getattr(solver_module, "FEM")
solver_parameters = {"no_parameters": no_parameters,
        "no_observations": 5,
        "no_configurations": 1,
        "n": 60,
        "quiet": True,
        "tolerance": 1e-6,
        "PC": "icc",
        "use_deflation": True,
        "deflation_imp": 1e-6,
        "threshold_iter": 10}
solver_instance = solver_init(**solver_parameters)
# reference_parameters = np.array([-1, -0.5, 0.5, 1])
# solver_instance.set_parameters(reference_parameters)
# reference_observations = solver_instance.get_observations()
# print('ref. obs.:',reference_observations)

# from surrDAMH.modules import visualization_and_analysis as va
# G = va.grf_eigenfunctions.GRF("surrDAMH/modules/unit30.pckl", truncate=100)
# n = 60
# G.plot_realization_interfaces(quantiles=[0.25, 0.5, 0.75, 1.0], nx_new=n, ny_new=n)
# solver_instance.all_solvers[0].plot_solution_image()

def show_data(data):
    print("residual_norm:")
    print(data[:,3])
    print("comp_time")
    print(data[:,2])
    print("W size:")
    print(data[:,0])
    print("iterations:")
    print(data[:,1])
    plt.figure()
    plt.plot(data[:,0])
    plt.plot(data[:,1])
    plt.plot(data[:,2]*1e3)
    plt.plot(data[:,3]*1e7)
    plt.legend(["W","iter","time*1e3","error*1e7"])
    plt.show()

def create_PETSc_Mat(M):
    nrows, ncols = M.shape
    W = petsc4py.PETSc.Mat()
    W.create(petsc4py.PETSc.COMM_WORLD)
    W.setSizes((nrows,ncols))
    W.setType("aij")
    W.setPreallocationNNZ(nrows*ncols)
    W.setValues(range(nrows),range(ncols),M[:,:])
    W.assemblyBegin()
    W.assemblyEnd()
    return W

def create_PETSc_Vec(v):
    nrows = v.shape[0]
    W = petsc4py.PETSc.Vec()
    W.create(petsc4py.PETSc.COMM_WORLD)
    W.setSizes(nrows)
    W.setType("seq")
    #W.setPreallocationNNZ(nrows)
    W.setValues(range(nrows),v)
    W.assemblyBegin()
    W.assemblyEnd()
    return W


## TEST 1
# N=100
# #np.random.seed(4)
# all_parameters = np.random.randn(N,no_parameters)
# data = np.zeros((N,4))
# for i in range(N):
#     new_parameters = all_parameters[i,:]
#     solver_instance.set_parameters(new_parameters)
#     reference_observations = solver_instance.get_observations()
#     no_iter = solver_instance.no_iter
#     residual_norm = solver_instance.residual_norm
#     comp_time = solver_instance.comp_time
#     size_W = solver_instance.ncols
#     data[i,:] = np.array([size_W,no_iter,comp_time,residual_norm])
# show_data(data)

## TEST 2
N = 150
np.random.seed(seed)
all_parameters = np.random.randn(N,no_parameters)
data_without = np.zeros((N,4))
solutions = np.zeros((3721,N))
solver_parameters["use_deflation"] = False
solver_instance = solver_init(**solver_parameters)
for i in range(N): # precomputation without DCG
    new_parameters = all_parameters[i,:]
    solver_instance.set_parameters(new_parameters)
    observations = solver_instance.get_observations()
    no_iter = solver_instance.no_iter
    residual_norm = solver_instance.residual_norm
    comp_time = solver_instance.comp_time
    size_W = solver_instance.ncols
    data_without[i,:] = np.array([size_W,no_iter,comp_time,residual_norm])
    solutions[:,i] = np.array(solver_instance.solution)
#show_data(data_without)

data_with = np.zeros((N,4))
data_with[0,:] = data_without[-1,:]
solver_parameters["use_deflation"] = True
solver_parameters["threshold_iter"] = -1
for i in range(N-1): # precomputation without DCG
    solver_instance = solver_init(**solver_parameters)
    for j in range(i+1):
        tmp = create_PETSc_Vec(solutions[:,j])
        solver_instance.deflation_extend_optional(tmp)
    #solver_instance.set_deflation_basis(solutions[:,:i])
    parameters = all_parameters[-1,:]
    solver_instance.set_parameters(parameters)
    observations = solver_instance.get_observations()
    no_iter = solver_instance.no_iter
    residual_norm = solver_instance.residual_norm
    comp_time = solver_instance.comp_time
    size_W = solver_instance.ncols
    data_with[i+1,:] = np.array([size_W,no_iter,comp_time,residual_norm])
#show_data(data_with)

# filename = "saved_tests/deflation_interfaces/data_without" + str(seed) + ".csv"
# np.savetxt(filename, data_without, delimiter=",")

filename = "saved_tests/deflation_grf/data_without" + str(seed) + ".csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
labels = ["W","iter","time","error"]
file = open(filename, 'w')
writer = csv.writer(file)
writer.writerow(labels)
writer.writerows(data_without)#.tolist())
file.close()

filename = "saved_tests/deflation_grf/data_with" + str(seed) + ".csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
labels = ["W","iter","time","error"]
file = open(filename, 'w')
writer = csv.writer(file)
writer.writerow(labels)
writer.writerows(data_with)#.tolist())
file.close()

print("SEED", seed, "DONE")