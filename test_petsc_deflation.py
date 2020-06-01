#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:21:47 2020

@author: simona
"""


## create sample FEM matrix:
import sys # REMOVE!
import time
#sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM") 
sys.path.append("/home/ber0061/Repositories_dom0015/Simple_Python_PETSc_FEM")
#sys.path.append("/home/ber0061/Repositories_dom0015/MCMC-Bayes-python")
import numpy as np
from petsc4py import PETSc
import matplotlib.pyplot as plt
from modules import FEM_wrapper

no_parameters = 5
no_observations = 5
n = 500
FEM_instance = FEM_wrapper.FEM(no_parameters, no_observations, n)

""" FIRST SOLUTION """
eta = 1.5 * np.ones(no_parameters)
FEM_instance.pass_parameters(eta)
A, b = FEM_instance.get_linear_system()

x = FEM_instance.solver.assembled_matrices.create_vec()
t = time.time()
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType('cg')
ksp_pc = ksp.getPC()
ksp_pc.setType("none")
ksp.setFromOptions()
ksp.solve(b, x)
print("duration:",time.time()-t)
print("iterations:",ksp.getIterationNumber())

fig, ax = plt.subplots(1, 1)
ax.plot(x)
#plt.show()

W = PETSc.Mat().create()
N = (n+1)*(n+1)
W.setSizes((N,1))
W.setType("dense")
W.setUp()
#W.assemblyBegin()
W.setValues(range(N),range(1),x)
W.assemblyEnd()

""" SECOND SOLUTION """
eta2 = 1.5 * np.ones(no_parameters)
FEM_instance.pass_parameters(eta2)
A, b = FEM_instance.get_linear_system()

x = FEM_instance.solver.assembled_matrices.create_vec()
t = time.time()
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType('cg')
ksp_pc = ksp.getPC()
ksp_pc.setType("deflation")
opts = PETSc.Options()
opts.setValue("deflation_pc_pc_type","icc")
#ksp_pc.setFromOptions()
#opts.setValue("pc_type","lu")
print(PETSc.Options().getAll())
ksp.setFromOptions()
ksp.setUp()
ksp.solve(b, x)
print("duration:",time.time()-t)
print("iterations:",ksp.getIterationNumber())

#fig, ax = plt.subplots(1, 1)
ax.plot(x)
plt.show()