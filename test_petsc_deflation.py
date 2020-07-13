#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:21:47 2020

@author: simona
"""


## create sample FEM matrix:
import sys # REMOVE!
import time
sys.path.append("/home/domesova/GIT/Simple_Python_PETSc_FEM") 
#sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM") 
# sys.path.append("/home/ber0061/Repositories_dom0015/Simple_Python_PETSc_FEM")
#sys.path.append("/home/ber0061/Repositories_dom0015/MCMC-Bayes-python")
import numpy as np
from petsc4py import PETSc
import matplotlib.pyplot as plt
from modules import FEM_wrapper
from modules import pcdeflation

no_parameters = 5
no_observations = 5
n = 100
FEM_instance = FEM_wrapper.FEM(no_parameters, no_observations, n)

""" SOLUTIONS FOR par0-2 (pc = none) """
par = [1.5, 1.6, 1.7]
npar = len(par)
A = [None] * npar
b = [None] * npar
sol = [None] * npar
fig, ax = plt.subplots(1, 1)
for i in range(npar):
    eta = par[i] * np.ones(no_parameters)
    FEM_instance.pass_parameters(eta)
    A[i], b[i] = FEM_instance.get_linear_system()
    sol[i] = FEM_instance.solver.assembled_matrices.create_vec()
    t = time.time()
    ksp = PETSc.KSP().create()
    ksp.setOperators(A[i])
    ksp.setType('cg')
    ksp_pc = ksp.getPC()
    ksp_pc.setType("none")
    ksp.setFromOptions()
    ksp.solve(b[i], sol[i])
    print("duration:",time.time()-t)
    print("iterations:",ksp.getIterationNumber())
    ax.plot(sol[i])
plt.show()

""" SOLUTION for par[0] (pc = deflation, W = none) """
x = FEM_instance.solver.assembled_matrices.create_vec()
t = time.time()
ksp = PETSc.KSP().create()
ksp.setOperators(A[0])
ksp.setType('cg')
ksp_pc = ksp.getPC()
ksp_pc.setType("deflation")
opts = PETSc.Options()
opts.setValue("deflation_pc_pc_type","icc")
print(PETSc.Options().getAll())
ksp.setFromOptions()
nrows = (n+1)*(n+1)
ksp.setUp()
ksp.solve(b[0], x)
print("duration:",time.time()-t)
print("iterations:",ksp.getIterationNumber())

""" SOLUTION for par[0] (pc = deflation, W = [x1]) """
x = FEM_instance.solver.assembled_matrices.create_vec()
t = time.time()
ksp = PETSc.KSP().create()
ksp.setOperators(A[0])
ksp.setType('cg')
ksp_pc = ksp.getPC()
ksp_pc.setType("deflation")
opts = PETSc.Options()
opts.setValue("deflation_pc_pc_type","icc")
print(PETSc.Options().getAll())
ksp.setFromOptions()
nrows = (n+1)*(n+1)
W = PETSc.Mat()
W.create(PETSc.COMM_WORLD)
W.setSizes((nrows,1))
W.setType("aij")
W.setPreallocationNNZ(nrows)
W.setValues(range(nrows),range(1),sol[1])
W.assemblyBegin()
W.assemblyEnd()
pcdeflation.setDeflationMat(ksp_pc,W,False);
ksp.setUp()
ksp.solve(b[0], x)
print("duration:",time.time()-t)
print("iterations:",ksp.getIterationNumber())

""" SOLUTION for par[0] (pc = deflation, W = [x1 x2]) """
x = FEM_instance.solver.assembled_matrices.create_vec()
t = time.time()
ksp = PETSc.KSP().create()
ksp.setOperators(A[0])
ksp.setType('cg')
ksp_pc = ksp.getPC()
ksp_pc.setType("deflation")
opts = PETSc.Options()
opts.setValue("deflation_pc_pc_type","icc")
print(PETSc.Options().getAll())
ksp.setFromOptions()
nrows = (n+1)*(n+1)
W = PETSc.Mat()
W.create(PETSc.COMM_WORLD)
W.setSizes((nrows,2))
W.setType("aij")
W.setPreallocationNNZ(nrows*2)
#W.setUp()
W.setValues(range(nrows),range(1),sol[1])
W.setValues(range(nrows),range(1,2),sol[2])
W.assemblyBegin()
W.assemblyEnd()
pcdeflation.setDeflationMat(ksp_pc,W,False);
ksp.setUp()
ksp.solve(b[0], x)
print("duration:",time.time()-t)
print("iterations:",ksp.getIterationNumber())
