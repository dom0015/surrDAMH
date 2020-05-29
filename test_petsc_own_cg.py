#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:21:47 2020

@author: simona
"""


## create sample FEM matrix:
import sys # REMOVE!
import time
sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM") 
#sys.path.append("/home/ber0061/Repositories_dom0015/Simple_Python_PETSc_FEM")
#sys.path.append("/home/ber0061/Repositories_dom0015/MCMC-Bayes-python")
import numpy as np
from MyFEM import Mesh, ProblemSetting, Assemble, Solvers
from modules import grf_eigenfunctions as grf
import matplotlib.pyplot as plt
from petsc4py import PETSc

""" ASSEMBLY A, b """
no_parameters = 5
no_observations = 5
n = 50
my_mesh = Mesh.RectUniTri(n, n)
my_problem = ProblemSetting.BoundaryValueProblem2D(my_mesh) 
bounds = np.linspace(0,1,no_observations)
dirichlet_boundary = [None] * no_observations
dirichlet_boundary[0] = ["left",[0, 1]]
for i in range(no_observations-1):
    dirichlet_boundary[i+1] = ["right",[bounds[i], bounds[i+1]]]
dirichlet_boundary_val = [1e1] + [0] * (no_observations-1)
my_problem.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
neumann_boundary = ["top"]  # select boundary
neumann_boundary_val = [0]  # boundary value
my_problem.set_neumann_boundary(neumann_boundary, neumann_boundary_val)
my_problem.set_rhs(0) # forcing term (rhs) setting:
no_parameters = no_parameters
grf_instance = grf.GRF('modules/unit50.pckl', truncate=no_parameters)
eta = 1.5 * np.ones(no_parameters)
f_grf = grf_instance.realization_as_function(eta)
def material_function(x,y):
    no_points = len(x)
    result = np.zeros((no_points,))
    for i in range(no_points):
        result[i] = np.exp(f_grf(x[i],y[i]))
    return result
my_problem.set_material(material_function)
# MATRIX ASSEMBLER (SYSTEM MAT + RHS) assemble all parts necessary for solution:
FEM_assembly = Assemble.LaplaceSteady(my_problem)  # init assemble obj
FEM_assembly.assemble_matrix_generalized()
FEM_assembly.assemble_rhs_force()
FEM_assembly.assemble_rhs_neumann()
FEM_assembly.assemble_rhs_dirichlet()
FEM_assembly.dirichlet_cut_and_sum_rhs(duplicate=True)
# SOLVING using KSP ----------------------------------------------------------
solver = Solvers.LaplaceSteady(FEM_assembly)  # init
A = solver.assembled_matrices.matrices["A_dirichlet"]
b = solver.assembled_matrices.rhss["final"]

""" SOLVE Ax=b USING NUMPY AND SHOW """
Anp = A.convert("dense")
Anp = Anp.getDenseArray()
bnp = np.array(b)
x = np.linalg.solve(Anp,bnp)
x_reshaped = x.reshape((n + 1, n + 1), order='F')
plt.imshow(x_reshaped, extent=[0, 1, 1, 0])
plt.gca().invert_yaxis()
plt.show()

""" IMPLEMENT OWN CG WITH PETSC MATRICES AND VECTORS """
""" INPUTS (A,b) defined above """
def Afun(x):
    res = A*x
    return res
""" OTHER INPUTS (x0, W, Q, M, tol, maxiter): """
N = A.size[1]
x0 = PETSc.Vec().create()
x0.setSizes(N)
x0.setType("mpi")
x0.set(0)
x0.setUp()
Wnp = np.zeros((N,1))
W = PETSc.Mat().create()
W.setSizes((N,1))
W.setType("dense")
W.setUp()
W.assemblyBegin()
W.setValues(range(N),range(1),Wnp)
W.assemblyEnd()
WTAW = PETSc.Mat.transposeMatMult(W,A).matMult(W)
WTAWksp = PETSc.KSP().create()
WTAWksp.setOperators(WTAW)
WTAWksp.setType('preonly')
WTAWkspPC = WTAWksp.getPC()
WTAWkspPC.setType('lu')
WTAWkspPC.setFactorSolverType('umfpack')
WTAWkspPC.setFactorSetUpSolverType()
WTAWksp.ksp.setFromOptions()
def Qfun(x):
    tmp = W.transposeMult(x)
    sol = x.copy()
    WTAWksp.solve(tmp, sol)
    return W*sol
def PCfun(x): # TO DO
    return x.copy()
maxiter = 10
""" PREPARATION: """
b_norm = b.norm()
x_old = x0.copy()
r_old = b - Afun(x0)
z_old = PCfun(r_old)
resvec = PETSc.Vec().create()
resvec.setSizes(maxiter)
resvec.setType("mpi")
resvec.set(0)
resvec.setUp()
gamma_old = r_old.dot(z_old)
tag = 3
""" CG ITERATIONS: """
#for i in range(maxiter):
#    t = Afun(p_old)

    
    
    
    