#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:28:42 2020

@author: simona
"""

import metis

## create sample FEM matrix:
import sys # REMOVE!
sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM") 
#sys.path.append("/home/ber0061/Repositories_dom0015/Simple_Python_PETSc_FEM")
#sys.path.append("/home/ber0061/Repositories_dom0015/MCMC-Bayes-python")
import numpy as np
from MyFEM import Mesh, ProblemSetting, Assemble, Solvers
from modules import grf_eigenfunctions as grf
no_parameters = 5
no_observations = 5
n = 100
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
solver.ksp_direct_type('petsc')
solver.plot_solution_image()  # plot solution as image
A = solver.assembled_matrices.matrices["A"]


A.convert("dense")
A = A.getDenseArray()
A[A!=0]=1

N = (n+1)*(n+1)
sqrtN = n+1
NS = 12 # number of subdomains
SO = 2 # size of overlap
#A = A - np.eye(N)
adjlist = [None] * N
for i in range(N):
    tmp = np.where(A[i,:]!=0)[0]
#    for j in range(N):
#        if A[i,j]!=0:
#            tmp.append(j)
    adjlist[i] = tmp
metis_graph = metis.adjlist_to_metis(adjlist)
[cost, parts] = metis.part_graph(metis_graph, nparts=NS, recursive=True)

import matplotlib.pyplot as plt
parts_numpy = np.array(parts)
tmp = parts_numpy.reshape((sqrtN,sqrtN))#,order='F')
plt.imshow(tmp, extent=[0, 1, 1, 0])
plt.gca().invert_yaxis()
plt.show()


indices = np.zeros((N,NS))
for i in range(NS):
    indices[:,i] = (parts_numpy==i)
for i in range(SO):
    indices = np.matmul(A,indices)>0
overlap = np.sum(indices,axis=1)

tmp = overlap.reshape((sqrtN,sqrtN))#,order='F')
plt.imshow(tmp, extent=[0, 1, 1, 0])
plt.gca().invert_yaxis()
plt.show()