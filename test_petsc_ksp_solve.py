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
#t = time.time()
#solver.ksp_direct_type('petsc')
#print(time.time()-t)
#solver.plot_solution_image()  # plot solution as image
#solver = Solvers.LaplaceSteady(FEM_assembly)  # init
#t = time.time()
#solver.ksp_cg_with_pc('mg')
#print(time.time()-t)
#solver.plot_solution_image()  # plot solution as image
#solver = Solvers.LaplaceSteady(FEM_assembly)  # init
#t = time.time()
#solver.ksp_cg_with_pc('deflation')
#print(time.time()-t)
#solver.plot_solution_image()  # plot solution as image
#solver = Solvers.LaplaceSteady(FEM_assembly)  # init
#t = time.time()
#solver.ksp_cg_with_pc('ilu')
#print(time.time()-t)
#solver.plot_solution_image()  # plot solution as image
#solver = Solvers.LaplaceSteady(FEM_assembly)  # init
#t = time.time()
#solver.ksp_cg_with_pc('icc')
#print(time.time()-t)
#solver.plot_solution_image()  # plot solution as image
#solver = Solvers.LaplaceSteady(FEM_assembly)  # init
#t = time.time()
#solver.ksp_cg_with_pc('jacobi')
#print(time.time()-t)
#solver.plot_solution_image()  # plot solution as image
#solver = Solvers.LaplaceSteady(FEM_assembly)  # init
#t = time.time()
#solver.ksp_cg_with_pc('none')
#print(time.time()-t)
#solver.plot_solution_image()  # plot solution as image

## CG
import matplotlib.pyplot as plt
A = solver.assembled_matrices.matrices["A_dirichlet"]
Anp = A.convert("dense")
Anp = Anp.getDenseArray()
b= solver.assembled_matrices.rhss["final"]
bnp = np.array(b)
x = np.linalg.solve(Anp,bnp)
x = x.reshape((n + 1, n + 1), order='F')
plt.imshow(x, extent=[0, 1, 1, 0])
plt.gca().invert_yaxis()
plt.show()