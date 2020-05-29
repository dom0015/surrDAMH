#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:48:13 2020

@author: simona
"""

import sys # REMOVE!
sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM") 
#sys.path.append("/home/ber0061/Repositories_dom0015/Simple_Python_PETSc_FEM")
#sys.path.append("/home/ber0061/Repositories_dom0015/MCMC-Bayes-python")
import petsc4py
import numpy as np
from MyFEM import Mesh, ProblemSetting, Assemble, Solvers
from modules import grf_eigenfunctions as grf

#class Solver_local_2to2:
#    def __init__(self, ):
#        self.no_parameters = 2
#        self.no_observations = 2
#        self.request_solved = True
#        self.max_requests = 1
#    
#    def pass_parameters(self, data_par):
#        self.data_par = data_par
#        
#    def get_solution(self, ):
#        x1 = self.data_par[0,0]
#        x2 = self.data_par[0,1]
#        y1 = (x1*x1 + x2 - 11)**2 + (x1 + x2*x2 - 7)**2
#        y2 = x1 + x2
#        return np.array([[y1,y2]])
#    
#    def terminate(self):
#        print("Terminate function is empty.")

class FEM:
    # FEM solver preparation
    def __init__(self, no_parameters = 5, no_observations = 5, n = 100):    
        petsc4py.init()
        
        # TRIANGULAR MESH SETTING -----------------------------------------------------
        my_mesh = Mesh.RectUniTri(n, n)
        
        # PROBLEM SETTING (BOUNDARY + RHS) --------------------------------------
        self.my_problem = ProblemSetting.BoundaryValueProblem2D(my_mesh)  # init ProblemSetting obj
        # Dirichlet boundary condition settins:
        bounds = np.linspace(0,1,no_observations)
        dirichlet_boundary = [None] * no_observations
        dirichlet_boundary[0] = ["left",[0, 1]]
        for i in range(no_observations-1):
            dirichlet_boundary[i+1] = ["right",[bounds[i], bounds[i+1]]]
        dirichlet_boundary_val = [1e1] + [0] * (no_observations-1)
        self.my_problem.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
        # Neumann boundary condition settins:
        neumann_boundary = ["top"]  # select boundary
        neumann_boundary_val = [0]  # boundary value
        self.my_problem.set_neumann_boundary(neumann_boundary, neumann_boundary_val)  # set
        # forcing term (rhs) setting:
        self.my_problem.set_rhs(0)
        
        self.no_parameters = no_parameters
        self.grf_instance = grf.GRF('modules/unit50.pckl', truncate=no_parameters)
        
    def pass_parameters(self, data_par):
        self.data_par = data_par

    def assemble(self):
        # material setting:
        f_grf = self.grf_instance.realization_as_function(self.data_par)
        def material_function(x,y):
            no_points = len(x)
            result = np.zeros((no_points,))
            for i in range(no_points):
                result[i] = np.exp(f_grf(x[i],y[i]))
            return result
        self.my_problem.set_material(material_function)
        
        # MATRIX ASSEMBLER (SYSTEM MAT + RHS) ---------------------------------
        # assemble all parts necessary for solution:
        FEM_assembly = Assemble.LaplaceSteady(self.my_problem) # init assemble obj
        FEM_assembly.assemble_matrix_generalized()
        FEM_assembly.assemble_rhs_force()
        FEM_assembly.assemble_rhs_neumann()
        FEM_assembly.assemble_rhs_dirichlet()
        FEM_assembly.dirichlet_cut_and_sum_rhs(duplicate=True)
#        print(FEM_assembly.times_assembly)
        self.solver = Solvers.LaplaceSteady(FEM_assembly)  # init
        
    def get_solution(self):
        """ assemble and solve"""
        self.assemble()

        # SOLVE using KSP -----------------------------------------------------
        self.solver.ksp_direct_type('petsc')
#        print(self.solver.times_assembly)
        self.solver.calculate_window_flow()
        return np.array(self.solver.window_flow)
    
    def get_linear_system(self): # TO DO: only temporary
        """ assemble but do not solve"""
        self.assemble()
        A = self.solver.assembled_matrices.matrices["A_dirichlet"]
        b = self.solver.assembled_matrices.rhss["final"]       
        return A, b
    
    def get_observations_from_solution(self):
        # TO DO
        pass
    
def demo_prepare_and_solve():
    no_parameters = 10
    FEM_instance = FEM(no_parameters)
    eta = np.random.randn(no_parameters)
    flow = FEM_instance.solve(eta)
    print('flow through windows:', flow, 'sum:', sum(flow))
