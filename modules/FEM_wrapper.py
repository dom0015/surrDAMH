#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:48:13 2020

@author: simona
"""

#import sys
#sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM")
import petsc4py
import numpy as np
from MyFEM import Mesh, ProblemSetting, Assemble, Solvers
from modules import grf_eigenfunctions as grf

class FEM:
    # FEM solver preparation
    def __init__(self, no_parameters = 5, n = 100):    
        petsc4py.init()
        
        # TRIANGULAR MESH SETTING -----------------------------------------------------
        my_mesh = Mesh.RectUniTri(n, n)
        
        # PROBLEM SETTING (BOUNDARY + RHS) --------------------------------------
        self.my_problem = ProblemSetting.BoundaryValueProblem2D(my_mesh)  # init ProblemSetting obj
        # Dirichlet boundary condition settins:
#        dirichlet_boundary = [["right",[0.2, 0.5]],
#                              ["top",[0.2, 0.5]],
#                              ["bottom",[0.4, 0.6]],
#                              ["left", [0.2, 0.8]]]  # select boundary
#        dirichlet_boundary_val = [0, 0, 0, 1]  # boundary value
        # Dirichlet boundary condition settins:
        dirichlet_boundary = [["left",[0, 1]],
                              ["right",[0.0, 0.2]],
                              ["right",[0.2, 0.4]],
                              ["right",[0.4, 0.6]],
                              ["right",[0.6, 0.8]],
                              ["right",[0.8, 1.0]]] # select boundary
        dirichlet_boundary_val = [1e6,0,0,0,0,0] # boundary value
        self.my_problem.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
        # Neumann boundary condition settins:
        neumann_boundary = ["top"]  # select boundary
        neumann_boundary_val = [0]  # boundary value
        self.my_problem.set_neumann_boundary(neumann_boundary, neumann_boundary_val)  # set
        # forcing term (rhs) setting:
        self.my_problem.set_rhs(0)
        
        self.no_parameters = no_parameters
        self.grf_instance = grf.GRF('modules/unit50.pckl', truncate=no_parameters)
    
    def solve(self,parameters):
        # material setting:
        f_grf = self.grf_instance.realization_as_function(parameters)
        def material_function(x,y):
            no_points = len(x)
            result = np.zeros((no_points,))
            for i in range(no_points):
                result[i] = np.exp(f_grf(x[i],y[i]))
            return result
        self.my_problem.set_material(material_function)
        
        # MATRIX ASSEMBLER (SYSTEM MAT + RHS) ----------------------------------------
        # assemble all parts necessary for solution:
        FEM_assembly = Assemble.LaplaceSteady(self.my_problem)  # init assemble obj
        FEM_assembly.assemble_matrix_generalized()
        FEM_assembly.assemble_rhs_force()
        FEM_assembly.assemble_rhs_neumann()
        FEM_assembly.assemble_rhs_dirichlet()
        FEM_assembly.dirichlet_cut_and_sum_rhs(duplicate=True)
        
        print(FEM_assembly.times_assembly)
        
        # SOLVING using KSP ----------------------------------------------------------
        self.solver = Solvers.LaplaceSteady(FEM_assembly)  # init
        self.solver.ksp_direct_type('petsc')
        print(self.solver.times_assembly)
        self.solver.calculate_window_flow()
        return self.solver.window_flow
    
def demo_prepare_and_solve():
    no_parameters = 10
    FEM_instance = FEM(no_parameters)
    eta = np.random.randn(no_parameters)
    flow = FEM_instance.solve(eta)
    print('flow through windows:', flow, 'sum:', sum(flow))