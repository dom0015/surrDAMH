#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:12:30 2020

@author: domesova
"""

import sys # REMOVE!
#sys.path.append("/home/domesova/GIT/Simple_Python_PETSc_FEM") 
sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM") 
#sys.path.append("/home/ber0061/Repositories_dom0015/Simple_Python_PETSc_FEM")
#sys.path.append("/home/ber0061/Repositories_dom0015/MCMC-Bayes-python")
import petsc4py
import numpy as np
from MyFEM import Mesh, ProblemSetting, Assemble, Solvers
import time
from modules import grf_eigenfunctions as grf
from modules import pcdeflation

#import grf_eigenfunctions as grf
#import pcdeflation

class FEM:
    # FEM solver preparation
    def __init__(self, no_parameters = 5, no_observations = 5, n = 100, quiet = True, tolerance = None, PC = "none", use_deflation = False, deflation_imp = None):    
        petsc4py.init()
        self.nrows = (n+1)*(n+1)
        # TRIANGULAR MESH SETTING -----------------------------------------------------
        my_mesh = Mesh.RectUniTri(n, n)
        
        # PROBLEM SETTING (BOUNDARY + RHS) --------------------------------------
        no_windows = int(no_observations/4)
        # 1) LEFT -> RIGHT
        self.my_problem_left = ProblemSetting.BoundaryValueProblem2D(my_mesh)  # init ProblemSetting obj
        # Dirichlet boundary condition settins:
        bounds = np.linspace(0,1,no_windows)
        dirichlet_boundary = [None] * no_windows
        dirichlet_boundary[0] = ["left",[0, 1]]
        for i in range(no_windows-1):
            dirichlet_boundary[i+1] = ["right",[bounds[i], bounds[i+1]]]
        dirichlet_boundary_val = [1e1] + [0] * (no_windows-1)
        self.my_problem_left.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
        # Neumann boundary condition settins:
        neumann_boundary = ["top"]  # select boundary
        neumann_boundary_val = [0]  # boundary value
        self.my_problem_left.set_neumann_boundary(neumann_boundary, neumann_boundary_val)  # set
        # forcing term (rhs) setting:
        self.my_problem_left.set_rhs(0)
        
        # 2) TOP -> BOTTOM
        self.my_problem_top = ProblemSetting.BoundaryValueProblem2D(my_mesh)  # init ProblemSetting obj
        # Dirichlet boundary condition settins:
        bounds = np.linspace(0,1,no_windows)
        dirichlet_boundary = [None] * no_windows
        dirichlet_boundary[0] = ["top",[0, 1]]
        for i in range(no_windows-1):
            dirichlet_boundary[i+1] = ["bottom",[bounds[i], bounds[i+1]]]
        dirichlet_boundary_val = [1e1] + [0] * (no_windows-1)
        self.my_problem_top.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
        # Neumann boundary condition settins:
        neumann_boundary = ["left"]  # select boundary
        neumann_boundary_val = [0]  # boundary value
        self.my_problem_top.set_neumann_boundary(neumann_boundary, neumann_boundary_val)  # set
        # forcing term (rhs) setting:
        self.my_problem_top.set_rhs(0)
        
        # 3) LEFT <- RIGHT
        self.my_problem_right = ProblemSetting.BoundaryValueProblem2D(my_mesh)  # init ProblemSetting obj
        # Dirichlet boundary condition settins:
        bounds = np.linspace(0,1,no_windows)
        dirichlet_boundary = [None] * no_windows
        dirichlet_boundary[0] = ["right",[0, 1]]
        for i in range(no_windows-1):
            dirichlet_boundary[i+1] = ["left",[bounds[i], bounds[i+1]]]
        dirichlet_boundary_val = [1e1] + [0] * (no_windows-1)
        self.my_problem_right.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
        # Neumann boundary condition settins:
        neumann_boundary = ["top"]  # select boundary
        neumann_boundary_val = [0]  # boundary value
        self.my_problem_right.set_neumann_boundary(neumann_boundary, neumann_boundary_val)  # set
        # forcing term (rhs) setting:
        self.my_problem_right.set_rhs(0)
        
        # 4) TOP <- BOTTOM
        self.my_problem_bottom = ProblemSetting.BoundaryValueProblem2D(my_mesh)  # init ProblemSetting obj
        # Dirichlet boundary condition settins:
        bounds = np.linspace(0,1,no_windows)
        dirichlet_boundary = [None] * no_windows
        dirichlet_boundary[0] = ["bottom",[0, 1]]
        for i in range(no_windows-1):
            dirichlet_boundary[i+1] = ["top",[bounds[i], bounds[i+1]]]
        dirichlet_boundary_val = [1e1] + [0] * (no_windows-1)
        self.my_problem_bottom.set_dirichlet_boundary(dirichlet_boundary, dirichlet_boundary_val)
        # Neumann boundary condition settins:
        neumann_boundary = ["left"]  # select boundary
        neumann_boundary_val = [0]  # boundary value
        self.my_problem_bottom.set_neumann_boundary(neumann_boundary, neumann_boundary_val)  # set
        # forcing term (rhs) setting:
        self.my_problem_bottom.set_rhs(0)
        
        
        self.no_parameters = no_parameters
        self.grf_instance = grf.GRF('modules/unit50.pckl', truncate=no_parameters)
        self.quiet = quiet
        self.tolerance = tolerance
        self.PC = PC
        self.use_deflation = use_deflation
        if self.use_deflation:
            self.deflation_init()
        self.deflation_imp = deflation_imp
        
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
        
        # 1) LEFT -> RIGHT
        self.my_problem_left.set_material(material_function)
        # MATRIX ASSEMBLER (SYSTEM MAT + RHS) ---------------------------------
        # assemble all parts necessary for solution:
        FEM_assembly_left = Assemble.LaplaceSteady(self.my_problem_left) # init assemble obj
        FEM_assembly_left.assemble_matrix_generalized()
        FEM_assembly_left.assemble_rhs_force()
        FEM_assembly_left.assemble_rhs_neumann()
        FEM_assembly_left.assemble_rhs_dirichlet()
        FEM_assembly_left.dirichlet_cut_and_sum_rhs(duplicate=True)
#        print(FEM_assembly.times_assembly)
        self.solver_left = Solvers.LaplaceSteady(FEM_assembly_left)  # init
        
        # 2) TOP -> BOTTOM
        self.my_problem_top.set_material(material_function)
        # MATRIX ASSEMBLER (SYSTEM MAT + RHS) ---------------------------------
        # assemble all parts necessary for solution:
        FEM_assembly_top = Assemble.LaplaceSteady(self.my_problem_top) # init assemble obj
        FEM_assembly_top.assemble_matrix_generalized()
        FEM_assembly_top.assemble_rhs_force()
        FEM_assembly_top.assemble_rhs_neumann()
        FEM_assembly_top.assemble_rhs_dirichlet()
        FEM_assembly_top.dirichlet_cut_and_sum_rhs(duplicate=True)
#        print(FEM_assembly.times_assembly)
        self.solver_top = Solvers.LaplaceSteady(FEM_assembly_top)  # init
        
        # 3) LEFT <- RIGHT
        self.my_problem_right.set_material(material_function)
        # MATRIX ASSEMBLER (SYSTEM MAT + RHS) ---------------------------------
        # assemble all parts necessary for solution:
        FEM_assembly_right = Assemble.LaplaceSteady(self.my_problem_right) # init assemble obj
        FEM_assembly_right.assemble_matrix_generalized()
        FEM_assembly_right.assemble_rhs_force()
        FEM_assembly_right.assemble_rhs_neumann()
        FEM_assembly_right.assemble_rhs_dirichlet()
        FEM_assembly_right.dirichlet_cut_and_sum_rhs(duplicate=True)
#        print(FEM_assembly.times_assembly)
        self.solver_right = Solvers.LaplaceSteady(FEM_assembly_right)  # init
        
        # 4) TOP <- BOTTOM
        self.my_problem_bottom.set_material(material_function)
        # MATRIX ASSEMBLER (SYSTEM MAT + RHS) ---------------------------------
        # assemble all parts necessary for solution:
        FEM_assembly_bottom = Assemble.LaplaceSteady(self.my_problem_bottom) # init assemble obj
        FEM_assembly_bottom.assemble_matrix_generalized()
        FEM_assembly_bottom.assemble_rhs_force()
        FEM_assembly_bottom.assemble_rhs_neumann()
        FEM_assembly_bottom.assemble_rhs_dirichlet()
        FEM_assembly_bottom.dirichlet_cut_and_sum_rhs(duplicate=True)
#        print(FEM_assembly.times_assembly)
        self.solver_bottom = Solvers.LaplaceSteady(FEM_assembly_bottom)  # init
        
    def get_observations(self):
        """ assemble and solve"""
        self.assemble()
        t = time.time()
        
        result = [None] * 4
        for i,solver in enumerate([self.solver_left, self.solver_top, self.solver_right, self.solver_bottom]):
## Prozatim vynechana moznost PC deflation - deflacni matice byla nastavovana
## pres wrapper. Je mozne, ze nyni je jiz primo soucasti petsc4py. 
            if self.use_deflation:
                self.get_solution_DCG(solver)
                self.deflation_extend_optional(solver.solution)
            else:
                solver.ksp_cg_with_pc(self.PC,self.tolerance)
                if self.quiet == False:
                    print("SOLVER iterations:",solver.ksp.getIterationNumber(),"normres:",solver.ksp.getResidualNorm())
## ---------
            if self.quiet == False:
                print("SOLVER time:", time.time()-t)
            solver.calculate_window_flow()
            result[i]=solver.window_flow
            # print(result[i])
        return np.concatenate((result[0],result[1],result[2],result[3]))
    
    def get_linear_system(self):
        """ assemble but do not solve"""
        self.assemble()
        A = self.solver.assembled_matrices.matrices["A_dirichlet"]
        b = self.solver.assembled_matrices.rhss["final"]       
        return A, b
    
    def get_observations_from_solution(self,solution):
        """ postprocessing of the solution A\b """
        self.solver.solution = solution
        self.solver.calculate_window_flow()
        return np.array(self.solver.window_flow)
    
    def deflation_init(self):
        self.W = petsc4py.PETSc.Mat()
        self.W.create(petsc4py.PETSc.COMM_WORLD)
        self.W.setSizes((self.nrows,0))
        self.W.setType("aij")
        self.W.setPreallocationNNZ(0)
        self.W.assemblyBegin()
        self.W.assemblyEnd()
        self.ncols = 0

    def deflation_extend_optional(self, v):
        # decide if v should be added to W
        vnp = v[:]/v.norm()
        if self.ncols == 0:
            self.deflation_extend(vnp)
            return
        for i in range(self.ncols):
            w = self.W[:,i]
            dotprod = vnp.dot(w)
            vnp = vnp - dotprod * w
        imp = np.linalg.norm(vnp)
        if imp>self.deflation_imp:
            self.deflation_extend(vnp/imp)

    def deflation_extend(self, v):
        W_old = self.W.copy()
        self.W = petsc4py.PETSc.Mat()
        self.W.create(petsc4py.PETSc.COMM_WORLD)
        self.W.setSizes((self.nrows,self.ncols+1))
        self.W.setType("aij")
        self.W.setPreallocationNNZ(self.nrows*(self.ncols+1))
        self.W.setValues(range(self.nrows),range(self.ncols),W_old[:,:])
        self.W.setValues(range(self.nrows),range(self.ncols,self.ncols+1),v[:])
        self.ncols += 1
        self.W.assemblyBegin()
        self.W.assemblyEnd()

## Prozatim vynechana moznost PC deflation - deflacni matice byla nastavovana
## pres wrapper. Je mozne, ze nyni je jiz primo soucasti petsc4py. 
    def get_solution_DCG(self, solver):
        if solver.solution == None:
            solver.init_solution_vec()
#        solution = self.solver.assembled_matrices.create_vec()
        ksp = petsc4py.PETSc.KSP().create()
        ksp.setOperators(solver.assembled_matrices.matrices["A_dirichlet"])
        ksp.setType('cg')
        v = ksp.getTolerances()
        ksp.setTolerances(self.tolerance,v[1],v[2],v[3])
        ksp_pc = ksp.getPC()
        ksp_pc.setType('deflation')
        opts = petsc4py.PETSc.Options()
        opts.setValue("deflation_pc_pc_type",self.PC)
        ksp.setFromOptions()
        if self.ncols > 0:
            pcdeflation.setDeflationMat(ksp_pc,self.W,False);
        ksp.setUp()
        ksp.solve(solver.assembled_matrices.rhss["final"], solver.solution)
        if self.quiet == False:
            print("iterations:",ksp.getIterationNumber(),"W size:",self.ncols,"normres:",ksp.getResidualNorm())
        return solver.solution
## ---------

def demo_prepare_and_solve():
    no_parameters = 10
    FEM_instance = FEM(no_parameters)
    eta = np.random.randn(no_parameters)
    flow = FEM_instance.solve(eta)
    print('flow through windows:', flow, 'sum:', sum(flow))
