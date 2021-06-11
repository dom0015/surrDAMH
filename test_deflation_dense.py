#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import petsc4py
import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
from MyFEM import Mesh, ProblemSetting, Assemble, Solvers
import surrDAMH.modules.grf_eigenfunctions as grf
from examples.solvers.pcdeflation import pcdeflation

class FEM:
    # FEM solver preparation
    def __init__(self, no_parameters = 5, no_observations = 5, no_configurations = 1, n = 100, grf_filename = None, quiet = True, tolerance = None, PC = "none", use_deflation = False, deflation_imp = None, threshold_iter = 0):    
        if grf_filename == None:
            grf_filename = 'surrDAMH/modules/unit30.pckl'
        petsc4py.init()
        self.nrows = (n+1)*(n+1)
        # TRIANGULAR MESH SETTING -----------------------------------------------------
        my_mesh = Mesh.RectUniTri(n, n)
        
        # PROBLEM SETTING (BOUNDARY + RHS) --------------------------------------
        no_windows = int(no_observations/no_configurations)
        self.no_configurations = no_configurations
        
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
        if self.no_configurations>1:
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
        if self.no_configurations>2:
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
        if self.no_configurations>3:
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
        self.quiet = quiet
        self.tolerance = tolerance
        self.PC = PC
        self.use_deflation = use_deflation
        if self.use_deflation:
            self.deflation_init()
        self.deflation_imp = deflation_imp
        self.threshold_iter = threshold_iter
        self.ncols = 0
        self.no_iter = 0
        
        truncate = 100
        self.grf_instance = grf.GRF(filename=grf_filename, truncate=truncate)
        quantiles = np.cumsum(np.ones((no_parameters,))/no_parameters)
        midpoints = self.my_problem_left.geometry.nodes[self.my_problem_left.geometry.elems].mean(axis=1)
        x = midpoints[:, 0]
        y = midpoints[:, 1]
        eta = np.ones((truncate,))
        # (Y,X):
        self.interfaces_vector = self.grf_instance.realization_interfaces_apply(eta, y, x, quantiles=quantiles)

    def get_observations(self):
        """ assemble and solve """
        self.assemble()
        t = time.time()
        result = [None] * self.no_configurations
        for i,solver in enumerate(self.all_solvers):
            if self.use_deflation and self.ncols > 0:
                self.get_solution_DCG(solver)
            else:
                solver.ksp_cg_with_pc(self.PC,self.tolerance)
                #self.solution = solver.solution # TEMP (just for testing)
                self.no_iter = solver.ksp.getIterationNumber()
                self.residual_norm = solver.ksp.getResidualNorm()
            if self.quiet == False:
                row = [self.ncols, self.no_iter, self.residual_norm]
                self.__writer.writerow(row)
                #print("iterations:",self.no_iter,"W size:",self.ncols,"normres:",self.residual_norm)
            if self.use_deflation:
                #self.solution = solver.solution # TEMP (just for testing)
                self.deflation_extend_optional(solver.solution)
            solver.calculate_window_flow()
            result[i]=solver.window_flow
        self.comp_time = time.time()-t
        # if self.quiet == False:
        #     print("SOLVER time:", self.comp_time)
        return np.concatenate(result)
    
    def get_linear_system(self):
        """ assemble but do not solve"""
        self.assemble()
        A = self.solver.assembled_matrices.matrices["A_dirichlet"]
        b = self.solver.assembled_matrices.rhss["final"]       
        return A, b
    
    def deflation_init(self):
        self.W = petsc4py.PETSc.Mat()
        self.W.create(petsc4py.PETSc.COMM_WORLD)
        self.W.setSizes((self.nrows,0))
        self.W.setType("aij")
        self.W.setPreallocationNNZ(0)
        self.W.assemblyBegin()
        self.W.assemblyEnd()

        # self.W = petsc4py.PETSc.Mat()
        # self.W.createDense((self.nrows,0))
        # self.W.setUp()
        # self.W.assemblyBegin()
        # self.W.assemblyEnd()
        
        # self.Wt = petsc4py.PETSc.Mat()
        # self.Wt.createDense((0,self.nrows))
        # self.Wt.setUp()
        # self.Wt.assemblyBegin()
        # self.Wt.assemblyEnd()
        
        self.ncols = 0

    def deflation_extend_optional(self, v):
        # decide if v should be added to W
        if self.ncols == 0:
            vnp = v[:]/v.norm()
            self.deflation_extend(vnp)
            return
        if self.no_iter>self.threshold_iter:
            vnp = v[:]/v.norm()
            for i in range(self.ncols):
                w = self.W[:,i]
                dotprod = vnp.dot(w)
                vnp = vnp - dotprod * w
            imp = np.linalg.norm(vnp)
            #print("IMP", str(imp))
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
        self.W.assemblyBegin()
        self.W.assemblyEnd()
        
        # W_old = self.W.copy()
        # self.W = petsc4py.PETSc.Mat()
        # self.W.createDense((self.nrows,self.ncols+1))
        # self.W.setUp()
        # self.W.setValues(range(self.nrows),range(self.ncols),W_old[:,:])
        # self.W.setValues(range(self.nrows),range(self.ncols,self.ncols+1),v[:])
        # self.W.assemblyBegin()
        # self.W.assemblyEnd()
        
        # Wt_old = self.Wt.copy()
        # self.Wt = petsc4py.PETSc.Mat()
        # self.Wt.createDense((self.ncols+1,self.nrows),(self.ncols+1,self.nrows))
        # self.Wt.setUp()
        # self.Wt.setValues(range(self.ncols),range(self.nrows),Wt_old[:,:])
        # self.Wt.setValues(range(self.ncols,self.ncols+1),range(self.nrows),v[:])
        # self.Wt.assemblyBegin()
        # self.Wt.assemblyEnd()
        
        self.ncols += 1
        
    # def set_deflation_basis(self, W):
    #     self.ncols = W.shape[1]
    #     self.W = petsc4py.PETSc.Mat()
    #     self.W.create(petsc4py.PETSc.COMM_WORLD)
    #     self.W.setSizes((self.nrows,self.ncols))
    #     self.W.setType("aij")
    #     self.W.setPreallocationNNZ(self.nrows*(self.ncols))
    #     self.W.setValues(range(self.nrows),range(self.ncols),W[:,:])
    #     self.W.assemblyBegin()
    #     self.W.assemblyEnd()

    #     # self.ncols = W.shape[1]
    #     # self.W = petsc4py.PETSc.Mat()
    #     # self.W.createSeqDense(petsc4py.PETSc.COMM_WORLD)
    #     # self.W.setSizes((self.nrows,self.ncols))
    #     # #self.W.setType("aij")
    #     # #self.W.setPreallocationNNZ(self.nrows*(self.ncols))
    #     # self.W.setValues(range(self.nrows),range(self.ncols),W[:,:])
    #     # self.W.assemblyBegin()
    #     # self.W.assemblyEnd()

## PC deflation - deflacni matice je nastavovana pres wrapper od Kuby,
## mozna, ze uz je to primo soucasti petsc4py. 
    def get_solution_DCG(self, solver):
        if solver.solution == None:
            solver.init_solution_vec()
        ksp = petsc4py.PETSc.KSP().create()
        ksp.setOperators(solver.assembled_matrices.matrices["A_dirichlet"])
        ksp.setType('cg')
        v = ksp.getTolerances()
        #ksp.setTolerances(self.tolerance,v[1],v[2],v[3]) # relative
        ksp.setTolerances(1e-50,self.tolerance,v[2],v[3]) # absolute
        ksp_pc = ksp.getPC()
        ksp_pc.setType('deflation')
        opts = petsc4py.PETSc.Options()
        opts.setValue("deflation_pc_pc_type",self.PC)
        ksp.setFromOptions()
        if self.ncols > 0:
            pcdeflation.setDeflationMat(ksp_pc,self.W,False)
        ksp.setNormType(petsc4py.PETSc.KSP.NormType.NORM_UNPRECONDITIONED) # optional
        ksp.setUp()
        ksp.solve(solver.assembled_matrices.rhss["final"], solver.solution)
        self.no_iter = ksp.getIterationNumber()
        self.residual_norm = ksp.getResidualNorm()
