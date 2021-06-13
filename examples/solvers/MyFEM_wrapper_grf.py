#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:12:30 2020

@author: domesova
"""

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
        
        self.grf_instance = grf.GRF(grf_filename, truncate=no_parameters)
        
    def init_file(self, folder, solver_id):
        filename = "saved_test/" + folder + "/deflation" + str(solver_id) + ".csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.__file = open(filename, 'w')
        self.__writer = csv.writer(self.__file)
        row = ["W", "iter", "error"]
        self.__writer.writerow(row)

    def close_file(self):
        self.__file.close()
        
    def set_parameters(self, data_par):
        self.data_par = data_par

    def get_observations(self):
        """ assemble and solve """
        self.assemble()
        self.comp_time = 0
        result = [None] * self.no_configurations
        for i,solver in enumerate(self.all_solvers):
            t = time.time()
            if self.use_deflation and self.ncols > 0:
                self.get_solution_DCG(solver)
            else:
                solver.ksp_cg_with_pc(self.PC,self.tolerance)
                self.solution = solver.solution # TEMP (just for testing)
                self.no_iter = solver.ksp.getIterationNumber()
                self.residual_norm = solver.ksp.getResidualNorm()
            self.comp_time += time.time()-t
            if self.quiet == False:
                row = [self.ncols, self.no_iter, self.residual_norm]
                self.__writer.writerow(row)
                #print("iterations:",self.no_iter,"W size:",self.ncols,"normres:",self.residual_norm)
            if self.use_deflation:
                #self.solution = solver.solution # TEMP (just for testing)
                self.deflation_extend_optional(solver.solution)
            solver.calculate_window_flow()
            result[i]=solver.window_flow
        # if self.quiet == False:
        #     print("SOLVER time:", self.comp_time)
        return np.concatenate(result)

    def assemble(self):
        # material setting (Y,X):
        f_grf = self.grf_instance.realization_as_function(self.data_par)
        def material_function(x,y):
            no_points = len(x)
            result = np.zeros((no_points,))
            for i in range(no_points):
                result[i] = np.exp(f_grf(y[i],x[i]))
                # if y[i]>0.2 and x[i]>0.6:
                #     result[i] = 1000
                # if result[i]>np.exp(0):
                #     result[i]=100
                # else:
                #     result[i]=1
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
        self.solver_left = Solvers.LaplaceSteady(FEM_assembly_left)  # init
        self.all_solvers = [self.solver_left]
        
        # 2) TOP -> BOTTOM
        if self.no_configurations>1:
            self.my_problem_top.set_material(material_function)
            FEM_assembly_top = Assemble.LaplaceSteady(self.my_problem_top) # init assemble obj
            FEM_assembly_top.assemble_matrix_generalized()
            FEM_assembly_top.assemble_rhs_force()
            FEM_assembly_top.assemble_rhs_neumann()
            FEM_assembly_top.assemble_rhs_dirichlet()
            FEM_assembly_top.dirichlet_cut_and_sum_rhs(duplicate=True)
            self.solver_top = Solvers.LaplaceSteady(FEM_assembly_top)  # init
            self.all_solvers.append(self.solver_top)
        
        # 3) LEFT <- RIGHT
        if self.no_configurations>2:
            self.my_problem_right.set_material(material_function)
            FEM_assembly_right = Assemble.LaplaceSteady(self.my_problem_right) # init assemble obj
            FEM_assembly_right.assemble_matrix_generalized()
            FEM_assembly_right.assemble_rhs_force()
            FEM_assembly_right.assemble_rhs_neumann()
            FEM_assembly_right.assemble_rhs_dirichlet()
            FEM_assembly_right.dirichlet_cut_and_sum_rhs(duplicate=True)
            self.solver_right = Solvers.LaplaceSteady(FEM_assembly_right)  # init
            self.all_solvers.append(self.solver_right)
        
        # 4) TOP <- BOTTOM
        if self.no_configurations>3:
            self.my_problem_bottom.set_material(material_function)
            FEM_assembly_bottom = Assemble.LaplaceSteady(self.my_problem_bottom) # init assemble obj
            FEM_assembly_bottom.assemble_matrix_generalized()
            FEM_assembly_bottom.assemble_rhs_force()
            FEM_assembly_bottom.assemble_rhs_neumann()
            FEM_assembly_bottom.assemble_rhs_dirichlet()
            FEM_assembly_bottom.dirichlet_cut_and_sum_rhs(duplicate=True)
            self.solver_bottom = Solvers.LaplaceSteady(FEM_assembly_bottom)  # init
            self.all_solvers.append(self.solver_bottom)
    
    def get_linear_system(self, conf = 0):
        """ assemble but do not solve"""
        self.assemble()
        A = self.all_solvers[conf].assembled_matrices.matrices["A_dirichlet"]
        b = self.all_solvers[conf].assembled_matrices.rhss["final"]       
        return A, b
    
    def get_observations_from_solution(self,solution):
        """ postprocessing of the solution A\b """
        self.solver.solution = solution
        self.solver.calculate_window_flow()
        return np.array(self.solver.window_flow)
    
    def deflation_init(self):
        # self.W = petsc4py.PETSc.Mat()
        # self.W.create(petsc4py.PETSc.COMM_WORLD)
        # self.W.setSizes((self.nrows,0))
        # self.W.setType("aij")
        # self.W.setPreallocationNNZ(0)
        # self.W.assemblyBegin()
        # self.W.assemblyEnd()
        
        self.Wt = petsc4py.PETSc.Mat()
        #self.Wt.createDense((0,self.nrows))
        self.Wt.create(petsc4py.PETSc.COMM_WORLD)
        self.Wt.setSizes((0,self.nrows))
        self.Wt.setType("dense")
        self.Wt.setUp()
        self.Wt.assemblyBegin()
        self.Wt.assemblyEnd()
        
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
                #w = self.W[:,i]
                w = self.Wt[i,:]
                dotprod = vnp.dot(w)
                vnp = vnp - dotprod * w
            imp = np.linalg.norm(vnp)
            if imp>self.deflation_imp:
                self.deflation_extend(vnp/imp)

    def deflation_extend(self, v):
        # W_old = self.W.copy()
        # self.W = petsc4py.PETSc.Mat()
        # self.W.create(petsc4py.PETSc.COMM_WORLD)
        # self.W.setSizes((self.nrows,self.ncols+1))
        # self.W.setType("aij")
        # self.W.setPreallocationNNZ(self.nrows*(self.ncols+1))
        # self.W.setValues(range(self.nrows),range(self.ncols),W_old[:,:])
        # self.W.setValues(range(self.nrows),range(self.ncols,self.ncols+1),v[:])
        # self.W.assemblyBegin()
        # self.W.assemblyEnd()
        
        Wt_old = self.Wt.copy()
        self.Wt = petsc4py.PETSc.Mat()
        #self.Wt.createDense((self.ncols+1,self.nrows),(self.ncols+1,self.nrows))
        self.Wt.create(petsc4py.PETSc.COMM_WORLD)
        self.Wt.setSizes((self.ncols+1,self.nrows))
        self.Wt.setType("dense")
        self.Wt.setUp()
        self.Wt.setValues(range(self.ncols),range(self.nrows),Wt_old[:,:])
        self.Wt.setValues(range(self.ncols,self.ncols+1),range(self.nrows),v[:])
        self.Wt.assemblyBegin()
        self.Wt.assemblyEnd()
        
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
            #pcdeflation.setDeflationMat(ksp_pc,self.W,False)
            pcdeflation.setDeflationMat(ksp_pc,self.Wt,True)
        ksp.setNormType(petsc4py.PETSc.KSP.NormType.NORM_UNPRECONDITIONED) # optional
        ksp.setUp()
        ksp.solve(solver.assembled_matrices.rhss["final"], solver.solution)
        self.no_iter = ksp.getIterationNumber()
        self.residual_norm = ksp.getResidualNorm()
        # if self.quiet == False:
        #     row = [self.ncols, self.no_iter, self.comp_time, self.residual_norm]
        #     self.__writer.writerow(row)
            #print("iterations:",self.no_iter,"W size:",self.ncols,"normres:",self.residual_norm)
## ---------

    def plot_problem_left(self, flow = False):
        n = int(np.sqrt(self.nrows))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mat = self.my_problem_left.material["permeability"]
        mat = 0.5*(mat[::2] + mat[1::2])
        mat = mat.reshape((n-1,n-1),order="F")
        im = ax.imshow(mat, extent=[0, 1, 1, 0])
        plt.gca().invert_yaxis()
        plt.colorbar(im)
        plt.show()
        
        print(np.unique(mat))
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        tmp = self.all_solvers[0].solution[:]
        tmp = tmp.reshape((n,-1), order='F')
        im = ax.imshow(tmp, extent=[0, 1, 1, 0])
        plt.gca().invert_yaxis()
        plt.colorbar(im)
        plt.show()
        
        if flow:
            flow_y = tmp[:-1,:] - tmp[1:,:]
            flow_y = 0.5*(flow_y[:,:-1] + flow_y[:,1:])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(mat*flow_y, extent=[0, 1, 1, 0])
            plt.gca().invert_yaxis()
            plt.colorbar(im)
            plt.show()
        
            flow_x = tmp[:,:-1] - tmp[:,1:]
            flow_x = 0.5*(flow_x[:-1,:] + flow_x[1:,:])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(mat*flow_x, extent=[0, 1, 1, 0])
            plt.gca().invert_yaxis()
            plt.colorbar(im)
            plt.show()
        return

def demo_prepare_and_solve():
    no_parameters = 10
    FEM_instance = FEM(no_parameters)
    eta = np.random.randn(no_parameters)
    flow = FEM_instance.solve(eta)
    print('flow through windows:', flow, 'sum:', sum(flow))
