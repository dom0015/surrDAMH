#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:48:13 2020

@author: simona
"""

#import sys # REMOVE!
#sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM") 
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
        
    def get_observations(self):
        """ assemble and solve"""
        self.assemble()
        t = time.time()
        if self.use_deflation:
            self.get_solution_DCG()
            self.deflation_extend_optional(self.solver.solution)
        else:
            self.solver.ksp_cg_with_pc(self.PC,self.tolerance)
            if self.quiet == False:
                print("SOLVER iterations:",self.solver.ksp.getIterationNumber(),"normres:",self.solver.ksp.getResidualNorm())
        if self.quiet == False:
            print("SOLVER time:", time.time()-t)
        self.solver.calculate_window_flow()
        return np.array(self.solver.window_flow)
    
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

    def get_solution_DCG(self):
        if self.solver.solution == None:
            self.solver.init_solution_vec()
#        solution = self.solver.assembled_matrices.create_vec()
        ksp = petsc4py.PETSc.KSP().create()
        ksp.setOperators(self.solver.assembled_matrices.matrices["A_dirichlet"])
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
        ksp.solve(self.solver.assembled_matrices.rhss["final"], self.solver.solution)
        if self.quiet == False:
            print("iterations:",ksp.getIterationNumber(),"W size:",self.ncols,"normres:",ksp.getResidualNorm())
        return self.solver.solution

def demo_prepare_and_solve():
    no_parameters = 10
    FEM_instance = FEM(no_parameters)
    eta = np.random.randn(no_parameters)
    flow = FEM_instance.solve(eta)
    print('flow through windows:', flow, 'sum:', sum(flow))
