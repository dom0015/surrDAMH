#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:15:15 2019

@author: simona
"""

from modules import classes_communication
import numpy as np

class Configuration:
    def __init__(self):

### MODEL PROBLEM CHOICE:
        problem = "simple" # simple / Darcy
        surr = "poly" # rbf / poly
        self.no_full_solvers = 4
    
### PROBLEM PARAMETERS:
        if problem == "Darcy":
            self.problem_name = "Darcy"
            self.no_parameters = 4
            self.no_observations = 3 # length of the vector of observations, not repeated observations
            self.problem_parameters = {'prior_mean': 0.0, # the same as [0.0] * no_parameters
                                       'prior_std': 1.0,
                                       'observations': [9.62638828, -5.90755323, -3.71883564],
                                       'noise_std': [0.2, 0.1, 0.1],
                                       }
        elif problem == "simple":
            self.problem_name = "simple"
            self.no_parameters = 2
            self.no_observations = 1
            self.problem_parameters = {'prior_mean': [5.0, 3.0],
                                       'prior_std': [[4,-2],[-2,4]],
                                       'observations': [-1e-3],
                                       'noise_std': 2e-4,
                                       }
            
### SOLVER SPECIFICATION:
        if problem == "Darcy":
            import sys
            sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM") 
            from modules import FEM_interfaces
            self.child_solver_init = FEM_interfaces.FEM
            self.child_solver_parameters = {'no_parameters': self.no_parameters,
                                            'no_observations': self.no_observations, 
                                            'no_configurations': 1,
                                            'n': 50,
                                            'quiet': False,
                                            'tolerance': 1e-8,
                                            'PC': "icc",
                                            'use_deflation': False, # True,
                                            'deflation_imp': 1e-2,
                                            }
        elif problem == "simple":
            from modules import solver_examples_MPI
            self.child_solver_init = solver_examples_MPI.Solver_linela2exp_MPI
            self.child_solver_parameters = {}
                
### SAMPLING PARAMETERS:
        self.no_samplers = 8
        algMH = {'type': 'MH', 
                 'max_samples': 200, 
                 'time_limit': 60, # seconds
                 'proposal_std': 0.2,
                 'surrogate_is_updated': True,
                 }
        algDAMHSMU = {'type': 'DAMH', 
                   'max_samples': 2000, 
                   'time_limit': 60, # seconds
                   'proposal_std': 0.2,
                   'surrogate_is_updated': True,
                   }
        algDAMH = {'type': 'DAMH', 
                    'max_samples': 10000, 
                    'time_limit': 60, # seconds
                    'proposal_std': 0.2,
                    'surrogate_is_updated': False,
                    }
        self.list_alg = [algMH, algDAMHSMU, algDAMH]

### SURROGATE MODEL SPECIFICATION:
        if surr == "rbf": # radial basis functions surrogate model
            from modules import surrogate_rbf as surr
            self.surr_solver_parameters = {'no_parameters':self.no_parameters,
                                            'no_observations':self.no_observations,
                                            'kernel_type':1,
                                            }
            self.surr_updater_parameters = {'no_parameters':self.no_parameters,
                                            'no_observations':self.no_observations,
                                            'kernel_type':1,
                                            'no_keep':None,
                                            'expensive':False,
                                            }
        else: # polynomial surrogate model
            from modules import surrogate_poly as surr
            self.surr_solver_parameters = {'no_parameters':self.no_parameters,
                                            'no_observations':self.no_observations,
                                            }
            self.surr_updater_parameters = {'no_parameters':self.no_parameters,
                                            'no_observations':self.no_observations,
                                            'max_degree':5,
                                            }
        
        self.surr_solver_init = surr.Surrogate_apply
        self.surr_updater_init = surr.Surrogate_update
        
### OTHER SETTINGS:
        self.max_buffer_size = 1<<30 #20
        self.rank_full_solver = self.no_samplers
        self.rank_surr_collector = self.no_samplers + 1
        
        # TYPE 1 - solvers are spawned:
        self.full_solver_init = classes_communication.Solver_MPI_parent
        self.full_solver_parameters = []
        for i in range(self.no_full_solvers):
            self.full_solver_parameters.append({'no_parameters':self.no_parameters,
                                                'no_observations':self.no_observations,
                                                'maxprocs':5})