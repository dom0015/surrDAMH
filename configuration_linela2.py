#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:15:15 2019

@author: simona
"""

from modules import classes_communication

class Configuration:
    def __init__(self):
    
### PROBLEM PARAMETERS
        self.problem_name = "linela2"
        self.no_parameters = 2
        self.no_observations = 1
        self.problem_parameters = {'prior_mean': 5.0,
                                   'prior_std': 1.0,
                                   'observations': [-1e-3],
                                   'noise_std': 1e-4,
                                   }

### SOLVER SPECIFICATION
# linela2_exp - simple example with 2 parameters
        from modules import full_solver_examples as fse
        self.child_solver_init = fse.Solver_local_linela2_exp # 2 param., 1 obs.
        self.child_solver_parameters = {}
        self.no_full_solvers = 8

### SAMPLING PARAMETERS
        self.no_samplers = 8
        algMH = {'type': 'MH', 
                 'max_samples': 100, 
                 'time_limit': 3*60*60, # in seconds
                 'proposal_std': 1.0,
                 'surrogate_is_updated': True,
                 }
        algDAMHSMU = {'type': 'DAMH', 
                   'max_samples': 1000, 
                   'time_limit': 3*60*60, # in seconds
                   'proposal_std': 1.0,
                   'surrogate_is_updated': True,
                   }
        algDAMH = {'type': 'DAMH', 
                    'max_samples': 1000, 
                    'time_limit': 3*60*60, # in seconds
                    'proposal_std': 1.0,
                    'surrogate_is_updated': False,
                    }
        self.list_alg = [algMH, algDAMHSMU, algDAMH]

### SURROGATE MODEL SPECIFICATION
        from modules import surrogate_solver_rbf as surr
        self.surr_solver_init = surr.Surrogate_apply
        self.surr_solver_parameters = {'no_parameters':self.no_parameters,
                                       'no_observations':self.no_observations,
                                       # 'kernel_type':1,
                                       }
        self.surr_updater_init = surr.Surrogate_update
        self.surr_updater_parameters = {'no_parameters':self.no_parameters,
                                        'no_observations':self.no_observations,
                                        # 'kernel_type':1,
                                        }
        
### OTHER SETTINGS
        self.max_buffer_size = 1<<30 #20
        self.rank_full_solver = self.no_samplers
        self.rank_surr_collector = self.no_samplers + 1
        
        # TYPE 1 - solvers are spawned:
        self.full_solver_init = classes_communication.Solver_MPI_parent
        self.full_solver_parameters = []
        for i in range(self.no_full_solvers):
            self.full_solver_parameters.append({'no_parameters':self.no_parameters,
                                                'no_observations':self.no_observations})