#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:15:15 2019

@author: simona
"""

from modules import classes_communication
from modules import surrogate_solver_rbf as surr
#from modules import full_solver_examples as fse
#from modules import surrogate_solver_examples as sse

class Configuration:
    def __init__(self):
        self.problem_name = "grf1e3"
        self.no_samplers = 3
        self.no_full_solvers = 2
        self.no_parameters = 10
        self.no_observations = 6
        self.rank_full_solver = self.no_samplers
        self.rank_surr_collector = self.no_samplers + 1
        algMH = {'type': 'MH', 
                 'max_samples': 100, 
                 'time_limit': 60*2,
                 'proposal_std': 0.8}
        algDAMH = {'type': 'DAMH', 
                   'max_samples': 1000, 
                   'time_limit': 60*15,
                   'proposal_std': 1.0}
        self.list_alg = [algMH, algDAMH]
        self.max_buffer_size = 1<<20
        self.surrogate_is_updated = True
        self.problem_parameters = {'noise_std': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                   'prior_mean': 0.0,
                                   'prior_std': 1.0,
                                   'observations': [ 7.4837606, -1.68701975, -1.81545499, -1.65487159, -1.28726334, -1.03915092], # 10 par
                                   }
### local solver?        
#        self.full_solver_init = fse.Solver_local_ntom
#        self.full_solver_parameters = []
#        for i in range(self.no_full_solvers):
#            self.full_solver_parameters.append({'no_parameters':self.no_parameters, 
#                                                'no_observations':self.no_observations})

### SOLVER TYPE 1 - solvers are spawned
        # TO DO: test if other options are also possible
        from modules import FEM_wrapper
        self.child_solver_init = FEM_wrapper.FEM
        self.child_solver_parameters = {'no_parameters': self.no_parameters,
                                        'no_observations': self.no_observations, 
                                        'n': 50,
                                        'quiet': False,
                                        'tolerance': 1e-8,
                                        'PC': "icc",
                                        'use_deflation': True,
                                        'deflation_imp': 1e-2}
        self.full_solver_init = classes_communication.Solver_MPI_parent
        self.full_solver_parameters = []
        for i in range(self.no_full_solvers):
            self.full_solver_parameters.append({'no_parameters':self.no_parameters,
                                                'no_observations':self.no_observations})

### TYPE 2 - solvers are in the same COMM_WORLD (TO DO)
#        self.full_solver_init = classes_communication.Solver_MPI_collector_MPI
#        self.full_solver_parameters = []
#        for i in range(self.no_full_solvers):
#            self.full_solver_parameters.append({'no_parameters':2, 
#                                                'no_observations':2, 
#                                                'rank_solver':i+5})

### SURROGATE
        # TO DO: use 1 iteration of deflation as surrogate
        self.surr_solver_init = surr.Surrogate_apply
        self.surr_solver_parameters = {'no_parameters':self.no_parameters,
                                       'no_observations':self.no_observations}
        self.surr_updater_init = surr.Surrogate_update
        self.surr_updater_parameters = {'no_parameters':self.no_parameters,
                                        'no_observations':self.no_observations}