#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:15:15 2019

@author: simona
"""

import full_solver_examples as fse
import surrogate_solver_examples as sse
import main_codes

class Configuration:
    def __init__(self,display=False):
        self.no_samplers = 3
        self.no_full_solvers = 2
        self.no_parameters = 2
        self.no_observations = 2
        self.rank_full_solver = self.no_samplers
        self.rank_surr_collector = self.no_samplers + 1

### local solver?        
#        self.full_solver_init = fse.Solver_local_2to2
#        self.full_solver_parameters = {}
        
### TYPE 1 - solvers are spawned
        self.full_solver_init = main_codes.Solver_MPI_parent
        self.full_solver_parameters = []
        for i in range(self.no_full_solvers):
            self.full_solver_parameters.append({'no_parameters':self.no_parameters, 'no_observations':self.no_observations})
    
### TYPE 2 - solvers are in the same COMM_WORLD
#        self.full_solver_init = main_codes.Solver_MPI_linker
#        self.full_solver_parameters = []
#        for i in range(self.no_full_solvers):
#            self.full_solver_parameters.append({'no_parameters':2, 'no_observations':2, 'rank_full_solver':i+5})

### SURROGATE
        self.surr_solver_init = sse.Surrogate_col
        self.surr_solver_parameters = {'no_parameters':self.no_parameters, 'no_observations':self.no_observations}
        self.surr_updater_init = sse.Surrogate_col
        self.surr_updater_parameters = {'no_parameters':self.no_parameters, 'no_observations':self.no_observations}
        
        if display:
            print("mpirun -n", self.no_samplers, "--oversubscribe python3 test_sampling_algorithms_MPI.py : -n 1 python3 full_solver.py")
            
# mpirun -n 2 --oversubscribe python3 test_sampling_algorithms_MPI.py : -n 2 python3 process_full_solver.py : -n 1 python3 data_collector.py : -n 2 python3 process_solver_wrapper.py

