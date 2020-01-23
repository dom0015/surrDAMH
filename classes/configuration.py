#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:15:15 2019

@author: simona
"""

import full_solver_examples as fse
import main_codes

class Configuration:
    def __init__(self,display=False):
        self.no_samplers = 5
        self.no_full_solvers = 3
        self.no_surr_solvers = 1
        
#        self.full_solver_init = fse.Solver_local_2to2
#        self.full_solver_parameters = {}
        
#        self.full_solver_init = main_codes.Solver_MPI_parent
#        self.full_solver_parameters = []
#        for i in range(self.no_full_solvers):
#            self.full_solver_parameters.append({'no_parameters':2, 'no_observations':2})
        
        self.full_solver_init = main_codes.Solver_MPI_linker
        self.full_solver_parameters = []
        for i in range(self.no_full_solvers):
            self.full_solver_parameters.append({'no_parameters':2, 'no_observations':2, 'rank_full_solver':i+8})

        self.surr_solver_init = fse.Solver_local_2to2
        self.surr_solver_parameters = {}
        if display:
            print("mpirun -n", self.no_algorithms, "--oversubscribe python3 test_sampling_algorithms_MPI.py : -n 1 python3 full_solver.py")