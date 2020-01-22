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
        self.no_full_solvers = 4
        self.no_surrogate_solvers = 1
        self.full_solver = fse.Solver_local_2to2 #external_2to2
#        self.full_solver = main_codes.external_solver
        self.surrogate_solver = fse.Solver_local_2to2
        if display:
            print("mpirun -n", self.no_algorithms, "--oversubscribe python3 test_sampling_algorithms_MPI.py : -n 1 python3 full_solver.py")