#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:15:15 2019

@author: simona
"""

class Configuration:
    def __init__(self,display=False):
        self.no_algorithms = 5
        self.no_solvers = 4
        if display:
            print("mpirun -n", self.no_algorithms, "--oversubscribe python3 test_sampling_algorithms_MPI.py : -n 1 python3 full_solver.py")