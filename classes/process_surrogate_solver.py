#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:55:37 2019

@author: simona
"""

from configuration import Configuration
from auxiliary_methods import initialize_and_manage_solvers

C = Configuration()
    
initialize_and_manage_solvers(C.surr_solver_init, C.surr_solver_parameters, C.no_surr_solvers, C.no_samplers)