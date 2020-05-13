#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:35:07 2019
Example for Darcy flow.
@author: ber0061
"""

import sys
sys.path.append("/home/simona/GIT/Simple_Python_PETSc_FEM")
from modules import FEM_wrapper
import numpy as np

n = 40
no_parameters = 2
no_observations = 2
FEM_instance = FEM_wrapper.FEM(no_parameters,no_observations,n)
eta = np.random.randn(no_parameters)
FEM_instance.pass_parameters(eta)
flow = FEM_instance.get_solution()
print('flow through windows:', flow, 'sum:', sum(flow))

if n <= 30:
    FEM_instance.solver.plot_solution()  # triplot the solution
else:
    FEM_instance.solver.plot_solution_image()  # plot solution as image

M = FEM_instance.solver.assembled_matrices.matrices['A']
pressure = FEM_instance.solver.solution
t = M*pressure

FEM_instance.solver.solution = t

if n <= 30:
    FEM_instance.solver.plot_solution()  # triplot the solution
else:
    FEM_instance.solver.plot_solution_image()  # plot solution as image

print(FEM_instance.solver.times_assembly)
