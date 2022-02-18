#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 08:34:58 2022

@author: domesova
"""
from mpi4py import MPI
import numpy as np
import sys
import os
import csv
from configuration import Configuration

# generate samples from prior, calculate outputs and save
no_samplers = 0 #int(sys.argv[1])
problem_path = sys.argv[1]
output_dir = sys.argv[2] + "/"
C = Configuration(no_samplers,problem_path)

comm_world = MPI.COMM_WORLD
rank = comm_world.Get_rank()

""" INITIALIZATION OF THE SOLVER """
constructor_parameters = C.child_solver_parameters.copy()
constructor_parameters["solver_id"] = rank
solver_instance = C.child_solver_init(**constructor_parameters)

""" PREPARE FILE """
filename = output_dir + "saved_samples/" + C.problem_name + "/prior_outputs/" + "solver_rank" + str(rank) + ".csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
file = open(filename, 'w')
writer = csv.writer(file)

prior_mean = C.problem_parameters["prior_mean"]
prior_std = C.problem_parameters["prior_std"]
RS = np.random.RandomState(rank)
for i in range(40):
    data = RS.normal(loc=prior_mean,scale=prior_std)
    transformed_data = C.transform(data)
    print(rank,":",transformed_data)
    solver_instance.set_parameters(transformed_data.reshape((C.no_parameters,)))
    if C.solver_returns_tag:
        [convergence_tag,outputs] = solver_instance.get_observations()
        if convergence_tag<0:
            outputs=np.zeros((C.no_observations,))
    else:
        outputs = solver_instance.get_observations()
        convergence_tag = 0
    print(rank,"=>",outputs)
    row = ['prior'] + list(transformed_data) + [convergence_tag] + list(outputs)
    writer.writerow(row)
    
file.close()