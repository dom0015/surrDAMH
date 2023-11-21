#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:47:09 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import sys
from configuration import Configuration

assert (len(sys.argv) == 5)
no_samplers = int(sys.argv[1])
problem_path = sys.argv[2]
solver_id = int(sys.argv[3])
output_dir = sys.argv[4]

C = Configuration(no_samplers, problem_path)

parent_comm = MPI.Comm.Get_parent()
rank = parent_comm.Get_rank()

""" INITIALIZATION OF THE SOLVER """
constructor_parameters = C.child_solver_parameters.copy()
constructor_parameters["solver_id"] = solver_id
constructor_parameters["output_dir"] = output_dir
solver_instance = C.child_solver_init(**constructor_parameters)

""" SOLVING INCOMING REQUESTS USING LINKED SOLVER """
# tag is broadcasted by parent
# parameters are broadcasted by parent
# methods "set_parameters" and "get_observations" are called by all ranks
# observation are sent to parent by rank 0
# received_data = np.empty((solver_instance.no_parameters,),dtype='float64')
received_data = np.empty(C.no_parameters, dtype='d')
# tag = np.empty((1,),dtype=int);
tag = np.array(0, dtype='i')
solver_is_active = True
counter = 0
while solver_is_active:
    # parent_comm.Barrier()
    parent_comm.Bcast([tag, MPI.INT], root=0)
    # if tag[0] == 0:
    if tag == 0:
        parent_comm.Barrier()
        parent_comm.Disconnect()
        solver_is_active = False
    else:
        parent_comm.Bcast([received_data, MPI.DOUBLE], root=0)
        transformed_data = C.transform(received_data)
        # print("RECEIVED: ", received_data)
        # print("TRANS: ", transformed_data)
        solver_instance.set_parameters(transformed_data.reshape((C.no_parameters,)))
        if C.solver_returns_tag:
            [convergence_tag, sent_data] = solver_instance.get_observations()
            if convergence_tag < 0:
                sent_data = np.zeros((C.no_observations,))
        else:
            sent_data = solver_instance.get_observations()
            convergence_tag = 0
        counter += 1
        if rank == 0:
            if C.pickled_observations:
                parent_comm.send([convergence_tag, sent_data], dest=0, tag=tag)
            else:
                parent_comm.Send(sent_data, dest=0, tag=convergence_tag)
print("CHILD: ", counter)
