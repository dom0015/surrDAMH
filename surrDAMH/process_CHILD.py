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

no_samplers = int(sys.argv[1])
problem_name = None
if len(sys.argv)>1:
    problem_name = sys.argv[2]
C = Configuration(no_samplers,problem_name)

comm = MPI.Comm.Get_parent()
rank = comm.Get_rank()

""" INITIALIZATION OF THE SOLVER """
solver_instance = C.child_solver_init(**C.child_solver_parameters)

""" SOLVING INCOMING REQUESTS USING LINKED SOLVER """
# tag is broadcasted by parent
# parameters are broadcasted by parent
# methods "set_parameters" and "get_observations" are called by all ranks
# observation are sent to parent by rank 0
received_data = np.empty((solver_instance.no_parameters,))
tag = np.empty((1,),dtype=int);
solver_is_active = True
while solver_is_active:
    comm.Bcast(tag, root=0)
    if tag[0] == 0:
        comm.Barrier()
        comm.Disconnect()
        solver_is_active = False
    else:
        comm.Bcast(received_data, root=0)
        solver_instance.set_parameters(received_data.reshape((solver_instance.no_parameters,)))
        sent_data = solver_instance.get_observations()
        if rank==0:
            comm.Send(sent_data, dest=0, tag=tag[0])
            