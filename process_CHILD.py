#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:47:09 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
#import full_solver_examples as fse
from configuration import Configuration
C = Configuration()

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

print("LAUNCHER", rank, size, rank_world, size_world)

"""
INITIALIZATION OF THE SOLVER
"""
#solver_instance = fse.Solver_local_2to2()
#solver_instance = fse.Solver_local_ntom(no_parameters = 10, no_observations = 6)
solver_instance = C.child_solver_init(**C.child_solver_parameters)

"""
SOLVING INCOMING REQUESTS USING LINKED SOLVER
"""
status = MPI.Status()
received_data = np.empty((solver_instance.no_parameters,))
solver_is_active = True
if rank_world==0:
    while solver_is_active:
        comm.Recv(received_data, source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == 0:
            comm.Barrier()
            comm.Disconnect()
            print("External solver disconnected.") #" W size:", solver_instance.ncols)
            solver_is_active = False
        else: # standard approach
            solver_instance.pass_parameters(received_data.reshape((solver_instance.no_parameters,)))
            sent_data = solver_instance.get_observations()
            comm.Send(sent_data, dest=0, tag=tag)
            # TO DO: monitor iterations, size of W, normres, time, etc.