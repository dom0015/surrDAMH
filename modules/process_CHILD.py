#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:47:09 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import full_solver_examples as fse
import FEM_wrapper
import time

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
#S = fse.Solver_local_2to2()
#S = fse.Solver_local_ntom(no_parameters = 3, no_observations = 6)
S = FEM_wrapper.FEM(no_parameters = 10, no_observations = 6, n = 50)

"""
SOLVING INCOMING REQUESTS USING LINKED SOLVER
"""

status = MPI.Status()
received_data = np.empty((S.no_parameters,))
solver_is_active = True
if rank_world==0:
    while solver_is_active:
        comm.Recv(received_data, source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
#        print('DEBUG - CHILD Recv request FROM parent', 0, 'TO child:', rank, "TAG:", tag)
        if tag == 0:
            comm.Disconnect()
            print("External solver disconnected.")
            solver_is_active = False
        else:
            S.pass_parameters(received_data.reshape((S.no_parameters,)))
#            time.sleep(1)
            sent_data = S.get_solution()
            comm.Send(sent_data, dest=0, tag=tag)
#            print('DEBUG - CHILD Send solution FROM child', rank, 'TO parent:', 0, "TAG:", tag)

