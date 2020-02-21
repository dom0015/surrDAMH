#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:47:09 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import full_solver_examples as fse
import time

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

print("PROCESS SOLVER WRAPPER AT RANK", rank_world, "of", size_world)

"""
INITIALIZATION OF THE SOLVER
"""
S = fse.Solver_local_2to2()

"""
SOLVING INCOMING REQUESTS USING LINKED SOLVER
"""

status = MPI.Status()
received_data = np.empty((1,S.no_parameters))
solver_is_active = True
while solver_is_active:
    comm_world.Recv(received_data, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    tag = status.Get_tag()
    source = status.Get_source()
    print('DEBUG - WRAPPER Recv request FROM', source, 'TO:', rank_world, "TAG:", tag)
    if tag == 0:
        print("External solver disconnected.")
        solver_is_active = False
    else:
        S.send_request(received_data)
        # simulates random computation time
        #time.sleep(np.random.rand())
        time.sleep(1)
        sent_data = S.get_solution()
        comm_world.Send(sent_data, dest=source, tag=tag)
        print('DEBUG - WRAPPER Send solution FROM', rank_world, 'TO:', source, "TAG:", tag)

comm_world.Barrier()
print("MPI process", rank_world, "(solver wrapper) terminated.")