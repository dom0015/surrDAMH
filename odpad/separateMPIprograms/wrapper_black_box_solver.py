#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:24:30 2019

@author: simona
"""

"""
This wrapper was spawned by the main process,
communicates via intercommunication "comm".

It is assumed that
    processes of all ranks need solver_input,
    the process of rank_world 0 has results.
"""

from mpi4py import MPI
import numpy as np
from black_box_solver import black_box_solver as bbs

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

wrapped_solver = bbs()

solver_input = np.zeros(1,)
solver_parameters = np.zeros(1,)
if rank_world==0:
    status = MPI.Status()
    
finish = False
tag = np.empty((1,), dtype='i')
while finish==False:
    if rank_world==0:
        comm.Recv(solver_input, source=0, tag=MPI.ANY_TAG, status=status)
        temp = status.Get_tag()
        print('TAG:',temp)
        tag[0]=temp
    comm_world.Bcast(tag, root=0)
    if tag[0]==0: # (re-)initialize solver
        if rank_world==0:
            comm.Recv(solver_parameters, source=0, tag=10)
        comm_world.Bcast(solver_parameters, root=0)
        wrapped_solver.prepare(solver_parameters)
    elif tag[0]==1: # solution required
        comm_world.Bcast(solver_input, root=0)
        result = wrapped_solver.solve(solver_input)
        # for all world ranks except 0, result is None
        if rank_world==0:
            comm.Send(result, dest=0, tag=22)
    else:
        print("Wrapper finishes")
        finish = True
comm.Disconnect()