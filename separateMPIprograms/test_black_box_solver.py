#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:54:48 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import sys

N = 3
comm = MPI.COMM_SELF.Spawn(sys.executable, args=['wrapper_black_box_solver.py'], maxprocs=N)
size = comm.Get_size()
rank = comm.Get_rank()

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

print("PARENT", rank, size, rank_world, size_world)

solver_parameters = 3*np.ones(1,)
solver_input = np.empty(1,)
comm.Send(solver_input, dest=0, tag=0)
comm.Send(solver_parameters, dest=0, tag=10)

solver_input = 6*np.ones(1,)
comm.Send(solver_input, dest=0, tag=1)
results = np.empty(N,)
comm.Recv(results, source=0, tag=22)
print('Results:', results)

solver_input = 7*np.ones(1,)
comm.Send(solver_input, dest=0, tag=1)
results = np.empty(N,)
comm.Recv(results, source=0, tag=22)
print('Results:', results)

solver_parameters = 4*np.ones(1,)
solver_input = np.empty(1,)
comm.Send(solver_input, dest=0, tag=0)
comm.Send(solver_parameters, dest=0, tag=10)

solver_input = 6*np.ones(1,)
comm.Send(solver_input, dest=0, tag=1)
results = np.empty(N,)
comm.Recv(results, source=0, tag=22)
print('Results:', results)

solver_input = 7*np.ones(1,)
comm.Send(solver_input, dest=0, tag=1)
results = np.empty(N,)
comm.Recv(results, source=0, tag=22)
print('Results:', results)

comm.Send(solver_input, dest=0, tag=2)

comm.Disconnect()