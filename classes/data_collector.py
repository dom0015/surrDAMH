#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:26:55 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import surrogate_solver_examples as sse

from configuration import Configuration
C = Configuration()
no_samplers = C.no_samplers
#no_parameters = C.no_parameters
#no_observations = C.no_observations

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()


Surrogate = sse.Surrogate_col(2,2)
algorithm_ranks = np.arange(no_samplers)
is_active = np.array([True] * len(algorithm_ranks))
is_free = True
status = MPI.Status()
snapshot_list = []
while any(is_active): # while at least 1 sampling algorithm is active
    print('is_active:',is_active)
    tmp = comm_world.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    if tmp: # if there is an incoming message from any sampling algorithm
        rank_source = status.Get_source()
        tag = status.Get_tag()
        snapshot = comm_world.recv(source=rank_source, tag=tag)
        if tag == 0: # if received message has tag 0, switch corresponding sampling alg. to inactive
            # assumes that there will be no other incoming message from that source 
            is_active[algorithm_ranks == rank_source] = False
        else: # put the request into queue (remember source and tag)
            snapshot_list.append(snapshot)
            snapshot.print()
    if not is_free:
        if Surrogate.update_finished:
            is_free = True
    if is_free: # if surrogate updater is free
        if len(snapshot_list)>0:
            Surrogate.add_data(snapshot_list)
            SOL, no_snapshots = Surrogate.calculate()
            print(no_snapshots)
            snapshot_list = []
            is_free = False
        
print("All collected snapshots:", len(snapshot_list), len(Surrogate.alldata_par))

comm_world.Barrier()
print("MPI process", rank_world, "(full solver) terminated.")