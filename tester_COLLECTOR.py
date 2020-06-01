#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 23:53:09 2020

@author: ber0061
"""

# run with:
# com = 'mpirun -n 1 python3 tester_COLLECTOR : -n 1 python3 process_COLLECTOR.py'

from modules import classes_SAMPLER
from modules import classes_communication
import numpy as np
import time
from configuration import Configuration

from mpi4py import MPI
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

C = Configuration()
no_parameters = C.no_parameters
no_observations = C.no_observations
rank_collector = 1


request_recv = comm_world.irecv(source=rank_collector, tag=2)
request_send = None
empty_buffer = np.zeros((1,))
comm_world.Isend(empty_buffer, dest=rank_collector, tag=1)
list_snapshots_to_send = []
status = MPI.Status()


for i in range(1000):
#    print("TESTER loop:",i)
    sample = np.arange(i,i+no_parameters)
    G_sample = np.arange(i,i+no_observations)
    weight = i
    snapshot_to_send = classes_SAMPLER.Snapshot(sample, G_sample, weight)
    
    list_snapshots_to_send.append(snapshot_to_send)
    tmp = comm_world.Iprobe(source=rank_collector, tag=1)
    if tmp: # if COLLECTOR is ready to receive new snapshots
        print("TESTER:",i)
        tmp = np.zeros((1,)) # TO DO: useless pointer / cancel message
        comm_world.Recv(tmp, source=rank_collector, tag=1)
        data_to_pickle = list_snapshots_to_send.copy()
        if request_send is not None:
            request_send.wait()
        request_send = comm_world.isend(data_to_pickle, dest=rank_collector, tag=2)
        list_snapshots_to_send = []
#    if np.random.rand()<0.49:
#        time.sleep(np.random.rand()/1000)
    time.sleep(np.random.rand()/500)
        

comm_world.Isend(empty_buffer, dest=rank_collector, tag=0)


comm_world.Barrier()
print("MPI process", rank_world, "(TESTER) terminated.")