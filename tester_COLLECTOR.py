#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 23:53:09 2020

@author: ber0061
"""

# run with:
# com = 'mpirun -n 1 python3 tester_COLLECTOR : -n 1 python3 process_COLLECTOR.py'

from modules import classes_SAMPLER
import numpy as np
import time
import sys
import pickle
from configuration import Configuration

from mpi4py import MPI
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

C = Configuration()
no_parameters = C.no_parameters
no_observations = C.no_observations
rank_collector = 1
request_send = None
list_snapshots_to_send = []

for i in range(1000):
    sample = np.arange(i,i+no_parameters)
    G_sample = np.arange(i,i+no_observations+300)
    weight = i
#    snapshot_to_send = classes_SAMPLER.Snapshot(sample, G_sample, weight)
    snapshot_to_send = [sample, G_sample, weight]
    list_snapshots_to_send.append(snapshot_to_send)
    tmp = comm_world.Iprobe(source=rank_collector, tag=1)
    if tmp: # if COLLECTOR is ready to receive new snapshots
        print("TESTER:",i)
        tmp = np.zeros((1,)) # TO DO: useless pointer / cancel message
        comm_world.Recv(tmp, source=rank_collector, tag=1)
        print("TESTER Recv tag 1")
        data_to_pickle = list_snapshots_to_send.copy()
        if i>20:
            data_to_pickle = data_to_pickle[:20]
        l0 = sys.getsizeof(data_to_pickle)
        l1 = sys.getsizeof(data_to_pickle[0])
        l10 = sys.getsizeof(data_to_pickle[0][0])
        l11 = sys.getsizeof(data_to_pickle[0][1])
        l12 = sys.getsizeof(data_to_pickle[0][2])
        print(l0,l1,l10,l11,l12)
        print("size", l0 + 20*(l10+l11+l12))
        ar = np.random.rand(no_parameters+300+no_observations,20)
        print(sys.getsizeof(ar)+20*l12)
        if request_send is not None:
            request_send.wait()
            print("TESTER wait isend tag 2")
        request_send = comm_world.isend([snapshot_to_send], dest=rank_collector, tag=2)
        print("TESTER isend tag 2")
        list_snapshots_to_send = []
#    if np.random.rand()<0.49:
#        time.sleep(np.random.rand()/1000)
#    time.sleep(np.random.rand()/500)
        
empty_buffer = np.zeros((1,))
comm_world.Isend(empty_buffer, dest=rank_collector, tag=0)
print("TESTER Isend tag 0")

comm_world.Barrier()
print("MPI process", rank_world, "(TESTER) terminated.")