#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:26:55 2019

@author: simona
"""

from mpi4py import MPI
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

from configuration import Configuration
C = Configuration()
no_parameters = C.no_parameters
no_observations = C.no_observations
no_samplers = 1

import numpy as np
import time

#samplers_ranks = np.arange(no_samplers) # TO DO: assumes ranks 0,1,...
is_active_sampler = np.array([True] * no_samplers)
list_received_data = []
request_irecv = [None] * no_samplers
request_Isend = [None] * no_samplers
recycled_buf = np.zeros((1000,))
if is_active_sampler[0]:
    # expects to receive data from this (active) sampler later:
    print("COLLECTOR irecv tag 2 (after)")
    request_irecv[0] = comm_world.irecv(30000,source=0, tag=2)
    print("COLLECTOR irecv tag 2")
    # sends signal to this (active) sampler that he is ready to receive data:
#    time.sleep(0.001)
    empty_buffer = np.zeros((1,))
    print("COLLECTOR Isend tag 1 (after)")
    request_Isend[0] = comm_world.Isend(empty_buffer, dest=0, tag=1)
    print("COLLECTOR Isend tag 1")

while is_active_sampler[0]:
    status = MPI.Status()
    probe = comm_world.Iprobe(source=0, tag=0, status=status)
    if probe:
        # if received message has tag_terminate, switch corresponding sampler to inactive
        # assumes that there will be no other incoming message from that source 
        is_active_sampler[0] = False
        print("sampler terminated",is_active_sampler)
        tmp = np.empty((1,)) # TO DO: useless pointer / cancel message
        print("COLLECTOR Recv tag 0 (after)")
        comm_world.Recv(tmp,source=0, tag=0)
        print("COLLECTOR Recv tag 0")
    if is_active_sampler[0]:
        if request_irecv[0].Get_status():
            print("COLLECTOR wait irecv tag 2: len (after)")
            try:
                received_data = request_irecv[0].wait()
            except:
                print("EXCEPTION")    
            print("COLLECTOR wait irecv tag 2:")#, print(received_data[-1]))
            # expects to receive data from this active sampler later:
            print("COLLECTOR irecv tag 2 (after)")
            request_irecv[0] = comm_world.irecv(30000,source=0, tag=2)
            print("COLLECTOR irecv tag 2")
            # sends signal to this active sampler that he is ready to receive data:
            if request_Isend[0] is not None:
                print("COLLECTOR Wait Isend tag 1 (after)")
                request_Isend[0].Wait()
                print("COLLECTOR Wait Isend tag 1")
#            time.sleep(0.001)
            empty_buffer = np.zeros((1,))
            print("COLLECTOR Isend tag 1 (after)")
            request_Isend[0] = comm_world.Isend(empty_buffer, dest=0, tag=1)
            print("COLLECTOR Isend tag 1")
            list_received_data.extend(received_data.copy())

comm_world.Barrier()
print("MPI process", rank_world, "(COLLECTOR) terminated.")