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
#updater_init = C.surr_updater_init
#updater_parameters = C.surr_updater_parameters
no_samplers = 1

import numpy as np
import time

#samplers_ranks = np.arange(no_samplers) # TO DO: assumes ranks 0,1,...
is_active_sampler = np.array([True] * no_samplers)
send_buffers = [None] * no_samplers
is_ready_sampler = np.array([False] * no_samplers) # ready to receive updated data
is_free_updater = True
list_received_data = []
empty_buffers = [None] * no_samplers
for i in range(no_samplers):
    empty_buffers[i] = np.zeros((1,))
request_irecv = [None] * no_samplers
request_isend = [None] * no_samplers
request_Isend = [None] * no_samplers
if any(is_active_sampler):
    for i in np.nditer(np.where(is_active_sampler)):
        # expects to receive data from this (active) sampler later:
        request_irecv[i] = comm_world.irecv(source=0, tag=2)
        # sends signal to this (active) sampler that he is ready to receive data:
        time.sleep(0.001)
        request_Isend[i] = comm_world.Isend(empty_buffers[i], dest=0, tag=1)
status = MPI.Status()
#progress_bar = tqdm(total=10000)
#no_snapshots_old = 0
while any(is_active_sampler): # while at least 1 sampling algorithm is active
    probe = comm_world.Iprobe(source=0, tag=0, status=status)
    if probe:
        # if received message has tag_terminate, switch corresponding sampler to inactive
        # assumes that there will be no other incoming message from that source 
        is_active_sampler[0] = False
        print("sampler terminated",is_active_sampler)
        tmp = np.empty((1,)) # TO DO: useless pointer / cancel message
        comm_world.Recv(tmp,source=0, tag=0)
    probe = comm_world.Iprobe(source=0, tag=1, status=status)
    if probe:
        # if received message has tag_ready_to_receive, the corresponding
        # sampler is ready to receive updated data
        is_ready_sampler[0] = True
        tmp = np.empty((1,)) # TO DO: useless pointer / cancel message
        comm_world.Recv(tmp,source=i, tag=1)
    if any(is_active_sampler):
        if request_irecv[0].Get_status():
            received_data = request_irecv[0].wait()
            print("COLLECTOR: unpickled data",len(received_data))
            # expects to receive data from this active sampler later:
            request_irecv[i] = comm_world.irecv(source=0, tag=2)
            # sends signal to this active sampler that he is ready to receive data:
            if request_Isend[0] is not None:
                request_Isend[0].Wait()
            time.sleep(0.001)
            request_Isend[i] = comm_world.Isend(empty_buffers[i], dest=0, tag=1)
            list_received_data.extend(received_data.copy())
    if len(list_received_data)>0:
        SOL = np.zeros((1,))
        list_received_data = []
        is_free_updater = False
        if any(is_active_sampler & is_ready_sampler):
            print("COLLECTOR ready", np.where(is_active_sampler & is_ready_sampler))
            print(list(id(k) for k in send_buffers))
            for i in np.nditer(np.where(is_active_sampler & is_ready_sampler)):
                print("COLLECTOR will send SOL to", i)
                if request_isend[i] is not None:
                    request_isend[i].wait()
                    print("COLLECTOR waited for", i)
                send_buffers[i] = SOL.copy() # TO DO: copy?
                request_isend[i] = comm_world.isend(send_buffers[i], dest=0, tag=2)
                print("COLLECTOR isent SOL to", i)
                is_ready_sampler[i] = False

print("RANK", rank_world, "all collected snapshots:", len(list_received_data))#, len(local_updater_instance.processed_par))

comm_world.Barrier()
print("MPI process", rank_world, "(DATA COLLECTOR) terminated.")