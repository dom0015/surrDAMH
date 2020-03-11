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
updater_init = C.surr_updater_init
updater_parameters = C.surr_updater_parameters
no_samplers = C.no_samplers
# updater_init ... initializes the object of the surr. updater
# updater_parameters ... list of dictionaries with initialization parameters
# no_solvers ... number of solvers to be created
# no_samplers ... number of samplers that request solutions

import numpy as np

local_updater_instance = updater_init(**updater_parameters) # TO DO: input parameter
samplers_ranks = np.arange(no_samplers) # TO DO: assumes ranks 0,1,...
is_active_sampler = np.array([True] * no_samplers)
send_buffers = [None] * no_samplers
is_ready_sampler = np.array([False] * no_samplers) # ready to receive updated data
is_free_updater = True
status = MPI.Status()
list_received_data = []
while any(is_active_sampler): # while at least 1 sampling algorithm is active
#    print(is_active_sampler,"while at least 1 sampling algorithm is active - rank", rank_world)
    tmp = comm_world.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    if tmp: # if there is an incoming message from any sampling algorithm
        rank_source = status.Get_source()
        tag = status.Get_tag()
        if tag == 0: # if received message has tag 0, switch corresponding sampler to inactive
            # assumes that there will be no other incoming message from that source 
            is_active_sampler[samplers_ranks == rank_source] = False
            received_data = comm_world.recv(source=rank_source, tag=tag)
        elif tag == 1: # if received message has tag 1, the corresponding sampler
            # is ready to receive updated data
            is_ready_sampler[rank_source] = True
            tmp = np.zeros(1) # TO DO: useless pointer / cancel message
            comm_world.Recv(tmp,source=rank_source, tag=tag)
        else: # put the request into queue (remember source and tag)
            received_data = comm_world.recv(source=rank_source, tag=tag)
            list_received_data.append(received_data)
            received_data.print() # DEBUG PRINT
    if not is_free_updater: # check if the surrogate update is done
        if local_updater_instance.update_finished:
            is_free_updater = True
    if is_free_updater: # if surrogate updater is free
        # TO DO: surrogate updater is always free...
        if len(list_received_data)>0:
            local_updater_instance.add_data(list_received_data)
            SOL, no_snapshots = local_updater_instance.update()
            print("RANK", rank_world, "collected snapshots:", no_snapshots)
            list_received_data = []
            is_free_updater = False
            for i in samplers_ranks[is_active_sampler & is_ready_sampler]:
                # TO DO: buffers (avoid memory copying)
                # TO DO: tag
                send_buffers[i]=SOL.copy() # TO DO: copy?
                comm_world.isend(send_buffers[i], dest=i, tag=no_snapshots)
                is_ready_sampler[i] = False

print("RANK", rank_world, "all collected snapshots:", len(list_received_data), len(local_updater_instance.alldata_par))

comm_world.Barrier()
print("MPI process", rank_world, "(DATA COLLECTOR) terminated.")