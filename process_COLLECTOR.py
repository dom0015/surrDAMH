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
tag_terminate = 0
tag_ready_to_receive = 1
tag_sent_data = 2
list_received_data = []
empty_buffer = np.zeros(1)
requests = [None] * no_samplers
if any(is_active_sampler):
    for i in np.nditer(np.where(is_active_sampler)):
        requests[i] = comm_world.irecv(source=samplers_ranks[i], tag=tag_sent_data)
        comm_world.Isend(empty_buffer, dest=samplers_ranks[i], tag=tag_ready_to_receive)
status = MPI.Status()
while any(is_active_sampler): # while at least 1 sampling algorithm is active
#    print(is_active_sampler,"while at least 1 sampling algorithm is active - rank", rank_world)
    tmp = comm_world.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    if tmp: # if there is an incoming message from any sampling algorithm
        rank_source = status.Get_source()
        tag = status.Get_tag()
        if tag == tag_terminate: # if received message has tag 0, switch corresponding sampler to inactive
            # assumes that there will be no other incoming message from that source 
            is_active_sampler[samplers_ranks == rank_source] = False
            received_data = comm_world.recv(source=rank_source, tag=tag)
        elif tag == tag_ready_to_receive: # if received message has tag 1, the corresponding sampler
            # is ready to receive updated data
            is_ready_sampler[rank_source] = True
            tmp = np.zeros(1) # TO DO: useless pointer / cancel message
            comm_world.Recv(tmp,source=rank_source, tag=tag)
    if any(is_active_sampler):
        for i in np.nditer(np.where(is_active_sampler)):
            tmp = requests[i].test(status=status)
            if tmp[0]:
                received_data = tmp[1]
                requests[i] = comm_world.irecv(source=samplers_ranks[i], tag=tag_sent_data)
                comm_world.Isend(empty_buffer, dest=rank_source, tag=tag_ready_to_receive)
                list_received_data.extend(received_data)
    if len(list_received_data)>0:
        local_updater_instance.add_data(list_received_data)
        SOL, no_snapshots = local_updater_instance.update()
        print("RANK", rank_world, "collected snapshots:", no_snapshots)
        list_received_data = []
        is_free_updater = False
        for i in samplers_ranks[is_active_sampler & is_ready_sampler]:
            send_buffers[i]=SOL.copy() # TO DO: copy?
            print("=========================PICKLE:")
            print("=========================PICKLE:",type(send_buffers[i]),len(send_buffers[i]))
            comm_world.isend(send_buffers[i], dest=i, tag=tag_sent_data)
            is_ready_sampler[i] = False

print("RANK", rank_world, "all collected snapshots:", len(list_received_data), len(local_updater_instance.alldata_par))

comm_world.Barrier()
print("MPI process", rank_world, "(DATA COLLECTOR) terminated.")