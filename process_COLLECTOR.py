#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:26:55 2019

@author: simona
"""

from mpi4py import MPI
#from tqdm import tqdm
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

from configuration import Configuration
no_samplers, problem_name = comm_world.recv(source=MPI.ANY_SOURCE)
C = Configuration(no_samplers, problem_name)

updater_init = C.surr_updater_init
updater_parameters = C.surr_updater_parameters
#no_samplers = C.no_samplers
max_buffer_size = C.max_buffer_size
# updater_init ... initializes the object of the surr. updater
# updater_parameters ... list of dictionaries with initialization parameters
# no_solvers ... number of solvers to be created
# no_samplers ... number of samplers that request solutions
# communicates with: SAMPLERs 

import numpy as np

local_updater_instance = updater_init(**updater_parameters)
samplers_ranks = np.arange(no_samplers) # TO DO: assumes ranks 0,1,...
is_active_sampler = np.array([True] * no_samplers)
send_buffers = [None] * no_samplers
is_ready_sampler = np.array([False] * no_samplers) # ready to receive updated data
is_free_updater = True
tag_terminate = 0
tag_ready_to_receive = 1
tag_sent_data = 2
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
        request_irecv[i] = comm_world.irecv(max_buffer_size,source=samplers_ranks[i], tag=tag_sent_data)
        # sends signal to this (active) sampler that he is ready to receive data:
        request_Isend[i] = comm_world.Isend(empty_buffers[i], dest=samplers_ranks[i], tag=tag_ready_to_receive)
status = MPI.Status()
#progress_bar = tqdm(total=10000)
no_snapshots_old = 0
while any(is_active_sampler): # while at least 1 sampling algorithm is active
    # checks if there is an incoming message from any sampling algorithm:
    # TO DO : check for all possible kinds of message in each loop
#    tmp = comm_world.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
#    if tmp:
#        rank_source = status.Get_source()
#        tag = status.Get_tag()
#        if tag == tag_terminate:
#            # if received message has tag 0, switch corresponding sampler to inactive
#            # assumes that there will be no other incoming message from that source 
#            is_active_sampler[samplers_ranks == rank_source] = False
##            tmp = comm_world.recv(source=rank_source, tag=tag) # useless received data
#            tmp = np.empty((1,)) # TO DO: useless pointer / cancel message
#            comm_world.Recv(tmp,source=rank_source, tag=tag)
#        elif tag == tag_ready_to_receive:
#            # if received message has tag 1, the corresponding sampler
#            # is ready to receive updated data
#            is_ready_sampler[samplers_ranks == rank_source] = True
#            tmp = np.empty((1,)) # TO DO: useless pointer / cancel message
#            comm_world.Recv(tmp,source=rank_source, tag=tag)
#        else:
#            print("TO DO: different option!")
    for i in samplers_ranks:
        probe = comm_world.Iprobe(source=i, tag=tag_terminate, status=status)
        if probe:
            # if received message has tag_terminate, switch corresponding sampler to inactive
            # assumes that there will be no other incoming message from that source 
            is_active_sampler[samplers_ranks == i] = False
            tmp = np.empty((1,)) # TO DO: useless pointer / cancel message
            comm_world.Recv(tmp,source=i, tag=tag_terminate)
        probe = comm_world.Iprobe(source=i, tag=tag_ready_to_receive, status=status)
        if probe:
            # if received message has tag_ready_to_receive, the corresponding
            # sampler is ready to receive updated data
            is_ready_sampler[samplers_ranks == i] = True
            tmp = np.empty((1,)) # TO DO: useless pointer / cancel message
            comm_world.Recv(tmp,source=i, tag=tag_ready_to_receive)
    if any(is_active_sampler):
        for i in np.nditer(np.where(is_active_sampler)):
            # checks if there are incoming data from this active sampler:
#            tmp = request_recv[i].test()#status=status)
#            if tmp[0]:
            if request_irecv[i].Get_status():
                received_data = request_irecv[i].wait()
                # expects to receive data from this active sampler later:
                request_irecv[i] = comm_world.irecv(max_buffer_size,source=samplers_ranks[i], tag=tag_sent_data)
                # sends signal to this active sampler that he is ready to receive data:
                if request_Isend[i] is not None:
                    request_Isend[i].Wait()
                request_Isend[i] = comm_world.Isend(empty_buffers[i], dest=samplers_ranks[i], tag=tag_ready_to_receive)
                list_received_data.extend(received_data.copy())
        if len(list_received_data)>0:
            # print("COLLECTOR:",len(list_received_data), is_active_sampler)
            local_updater_instance.add_data(list_received_data)
            SOL, no_snapshots = local_updater_instance.update()
    #            print("COLLECTOR computed new update:",SOL[0].shape,SOL[1].shape)
    #        print("RANK", rank_world, "collected snapshots:", no_snapshots)
    #        progress_bar.update(no_snapshots-no_snapshots_old)
            no_snapshots_old = no_snapshots
            list_received_data = []
            is_free_updater = False
            if any(is_active_sampler & is_ready_sampler):
                for i in np.nditer(np.where(is_active_sampler & is_ready_sampler)):
                    if request_isend[i] is not None:
                        request_isend[i].wait()
                    send_buffers[i] = SOL.copy() # TO DO: copy?
                    request_isend[i] = comm_world.isend(send_buffers[i], dest=samplers_ranks[i], tag=tag_sent_data)
                    is_ready_sampler[i] = False

print("RANK", rank_world, "- all collected snapshots:", len(list_received_data), len(local_updater_instance.processed_par))

comm_world.Barrier()
print("RANK", rank_world, "(DATA COLLECTOR) terminated.")