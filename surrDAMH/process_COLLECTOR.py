#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:26:55 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
from configuration import Configuration

# try FIX unpickled error
import mpi4py
mpi4py.rc.recv_mprobe = False

# communicates with: SAMPLERs 
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

# comm_sampler = comm_world.Split(color=2, key=rank_world)

# print(rank_world,size_world)
no_samplers, problem_path = comm_world.recv(source=MPI.ANY_SOURCE, tag=100)
comm_world.Barrier() # barrier probably not necessary when send/recv paired by tag=100
# data = None
# data = comm_world.bcast(data,root=1)
# no_samplers, problem_name = data
# print(rank_world,size_world,no_samplers,problem_name)
C = Configuration(no_samplers, problem_path)
updater_init = C.surr_updater_init
updater_parameters = C.surr_updater_parameters
max_buffer_size = C.max_buffer_size
# updater_init ... initializes the object of the surr. updater
# updater_parameters ... list of dictionaries with initialization parameters
# no_samplers ... number of samplers that request solutions

local_updater_instance = updater_init(**updater_parameters)
samplers_ranks = np.arange(no_samplers) # TO DO: assumes ranks 0,1,...
sampler_is_active = np.array([True] * no_samplers)
send_buffers = [None] * no_samplers
sampler_can_recv = np.array([False] * no_samplers) # ready to receive updated data
sampler_got_last_surrogate = np.array([True] * no_samplers)
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
if any(sampler_is_active):
    for r in samplers_ranks[sampler_is_active]:
    #for i in np.nditer(np.where(sampler_is_active)):
        i = samplers_ranks[samplers_ranks==r][0]
        # expects to receive data from this (active) sampler later:
        request_irecv[i] = comm_world.irecv(max_buffer_size,source=r, tag=tag_sent_data)
        # sends signal to this (active) sampler that he is ready to receive data:
        request_Isend[i] = comm_world.Isend(empty_buffers[i], dest=r, tag=tag_ready_to_receive)
status = MPI.Status()
while any(sampler_is_active): # while at least 1 sampling algorithm is active
    # if all(sampler_got_last_surrogate|(~sampler_is_active)): # block until a new incoming message
    #     comm_world.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    # checks all active samplers for an incoming message:
    for r in samplers_ranks[sampler_is_active]:
        i = samplers_ranks[samplers_ranks==r][0]
        status = MPI.Status()
        probe = comm_world.Iprobe(source=r, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if probe:
            if C.debug:
                print("debug - RANK", rank_world, "COLLECTOR Iprobe True", tag, i, r, samplers_ranks, sampler_is_active)
            if tag==tag_terminate:
                # if received message has tag_terminate, switch corresponding sampler to inactive
                # assumes that there will be no other incoming message from that source 
                sampler_is_active[i] = False
                tmp = np.empty((1,)) # TO DO: useless pointer / cancel message
                comm_world.Recv(tmp,source=r, tag=tag_terminate)
                if C.debug:
                    print("debug - RANK", rank_world, "COLLECTOR Recv terminate", tag, r)
            elif tag==tag_ready_to_receive:
                # if received message has tag_ready_to_receive, the corresponding
                # sampler is ready to receive updated data
                sampler_can_recv[i] = True
                tmp = np.empty((1,)) # TO DO: useless pointer / cancel message
                comm_world.Recv(tmp,source=r, tag=tag_ready_to_receive)
                if C.debug:
                    print("debug - RANK", rank_world, "COLLECTOR Recv ready_to_receive", tag, r)
        elif C.debug:
            print("debug - RANK", rank_world, "COLLECTOR IProbe False", r)
    for r in samplers_ranks[sampler_is_active]:
    #for i in np.nditer(np.where(sampler_is_active)):
        i = samplers_ranks[samplers_ranks==r][0]
        # checks if there are incoming data from this active sampler:
        if request_irecv[i].Get_status():
            received_data = request_irecv[i].wait()
            # expects to receive data from this active sampler later:
            request_irecv[i] = comm_world.irecv(max_buffer_size,source=r, tag=tag_sent_data)
            # sends signal to this active sampler that he is ready to receive data:
            if request_Isend[i] is not None:
                request_Isend[i].Wait()
            request_Isend[i] = comm_world.Isend(empty_buffers[i], dest=r, tag=tag_ready_to_receive)
            list_received_data.extend(received_data.copy())
    if len(list_received_data)>0 and any(sampler_is_active):
        local_updater_instance.add_data(list_received_data)
        SOL, no_snapshots = local_updater_instance.update()
        list_received_data = []
        sampler_got_last_surrogate = np.array([False] * no_samplers)
    for r in samplers_ranks[sampler_is_active & sampler_can_recv & ~sampler_got_last_surrogate]:
    #for i in np.nditer(np.where(sampler_is_active & sampler_can_recv & ~sampler_got_last_surrogate)):
        i = samplers_ranks[samplers_ranks==r][0]
        if request_isend[i] is not None:
            request_isend[i].wait()
        send_buffers[i] = SOL.copy() # TO DO: copy?
        request_isend[i] = comm_world.isend(send_buffers[i], dest=r, tag=tag_sent_data)
        sampler_can_recv[i] = False
        sampler_got_last_surrogate[i] = True

print("RANK", rank_world, "- all collected snapshots:", len(local_updater_instance.processed_par))

comm_world.Barrier()
print("RANK", rank_world, "(DATA COLLECTOR) terminated.")