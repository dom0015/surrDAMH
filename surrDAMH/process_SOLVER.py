#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:55:37 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import sys
from collections import deque
from configuration import Configuration

# try FIX unpickled error
import mpi4py
mpi4py.rc.recv_mprobe = False

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

no_samplers = rank_world

assert(len(sys.argv) == 3)
problem_path = sys.argv[1]
output_dir = sys.argv[2]

print("process_SOLVER: ", no_samplers, problem_path, flush=True)
for i in range(size_world):
    if i != rank_world:
        list_to_send = [no_samplers, problem_path]
        print("comm_world.send(dest=", i, ") params: ", list_to_send, flush=True)
        comm_world.send(list_to_send, dest=i, tag=100)
C = Configuration(no_samplers, problem_path)

solver_init = C.solver_parent_init
solver_parameters = C.solver_parent_parameters
no_solvers = C.no_full_solvers
# solver_init ... initializes the object of the (full, surrogate) solver
# solver_parameters ... list of dictionaries with initialization parameters
# no_solvers ... number of solvers to be created
# no_samplers ... number of samplers that request solutions

Solvers = []
for i in range(no_solvers):
    Solvers.append(solver_init(**solver_parameters[i], output_dir=output_dir))
samplers_rank = np.arange(no_samplers)
sampler_is_active = np.array([True] * no_samplers)
sampler_can_send = np.array([True] * no_samplers)
occupied_by_source = [None] * no_solvers
occupied_by_tag = [None] * no_solvers
child_can_solve = np.array([True] * no_solvers)
no_parameters = C.no_parameters
received_data = np.zeros(no_parameters)
status = MPI.Status()
request_queue = deque()

def receive_observations_from_child(i):
    convergence_tag, sent_data = Solvers[i].recv_observations()
    child_can_solve[i] = True # mark the solver as free
    for j in range(len(occupied_by_source[i])):
        rank_dest = occupied_by_source[i][j]
        if C.pickled_observations:
            comm_world.send([convergence_tag, sent_data[j,:].copy()], dest=rank_dest, tag=occupied_by_tag[i][j])
        else:
            comm_world.Send(sent_data[j,:].copy(), dest=rank_dest, tag=convergence_tag) #occupied_by_tag[i][j])
        # if C.debug:
        #     print("debug - RANK", rank_world, "POOL Send", rank_dest, occupied_by_tag[i][j])
        sampler_can_send[samplers_rank == rank_dest] = True

while any(sampler_is_active): # while at least 1 sampling algorithm is active
    if any(sampler_can_send) and any(child_can_solve):
        if all(child_can_solve): # no child is busy
            probe = comm_world.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        else:
            probe = comm_world.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        if probe: # if there is an incoming message from any sampler
            # receive one message from one sampler
            rank_source = status.Get_source()
            tag = status.Get_tag()
            comm_world.Recv(received_data, source=rank_source, tag=tag)
            # if C.debug:
            #     print("debug - RANK", rank_world, "POOL Recv", rank_source)
            if tag == 0: # if received message has tag 0, switch corresponding sampling alg. to inactive
                # assumes that there will be no other incoming message from that source 
                sampler_is_active[samplers_rank == rank_source] = False
                sampler_can_send[samplers_rank == rank_source] = False
            else: # put the request into queue (remember source and tag)
                request_queue.append([rank_source,tag,received_data.copy()])
                # nothing else will come from this sampler until completion of this request
                sampler_can_send[samplers_rank == rank_source] = False
        elif C.debug:
                print("debug - RANK", rank_world, "POOL Iprobe False - S:", sampler_can_send, "- CH:", child_can_solve)
    # if no sampler can send and only one child is busy, wait for this child:
    # if not any(sampler_can_send):
    #     print("SAMPLER CAN SEND --", sampler_can_send)
    #     if sum(child_can_solve==False)==1:
    #         i = np.nonzero(child_can_solve==False)[0][0]
    #         receive_observations_from_child(i)
    for i in range(no_solvers):
        if not child_can_solve[i]: # check all busy child solvers if they finished the request
            if Solvers[i].is_solved(): # if so, send solution to the sampling algorithm
                receive_observations_from_child(i)
            elif C.debug:
                    print("debug - RANK", rank_world, "PARENT Iprobe False - S:", sampler_can_send, "- CH:", child_can_solve, i)
        if child_can_solve[i]:
            len_queue = len(request_queue)
            if len_queue>0:
                occupied_by_source[i] = []
                occupied_by_tag[i] = []
                temp_received_data = np.empty((0,no_parameters))
                for j in range(min(len_queue,Solvers[i].max_requests)):
                    rank_source, tag, received_data = request_queue.popleft()
                    temp_received_data = np.vstack((temp_received_data,received_data.copy()))
                    occupied_by_source[i].append(rank_source)
                    occupied_by_tag[i].append(tag)
                Solvers[i].send_parameters(temp_received_data)
                child_can_solve[i] = False
    
for i in range(no_solvers):
    f = getattr(Solvers[i],"terminate",None)
    if callable(f):
        Solvers[i].terminate()
        
comm_world.Barrier()
print("RANK", rank_world, "(SOLVERS POOL) terminated.")
