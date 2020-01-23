#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:36:27 2020

@author: simona
"""

import numpy as np
from mpi4py import MPI
from collections import deque

# waits for a request from a sampling algorithm
# (possibly from one of several sampling algorithms)
# - either solves the request localy 
#   and immediately sends the solution back to the sampling algorithm
# - or sends it to a free spawned child,
#   receives solution from the child (non-blocking)
#   and sends the solution back to the sampling algorithm

# knows ranks of all connected algorithms,   ... is this necessary?
# expects messages with increasing tag (1,2,...),   ... is this necessary?
# finishes when received signal to exit from all connected algorithms (tag=0)

# full solver spawns external solvers that solve the requests
# in a loop (while some alg. is active, queue is empty and solvers are free):
# - check for new messages on each communicator (using Iprobe)
#   - world communicator (algorithms) - any source
#     - put the request into a queue with the identification of the algorithm
#     - if the tag is 0, mark the algorithm as inactive
#   - several spawned communicators - source 0 
#     - if there is a new message (solution), free the solver
#       and (I)send the solution to the corresponding algorithm 
# - check the request queue
#   - if the queue is not empty, send a requests to a free solver, 
#     assign the rank of the algorithms to this solver
def initialize_and_manage_solvers(solver_init, solver_parameters, no_solvers, no_samplers):
    # solver_init ... initializes the object of the (full, surrogate) solver
    # no_solvers ... number of solvers to be created
    # no_samplers ... number of samplers that request solutions
    
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    
    Solvers = []
    for i in range(no_solvers):
        Solvers.append(solver_init(**solver_parameters[i]))
    algorithm_ranks = np.arange(no_samplers)
    is_active = np.array([True] * len(algorithm_ranks))
    occupied_by_source = [None] * no_solvers
    occupied_by_tag = [None] * no_solvers
    is_free = np.array([True] * no_solvers)
    no_parameters = Solvers[0].no_parameters
    status = MPI.Status()
    received_data = np.zeros(no_parameters)
    request_queue = deque()
    #temp_received_data = [received_data] * no_solvers
    while any(is_active): # while at least 1 sampling algorithm is active
        tmp = comm_world.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        if tmp: # if there is an incoming message from any sampling algorithm
            rank_source = status.Get_source()
            tag = status.Get_tag()
            comm_world.Recv(received_data, source=rank_source, tag=tag)
            print('DEBUG --- RANK --- SOURCE --- TAG:', rank_world, rank_source, tag)
            if tag == 0: # if received message has tag 0, switch corresponding sampling alg. to inactive
                # assumes that there will be no other incoming message from that source 
                is_active[algorithm_ranks == rank_source] = False
            else: # put the request into queue (remember source and tag)
                request_queue.append([rank_source,tag,received_data.copy()])
        for i in range(no_solvers):
            if not is_free[i]: # check all busy solvers if they finished the request
                if Solvers[i].request_solved: # if so, send solution to the sampling algorithm
                    sent_data = Solvers[i].get_solution()
                    is_free[i] = True # mark the solver as free
                    for j in range(len(occupied_by_source[i])):
                        comm_world.Send(sent_data[j].copy(), dest=occupied_by_source[i][j], tag=occupied_by_tag[i][j])
            if is_free[i]:
                occupied_by_source[i] = []
                occupied_by_tag[i] = []
                temp_received_data = np.empty((0,no_parameters))
                len_queue = len(request_queue)
                if len_queue>0:
                    for j in range(min(len_queue,Solvers[i].max_requests)):
                        rank_source, tag, received_data = request_queue.popleft()
                        temp_received_data = np.vstack((temp_received_data,received_data.copy()))
                        occupied_by_source[i].append(rank_source)
                        occupied_by_tag[i].append(tag)
                    is_free[i] = False
                    Solvers[i].send_request(temp_received_data)
        
    for i in range(no_solvers):
        f = getattr(Solvers[i],"terminate",None)
        if callable(f):
            Solvers[i].terminate()
    
    comm_world.Barrier()
    print("MPI process", rank_world, "(full solver) terminated.")