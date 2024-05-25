#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:55:37 2019

@author: simona
"""

import os
import sys
from collections import deque
from typing import Any, List

import numpy as np
from mpi4py import MPI

from surrDAMH.configuration import Configuration
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.solver_specification import SolverSpec

# SOLVERS POOL communicates with SAMPLERs and CHILD SOLVERs
# with SAMPLER:
# receives parameters ( , tag=1,2,...)
# receives signal that sampler terminated ( , tag=0)
# sends observations and conv_tag ( , tag=1,2,...) or ( , tag=conv_tag)
# with CHILD:


class CommunicationWithChild:
    def __init__(self, conf: Configuration, solver_spec: SolverSpec, solver_output_dir: str, solver_id: int) -> None:
        self.pickled_observations = conf.pickled_observations
        child_process_path = os.path.dirname(os.path.abspath(__file__))
        self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                                        args=[child_process_path+'/process_CHILD.py', str(solver_id), solver_output_dir],
                                        maxprocs=conf.solver_maxprocs)
        self.tag = 0
        self.received_data = np.zeros((conf.no_observations,))
        self.status = MPI.Status()
        self.comm.bcast([conf, solver_spec], root=MPI.ROOT)

    def send_parameters(self, data_par):
        self.tag += 1
        self.comm.Bcast([np.array(self.tag, 'i'), MPI.INT], root=MPI.ROOT)
        self.comm.Bcast([data_par, MPI.DOUBLE], root=MPI.ROOT)

    def recv_observations(self):
        if self.pickled_observations:
            self.received_data, solver_tag = self.comm.recv(source=0, tag=self.tag)
        else:
            self.comm.Recv(self.received_data, source=0, tag=MPI.ANY_TAG, status=self.status)
            solver_tag = self.status.Get_tag()
        return self.received_data.flatten().copy(), solver_tag

    def is_solved(self):
        # check the parent-child communicator if there is an incoming message
        if self.pickled_observations:
            tmp = self.comm.Iprobe(source=0, tag=self.tag)
        else:
            tmp = self.comm.Iprobe(source=0, tag=MPI.ANY_TAG)  # tag=self.tag)
        if tmp:
            return True
        else:
            return False

    def terminate(self):
        # self.comm.Barrier()
        self.comm.Bcast([np.array(0, 'i'), MPI.INT], root=MPI.ROOT)
        self.comm.Barrier()
        self.comm.Disconnect()
        # print("Solver spawned by rank", MPI.COMM_WORLD.Get_rank(), "disconnected.", flush=True)


def run_SOLVER(conf: Configuration, solver_spec: SolverSpec):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    comm_world.Split(color=1, key=rank_world)

    comm_with_child = []
    for i in range(conf.no_solvers):
        solver_output_dir = ensure_dir(os.path.join(conf.output_dir, "solver_output", "rank{}".format(i)))
        comm_with_child.append(CommunicationWithChild(conf=conf, solver_spec=solver_spec,
                                                      solver_output_dir=solver_output_dir, solver_id=i))
    samplers_rank = np.arange(conf.no_samplers)
    sampler_is_active = np.array([True] * conf.no_samplers)
    sampler_can_send = np.array([True] * conf.no_samplers)
    occupied_by_source: List[Any] = [None] * conf.no_solvers
    occupied_by_tag: List[Any] = [None] * conf.no_solvers
    child_can_solve = np.array([True] * conf.no_solvers)
    no_parameters = conf.no_parameters
    received_data = np.zeros(no_parameters)
    status = MPI.Status()
    parameters_queue = deque()

    def receive_observations_and_resend(i):
        sent_data, solver_tag = comm_with_child[i].recv_observations()
        child_can_solve[i] = True  # mark the solver as free
        rank_dest = occupied_by_source[i]
        if conf.pickled_observations:
            comm_world.send([sent_data.copy(), solver_tag], dest=rank_dest, tag=occupied_by_tag[i])
        else:
            comm_world.Send(sent_data.copy(), dest=rank_dest, tag=solver_tag)  # occupied_by_tag[i])
        sampler_can_send[samplers_rank == rank_dest] = True

    def receive_parameters_from_sampler():
        sources = samplers_rank[sampler_can_send]
        sources = np.random.permutation(sources)
        for rank in sources:
            # if any(sampler_can_send):  # and any(child_can_solve):
            if False and all(child_can_solve):  # no child is busy, wait for an incoming message from any sampler
                probe = comm_world.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            else:
                probe = comm_world.Iprobe(source=rank, tag=MPI.ANY_TAG, status=status)
            if probe:  # if there is an incoming message from any sampler
                # receive this message (one message from one sampler)
                rank_source = status.Get_source()
                tag = status.Get_tag()
                comm_world.Recv(received_data, source=rank_source, tag=tag)
                sampler_can_send[samplers_rank == rank_source] = False
                if tag == 0:  # if received message has tag 0, switch corresponding sampler to inactive
                    # there will be no other message from that sampler
                    sampler_is_active[samplers_rank == rank_source] = False
                else:  # put the request into queue (remember source and tag)
                    parameters_queue.append([rank_source, tag, received_data.copy()])
                    # nothing else will come from this sampler until completion of this request

    while any(sampler_is_active):  # while at least 1 sampling algorithm is active
        receive_parameters_from_sampler()
        for i in range(conf.no_solvers):  # for all child solvers
            if not child_can_solve[i]:  # if the child is busy, check if it finished its request
                if comm_with_child[i].is_solved():  # if finished, send solution to sampler
                    receive_observations_and_resend(i)
            if child_can_solve[i]:
                if parameters_queue:  # if the queue is not empty
                    rank_source, tag, received_data = parameters_queue.popleft()
                    occupied_by_source[i] = rank_source
                    occupied_by_tag[i] = tag
                    comm_with_child[i].send_parameters(received_data)
                    child_can_solve[i] = False
    for i in range(conf.no_solvers):
        f = getattr(comm_with_child[i], "terminate", None)
        if callable(f):
            comm_with_child[i].terminate()

    comm_world.Barrier()
    comm_world.Barrier()
    return []
