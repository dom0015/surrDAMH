#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:55:37 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
from collections import deque
from surrDAMH.configuration import Configuration
from surrDAMH.priors.parent import Prior
from surrDAMH.solver_specification import SolverSpec
from surrDAMH.modules.tools import ensure_dir
import os
import sys

# SOLVERS POOL communicates with SAMPLERs and CHILD SOLVERs
# with SAMPLER:
# receives parameters ( , tag=1,2,...)
# receives signal that sampler terminated ( , tag=0)
# sends observations and conv_tag ( , tag=1,2,...) or ( , tag=conv_tag)
# with CHILD:


class CommunicationWithChild:
    def __init__(self, conf, transform, solver_spec, solver_output_dir, solver_id):
        self.pickled_observations = conf.pickled_observations
        self.max_requests = 1
        child_process_path = os.path.dirname(os.path.abspath(__file__))
        self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                                        args=[child_process_path+'/process_CHILD.py', str(solver_id), solver_output_dir],
                                        maxprocs=conf.solver_maxprocs)
        self.tag = 0
        self.received_data = np.zeros(conf.no_observations)
        self.status = MPI.Status()
        self.comm.bcast([conf, transform, solver_spec], root=MPI.ROOT)

    def send_parameters(self, data_par):
        self.tag += 1
        self.data_par = data_par.copy()
        self.comm.Bcast([np.array(self.tag, 'i'), MPI.INT], root=MPI.ROOT)
        self.comm.Bcast([self.data_par, MPI.DOUBLE], root=MPI.ROOT)

    def recv_observations(self):
        if self.pickled_observations:
            convergence_tag, self.received_data = self.comm.recv(source=0, tag=self.tag)
        else:
            self.comm.Recv(self.received_data, source=0, tag=MPI.ANY_TAG, status=self.status)
            convergence_tag = self.status.Get_tag()
        return convergence_tag, self.received_data.reshape((1, -1)).copy()

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
        print("Solver spawned by rank", MPI.COMM_WORLD.Get_rank(), "disconnected.")


def run_SOLVER(conf: Configuration, prior: Prior, solver_spec: SolverSpec):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    comm_world.Split(color=1, key=rank_world)

    comm_with_child = []
    for i in range(conf.no_solvers):
        solver_output_dir = ensure_dir(os.path.join(conf.output_dir, "solver_output", "rank{}".format(i)))
        comm_with_child.append(CommunicationWithChild(conf=conf, transform=prior.transform, solver_spec=solver_spec,
                                                      solver_output_dir=solver_output_dir, solver_id=i))
    samplers_rank = np.arange(conf.no_samplers)
    sampler_is_active = np.array([True] * conf.no_samplers)
    sampler_can_send = np.array([True] * conf.no_samplers)
    occupied_by_source = [None] * conf.no_solvers
    occupied_by_tag = [None] * conf.no_solvers
    child_can_solve = np.array([True] * conf.no_solvers)
    no_parameters = conf.no_parameters
    received_data = np.zeros(no_parameters)
    status = MPI.Status()
    parameters_queue = deque()

    def receive_observations_and_resend(i):
        convergence_tag, sent_data = comm_with_child[i].recv_observations()
        child_can_solve[i] = True  # mark the solver as free
        for j in range(len(occupied_by_source[i])):
            rank_dest = occupied_by_source[i][j]
            if conf.pickled_observations:
                comm_world.send([convergence_tag, sent_data[j, :].copy()], dest=rank_dest, tag=occupied_by_tag[i][j])
            else:
                comm_world.Send(sent_data[j, :].copy(), dest=rank_dest, tag=convergence_tag)  # occupied_by_tag[i][j])
            sampler_can_send[samplers_rank == rank_dest] = True

    def receive_parameters_from_sampler():
        if any(sampler_can_send) and any(child_can_solve):
            if all(child_can_solve):  # no child is busy
                probe = comm_world.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            else:
                probe = comm_world.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if probe:  # if there is an incoming message from any sampler
                # receive this message (one message from one sampler)
                rank_source = status.Get_source()
                tag = status.Get_tag()
                comm_world.Recv(received_data, source=rank_source, tag=tag)
                if tag == 0:  # if received message has tag 0, switch corresponding sampler to inactive
                    # there will be no other message from that sampler
                    sampler_is_active[samplers_rank == rank_source] = False
                    sampler_can_send[samplers_rank == rank_source] = False
                else:  # put the request into queue (remember source and tag)
                    parameters_queue.append([rank_source, tag, received_data.copy()])
                    # nothing else will come from this sampler until completion of this request
                    sampler_can_send[samplers_rank == rank_source] = False
            elif conf.debug:
                print("debug - RANK", rank_world, "POOL Iprobe False - S:", sampler_can_send, "- CH:", child_can_solve)

    while any(sampler_is_active):  # while at least 1 sampling algorithm is active
        receive_parameters_from_sampler()
        for i in range(conf.no_solvers):
            if not child_can_solve[i]:  # check all busy child solvers if they finished the request
                if comm_with_child[i].is_solved():  # if so, send solution to the sampling algorithm
                    receive_observations_and_resend(i)
                elif conf.debug:
                    print("debug - RANK", rank_world, "PARENT Iprobe False - S:", sampler_can_send, "- CH:", child_can_solve, i)
            if child_can_solve[i]:
                len_queue = len(parameters_queue)
                if len_queue > 0:
                    occupied_by_source[i] = []
                    occupied_by_tag[i] = []
                    temp_received_data = np.empty((0, no_parameters))
                    for j in range(min(len_queue, comm_with_child[i].max_requests)):
                        rank_source, tag, received_data = parameters_queue.popleft()
                        temp_received_data = np.vstack((temp_received_data, received_data.copy()))
                        occupied_by_source[i].append(rank_source)
                        occupied_by_tag[i].append(tag)
                    comm_with_child[i].send_parameters(temp_received_data)
                    child_can_solve[i] = False

    for i in range(conf.no_solvers):
        f = getattr(comm_with_child[i], "terminate", None)
        if callable(f):
            comm_with_child[i].terminate()

    comm_world.Barrier()
    comm_world.Barrier()
    return []
