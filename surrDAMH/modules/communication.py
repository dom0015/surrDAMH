#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import sys
import os
from surrDAMH.configuration import Configuration
from surrDAMH.surrogates.parent import Evaluator
import copy

TAG_TERMINATE = 0
TAG_READY_TO_RECEIVE = 1
TAG_DATA = 2
TAG_ISEND_START = 10


class SolverMPI:  # initiated by SAMPLERs
    # initiated by SAMPLERs
    # communicates with SOLVERS POOL
    # sends parameters (Send, tag=1,2,...)
    # sends signal that sampler terminated (Send, tag=0)
    # receives observations and conv_tag (recv, tag=1,2,...) or (Recv, tag=conv_tag)
    def __init__(self, conf: Configuration):
        self.solver_pool_rank = conf.solver_pool_rank
        self.rank_collector = conf.rank_collector
        self.pickled_observations = conf.pickled_observations
        self.max_requests = 1
        self.tag_solver = 0
        self.observations = np.zeros(conf.no_observations)
        self.buffer_empty_signal = np.zeros((1,))
        self.terminated = False
        self.comm_world = MPI.COMM_WORLD
        self.status = MPI.Status()

    def send_parameters(self, parameters):
        self.tag_solver += 1
        self.comm_world.Send(parameters, dest=self.solver_pool_rank, tag=self.tag_solver)  # TODO fixed tag

    def recv_observations(self, ):
        if self.pickled_observations:
            [convergence_tag, self.observations] = self.comm_world.recv(source=self.solver_pool_rank)  # , tag=self.tag_solver)
        else:
            self.comm_world.Recv(self.observations, source=self.solver_pool_rank, tag=MPI.ANY_TAG, status=self.status)  # tag=self.tag_solver)
            convergence_tag = self.status.Get_tag()
        return convergence_tag, self.observations.copy()

    def terminate(self, ):
        if not self.terminated:
            self.comm_world.Send(self.buffer_empty_signal, dest=self.solver_pool_rank, tag=TAG_TERMINATE)
            self.terminated = True


class SurrogateLocal_CollectorMPI:
    # initiated by SAMPLERs
    # communicates with COLLECTOR
    # local surrogate evaluator (evaluated on SAMPLERs)
    # receives signals that collector is ready to receive snapshots (Recv, tag=1)
    # receives evaluator instances (irecv, tag=2)
    # sends snapshots (isend, tag=2)
    # sends signals that sampler is ready to receive evaluator instance (Isend, tag=1)
    # sends signal that sampler terminated (Send, tag=0)
    def __init__(self, conf: Configuration, evaluator: Evaluator = None):
        self.conf = conf
        self.evaluators_buffer = [None] * 2       # double buffer
        self.evaluators_buffer_idx = 0            # idx of current buffer 0/1
        self.tag = TAG_ISEND_START
        if self.conf.rank_collector is None:
            self.evaluators_buffer[self.evaluators_buffer_idx] = evaluator
            self.terminated_collector = True
        else:
            self.terminated_collector = False
            self.comm_world = MPI.COMM_WORLD
            self.empty_buffer = np.zeros((1,))
            self.empty_buffer_Isend = np.zeros((1,))
            self.requests = []  # buffer for isend requests
            # self.request_send = None
            self.request_Isend_signal = None
            self.request_evaluator_from_collector()

    def wait_for_evaluator_and_request_new(self, ):
        evaluator_instance = self.request_recv.wait()
        self.evaluators_buffer[1 - self.evaluators_buffer_idx] = evaluator_instance
        self.evaluators_buffer_idx = 1 - self.evaluators_buffer_idx
        self.request_evaluator_from_collector()

    def request_evaluator_from_collector(self, ):
        # sampler expects to receive evaluator later:
        self.request_recv = self.comm_world.irecv(self.conf.max_buffer_size, source=self.conf.rank_collector, tag=TAG_DATA)
        # sends signal to collector that the sampler is ready to receive evaluator
        if self.request_Isend_signal is not None:
            self.request_Isend_signal.Wait()
        self.request_Isend_signal = self.comm_world.Isend(self.empty_buffer_Isend, dest=self.conf.rank_collector, tag=TAG_READY_TO_RECEIVE)

        self.list_of_snapshots = []

    def send_parameters(self, parameters):
        self.parameters = parameters.copy()  # TO DO: copy?

    def recv_observations(self, ):
        if self.evaluators_buffer[self.evaluators_buffer_idx] is None:
            self.wait_for_evaluator_and_request_new()
        computed_observations = self.evaluators_buffer[self.evaluators_buffer_idx](self.parameters)
        return 1, computed_observations

    def send_to_collector(self, snapshot):
        # Adds new snapshot to a list; if COLLECTOR is ready to receive new
        # snapshots, sends list of snapshots to COLLECTOR and empties the list.
        # (only if is_updated == True)

        # SIMPLE VERSION
        data_to_pickle = copy.deepcopy(snapshot)
        # print("COMMUNICATION", self.comm_world.Get_rank(), data_to_pickle, flush=True)

        # if self.request_send is not None:
        #     self.request_send.wait()
        request_send = self.comm_world.isend(data_to_pickle, dest=self.conf.rank_collector, tag=self.tag)

        if self.tag-TAG_ISEND_START >= self.conf.max_sampler_isend_requests:
            self.requests[(self.tag-TAG_ISEND_START) % self.conf.max_sampler_isend_requests].wait()
            self.requests[(self.tag-TAG_ISEND_START) % self.conf.max_sampler_isend_requests] = request_send
        else:
            self.requests.append(request_send)
        self.tag += 1
        ################################################################

        # PREVIOUS VERSION
        # if self.list_of_snapshots:
        #     self.list_of_snapshots = [np.vstack((self.list_of_snapshots[i], snapshot[i])) for i in range(3)]
        # else:
        #     self.list_of_snapshots = snapshot.copy()
        # probe = self.comm_world.Iprobe(source=self.rank_collector, tag=TAG_READY_TO_RECEIVE)
        # if probe:  # if COLLECTOR is ready to receive new snapshots
        #     self.comm_world.Recv(self.empty_buffer, source=self.rank_collector, tag=TAG_READY_TO_RECEIVE)
        #     data_to_pickle = self.list_of_snapshots.copy()
        #     if self.request_send is not None:
        #         self.request_send.wait()
        #     self.request_send = self.comm_world.isend(data_to_pickle, dest=self.rank_collector, tag=TAG_DATA)
        #     self.list_of_snapshots = []
        ################################################################

        # check COMM_WORLD if there is an incoming message with TAG_DATA,
        # if so, receive updated surrogate model evaluator:
        status = self.request_recv.Get_status()
        if status:
            self.wait_for_evaluator_and_request_new()

    def terminate(self, ):
        if not self.terminated_collector:
            # if self.request_send is not None:
            #     self.request_send.wait()
            MPI.Request.waitall(self.requests)
            MPI.Request.waitall([])
            self.comm_world.Send(self.empty_buffer, dest=self.conf.rank_collector, tag=TAG_TERMINATE)
            self.terminated_collector = True
