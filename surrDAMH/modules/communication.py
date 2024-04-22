#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import numpy.typing as npt
from typing import Tuple
from surrDAMH.configuration import Configuration
from surrDAMH.surrogates.parent import Evaluator
import copy
import time

TAG_TERMINATE = 0
TAG_READY_TO_RECEIVE = 1
TAG_DATA = 2
TAG_ISEND_START = 10


class Communicator:
    """
    Parent class for communication between samplers and
    full/surrogate solver (and collector, optionally).
    """

    def __init__(self) -> None:
        pass

    def set_parameters(self, parameters: npt.ArrayLike) -> None:
        """
        Sets sample for which the observations will be computed later
        (by the full/surrogate solver).
        """
        pass

    def get_observations(self) -> Tuple[npt.NDArray, int]:
        """
        Gets observations computed by the full/surrogate solver and solver tag.
        """
        raise NotImplementedError

    def send_to_collector(self, data: list) -> None:
        """
        Sends triplets [sample, observations, weight] to the collector.
        Optional.
        """
        pass


class SolverMPI(Communicator):
    # initiated by SAMPLERs
    # communicates with SOLVERS POOL
    # sends parameters (Send, tag=1,2,...)
    # sends signal that sampler terminated (Send, tag=0)
    # receives observations and conv_tag (recv, tag=1,2,...) or (Recv, tag=conv_tag)
    def __init__(self, conf: Configuration) -> None:
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
        self.tt_recv = 0
        self.tt_send = 0

    def set_parameters(self, parameters: npt.ArrayLike) -> None:
        self.tag_solver += 1
        tmp = time.time()
        self.comm_world.Send(parameters, dest=self.solver_pool_rank, tag=self.tag_solver)  # TODO fixed tag
        tmp2 = time.time() - tmp
        self.tt_send = self.tt_send + tmp2

    def get_observations(self, ):
        tmp = time.time()
        if self.pickled_observations:
            [self.observations, solver_tag] = self.comm_world.recv(source=self.solver_pool_rank)  # , tag=self.tag_solver)
        else:
            self.comm_world.Recv(self.observations, source=self.solver_pool_rank, tag=MPI.ANY_TAG, status=self.status)  # tag=self.tag_solver)
            solver_tag = self.status.Get_tag()
        tmp2 = time.time() - tmp
        self.tt_recv = self.tt_recv + tmp2
        return self.observations.copy(), solver_tag

    def terminate(self, ):
        print("communication RECV", self.tt_recv, "*************")
        print("communication SEND", self.tt_send, "*************")
        if not self.terminated:
            self.comm_world.Send(self.buffer_empty_signal, dest=self.solver_pool_rank, tag=TAG_TERMINATE)
            self.terminated = True


class SurrogateLocal_CollectorMPI(Communicator):
    # initiated by SAMPLERs
    # communicates with COLLECTOR
    # local surrogate evaluator (evaluated on SAMPLERs)
    # receives signals that collector is ready to receive snapshots (Recv, tag=1)
    # receives evaluator instances (irecv, tag=2)
    # sends snapshots (isend, tag=2)
    # sends signals that sampler is ready to receive evaluator instance (Isend, tag=1)
    # sends signal that sampler terminated (Send, tag=0)
    def __init__(self, conf: Configuration, evaluator: Evaluator | None = None) -> None:
        self.conf = conf
        self.evaluators_buffer: list[Evaluator | None] = [None] * 2  # double buffer
        self.evaluators_buffer_idx = 0  # idx of current buffer 0/1
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
        assert self.conf.rank_collector is not None
        # sampler expects to receive evaluator later:
        self.request_recv = self.comm_world.irecv(self.conf.max_buffer_size, source=self.conf.rank_collector, tag=TAG_DATA)
        # sends signal to collector that the sampler is ready to receive evaluator
        if self.request_Isend_signal is not None:
            self.request_Isend_signal.Wait()
        self.request_Isend_signal = self.comm_world.Isend(self.empty_buffer_Isend, dest=self.conf.rank_collector, tag=TAG_READY_TO_RECEIVE)

        self.list_of_snapshots = []

    def set_parameters(self, parameters: npt.NDArray) -> None:
        self.parameters = parameters.copy()  # TO DO: copy?

    def get_observations(self, ):
        if self.evaluators_buffer[self.evaluators_buffer_idx] is None:
            self.wait_for_evaluator_and_request_new()
        evaluator = self.evaluators_buffer[self.evaluators_buffer_idx]
        assert evaluator is not None
        computed_observations = evaluator(self.parameters)
        return computed_observations, 1

    def send_to_collector(self, snapshot):
        assert self.conf.rank_collector is not None
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
            assert self.conf.rank_collector is not None
            # if self.request_send is not None:
            #     self.request_send.wait()
            MPI.Request.waitall(self.requests)
            MPI.Request.waitall([])
            self.comm_world.Send(self.empty_buffer, dest=self.conf.rank_collector, tag=TAG_TERMINATE)
            self.terminated_collector = True
