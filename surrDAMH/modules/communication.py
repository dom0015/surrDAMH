#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import time
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from surrDAMH.configuration import Configuration
from surrDAMH.surrogates.parent import Evaluator

TAG_TERMINATE = 0
TAG_READY_TO_RECEIVE = 1
TAG_DATA = 2
TAG_UPDATE = 3  # signal that sampler wants new evaluator if available
TAG_EVALUATOR_OBJECT = 4  # this message contains evaluator object (or None if sampler terminated)
TAG_STOP_UPDATING = 5  # when sampler knows that it will not want new evaluator later
TAG_FIRST_SNAPSHOT = 10


class CommEvaluator_sampler:
    """
    MPI communication between one sampler and collector regarding surrogate model evaluators,
    sampler side.
    """

    def __init__(self, rank_collector: int, max_buffer_size: int = 1 << 30) -> None:
        self.rank_collector = rank_collector
        self.max_buffer_size = max_buffer_size
        self.comm_world = MPI.COMM_WORLD
        self.idx = 0
        self.evaluator = None
        self.request_Isend = None

    def request_evaluator(self) -> None:
        """
        Sends signal to collector, that he is ready to receive a new evaluator.
        Prepares irecv request for the new evaluator.
        """
        self.idx += 1
        buf = np.array([self.idx], dtype=int)
        if self.request_Isend is not None:
            self.request_Isend.Wait()
        self.request_Isend = self.comm_world.Isend(buf=buf, dest=self.rank_collector, tag=TAG_UPDATE)
        self.request_irecv = self.comm_world.irecv(buf=self.max_buffer_size, source=self.rank_collector, tag=TAG_EVALUATOR_OBJECT)

    def evaluator_is_available(self) -> bool:
        """
        Checks if the surrogate model evaluator can be received without waiting.
        """
        return self.request_irecv.get_status()

    def get_evaluator(self) -> Evaluator:
        """
        Waits for current evaluator.
        """
        evaluator = self.request_irecv.wait()
        self.evaluator = evaluator
        return evaluator

    def get_evaluator_and_terminate(self) -> Evaluator:
        """
        Sends termination signal to the collector.
        Receives last evaluator (or None if a new evaluator is not available).
        """
        buf = np.array([self.idx], dtype=int)
        self.comm_world.Send(buf=buf, dest=self.rank_collector, tag=TAG_STOP_UPDATING)  # TODO: Isend?
        evaluator = self.request_irecv.wait()
        if evaluator is None:  # this means that at least one evaluator has been received before
            return self.evaluator
        return evaluator


class CommEvaluator_collector():
    """
    MPI communication between one sampler and collector regarding surrogate model evaluators,
    collector side.
    On initialization, prepares irecv request for a signal from sampler.
    """

    def __init__(self, rank_sampler: int) -> None:
        self.rank_sampler = rank_sampler
        self.comm_world = MPI.COMM_WORLD
        self.active = True
        self.current_idx = 0
        self.recv_buffer_update = np.zeros(shape=(1,), dtype=int)
        self.request_update_signal = self.comm_world.Irecv(buf=self.recv_buffer_update, source=self.rank_sampler, tag=TAG_UPDATE)
        self.recv_buffer_stop = np.zeros(shape=(1,), dtype=int)
        self.request_stop_signal = self.comm_world.Irecv(buf=self.recv_buffer_stop, source=self.rank_sampler, tag=TAG_STOP_UPDATING)
        self.isend_request = None

    def sampler_requests_evaluator(self) -> bool:
        """
        Checks if there is an incoming "update" signal from the sampler,
        receives it and prepares a new request.
        """
        if self.request_update_signal.Get_status():
            self.request_update_signal.Wait()
            self.request_update_signal = self.comm_world.Irecv(buf=self.recv_buffer_update, source=self.rank_sampler, tag=TAG_UPDATE)
            self.current_idx = self.recv_buffer_update[0]
            return True
        return False

    def sampler_stops(self) -> bool:
        """
        Checks if there is a "stop updating" signal from the sampler and receives it.
        """
        if self.request_stop_signal.Get_status():
            self.request_update_signal.Cancel()
            self.request_stop_signal.Wait()
            self.max_idx = self.recv_buffer_stop[0]
            self.active = False
            return True
        return False

    def send_evaluator(self, evaluator: Evaluator | None) -> None:
        """
        Sends evaluator to the sampler.
        """
        if self.isend_request is not None:
            self.isend_request.wait()
        self.isend_request = self.comm_world.isend(obj=evaluator, dest=self.rank_sampler, tag=TAG_EVALUATOR_OBJECT)

    def terminate(self, last_evaluator: Evaluator | None) -> None:
        """
        If all update signals have been received,
        cancel last update signal request.
        If the last update signal haven't been received yet,
        send the last evaluator (or None, if this one was sent already),
        wait for the last update signal.
        IF NO EVALUATOR HAVE BEEN SENT YET, MAKE SURE TO DO IT NOW.
        """
        if self.current_idx == self.max_idx:
            self.request_update_signal.Cancel()
        else:
            # TODO: isend?
            self.comm_world.send(obj=last_evaluator, dest=self.rank_sampler, tag=TAG_EVALUATOR_OBJECT)
            self.request_update_signal.Wait()


class CommSnapshot_sampler:
    """
    MPI communication between one sampler and collector regarding collecting snapshots,
    sampler side.
    Snapshot is a list containing:
        parameters (npt.NDArray)
        observations (npt.NDArray)
        weight (int)
        likelihood (float)
    """

    def __init__(self, rank_collector: int, max_sampler_isend_requests: int = 100) -> None:
        self.rank_collector = rank_collector
        self.max_sampler_isend_requests = max_sampler_isend_requests
        self.comm_world = MPI.COMM_WORLD
        self.requests = []  # buffer for isend requests
        self.idx = TAG_FIRST_SNAPSHOT

    def send_to_collector(self, snapshot: List[Any]):
        """
        Sends snapshot to collector.
        """
        data_to_pickle = copy.deepcopy(snapshot)
        # create isend request for the snapshot
        request_send = self.comm_world.isend(data_to_pickle, dest=self.rank_collector, tag=self.idx)
        # store the isend request in a list
        # if the number of unsent messages is above maximum, wait for one of them
        if (self.idx-TAG_FIRST_SNAPSHOT) >= self.max_sampler_isend_requests:
            idx_sent, _ = MPI.Request.waitany(self.requests)
            self.requests[idx_sent] = request_send
        else:
            self.requests.append(request_send)
        self.idx += 1

    def terminate(self) -> None:
        """
        Sends message to collector, tag indicates that there will be no more snapshots from this sampler,
        message is the tag of the last snapshot sent by this sampler.
        """
        MPI.Request.waitall(self.requests)
        # sends the number of sent snapshots
        buf = np.array([self.idx-1], dtype=int)
        # TODO: Isend?
        self.comm_world.Send(buf=buf, dest=self.rank_collector, tag=TAG_TERMINATE)


class CommSnapshot_collector:
    """
    MPI communication between one sampler and collector regarding collecting snapshots,
    collector side.
    Snapshot is a list containing:
        parameters (npt.NDArray)
        observations (npt.NDArray)
        weight (int)
        likelihood (float)
    """

    def __init__(self, rank_sampler: int) -> None:
        self.rank_sampler = rank_sampler
        self.comm_world = MPI.COMM_WORLD
        self.sampler_terminated = False  # switch to True after receiving termination signal
        self.all_snapshots_were_received = False  # switch to True after receiving last snapshot
        self.max_idx = np.inf
        # prepare for receiving termination signal
        self.buffer_terminate = np.zeros(shape=(1,), dtype=int)
        self.request_terminate = self.comm_world.Irecv(buf=self.buffer_terminate, source=MPI.ANY_SOURCE, tag=TAG_TERMINATE)
        # prepare for receiving first snapshot
        self.current_idx = TAG_FIRST_SNAPSHOT
        self.request_snapshot = self.comm_world.irecv(source=self.rank_sampler, tag=self.current_idx)

    def snapshot_is_available(self) -> bool:
        """
        Return:
            bool
                True (a snapshot can be received without waiting),
                False (otherwise)
        Check if the sampler terminated. If so, get the index of the last snapshot.
        If all snapshots have been received, cancel the last request and set active to False.
        If there is some undelivered snapshot, check if it is ready to be received.
        """
        if not self.sampler_terminated:
            # if the sampler terminated
            if self.request_terminate.Get_status():
                # receive the message
                self.request_terminate.Wait()
                # get the index of the last snapshot that has to be received
                self.max_idx = self.buffer_terminate[0]
                # the sampler terminated but there may be undelivered snapshots
                self.sampler_terminated = True
        # if there is at least one snapshots to be received
        if self.current_idx < self.max_idx:
            if self.request_snapshot.get_status():
                return True
            else:
                return False
        else:
            # all snapshots have been received
            self.all_snapshots_were_received = True
            # cancel the request
            self.request_snapshot.cancel()
            return False

    def all_snapshots_received(self):
        return self.all_snapshots_were_received

    def get_snapshot_and_request_new(self) -> List[Any]:
        """
        Receive snapshot and request new.
        """
        snapshot = self.request_snapshot.wait()
        self.current_idx += 1
        self.request_snapshot = self.comm_world.irecv(source=self.rank_sampler, tag=self.current_idx)
        return snapshot

    def terminate(self) -> None:
        """
        Receives all remaining snapshots.
        Receives termination signal if not received yet.
        """
        # wait for termination signal
        # meanwhile, receive_snapshots
        while not self.sampler_terminated:
            if self.snapshot_is_available():
                self.get_snapshot_and_request_new()
        # Termination signal received,
        # the total number of snapshots is known.
        # If they were all received, the last request was cancelled
        # and active was set to False.
        # TODO: If not, is it necessary to receive remaining snapshots? Cancel the last request.
        if not self.all_snapshots_were_received:
            """
            while self.current_idx < self.max_idx:
                self.get_snapshot_and_request_new()
            """
            self.request_snapshot.cancel()


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
        assert conf.rank_solvers_pool is not None, "rank_solvers_pool is None"
        self.rank_solvers_pool = conf.rank_solvers_pool
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
        self.comm_world.Send(parameters, dest=self.rank_solvers_pool, tag=self.tag_solver)  # TODO fixed tag
        tmp2 = time.time() - tmp
        self.tt_send = self.tt_send + tmp2

    def get_observations(self, ):
        tmp = time.time()
        if self.pickled_observations:
            [self.observations, solver_tag] = self.comm_world.recv(source=self.rank_solvers_pool)  # , tag=self.tag_solver)
        else:
            self.comm_world.Recv(self.observations, source=self.rank_solvers_pool, tag=MPI.ANY_TAG, status=self.status)  # tag=self.tag_solver)
            solver_tag = self.status.Get_tag()
        tmp2 = time.time() - tmp
        self.tt_recv = self.tt_recv + tmp2
        return self.observations.copy(), solver_tag

    def terminate(self, ):
        if not self.terminated:
            self.comm_world.Send(self.buffer_empty_signal, dest=self.rank_solvers_pool, tag=TAG_TERMINATE)
            self.terminated = True
