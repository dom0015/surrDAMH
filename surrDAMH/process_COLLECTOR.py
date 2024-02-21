#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:26:55 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
from surrDAMH.configuration import Configuration
from surrDAMH.surrogates.parent import Updater
from dataclasses import dataclass

# communicates with: SAMPLERs
# sends signals that collector is ready to receive snapshots (Isend, tag=1)
# sends evaluator instances (isend, tag=2)
# receives snapshots (irecv, tag=2)
# receives signals that sampler is ready to receive evaluator instance (Recv, tag=1)
# receives signal that sampler terminated (Recv, tag=0)

TAG_TERMINATE = 0
TAG_READY_TO_RECEIVE = 1
TAG_DATA = 2

ANALYZE = False


def run_COLLECTOR(conf: Configuration, surrogate_updater: Updater, surrogate_delayed_init_data=None):
    surrogate_updater.delayed_init(surrogate_delayed_init_data)

    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    comm_world.Split(color=2, key=rank_world)

    # general:
    sampler_ranks = conf.sampler_ranks
    sampler_is_active = np.array([True] * conf.no_samplers)  # active until it terminates

    # related to surrogate evaluators:
    buffers_evaluator = [None] * conf.no_samplers
    sampler_can_recv_evaluator = np.array([False] * conf.no_samplers)  # ready to receive updated evaluator
    sampler_got_last_evaluator = np.array([True] * conf.no_samplers)
    request_isend_evaluator = [None] * conf.no_samplers

    # related to received snapshots
    list_received_snapshots = []
    request_irecv_snapshots = [None] * conf.no_samplers

    # related to signals
    buffer_empty_signal = np.zeros((1,))
    buffers_empty_signal = [None] * conf.no_samplers  # when only tag is important
    for i in range(conf.no_samplers):
        buffers_empty_signal[i] = np.zeros((1,))
    request_Isend_signal = [None] * conf.no_samplers

    def request_snapshots_from_sampler():
        # collector expects to receive snapshots from this (active) sampler later:
        request_irecv_snapshots[i] = comm_world.irecv(conf.max_buffer_size, source=rank, tag=TAG_DATA)
        # sends signal to this (active) sampler that collector is ready to receive snapshots:
        if request_Isend_signal[i] is not None:
            request_Isend_signal[i].Wait()
        request_Isend_signal[i] = comm_world.Isend(buffers_empty_signal[i], dest=rank, tag=TAG_READY_TO_RECEIVE)

    if any(sampler_is_active):
        for rank in sampler_ranks[sampler_is_active]:
            i = sampler_ranks[sampler_ranks == rank][0]
            request_snapshots_from_sampler()

    if ANALYZE:
        list_all_snapshots = []
        list_all_evaluators = []

    while any(sampler_is_active):  # while at least 1 sampling algorithm is active
        for rank in sampler_ranks[sampler_is_active]:
            i = sampler_ranks[sampler_ranks == rank][0]
            # check if there are incoming snapshots from this active sampler;
            # if so, receive the snapshots and create new request:
            if request_irecv_snapshots[i].Get_status():
                # receive snapshots and add to list:
                list_received_part = request_irecv_snapshots[i].wait()
                if list_received_snapshots:
                    list_received_snapshots = [np.vstack((list_received_snapshots[j], list_received_part[j])) for j in range(3)]
                else:
                    list_received_snapshots = list_received_part.copy()
                if ANALYZE:
                    if list_all_snapshots:
                        list_all_snapshots = [np.vstack((list_all_snapshots[j], list_received_part[j])) for j in range(3)]
                    else:
                        list_all_snapshots = list_received_part.copy()
                request_snapshots_from_sampler()
            # check if there is an incoming signal from this active sampler
            # (tag_terminate | tag_ready_to_receive (updated) evaluator):
            status = MPI.Status()
            probe = comm_world.Iprobe(source=rank, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if probe:  # there is an incomming signal from this sampler
                if tag == TAG_TERMINATE:
                    # if received message has tag_terminate, switch corresponding sampler to inactive,
                    # there will be no other incoming message from this sampler
                    sampler_is_active[i] = False
                    comm_world.Recv(buffer_empty_signal, source=rank, tag=TAG_TERMINATE)  # TODO: cancel message
                elif tag == TAG_READY_TO_RECEIVE:
                    # if received message has tag_ready_to_receive,
                    # the corresponding sampler is ready to receive (updated) evaluator
                    sampler_can_recv_evaluator[i] = True
                    comm_world.Recv(buffer_empty_signal, source=rank, tag=TAG_READY_TO_RECEIVE)  # TODO: cancel message
        # use received snapshots to update the surorgate model, the updater instance is local:
        if list_received_snapshots and any(sampler_is_active):
            surrogate_updater.add_data(list_received_snapshots[0], list_received_snapshots[1], list_received_snapshots[2])
            evaluator_instance = surrogate_updater.get_evaluator()
            list_received_snapshots = []
            sampler_got_last_evaluator = np.array([False] * conf.no_samplers)
            if ANALYZE:
                list_all_evaluators.append(evaluator_instance)
        for rank in sampler_ranks[sampler_is_active & sampler_can_recv_evaluator & ~sampler_got_last_evaluator]:
            i = sampler_ranks[sampler_ranks == rank][0]
            if request_isend_evaluator[i] is not None:
                request_isend_evaluator[i].wait()
            buffers_evaluator[i] = evaluator_instance  # TODO: copy?
            request_isend_evaluator[i] = comm_world.isend(buffers_evaluator[i], dest=rank, tag=TAG_DATA)
            sampler_can_recv_evaluator[i] = False
            sampler_got_last_evaluator[i] = True

    comm_world.Barrier()

    print("RANK", rank_world, "(DATA COLLECTOR) terminated.")

    if ANALYZE:
        @dataclass
        class Output:
            list_all_snapshots: list
            list_all_evaluators: list
        output = Output(list_all_snapshots, list_all_evaluators)
        return output
    else:
        return []
