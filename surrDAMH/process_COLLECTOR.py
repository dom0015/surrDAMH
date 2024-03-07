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
    status = MPI.Status()

    # general:
    sampler_ranks = conf.sampler_ranks
    sampler_is_active = np.array([True] * conf.no_samplers)  # active until it terminates
    no_snapshots_received = 0  # how many snapshots the surrogate model updater got
    no_snapshots_current = 0  # using how many snapshots the current evaluator was created

    # related to surrogate evaluators:
    buffers_evaluator = [None] * conf.no_samplers
    sampler_can_recv_evaluator = np.array([False] * conf.no_samplers)  # ready to receive updated evaluator
    sampler_got_current_evaluator = np.array([True] * conf.no_samplers)
    request_isend_evaluator = [None] * conf.no_samplers

    # related to received snapshots
    list_received_snapshots = [np.empty((0, conf.no_parameters)), np.empty((0, conf.no_observations)), np.empty((0, 1))]
    # request_irecv_snapshots = [None] * conf.no_samplers

    # related to signals
    buffer_empty_signal = np.zeros((1,))
    buffers_empty_signal = [None] * conf.no_samplers  # when only tag is important
    for i in range(conf.no_samplers):
        buffers_empty_signal[i] = np.zeros((1,))
    # request_Isend_signal = [None] * conf.no_samplers

    # PREVIOUS VERSION
    # def request_snapshot_from_sampler():
    #     # collector expects to receive snapshot from this (active) sampler later:
    #     request_irecv_snapshots[i] = comm_world.irecv(conf.max_buffer_size, source=rank, tag=TAG_DATA)
    #     # sends signal to this (active) sampler that collector is ready to receive snapshots:
    #     if request_Isend_signal[i] is not None:
    #         request_Isend_signal[i].Wait()
    #     request_Isend_signal[i] = comm_world.Isend(buffers_empty_signal[i], dest=rank, tag=TAG_READY_TO_RECEIVE)

    # if any(sampler_is_active):
    #     for rank in sampler_ranks[sampler_is_active]:
    #         i = sampler_ranks[sampler_ranks == rank][0]
    #         request_snapshot_from_sampler()
    ################################

    if ANALYZE:
        list_all_snapshots = []
        list_all_evaluators = []

    while any(sampler_is_active):  # while at least 1 sampling algorithm is active

        # SIMPLE VERSION
        while comm_world.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
            # it is expected that only samplers send messages to collector
            # while there is an incoming message from any sampler
            # the message is one of these: new snapshot, terminate signal, ready_to_receive signal
            tag = status.Get_tag()
            source = status.Get_source()
            if tag >= TAG_DATA:
                # if received message is a new snapshot, it is added to the list
                received_snapshot = comm_world.recv(source=source, tag=tag)
                list_received_snapshots = [np.vstack((list_received_snapshots[j], received_snapshot[j])) for j in range(3)]
                no_snapshots_received += len(list_received_snapshots[2])
            else:
                i = sampler_ranks[sampler_ranks == source][0]
                if tag == TAG_TERMINATE:
                    # if received message has tag_terminate, switch corresponding sampler to inactive,
                    # there will be no other incoming message from this sampler
                    sampler_is_active[i] = False
                    comm_world.Recv(buffer_empty_signal, source=source, tag=tag)  # TODO: cancel message
                elif tag == TAG_READY_TO_RECEIVE:
                    # if received message has tag_ready_to_receive,
                    # the corresponding sampler is ready to receive (updated) evaluator
                    sampler_can_recv_evaluator[i] = True
                    comm_world.Recv(buffer_empty_signal, source=source, tag=tag)  # TODO: cancel message
            ################################

        # PREVIOUS VERSION
        # for rank in sampler_ranks[sampler_is_active]:
        #     i = sampler_ranks[sampler_ranks == rank][0]
        #     # check if there are incoming snapshots from this active sampler;
        #     # if so, receive the snapshots and create new request:
        #     if request_irecv_snapshots[i].Get_status():
        #         # receive snapshots and add to list:
        #         list_received_part = request_irecv_snapshots[i].wait()
        #         if list_received_snapshots:
        #             list_received_snapshots = [np.vstack((list_received_snapshots[j], list_received_part[j])) for j in range(3)]
        #         else:
        #             list_received_snapshots = list_received_part.copy()
        #         request_snapshot_from_sampler()
        #     # check if there is an incoming signal from this active sampler
        #     # (tag_terminate | tag_ready_to_receive (updated) evaluator):
        #     status = MPI.Status()
        #     probe = comm_world.Iprobe(source=rank, tag=MPI.ANY_TAG, status=status)
        #     tag = status.Get_tag()
        #     if probe:  # there is an incomming signal from this sampler
        #         if tag == TAG_TERMINATE:
        #             # if received message has tag_terminate, switch corresponding sampler to inactive,
        #             # there will be no other incoming message from this sampler
        #             sampler_is_active[i] = False
        #             comm_world.Recv(buffer_empty_signal, source=rank, tag=TAG_TERMINATE)  # TODO: cancel message
        #         elif tag == TAG_READY_TO_RECEIVE:
        #             # if received message has tag_ready_to_receive,
        #             # the corresponding sampler is ready to receive (updated) evaluator
        #             sampler_can_recv_evaluator[i] = True
        #             comm_world.Recv(buffer_empty_signal, source=rank, tag=TAG_READY_TO_RECEIVE)  # TODO: cancel message
            ################################

        # use received snapshots to update the surorgate model, the updater instance is local:
        if no_snapshots_received > 0 and any(sampler_is_active):
            surrogate_updater.add_data(list_received_snapshots[0], list_received_snapshots[1], list_received_snapshots[2])
            if ANALYZE:
                if list_all_snapshots:
                    list_all_snapshots = [np.vstack((list_all_snapshots[j], list_received_snapshots[j])) for j in range(3)]
                else:
                    list_all_snapshots = list_received_snapshots.copy()
            list_received_snapshots = [np.empty((0, conf.no_parameters)), np.empty((0, conf.no_observations)), np.empty((0, 1))]
        cond_init = no_snapshots_current == 0 and no_snapshots_received >= conf.no_snapshots_initial  # initial surrogate model
        cond_update = no_snapshots_current > 0 and no_snapshots_received - \
            no_snapshots_current >= conf.no_snapshots_to_update  # update if at least NO_SNAPSHOTS_TO_UPDATE was added
        if (cond_init or cond_update) and any(sampler_is_active):
            evaluator_instance = surrogate_updater.get_evaluator()
            no_snapshots_current = no_snapshots_received
            if ANALYZE:
                list_all_evaluators.append(evaluator_instance)
            sampler_got_current_evaluator = np.array([False] * conf.no_samplers)
        for rank in sampler_ranks[sampler_is_active & sampler_can_recv_evaluator & ~sampler_got_current_evaluator]:
            i = sampler_ranks[sampler_ranks == rank][0]
            if request_isend_evaluator[i] is not None:
                request_isend_evaluator[i].wait()
            buffers_evaluator[i] = evaluator_instance  # TODO: copy?
            request_isend_evaluator[i] = comm_world.isend(buffers_evaluator[i], dest=rank, tag=TAG_DATA)
            sampler_can_recv_evaluator[i] = False
            sampler_got_current_evaluator[i] = True

    comm_world.Barrier()

    while comm_world.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
        tag = status.Get_tag()
        source = status.Get_source()
        print("COLLECTOR Barrier", tag, source)
        comm_world.recv(source=source, tag=tag)

    comm_world.Barrier()

    if ANALYZE:
        @dataclass
        class Output:
            list_all_snapshots: list
            list_all_evaluators: list
        output = Output(list_all_snapshots, list_all_evaluators)
        return output
    else:
        return []
