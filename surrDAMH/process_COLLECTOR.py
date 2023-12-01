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

# communicates with: SAMPLERs


def run_COLLECTOR(conf: Configuration, surrogate_updater: Updater = None, surrogate_delayed_init_data=None):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    comm_world.Split(color=2, key=rank_world)

    max_buffer_size = conf.max_buffer_size
    samplers_ranks = np.arange(conf.no_samplers)  # TODO: assumes ranks 0,1,...
    sampler_is_active = np.array([True] * conf.no_samplers)
    send_buffers = [None] * conf.no_samplers
    sampler_can_recv = np.array([False] * conf.no_samplers)  # ready to receive updated data
    sampler_got_last_surrogate = np.array([True] * conf.no_samplers)
    tag_terminate = 0
    tag_ready_to_receive = 1
    tag_sent_data = 2
    list_received = []
    empty_buffers = [None] * conf.no_samplers
    for i in range(conf.no_samplers):
        empty_buffers[i] = np.zeros((1,))
    request_irecv = [None] * conf.no_samplers
    request_isend = [None] * conf.no_samplers
    request_Isend = [None] * conf.no_samplers
    surrogate_updater.delayed_init(surrogate_delayed_init_data)
    if any(sampler_is_active):
        for r in samplers_ranks[sampler_is_active]:
            # for i in np.nditer(np.where(sampler_is_active)):
            i = samplers_ranks[samplers_ranks == r][0]
            # expects to receive data from this (active) sampler later:
            request_irecv[i] = comm_world.irecv(max_buffer_size, source=r, tag=tag_sent_data)
            # sends signal to this (active) sampler that he is ready to receive data:
            request_Isend[i] = comm_world.Isend(empty_buffers[i], dest=r, tag=tag_ready_to_receive)
    status = MPI.Status()
    while any(sampler_is_active):  # while at least 1 sampling algorithm is active
        # if all(sampler_got_last_surrogate|(~sampler_is_active)): # block until a new incoming message
        #     comm_world.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        # checks all active samplers for an incoming message:
        for r in samplers_ranks[sampler_is_active]:
            i = samplers_ranks[samplers_ranks == r][0]
            status = MPI.Status()
            probe = comm_world.Iprobe(source=r, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if probe:
                if conf.debug:
                    print("debug - RANK", rank_world, "COLLECTOR Iprobe True", tag, i, r, samplers_ranks, sampler_is_active)
                if tag == tag_terminate:
                    # if received message has tag_terminate, switch corresponding sampler to inactive
                    # assumes that there will be no other incoming message from that source
                    sampler_is_active[i] = False
                    tmp = np.empty((1,))  # TO DO: useless pointer / cancel message
                    comm_world.Recv(tmp, source=r, tag=tag_terminate)
                    if conf.debug:
                        print("debug - RANK", rank_world, "COLLECTOR Recv terminate", tag, r)
                elif tag == tag_ready_to_receive:
                    # if received message has tag_ready_to_receive, the corresponding
                    # sampler is ready to receive updated data
                    sampler_can_recv[i] = True
                    tmp = np.empty((1,))  # TO DO: useless pointer / cancel message
                    comm_world.Recv(tmp, source=r, tag=tag_ready_to_receive)
                    if conf.debug:
                        print("debug - RANK", rank_world, "COLLECTOR Recv ready_to_receive", tag, r)
            elif conf.debug:
                print("debug - RANK", rank_world, "COLLECTOR IProbe False", r)
        for r in samplers_ranks[sampler_is_active]:
            # for i in np.nditer(np.where(sampler_is_active)):
            i = samplers_ranks[samplers_ranks == r][0]
            # checks if there are incoming data from this active sampler:
            if request_irecv[i].Get_status():
                list_received_part = request_irecv[i].wait()
                # expects to receive data from this active sampler later:
                request_irecv[i] = comm_world.irecv(max_buffer_size, source=r, tag=tag_sent_data)
                # sends signal to this active sampler that he is ready to receive data:
                if request_Isend[i] is not None:
                    request_Isend[i].Wait()
                request_Isend[i] = comm_world.Isend(empty_buffers[i], dest=r, tag=tag_ready_to_receive)
                # TODO: CHECK 5 LINES BELOW
                # list_received_data.extend(received_data.copy())
                if list_received:
                    list_received = [np.vstack((list_received[j], list_received_part[j])) for j in range(3)]
                else:
                    list_received = list_received_part.copy()

        if list_received and any(sampler_is_active):
            surrogate_updater.add_data(list_received[0], list_received[1], list_received[2])
            evaluator_instance = surrogate_updater.get_evaluator()
            list_received = []
            sampler_got_last_surrogate = np.array([False] * conf.no_samplers)
        for r in samplers_ranks[sampler_is_active & sampler_can_recv & ~sampler_got_last_surrogate]:
            # for i in np.nditer(np.where(sampler_is_active & sampler_can_recv & ~sampler_got_last_surrogate)):
            i = samplers_ranks[samplers_ranks == r][0]
            if request_isend[i] is not None:
                request_isend[i].wait()
            send_buffers[i] = evaluator_instance  # .copy()  # TODO: copy?
            request_isend[i] = comm_world.isend(send_buffers[i], dest=r, tag=tag_sent_data)
            sampler_can_recv[i] = False
            sampler_got_last_surrogate[i] = True

    print("RANK", rank_world, "- all collected snapshots:")  # , len(local_updater_instance.processed_par))

    comm_world.Barrier()
    print("RANK", rank_world, "(DATA COLLECTOR) terminated.")
