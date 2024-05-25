#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:26:55 2019

@author: simona
"""

import numpy as np
from mpi4py import MPI

from surrDAMH.configuration import Configuration
from surrDAMH.modules.communication import (CommEvaluator_collector,
                                            CommSnapshot_collector)
from surrDAMH.surrogates.parent import Updater


def run_COLLECTOR(conf: Configuration, surrogate_updater: Updater, surrogate_delayed_init_data=None):
    surrogate_updater.delayed_init(surrogate_delayed_init_data)

    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    comm_world.Split(color=2, key=rank_world)

    comms_snapshots: list[CommSnapshot_collector] = []  # communicators for receiving snapshots
    comms_evaluators: list[CommEvaluator_collector] = []  # communicators for sending evaluators
    sampler_ranks = conf.sampler_ranks
    for r in sampler_ranks:
        comms_snapshots.append(CommSnapshot_collector(r))
        comms_evaluators.append(CommEvaluator_collector(r))

    # indicates if the samplers will require evaluator update later:
    needs_evaluator = [True] * conf.no_samplers

    no_snapshots_total = 0  # how many snapshots the surrogate model updater got
    no_snapshots_used = 0  # using how many snapshots the current evaluator was created

    # related to surrogate evaluators:
    sampler_got_last_evaluator = np.array([True] * conf.no_samplers)

    # related to received snapshots
    list_new_snapshots = [np.empty((0, conf.no_parameters)), np.empty((0, conf.no_observations)), np.empty((0, 1))]

    while any(needs_evaluator):  # while at least 1 sampling algorithm still requires updates

        # receiving snapshots from samplers:
        num_new_snapshots = 0
        while True:
            # try to receive one snapshot from each sampler
            counter = 0
            for comm_s in comms_snapshots:
                if not comm_s.all_snapshots_received():
                    if comm_s.snapshot_is_available():
                        snapshot = comm_s.get_snapshot_and_request_new()
                        # TODO: likelihood
                        list_new_snapshots = [np.vstack((list_new_snapshots[j], snapshot[j])) for j in range(3)]
                        counter += 1
            num_new_snapshots += counter
            if counter == 0:  # if no snapshots received (from any sampler), break
                break
            if num_new_snapshots > conf.max_collected_snapshots_per_loop:
                break
        # add received snapshots to the surorgate model updater:
        if num_new_snapshots > 0:
            # print("++++++++++++ new:",  num_new_snapshots, flush=True)
            no_snapshots_total += num_new_snapshots
            surrogate_updater.add_data(list_new_snapshots[0], list_new_snapshots[1], list_new_snapshots[2])
            list_new_snapshots = [np.empty((0, conf.no_parameters)), np.empty((0, conf.no_observations)), np.empty((0, 1))]
        # create initial evaluator or update:
        cond_init = no_snapshots_used == 0 and no_snapshots_total >= conf.min_snapshots_initial  # initial surrogate model
        cond_update = no_snapshots_used > 0 and no_snapshots_total - no_snapshots_used >= conf.min_snapshots_to_update
        if (cond_init or cond_update):
            surrogate_updater.train()
            no_snapshots_used = no_snapshots_total
            # evaluator changed
            sampler_got_last_evaluator = [False] * conf.no_samplers
            evaluator_instance = None
        # send evaluator to samplers:
        for i in range(conf.no_samplers):
            if needs_evaluator[i]:
                comm: CommEvaluator_collector = comms_evaluators[i]
                if not sampler_got_last_evaluator[i]:
                    if comm.sampler_requests_evaluator():
                        # print("COLLECTOR: sampler requests evaluator", flush=True)
                        if evaluator_instance is None:
                            evaluator_instance = surrogate_updater.get_evaluator()
                        # add: evaluator_instance = surrogate_updater.get_evaluator()
                        comm.send_evaluator(evaluator_instance)
                        # print("COLLECTOR - evaluator sent", no_snapshots_used, no_snapshots_total)
                        sampler_got_last_evaluator[i] = True
                # print("COLLECTOR: no_snapshots used, total", no_snapshots_used, no_snapshots_total)
                if no_snapshots_used > 0:
                    if comm.sampler_stops():
                        # print("COLLECTOR: SAMPLER stopped", i, flush=True)
                        needs_evaluator[i] = False
                        if sampler_got_last_evaluator[i]:
                            comm.terminate(None)
                        else:
                            """
                            if evaluator_instance is None:
                                print("COLLECTOR DEBUG", 444, flush=True)
                                evaluator_instance = surrogate_updater.get_evaluator()
                                """
                            comm.terminate(evaluator_instance)
    idx = 0
    for comm_s in comms_snapshots:
        idx += 1
        comm_s.terminate()
    comm_world.Barrier()
    comm_world.Barrier()
