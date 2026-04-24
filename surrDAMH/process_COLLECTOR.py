#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:26:55 2019

@author: simona
"""

import csv
import os

import numpy as np
from mpi4py import MPI

from surrDAMH.configuration import Configuration
from surrDAMH.modules.communication import (CommEvaluator_collector,
                                            CommSnapshot_collector)
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.surrogates.parent import Updater


def run_COLLECTOR(conf: Configuration, surrogate_updater: Updater, surrogate_delayed_init_data=None,
                  initial_snapshots: list | None = None):
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
    if initial_snapshots is None:
        list_new_snapshots = [np.empty((0, conf.no_parameters)), np.empty((0, conf.no_observations)), np.empty((0, 1))]
    else:
        list_new_snapshots = initial_snapshots

    # surrogate quality monitoring:
    monitoring_evaluator = None  # evaluator used for out-of-sample quality assessment
    surrogate_quality_csv_path = os.path.join(conf.output_dir, "sampling_output", "surrogate_quality.csv")
    surrogate_quality_rows = []  # accumulate rows: [no_snapshots_total, no_new_snapshots, rmse, max_abs_error]

    while any(needs_evaluator):  # while at least 1 sampling algorithm still requires updates

        # receiving snapshots from samplers:
        num_new_snapshots = list_new_snapshots[0].shape[0]
        while True:
            # try to receive one snapshot from each sampler
            counter = 0
            for comm_s in comms_snapshots:
                if not comm_s.all_snapshots_received():
                    if comm_s.snapshot_is_available():
                        snapshot = comm_s.get_snapshot_and_request_new()
                        list_new_snapshots = [np.vstack((list_new_snapshots[j], snapshot[j])) for j in range(3)]
                        counter += 1
            num_new_snapshots += counter
            if counter == 0:  # if no snapshots received (from any sampler), break
                break
            if num_new_snapshots > conf.max_collected_snapshots_per_loop:
                break

        # surrogate quality monitoring: evaluate new snapshots with current evaluator
        # before adding them to the training set (out-of-sample assessment)
        if num_new_snapshots > 0 and monitoring_evaluator is not None:
            try:
                new_parameters = list_new_snapshots[0]
                true_observations = list_new_snapshots[1]
                predicted_observations = monitoring_evaluator(new_parameters)
                predicted_observations = predicted_observations.reshape(true_observations.shape)
                errors = true_observations - predicted_observations
                rmse = float(np.sqrt(np.mean(errors ** 2)))
                max_abs_error = float(np.max(np.abs(errors)))
                surrogate_quality_rows.append([no_snapshots_total, num_new_snapshots, rmse, max_abs_error])
                print(f"Surrogate quality (out-of-sample): RMSE={rmse:.4e}, MaxAbsErr={max_abs_error:.4e}, "
                      f"snapshots_total={no_snapshots_total}, batch_size={num_new_snapshots}", flush=True)
            except Exception as e:
                print(f"Surrogate quality monitoring error: {e}", flush=True)

        # add received snapshots to the surorgate model updater:
        if num_new_snapshots > 0:
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
            evaluator_instance = surrogate_updater.get_evaluator()
            monitoring_evaluator = evaluator_instance  # update monitoring evaluator
        # send evaluator to samplers:
        for i in range(conf.no_samplers):
            if needs_evaluator[i]:
                comm: CommEvaluator_collector = comms_evaluators[i]
                if not sampler_got_last_evaluator[i]:
                    if comm.sampler_requests_evaluator():
                        comm.send_evaluator(evaluator_instance)
                        sampler_got_last_evaluator[i] = True
                if no_snapshots_used > 0:
                    if comm.sampler_stops():
                        needs_evaluator[i] = False
                        if sampler_got_last_evaluator[i]:
                            comm.terminate(None)
                        else:
                            comm.terminate(evaluator_instance)

    # save surrogate quality metrics to CSV:
    if len(surrogate_quality_rows) > 0:
        ensure_dir(os.path.dirname(surrogate_quality_csv_path))
        with open(surrogate_quality_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["snapshots_total", "batch_size", "rmse", "max_abs_error"])
            writer.writerows(surrogate_quality_rows)
        print(f"Surrogate quality metrics saved to {surrogate_quality_csv_path}", flush=True)

    idx = 0
    for comm_s in comms_snapshots:
        idx += 1
        comm_s.terminate()
    comm_world.Barrier()
    comm_world.Barrier()
