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


def _normalize_surrogate_test_data(conf: Configuration, surrogate_test_data):
    if surrogate_test_data is None:
        return None

    if len(surrogate_test_data) not in (2, 4):
        raise ValueError(
            "surrogate_test_data must contain either (parameters, observations) or "
            "(parameters, observations, log_posterior, weights)"
        )

    test_parameters, test_observations = surrogate_test_data[:2]
    test_parameters = np.asarray(test_parameters)
    test_observations = np.asarray(test_observations)

    if test_parameters.ndim == 1:
        test_parameters = test_parameters.reshape(1, -1)
    if test_observations.ndim == 1:
        test_observations = test_observations.reshape(1, -1)

    if test_parameters.ndim != 2:
        raise ValueError("surrogate_test_data parameters must have shape (n_test, no_parameters)")
    if test_observations.ndim != 2:
        raise ValueError("surrogate_test_data observations must have shape (n_test, no_observations)")
    if test_parameters.shape[0] != test_observations.shape[0]:
        raise ValueError("surrogate_test_data parameters and observations must have the same number of rows")
    if test_parameters.shape[1] != conf.no_parameters:
        raise ValueError(
            f"surrogate_test_data parameters have width {test_parameters.shape[1]}, expected {conf.no_parameters}"
        )
    if test_observations.shape[1] != conf.no_observations:
        raise ValueError(
            f"surrogate_test_data observations have width {test_observations.shape[1]}, expected {conf.no_observations}"
        )
    if test_parameters.shape[0] == 0:
        return None

    if len(surrogate_test_data) == 2:
        test_log_posterior = np.zeros((test_parameters.shape[0], 1), dtype=float)
        test_weights = np.full((test_parameters.shape[0], 1), 1.0 / test_parameters.shape[0], dtype=float)
    else:
        test_log_posterior = np.asarray(surrogate_test_data[2], dtype=float).reshape(-1, 1)
        test_weights = np.asarray(surrogate_test_data[3], dtype=float).reshape(-1, 1)

        if test_log_posterior.shape[0] != test_parameters.shape[0]:
            raise ValueError("surrogate_test_data log_posterior must match the number of test points")
        if test_weights.shape[0] != test_parameters.shape[0]:
            raise ValueError("surrogate_test_data weights must match the number of test points")

        weight_sum = float(np.sum(test_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("surrogate_test_data weights must sum to a positive finite value")
        test_weights = test_weights / weight_sum

    return test_parameters, test_observations, test_log_posterior, test_weights


def _compute_surrogate_quality_metrics(evaluator, parameters: np.ndarray, true_observations: np.ndarray,
                                       weights: np.ndarray | None = None):
    predicted_observations = np.asarray(evaluator(parameters)).reshape(true_observations.shape)
    errors = true_observations - predicted_observations
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_abs_error = float(np.max(np.abs(errors)))
    if weights is None:
        weighted_rmse = rmse
        weighted_mean_abs_error = float(np.mean(np.abs(errors)))
    else:
        normalized_weights = np.asarray(weights, dtype=float).reshape(-1, 1)
        normalized_weights = normalized_weights / np.sum(normalized_weights)
        weighted_rmse = float(np.sqrt(np.sum(normalized_weights * (errors ** 2))))
        weighted_mean_abs_error = float(np.sum(normalized_weights * np.abs(errors)))
    return rmse, max_abs_error, weighted_rmse, weighted_mean_abs_error


def run_COLLECTOR(conf: Configuration, surrogate_updater: Updater, surrogate_delayed_init_data=None,
                  initial_snapshots: list | None = None,
                  surrogate_test_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None):
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
    evaluator_instance = None

    # related to surrogate evaluators:
    sampler_got_last_evaluator = np.array([True] * conf.no_samplers)

    # related to received snapshots
    preloaded_snapshots = None
    if initial_snapshots is None:
        list_new_snapshots = [np.empty((0, conf.no_parameters)), np.empty((0, conf.no_observations)), np.empty((0, 1))]
    else:
        preloaded_snapshots = initial_snapshots
        list_new_snapshots = initial_snapshots

    if preloaded_snapshots is not None:
        no_snapshots_total = int(preloaded_snapshots[0].shape[0])
        if getattr(surrogate_updater, "training_data_loaded", False):
            list_new_snapshots = [np.empty((0, conf.no_parameters)), np.empty((0, conf.no_observations)), np.empty((0, 1))]
        if getattr(surrogate_updater, "pretrained_ready", False) and no_snapshots_total > 0:
            no_snapshots_used = no_snapshots_total
            evaluator_instance = surrogate_updater.get_evaluator()
            sampler_got_last_evaluator = [False] * conf.no_samplers

    # surrogate quality monitoring:
    monitoring_evaluator = evaluator_instance  # evaluator used for out-of-sample quality assessment
    surrogate_quality_csv_path = os.path.join(conf.output_dir, "sampling_output", "surrogate_quality.csv")
    surrogate_quality_rows = []  # accumulate rows: [no_snapshots_total, no_new_snapshots, rmse, max_abs_error]
    surrogate_test_data = _normalize_surrogate_test_data(conf, surrogate_test_data)
    surrogate_test_csv_path = os.path.join(conf.output_dir, "sampling_output", "surrogate_quality_test.csv")
    surrogate_test_rows = []  # accumulate rows with unweighted and posterior-weighted metrics
    surrogate_update_index = 1 if evaluator_instance is not None else 0

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
            surrogate_update_index += 1

            if surrogate_test_data is not None:
                try:
                    test_parameters, test_observations, test_log_posterior, test_weights = surrogate_test_data
                    rmse, max_abs_error, weighted_rmse, weighted_mean_abs_error = _compute_surrogate_quality_metrics(
                        evaluator_instance, test_parameters, test_observations, test_weights
                    )
                    surrogate_test_rows.append([
                        surrogate_update_index,
                        no_snapshots_total,
                        test_parameters.shape[0],
                        float(np.max(test_log_posterior)),
                        rmse,
                        max_abs_error,
                        weighted_rmse,
                        weighted_mean_abs_error,
                    ])
                    print(
                        f"Surrogate quality on fixed test set: RMSE={rmse:.4e}, MaxAbsErr={max_abs_error:.4e}, "
                        f"WeightedRMSE={weighted_rmse:.4e}, WeightedMAE={weighted_mean_abs_error:.4e}, "
                        f"update_index={surrogate_update_index}, snapshots_total={no_snapshots_total}, "
                        f"n_test={test_parameters.shape[0]}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"Surrogate fixed-test monitoring error: {e}", flush=True)
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

    if len(surrogate_test_rows) > 0:
        ensure_dir(os.path.dirname(surrogate_test_csv_path))
        with open(surrogate_test_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "update_index",
                "snapshots_total",
                "n_test",
                "max_log_posterior",
                "rmse",
                "max_abs_error",
                "weighted_rmse",
                "weighted_mean_abs_error",
            ])
            writer.writerows(surrogate_test_rows)
        print(f"Surrogate fixed-test metrics saved to {surrogate_test_csv_path}", flush=True)

    idx = 0
    for comm_s in comms_snapshots:
        idx += 1
        comm_s.terminate()
    comm_world.Barrier()
    comm_world.Barrier()
