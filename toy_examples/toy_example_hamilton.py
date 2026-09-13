#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 4 with required number of MPI processes):
    mpiexec -n 4 python3 -m mpi4py toy_example_hamilton.py
"""

import os
from typing import cast
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import solver_examples.solver_examples

import surrDAMH
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.torch_perceptron_minibatches import NeuralNetworkUpdaterMinibatches as Updater

from surrDAMH.modules.tools import (
    ensure_dir,
    surrogate_restart_state_has_snapshots, 
    generate_surrogate_test_data, 
    compute_test_log_posterior, 
    load_surrogate_restart_state_if_available, 
    normalized_weights_from_log_posterior,
)


output_dir = "out_toy_example_hamilton"
SAMPLE_TEST_DATA = True
LOAD_SURROGATE_STATE = False
LOAD_SURROGATE_DATA_ONLY = False
SAVE_SURROGATE_STATE = True
SURROGATE_TEST_SET_SIZE = 128
SURROGATE_TEST_SET_SEED = 11

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

no_parameters = 2
no_observations = 1

# solver example, takes 2 parameters, returns 1 observation:
solver_instance = solver_examples.solver_examples.Solver_illustrative_local(sleep_time=0.01)

# Gaussian prior distribution:
prior = surrDAMH.distributions.PriorIndependentComponents([
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=2.0),
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=2.0),
])

# reference observations:
#observations = surrDAMH.solvers.calculate_artificial_observations(solver_instance=solver_instance, parameters=[-2, 2])
observations = np.array([2.0], dtype=np.float32).reshape(no_observations)

# likelihood (additive Gaussian noise):
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)

SURROGATE_OUTPUT_MEAN = np.asarray(observations, dtype=np.float32).reshape(no_observations)
SURROGATE_OUTPUT_SCALE = np.full((no_observations,), 1.0, dtype=np.float32)


# ============================================================================
# Configuration
# ============================================================================
conf = surrDAMH.Configuration(
    output_dir=output_dir,
    no_parameters=no_parameters,
    no_observations=no_observations,
    use_solvers_pool=False,      # solver runs locally on each sampler process
    min_snapshots_to_update=0,
    min_snapshots_initial=0,
)

# ============================================================================
# Surrogate model (neural network)
# ============================================================================
updater = Updater(
    no_parameters=no_parameters,
    no_observations=no_observations,
    hidden_layer_sizes=(8, 8, 8),
    solver="adamw",
    activation="silu",
    learning_rate=1e-4,
    iterations_batch=100,
    loss_target=1e-6,
    device="cpu",
    verbose=False,
    # seed=42,
    output_mean=SURROGATE_OUTPUT_MEAN,
    output_scale=SURROGATE_OUTPUT_SCALE,
    batch_size=1024,
    replay_ratio=1.0,
    replay_max_old_samples=4096,
    train_on_added_data=False,
    shuffle_batches=True,
    gradient_clip_norm=2.0,
    weight_decay=1e-3,
)

surrogate_state_dir = os.path.join(conf.output_dir, "sampling_output")
surrogate_checkpoint_path = os.path.join(surrogate_state_dir, "surrogate_checkpoint.pt")
surrogate_training_data_path = os.path.join(surrogate_state_dir, "surrogate_training_data.npz")

USE_SURROGATE_RESTART = surrogate_restart_state_has_snapshots(LOAD_SURROGATE_STATE, LOAD_SURROGATE_DATA_ONLY, surrogate_state_dir, surrogate_checkpoint_path, surrogate_training_data_path, rank_world)


surrogate_test_data = None
surrogate_test_data_path = os.path.join(conf.output_dir, "sampling_output", "surrogate_test_data.npz")
if rank_world == conf.rank_collector:
    if SAMPLE_TEST_DATA:
        ensure_dir(os.path.dirname(surrogate_test_data_path))
        test_parameters_internal, test_parameters, test_observations = generate_surrogate_test_data(
            prior_distribution=prior,
            solver=solver_instance,
            n_test=SURROGATE_TEST_SET_SIZE,
            seed=SURROGATE_TEST_SET_SEED,
            transform_before_surrogate=conf.transform_before_surrogate,
            conf=conf
        )
        test_log_posterior = compute_test_log_posterior(
            prior_distribution=prior,
            likelihood_distribution=likelihood,
            internal_parameters=test_parameters_internal,
            observations=test_observations,
        )
        test_weights = normalized_weights_from_log_posterior(test_log_posterior)
        np.savez(
            surrogate_test_data_path,
            test_parameters=test_parameters,
            test_observations=test_observations,
            test_log_posterior=test_log_posterior,
            test_weights=test_weights,
        )
    
    with np.load(surrogate_test_data_path) as loaded_test_data:
        surrogate_test_data = (
            loaded_test_data["test_parameters"],
            loaded_test_data["test_observations"],
            loaded_test_data["test_log_posterior"],
            loaded_test_data["test_weights"],
        )
    print(
        f"Collector prepared surrogate test data at {surrogate_test_data_path} "
        f"with {surrogate_test_data[0].shape[0]} samples.",
        flush=True,
    )


# ============================================================================
# Sampling stages
# ============================================================================
list_of_stages = []
list_of_stages.append(Stage(
    algorithm_type="MH",
    proposal_type="pCN",
    pcn_beta=0.2,
    time_limit=60,  # 30 minutes
    surrogate_model_updates=True,
    is_excluded=False,
))

list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="HamiltonianInfinite",
    time_limit=60,  # 8 hours
    hamiltonian_num_steps=100,
    hamiltonian_step_size=0.05,
    subchain_max_length=1,
    # proposal_sd_or_cov=0.1, # 
    surrogate_model_updates=True,
    is_excluded=True,
))


assert updater is not None
initial_snapshots = load_surrogate_restart_state_if_available(
    updater=updater,
    use_surrogate_restart=USE_SURROGATE_RESTART,
    load_surrogate_state=LOAD_SURROGATE_STATE,
    load_surrogate_data_only=LOAD_SURROGATE_DATA_ONLY,
    rank_world=rank_world,
    conf=conf,
    surrogate_state_dir=surrogate_state_dir,
    surrogate_checkpoint_path=surrogate_checkpoint_path,
    surrogate_training_data_path=surrogate_training_data_path,
)

sam = surrDAMH.SamplingFramework(
    conf,
    prior=prior,
    likelihood=likelihood,
    surrogate_updater=updater,
    list_of_stages=list_of_stages,
    solver_instance=cast(Solver, solver_instance),
    initial_snapshots=initial_snapshots,
    surrogate_test_data=surrogate_test_data,
)

sam.run()
if rank_world == conf.rank_collector and SAVE_SURROGATE_STATE:
    if updater.no_snapshots > 0:
        surrogate_checkpoint_path_after = os.path.join(surrogate_state_dir, "surrogate_checkpoint_after.pt")
        surrogate_training_data_path_after = os.path.join(surrogate_state_dir, "surrogate_training_data_after.npz")
        updater.save_state(surrogate_checkpoint_path_after, surrogate_training_data_path_after)
        print(f"Collector saved surrogate restart state to {surrogate_state_dir}", flush=True)
    else:
        print(
            f"Collector skipped saving surrogate restart state because no snapshots were collected in {surrogate_state_dir}.",
            flush=True,
        )


def generate_html_report() -> None:
    report_stages = [stage_idx for stage_idx in range(len(list_of_stages))]
    if not report_stages:
        raise ValueError("No report stages are available.")

    # Display parameters: first 3 K_h KL + first 3 porosity KL + all 5 scalar params
    parameters_to_disp = list(range(conf.no_parameters))

    out_pp = os.path.join(conf.output_dir, "post_processing_output")
    ensure_dir(out_pp)
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # Summary statistics
    samples.calculate_CpUS([[i] for i in report_stages])
    setattr(samples, "summary", samples.summary.iloc[report_stages, :].copy())
    summary = samples.get_summary()
    summary.to_csv(os.path.join(out_pp, "summary.csv"))

    par_names_full = ["par0", "par1"]

    output_html = os.path.join(out_pp, "report_extended.html")
    samples.html_report_extended(
        no_observations=no_observations,
        stages_to_disp=report_stages,
        observations=observations,
        output_file=output_html,
        bins1d=20,
        bins2d=20,
        par_names=par_names_full,
        parameters_to_disp=parameters_to_disp,
        prior=prior,
        # Best-fit search scans raw snapshots and is too costly for this output set.
        no_best_fits=10,
        include_expensive_sections=True,
        field_statistics=None,
        configuration=conf,
    )
    print(f"\nExtended HTML report saved to {output_html}")

if rank_world == 0:
    generate_html_report()