#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bayesian inversion for the GRF-based TSX poroelastic model — Experiment 1.

Uses the surrDAMH framework with DAMH-SMU (Delayed Acceptance
Metropolis-Hastings with Surrogate Model Updates).

Parameter layout (45 total):
  par[ 0:20]   KL coefficients for K_h GRF     — prior: N(0, 1) each
  par[20:40]   KL coefficients for porosity GRF — prior: N(0, 1) each
  par[40]      K_d inner  [GPa]                 — prior: N(11.0e9, 1.0e9)
  par[41]      K_d outer  [GPa]                 — prior: N(13.5e9, 1.0e9)
  par[42]      G inner    [GPa]                 — prior: N(14.0e9, 1.0e9)
  par[43]      G outer    [GPa]                 — prior: N(15.5e9, 1.0e9)
  par[44]      K_s        [GPa]                 — prior: N(47.0e9, 2.0e9)

Time stepping: 0.5 d × 34  +  2.0 d × 174  =  208 steps, 365 days
Observation window: all 18 observation times (17, 37, ..., 357 days)
Observations: 18 times × 4 sensors = 72 values (Chandler 2002 field data)
Observation noise: sigma = 20.0 m water head

Run with (replace 4 with required number of MPI processes):
    mpiexec -n 4 python3 -m mpi4py sampling_TSX_grf2_pCN.py
"""

import os
import numpy as np
from mpi4py import MPI

import surrDAMH
from surrDAMH.surrogates.torch_perceptron_minibatches import (
    PyTorchNNOngoingUpdater as PyTorchNNMinibatchUpdater,
)
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from toy_examples.solver_examples import solver_examples

RUN = False
LOAD_SURROGATE_STATE = False
SAVE_SURROGATE_STATE = True
SURROGATE_TEST_SET_SIZE = 128
SURROGATE_TEST_SET_SEED = 12345
SURROGATE_OUTPUT_MEAN = np.array([2.0], dtype=np.float32)
SURROGATE_OUTPUT_SCALE = np.array([1.0], dtype=np.float32)

no_parameters = 2
no_observations = 1

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

# ============================================================================
# Create solver instance
# ============================================================================
solver_instance = solver_examples.Solver_illustrative_local(sleep_time=1e-1)

# ============================================================================
# Likelihood (additive Gaussian noise)
# ============================================================================
ref_obs = SURROGATE_OUTPUT_MEAN.copy()
likelihood = surrDAMH.distributions.Normal(mean=ref_obs, sd=1.0)

# ============================================================================
# Configuration
# ============================================================================
conf = surrDAMH.Configuration(
    output_dir="out_minibatch_training0",
    no_parameters=no_parameters,
    no_observations=no_observations,
    use_solvers_pool=False,         # solver runs locally on each sampler
    use_collector=True,             # build surrogate model
    min_snapshots_to_update=0,
    min_snapshots_initial=1,
    max_collected_snapshots_per_loop=100,
    state_dependent_approximation=False,
    save_snapshots_to_file=True
)

# ============================================================================
# Surrogate model (neural network)
# ============================================================================
updater = PyTorchNNMinibatchUpdater(
    no_parameters=no_parameters,
    no_observations=no_observations,
    hidden_layer_sizes=(16, 16, 16),
    solver="adamw",
    activation="silu",
    learning_rate=1e-3,
    iterations_batch=100,
    loss_target=1e-6,
    device="cpu",
    verbose=False,
    seed=42,
    output_mean=SURROGATE_OUTPUT_MEAN,
    output_scale=SURROGATE_OUTPUT_SCALE,
    batch_size=32,
    replay_ratio=1.0,
    replay_max_old_samples=256,
    train_on_added_data=False,
    shuffle_batches=True,
    gradient_clip_norm=10.0,
    weight_decay=1e-4,
)

surrogate_state_dir = os.path.join(conf.output_dir, "sampling_output")
surrogate_checkpoint_path = os.path.join(surrogate_state_dir, "surrogate_checkpoint.pt")
surrogate_training_data_path = os.path.join(surrogate_state_dir, "surrogate_training_data.npz")


def surrogate_restart_state_has_snapshots() -> bool:
    if not LOAD_SURROGATE_STATE:
        return False
    if not (os.path.exists(surrogate_checkpoint_path) and os.path.exists(surrogate_training_data_path)):
        return False
    try:
        with np.load(surrogate_training_data_path) as loaded:
            num_snapshots = int(loaded["num_snapshots"])
    except Exception as exc:
        if rank_world == 0:
            print(
                f"Could not inspect surrogate restart state in {surrogate_state_dir}: {exc}. Starting cold.",
                flush=True,
            )
        return False
    if num_snapshots <= 0:
        if rank_world == 0:
            print(
                f"Ignoring surrogate restart state in {surrogate_state_dir} because it contains 0 snapshots. Starting cold.",
                flush=True,
            )
        return False
    return True


USE_SURROGATE_RESTART = surrogate_restart_state_has_snapshots()


def load_surrogate_restart_state_if_available():
    if not USE_SURROGATE_RESTART or rank_world != conf.rank_collector:
        return None
    if not (os.path.exists(surrogate_checkpoint_path) and os.path.exists(surrogate_training_data_path)):
        print(
            f"Collector restart requested, but surrogate state files are missing in {surrogate_state_dir}. Starting cold.",
            flush=True,
        )
        return None
    loaded_arrays = updater.load_state(
        checkpoint_path=surrogate_checkpoint_path,
        data_path=surrogate_training_data_path,
        load_optimizer=True,
    )
    if loaded_arrays is None:
        return None
    print(
        f"Collector loaded surrogate restart state from {surrogate_state_dir} "
        f"with {loaded_arrays[0].shape[0]} snapshots.",
        flush=True,
    )
    return list(loaded_arrays)

# ============================================================================
# Prior distribution (45 independent components)
# ============================================================================
list_of_components = []
for _ in range(2):
    list_of_components.append(
        surrDAMH.distributions.independent_components.Normal(mu=0.0, sigma=2.0)
    )

prior = surrDAMH.distributions.PriorIndependentComponents(list_of_components)


def generate_surrogate_test_data(prior_distribution, solver, n_test: int, seed: int,
                                 transform_before_surrogate: bool):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    test_parameters = np.vstack([prior_distribution.rvs() for _ in range(n_test)])
    np.random.set_state(rng_state)

    if transform_before_surrogate:
        surrogate_test_parameters = np.vstack([
            prior_distribution.transform(parameters.copy())
            for parameters in test_parameters
        ])
    else:
        surrogate_test_parameters = test_parameters.copy()

    test_observations = np.zeros((n_test, no_observations))
    for i, parameters in enumerate(test_parameters):
        solver.set_parameters(prior_distribution.transform(parameters.copy()))
        test_observations[i, :] = np.asarray(solver.get_observations()).reshape(-1)
    return test_parameters, surrogate_test_parameters, test_observations


def compute_test_log_posterior(prior_distribution, likelihood_distribution,
                               internal_parameters: np.ndarray, observations: np.ndarray):
    log_posterior = np.zeros((internal_parameters.shape[0], 1), dtype=float)
    for i, (parameters, obs) in enumerate(zip(internal_parameters, observations)):
        log_posterior[i, 0] = prior_distribution.logpdf(parameters) + likelihood_distribution.logpdf(obs)
    return log_posterior


def normalized_weights_from_log_posterior(log_posterior: np.ndarray):
    shifted = log_posterior - np.max(log_posterior)
    weights = np.exp(shifted)
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.full_like(log_posterior, 1.0 / log_posterior.shape[0])
    return weights / weight_sum

surrogate_test_data = None
surrogate_test_data_path = os.path.join(conf.output_dir, "sampling_output", "surrogate_test_data.npz")
if RUN and rank_world == conf.rank_collector:
    
    ensure_dir(os.path.dirname(surrogate_test_data_path))
    test_parameters_internal, test_parameters, test_observations = generate_surrogate_test_data(
        prior_distribution=prior,
        solver=solver_instance,
        n_test=SURROGATE_TEST_SET_SIZE,
        seed=SURROGATE_TEST_SET_SEED,
        transform_before_surrogate=conf.transform_before_surrogate,
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

if USE_SURROGATE_RESTART:
    list_of_stages.append(Stage(
        algorithm_type="MH",
        proposal_type="pCN",
        pcn_beta=0.8,
        time_limit=60 * 1,
        surrogate_model_updates=True,
    ))
else:
    # Stage 1 — MH: burn-in + initial surrogate construction
    list_of_stages.append(Stage(
        algorithm_type="MH",
        proposal_type="pCN",
        pcn_beta=0.8,
        time_limit=60 * 1,  # 1 minute
        surrogate_model_updates=True,
    ))
list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="Hamiltonian",
    time_limit=60 * 1,  # 3 hours
    hamiltonian_num_steps=20,
    hamiltonian_step_size=0.1,
    subchain_max_length=20,
    surrogate_model_updates=True,
))
list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="HamiltonianInfinite",
    time_limit=60 * 1,  # 3 hours
    hamiltonian_num_steps=20,
    hamiltonian_step_size=0.1,
    subchain_max_length=20,
    surrogate_model_updates=False,
))
"""list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="Hamiltonian",
        time_limit=60 * 5,  # 3 hours
    hamiltonian_num_steps=10,
    hamiltonian_step_size=0.1,
    subchain_max_length=10,
    surrogate_model_updates=True,
))
list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="HamiltonianInfinite",
        time_limit=60 * 5,  # 3 hours
    hamiltonian_num_steps=10,
    hamiltonian_step_size=0.1,
    subchain_max_length=10,
    surrogate_model_updates=True,
))

# Stage 2 — DAMH-SMU: surrogate is used and updated
list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="pCN",
    pcn_beta=0.1,
    time_limit=60 * 60,  # 3 hours
    subchain_max_length=100,
    surrogate_model_updates=True,
))

# Stage 3 — DAMH: surrogate is used but frozen (production samples)
list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="pCN",
    pcn_beta=0.1,
    time_limit=60 * 10,  # 3 hours
    subchain_max_length=100,
    surrogate_model_updates=False,
))
"""
# ============================================================================
# Run sampling
# ============================================================================
if RUN:
    initial_snapshots = load_surrogate_restart_state_if_available()

    sam = surrDAMH.SamplingFramework(
        conf,
        prior=prior,
        likelihood=likelihood,
        surrogate_updater=updater,
        list_of_stages=list_of_stages,
        solver_instance=solver_instance,
        initial_snapshots=initial_snapshots,
        surrogate_test_data=surrogate_test_data,
    )

    sam.run()
    if rank_world == conf.rank_collector and SAVE_SURROGATE_STATE:
        if updater.no_snapshots > 0:
            updater.save_state(surrogate_checkpoint_path, surrogate_training_data_path)
            print(
                f"Collector saved surrogate restart state to {surrogate_state_dir}",
                flush=True,
            )
        else:
            print(
                f"Collector skipped saving surrogate restart state because no snapshots were collected in {surrogate_state_dir}.",
                flush=True,
            )
else:
    # ============================================================================
    # Post-processing (rank 0 only)
    # ============================================================================
    no_stages = len(list_of_stages)
    if rank_world == 0:
        out_pp = os.path.join(conf.output_dir, "post_processing_output")
        ensure_dir(out_pp)
        samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

        # Summary statistics
        samples.calculate_CpUS([[i] for i in range(no_stages)])
        summary = samples.get_summary()
        summary.to_csv(os.path.join(out_pp, "summary.csv"))

        # Chain traces and histograms
        for i in range(no_stages):
            fig, _ = samples.plot_chains(
                stages_to_disp=[i])
            fig.savefig(os.path.join(out_pp, f"chains_{i}.pdf"), bbox_inches="tight")

            fig, _ = samples.plot_chains(
                average=True, stages_to_disp=[i])
            fig.savefig(os.path.join(out_pp, f"averages_{i}.pdf"), bbox_inches="tight")
            """
            fig, _ = samples.plot_hist_grid(
                bins1d=30, bins2d=30, stages_to_disp=[i],
                prior=prior)
            fig.savefig(os.path.join(out_pp, f"histograms_{i}.pdf"), bbox_inches="tight")
            """

        print(f"\nPost-processing plots saved to {out_pp}/", flush=True)


        # ========================================================================
        # Extended HTML Report (includes prior overlays, best fits, field stats)
        # ========================================================================
        output_html = os.path.join(out_pp, "report_extended.html")
        samples.html_report_extended(
            no_observations=no_observations,
            stages_to_disp=list(range(no_stages)),
            observations=ref_obs,
            output_file=output_html,
            bins1d=40,
            bins2d=40,
            prior=prior,
            best_fits_n=0,
            include_expensive_sections=True
        )
        print(f"\nExtended HTML report saved to {output_html}", flush=True)
    else:
        print(f"Rank {rank_world}: skipping post-processing because RUN=False and only rank 0 handles it.", flush=True)