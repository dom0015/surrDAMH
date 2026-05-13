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
from typing import cast
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import surrDAMH
import surrDAMH.modules
from surrDAMH.modules import proposals
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.torch_perceptron_minibatches import (
    PyTorchNNOngoingUpdater as PyTorchNNMinibatchUpdater,
)

from wrapper_grf import (
    SolverTSX_grf,
    DEFAULT_OBSERVATION_TIMES_DAYS,
    observations as ref_observations_full,
    N_KL,
    SEC_IN_DAY,
)
from tunnel_grf import piecewise_uniform_time_steps, kl_expansion_at_vertices

RUN = False
SAMPLE_TEST_DATA = False
LOAD_SURROGATE_STATE = True
SAVE_SURROGATE_STATE = True
SURROGATE_TEST_SET_SIZE = 256
SURROGATE_TEST_SET_SEED = 12346

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

# ============================================================================
# Time stepping: whole year (365 days)
# ============================================================================
time_steps = piecewise_uniform_time_steps([
    (0.5 * SEC_IN_DAY, 34),    # fine:   0–17 days
    (2.0 * SEC_IN_DAY, 174),   # coarse: 17–365 days
])
SIM_DAYS = time_steps.sum() / SEC_IN_DAY   # ~365 days

# All observation times (full year)
obs_days = DEFAULT_OBSERVATION_TIMES_DAYS       # 17, 37, ..., 357
n_obs_times = len(obs_days)                     # 18
n_sensors = 4
ref_obs = ref_observations_full                 # 72 values

no_parameters = 2 * N_KL + 5                   # 45
no_observations = n_obs_times * n_sensors       # 72
SURROGATE_OUTPUT_MEAN = np.asarray(ref_obs, dtype=np.float32).reshape(no_observations)
SURROGATE_OUTPUT_SCALE = np.full((no_observations,), 40.0, dtype=np.float32)

# ============================================================================
# Create solver instance
# ============================================================================
solver_instance = SolverTSX_grf(time_steps=time_steps)
solver_instance.observation_times_days = obs_days
solver_instance.no_observations = no_observations


# ============================================================================
# Configuration
# ============================================================================
conf = surrDAMH.Configuration(
    output_dir="out_block_prop",
    no_parameters=no_parameters,
    no_observations=no_observations,
    use_solvers_pool=False,         # solver runs locally on each sampler
    use_collector=True,             # build surrogate model
    min_snapshots_to_update=0,
    min_snapshots_initial=0,
    state_dependent_approximation=False,
    save_snapshots_to_file=True
)

# ============================================================================
# Surrogate model (neural network)
# ============================================================================
updater = PyTorchNNMinibatchUpdater(
    no_parameters=no_parameters,
    no_observations=no_observations,
    hidden_layer_sizes=(64, 64, 64),
    solver="adamw",
    activation="silu",
    learning_rate=1e-3, #3e-4,
    iterations_batch=100,
    loss_target=1e-6,
    device="cpu",
    verbose=False,
    # seed=42,
    output_mean=SURROGATE_OUTPUT_MEAN,
    output_scale=SURROGATE_OUTPUT_SCALE,
    batch_size=256,
    replay_ratio=1.0,
    replay_max_old_samples=4096,
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

# KL coefficients for K_h (20): standard normal N(0, 1)
for _ in range(N_KL):
    list_of_components.append(
        surrDAMH.distributions.independent_components.Normal(mu=0, sigma=1)
    )

# KL coefficients for porosity (20): standard normal N(0, 1)
for _ in range(N_KL):
    list_of_components.append(
        surrDAMH.distributions.independent_components.Normal(mu=0, sigma=1)
    )

# K_d inner [Pa]: N(11.0e9, 1.0e9)  — range ~8–14 GPa
list_of_components.append(
    surrDAMH.distributions.independent_components.Normal(mu=11.0e9, sigma=1.0e9)
)

# K_d outer [Pa]: N(13.5e9, 1.0e9)  — range ~10.5–16.5 GPa
list_of_components.append(
    surrDAMH.distributions.independent_components.Normal(mu=13.5e9, sigma=1.0e9)
)

# G inner [Pa]: N(14.0e9, 1.0e9)  — range ~11–17 GPa
list_of_components.append(
    surrDAMH.distributions.independent_components.Normal(mu=14.0e9, sigma=1.0e9)
)

# G outer [Pa]: N(15.5e9, 1.0e9)  — range ~12.5–18.5 GPa
list_of_components.append(
    surrDAMH.distributions.independent_components.Normal(mu=15.5e9, sigma=1.0e9)
)

# K_s [Pa]: N(47.0e9, 2.0e9)  — range ~41–53 GPa
list_of_components.append(
    surrDAMH.distributions.independent_components.Normal(mu=47.0e9, sigma=2.0e9)
)

prior = surrDAMH.distributions.PriorIndependentComponents(list_of_components)


# ============================================================================
# Likelihood (additive Gaussian noise, sigma = 20.0)
# ============================================================================
likelihood = surrDAMH.distributions.Normal(mean=ref_obs, sd=40.0) # ORIG 20.0


# ---------
# test data
# ---------
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
if rank_world == conf.rank_collector:
    if SAMPLE_TEST_DATA:
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
"""
if USE_SURROGATE_RESTART:
    list_of_stages.append(Stage(
        algorithm_type="MH",
        proposal_type="pCN",
        pcn_beta=0.1,
        time_limit=60 * 60 * 4,  # 1 hour
    ))
else:
    # Stage 1 — MH: burn-in + initial surrogate construction
    list_of_stages.append(Stage(
        algorithm_type="MH",
        proposal_type="pCN",
        pcn_beta=0.1,
        time_limit=60 * 60 * 4,  # 1 hour
    ))"""
"""list_of_stages.append(Stage(
    algorithm_type="MH",
    proposal_type="pCN",
    pcn_beta=0.2,
    time_limit=60 * 60 * 1,  # 1 hour
    surrogate_model_updates=True,
))
list_of_stages.append(Stage(
    algorithm_type="MH",
    proposal_type="pCN",
    pcn_beta=0.4,
    time_limit=60 * 60 * 1,  # 1 hour
    surrogate_model_updates=True,
))"""
"""list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="pCN",
    pcn_beta=0.1,
    time_limit=60 * 60 * 4,  # 2 hours
    subchain_max_length=10,
    surrogate_model_updates=True,
))"""
list_of_stages.append(Stage(
    algorithm_type="MH",
    proposal_type="pCN",
    pcn_beta=0.2,
    time_limit=60 * 20,  # 2 hours
    surrogate_model_updates=False,
))
# groups 0-19, 20-39, 40-44 (KL coeffs for K_h, KL coeffs for porosity, scalar params)
list_of_groups = [slice(0, 20), slice(20, 40), slice(40, 45)]
list_of_proposals = []
for _ in range(2):
    list_of_proposals.append(proposals.PCN(
        no_parameters=20,
        beta=0.1,
        prior_mean=np.zeros((20,),dtype=np.float32),
        prior_sd_or_cov=1.0,
    ))
list_of_proposals.append(proposals.Hamiltonian(
    no_parameters=5,
    step_size=0.01,
    num_steps=20,
))
list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="block",
    block_proposal_groups=list_of_groups,
    block_proposal_list=list_of_proposals,
    subchain_max_length=10,
    time_limit=60 * 60 * 7,  # 2 hours
    surrogate_model_updates=False,
))
"""list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="pCN",
    pcn_beta=0.1,
    time_limit=60 * 60 * 2,  # 2 hours
    subchain_max_length=100,
    surrogate_model_updates=False,
))
list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="Hamiltonian",
    time_limit=60 * 10,  # 2 hours
    hamiltonian_num_steps=20,
    hamiltonian_step_size=0.01,
    subchain_max_length=5,
    surrogate_model_updates=False,
))
list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="HamiltonianInfinite",
    time_limit=60 * 10,  # 2 hours
    hamiltonian_num_steps=20,
    hamiltonian_step_size=0.01,
    subchain_max_length=5,
    surrogate_model_updates=False,
))"""
"""list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="Hamiltonian",
    time_limit=60 * 30,  # 2 hours
    hamiltonian_num_steps=10,
    hamiltonian_step_size=0.05,
    subchain_max_length=1,
    send_snapshots_to_collector=False,
    surrogate_model_updates=False,
))"""
"""
# Stage 2 — DAMH-SMU: surrogate is used and updated
list_of_stages.append(Stage(
    algorithm_type="DAMH",
    proposal_type="pCN",
    pcn_beta=0.1,
    time_limit=60 * 60,  # 3 hours
    subchain_max_length=100,
    surrogate_model_updates=True,
))
"""


# ============================================================================
# Run sampling
# ============================================================================
initial_snapshots = load_surrogate_restart_state_if_available()

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

if RUN:
    sam.run()
    if rank_world == conf.rank_collector and SAVE_SURROGATE_STATE:
        if updater.no_snapshots > 0:
            updater.save_state(surrogate_checkpoint_path, surrogate_training_data_path)
            print(f"Collector saved surrogate restart state to {surrogate_state_dir}", flush=True)
        else:
            print(
                f"Collector skipped saving surrogate restart state because no snapshots were collected in {surrogate_state_dir}.",
                flush=True,
            )
else:

    # ============================================================================
    # Post-processing (rank 0 only)
    # ============================================================================
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()

    print(["DEBUG PRINT 1 rank" + str(rank_world)], flush=True)

    if rank_world == conf.rank_collector:
        updater.save_snapshots()
        if SAVE_SURROGATE_STATE:
            if updater.no_snapshots > 0:
                updater.save_state(surrogate_checkpoint_path, surrogate_training_data_path)
                print(f"Collector saved surrogate restart state to {surrogate_state_dir}", flush=True)
            else:
                print(
                    f"Collector skipped saving surrogate restart state because no snapshots were collected in {surrogate_state_dir}.",
                    flush=True,
                )

    no_stages = len(list_of_stages)

    # Display parameters: first 3 K_h KL + first 3 porosity KL + all 5 scalar params
    parameters_to_disp = (list(range(3))
                        + list(range(N_KL, N_KL + 3))
                        + list(range(2 * N_KL, 2 * N_KL + 5)))
    param_labels = (
        [f"xi_Kh_{i+1}" for i in range(N_KL)]
        + [f"xi_por_{i+1}" for i in range(N_KL)]
        + ["K_d_inner", "K_d_outer", "G_inner", "G_outer", "K_s"]
    )

    if rank_world == 0:
        out_pp = os.path.join(conf.output_dir, "post_processing_output")
        ensure_dir(out_pp)
        samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

        # Summary statistics
        samples.calculate_CpUS([[i] for i in range(no_stages)])
        summary = samples.get_summary()
        summary.to_csv(os.path.join(out_pp, "summary.csv"))

        # Skip standalone PDF generation in the lightweight post-processing
        # path. The HTML report below is the primary artifact.

        # ========================================================================
        # Optional names for the HTML report
        # ========================================================================
        par_names_full = (
            [f"xi_Kh_{i+1}" for i in range(N_KL)]
            + [f"xi_por_{i+1}" for i in range(N_KL)]
            + ["K_d_inner", "K_d_outer", "G_inner", "G_outer", "K_s"]
        )

        # ========================================================================
        # Extended HTML Report (includes prior overlays, best fits, field stats)
        # ========================================================================
        output_html = os.path.join(out_pp, "report_extended.html")
        samples.html_report_extended(
            no_observations=no_observations,
            stages_to_disp=list(range(no_stages)),
            observations=ref_obs,
            output_file=output_html,
            bins1d=20,
            bins2d=20,
            par_names=par_names_full,
            parameters_to_disp=parameters_to_disp,
            prior=prior,
            best_fits_n=10,
            obs_grid=obs_days,
            n_sensors=n_sensors,
            include_expensive_sections=True,
            field_statistics=None,
        )
        print(f"\nExtended HTML report saved to {output_html}")

    print(["DEBUG PRINT 2 rank" + str(rank_world)], flush=True)