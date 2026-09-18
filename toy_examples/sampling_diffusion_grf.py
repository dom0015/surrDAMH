#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reference "advanced" example.

Forward model: diffusion equation with Gaussian-random-field coefficients (grf_diffusion.py,
needs FEniCSx/dolfinx -- see docs/README.md; ``toy_example_hamilton.py`` is the same workflow
on a trivial solver and runs anywhere).
Observations: 2D field values at a grid of points.
Sampling: pCN warm-up (MH), then DAMH with a Hamiltonian proposal on a minibatch-trained MLP
surrogate; held-out test data, surrogate restart, extended HTML report with field statistics.

Run with (replace 4 with required number of MPI processes):
    mpiexec -n 4 python3 -m mpi4py sampling_diffusion_grf.py

(here: 3 samplers + 1 collector; the solver runs in-process on each sampler).
"""

from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.surrogate_restart import SurrogateRestart
from surrDAMH.modules.test_data import TestData
from surrDAMH.stages import Stage
from surrDAMH.surrogates.torch_perceptron_minibatches import NeuralNetworkUpdaterMinibatches as NNUpdater

# --- 1. knobs ----------------------------------------------------------------
output_dir = "out_diffusion_grf3"
SURROGATE_RESTART_MODE = "none"   # "state" = reuse a previous run's network + snapshots,
                                  # "data" = reuse its snapshots only, "none" = start cold
SAVE_SURROGATE_STATE = True       # write the surrogate state so a later run can restart from it
SURROGATE_TEST_SET_SIZE = 128
SURROGATE_TEST_SET_SEED = 11

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

no_parameters = 20
observations_per_dim = 2
no_observations = observations_per_dim**2

# --- 2. forward model ---------------------------------------------------------
from grf_diffusion import Solver_diffusion_GRF  # noqa: E402  (needs FEniCSx; imported after the constants above)

solver_instance = Solver_diffusion_GRF(
    xa=-1.0, xb=1.0, ya=-1.0, yb=1.0, nx=20, ny=20, solver_id=rank_world, output_dir=None,
    sleep_time=0.0, covariance_type='squared_exponential', length_scale=0.1, sigma=0.5, nu=1.0,
    no_parameters=no_parameters, observations_per_dim=observations_per_dim,
    positivity_transform_factor=1.0, u_left=1.0, u_right=0.0, source_strength=0.0)

# --- 3. prior and likelihood ---------------------------------------------------
prior = surrDAMH.distributions.PriorIndependentComponents(
    [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(no_parameters)]
)
observations = solver_instance.generate_artificial_observations(seed=11)
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=0.03)

# --- 4. configuration -----------------------------------------------------------
conf = surrDAMH.Configuration(
    output_dir=output_dir,
    no_parameters=no_parameters,
    no_observations=no_observations,
    use_solvers_pool=False,      # solver runs locally on each sampler process
    min_snapshots_to_update=0,
    min_snapshots_initial=0,
    save_snapshots_to_file=True,
)

# --- 5. surrogate model (neural network) -----------------------------------------
updater = NNUpdater(
    no_parameters=no_parameters,
    no_observations=no_observations,
    hidden_layer_sizes=(16, 16, 16),
    solver="adamw",
    activation="silu",
    learning_rate=1e-4,
    iterations_batch=100,
    loss_target=1e-6,
    device="cpu",
    verbose=False,
    batch_size=1024,
    replay_ratio=1.0,
    replay_max_old_samples=4096,
    train_on_added_data=False,
    shuffle_batches=True,
    gradient_clip_norm=2.0,
    weight_decay=1e-3,
)
# restored on the collector rank by SamplingFramework.run(), before the collector loop starts:
surrogate_restart = SurrogateRestart(state_dir=conf.output_dir + "/sampling_output",
                                     mode=SURROGATE_RESTART_MODE)

# --- 6. held-out test data for surrogate-quality monitoring -------------------------
# Only the collector needs it; it writes surrogate_quality_test.csv from it. Generated once
# and reused by every later run against the same output directory.
surrogate_test_data = None
if rank_world == conf.rank_collector:
    try:
        surrogate_test_data = TestData.reuse(conf.output_dir)
    except FileNotFoundError:
        surrogate_test_data = TestData.generate(prior, likelihood, solver_instance, conf,
                                                size=SURROGATE_TEST_SET_SIZE, seed=SURROGATE_TEST_SET_SEED)
        surrogate_test_data.save(conf.output_dir)
    # log_posterior/weights are filled in by SamplingFramework if they are still missing
    print(f"Collector - surrogate test set: {surrogate_test_data.get_size()} points.", flush=True)

# --- 7. sampling stages -------------------------------------------------------------
list_of_stages = [
    # pCN warm-up on the full model; produces the first snapshots for the surrogate.
    Stage(algorithm_type="MH", proposal_type="pCN", pcn_beta=0.2, time_limit=60,
          surrogate_model_updates=True, is_excluded=False),
    # DAMH with a gradient-based proposal on the surrogate (surrogate keeps learning).
    Stage(algorithm_type="DAMH", proposal_type="HamiltonianInfinite", time_limit=60,
          hamiltonian_num_steps=100, hamiltonian_step_size=0.05, subchain_max_length=1,
          surrogate_model_updates=True, is_excluded=True),
]

# --- 8. run ----------------------------------------------------------------------------
sam = surrDAMH.SamplingFramework(
    conf,
    prior=prior,
    likelihood=likelihood,
    surrogate_updater=updater,
    list_of_stages=list_of_stages,
    solver_instance=solver_instance,
    surrogate_test_data=surrogate_test_data,
    surrogate_restart=surrogate_restart,
)
sam.run()

if rank_world == conf.rank_collector and SAVE_SURROGATE_STATE:
    surrogate_restart.save(updater)

# --- 9. report ---------------------------------------------------------------------------
# called on every rank: rank 0 writes post_processing_output/report_extended.html (including the
# posterior field statistics, because Solver_diffusion_GRF exposes field_builder/coords/
# measurement_points) and summary.csv; the other ranks only wait in the internal barrier.
# field_statistics_max_samples: the field statistics evaluate the solver once per posterior
# state; 20 000 random states keep the report to seconds (all states: minutes per million).
sam.write_report(observations=observations, ranking_mode="likelihood", field_statistics_max_samples=20_000)
