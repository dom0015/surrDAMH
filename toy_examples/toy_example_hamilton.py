#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced workflow (pCN warm-up -> DAMH with a Hamiltonian proposal on a minibatch-trained
MLP surrogate, held-out test data, surrogate restart, extended HTML report) on the trivial
2-parameter solver. This is ``sampling_diffusion_grf.py`` with the FEniCSx forward model
replaced by a toy one, so the same workflow can be smoke-tested without dolfinx.

Run with (replace 4 with required number of MPI processes):
    mpiexec -n 4 python3 -m mpi4py toy_example_hamilton.py

(here: 3 samplers + 1 collector; the solver runs in-process on each sampler).
"""

import numpy as np
from mpi4py import MPI

import surrDAMH
import solver_examples.solver_examples
from surrDAMH.modules.surrogate_restart import SurrogateRestart
from surrDAMH.modules.test_data import TestData
from surrDAMH.stages import Stage
from surrDAMH.surrogates.torch_perceptron_minibatches import NeuralNetworkUpdaterMinibatches as Updater

# --- 1. knobs ----------------------------------------------------------------
output_dir = "out_toy_example_hamilton"
SURROGATE_RESTART_MODE = "none"   # "state" = reuse a previous run's network + snapshots,
                                  # "data" = reuse its snapshots only, "none" = start cold
SAVE_SURROGATE_STATE = True       # write the surrogate state so a later run can restart from it
SURROGATE_TEST_SET_SIZE = 128
SURROGATE_TEST_SET_SEED = 11

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

no_parameters = 2
no_observations = 1

# --- 2. forward model ---------------------------------------------------------
# takes 2 parameters, returns 1 observation:
solver_instance = solver_examples.solver_examples.Solver_illustrative_local(sleep_time=0.01)

# --- 3. prior and likelihood ---------------------------------------------------
prior = surrDAMH.distributions.PriorIndependentComponents([
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=2.0),
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=2.0),
])
observations = np.array([2.0], dtype=np.float32).reshape(no_observations)
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)

# --- 4. configuration -----------------------------------------------------------
conf = surrDAMH.Configuration(
    output_dir=output_dir,
    no_parameters=no_parameters,
    no_observations=no_observations,
    use_solvers_pool=False,      # solver runs locally on each sampler process
    min_snapshots_to_update=0,
    min_snapshots_initial=0,
)

# --- 5. surrogate model (neural network) -----------------------------------------
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
# called on every rank: rank 0 writes post_processing_output/report_extended.html and
# summary.csv, the other ranks only wait in the internal barrier.
sam.write_report(observations=observations, par_names=["par0", "par1"],
                 parameters_to_disp=list(range(conf.no_parameters)))
