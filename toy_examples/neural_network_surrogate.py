#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 4 with required number of MPI processes):
mpiexec -n 4 python3 -m mpi4py neural_network_surrogate.py

(Here, one process will be used as collector,
and the remaining processes will be used as samplers.)
"""

import os

import solver_examples.solver_spec_examples
from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

# solver example, takes 2 parameters, returns 1 observation:
solver_spec = solver_examples.solver_spec_examples.SolverSpecExample1(sleep_time=0.1)

# configuration (specification of basic settings of the sampling framework):
conf = surrDAMH.Configuration(output_dir="out_nn_surrogate", no_parameters=2, no_observations=1, min_snapshots_to_update=0, use_solvers_pool=False)

# Gaussian prior distribution:
prior = surrDAMH.distributions.PriorIndependentComponents([
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
])

# likelihood (additive Gaussian noise):
observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=[-2, 2])
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)

# neural network surrogate model:
updater = surrDAMH.surrogates.NeuralNetworkUpdaterBasic(no_parameters=conf.no_parameters, no_observations=conf.no_observations,
                                                      hidden_layer_sizes=(4, ), solver="adam", activation="tanh", learning_rate=1e-3,
                                                      iterations_batch=100, loss_target=1e-6, device="cpu", verbose=False)

# sampling process stages:
list_of_stages = []
# during MH stage, initial surrogate model is constructed:
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=50))
# during DAMH-SMU stage, surrogate model is further updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5, max_evaluations=50, surrogate_model_updates=True))
# during DAMH stage, surrogate model is used but not updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5, max_evaluations=50,
                      surrogate_model_updates=False, send_snapshots_to_collector=False))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec)
sam.run()

# post processing:
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.get_summary()

    # save histograms grid to file:
    fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30)
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")
