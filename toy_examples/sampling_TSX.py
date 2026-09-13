#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 4 with required number of MPI processes):
mpirun -n 4 python3 -m mpi4py own_solver.py

(Here, one process will be used as collector,
and the remaining processes will be used as samplers.)
"""

import os

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from wrapper import SolverTSX, observations

no_subdomains = 9
no_parameters = no_subdomains*4+2  # 38
no_observations = 18*4  # 72

# create instance of own local solver
solver_instance = SolverTSX()

# configuration (specification of basic settings of the sampling framework),
# since local solver instance was specified, solvers pool cannot be used,
# the solver will be evaluated directly on Samplers:
conf = surrDAMH.Configuration(output_dir="out_tsx", no_parameters=no_parameters, no_observations=no_observations,
                              use_solvers_pool=False, use_collector=True,
                              min_snapshots_to_update=0, min_snapshots_initial=0,
                              state_dependent_approximation=False)

# NN updater:
updater = surrDAMH.surrogates.NeuralNetworkUpdaterBasic(no_parameters=conf.no_parameters, no_observations=conf.no_observations,
                                                      hidden_layer_sizes=(48, 60), solver="adam", activation="tanh", learning_rate=1e-3,
                                                      iterations_batch=100, loss_target=1e-6, device="cpu", verbose=False, seed=15)

# Gaussian prior distribution:
list_of_components = []
for i in range(no_subdomains):
    list_of_components.append(surrDAMH.distributions.independent_components.Lognormal(-40, 3))
for i in range(no_subdomains):
    list_of_components.append(surrDAMH.distributions.independent_components.Lognormal(-25, 3))
for i in range(no_subdomains):
    list_of_components.append(surrDAMH.distributions.independent_components.Lognormal(26, 2))
for i in range(no_subdomains):
    list_of_components.append(surrDAMH.distributions.independent_components.Uniform(0, 0.5))
for i in range(2):
    list_of_components.append(surrDAMH.distributions.independent_components.Lognormal(16, 2))
prior = surrDAMH.distributions.PriorIndependentComponents(list_of_components)

# likelihood (additive Gaussian noise):
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=10.0)

# sampling process stages:
list_of_stages = []
# during MH stage, initial surrogate model is constructed:
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.3, time_limit=60*60))
# during DAMH-SMU stage, surrogate model is further updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.3, time_limit=60*60*6,
                            subchain_max_length=10, surrogate_model_updates=True))
# during DAMH stage, surrogate model is used but not updated:
# list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5, max_evaluations=500, surrogate_model_updates=False))

sam = surrDAMH.SamplingFramework(conf, prior=prior, likelihood=likelihood, surrogate_updater=updater,
                                 list_of_stages=list_of_stages, solver_instance=solver_instance)
sam.run()

# post processing:
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

if rank_world == conf.rank_collector:
    updater.save_snapshots()

no_stages = len(list_of_stages)
parameters_to_disp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
if rank_world == 0:
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.get_summary()

    samples.calculate_CpUS([[i] for i in range(no_stages)])
    summary = samples.get_summary()
    summary_path = os.path.join(conf.output_dir, "post_processing_output", "summary.csv")
    summary.to_csv(summary_path)

    # save plot of chains to file:
    for i in range(no_stages):
        fig, _ = samples.plot_chains(stages_to_disp=[i], parameters_to_disp=parameters_to_disp)
        file_path = os.path.join(conf.output_dir, "post_processing_output", "chains_" + str(i) + ".pdf")
        fig.savefig(file_path, bbox_inches="tight")

        # save plot of chain cummulative averages to file:
        fig, _ = samples.plot_chains(average=True, stages_to_disp=[i], parameters_to_disp=parameters_to_disp)
        file_path = os.path.join(conf.output_dir, "post_processing_output", "averages_" + str(i) + ".pdf")
        fig.savefig(file_path, bbox_inches="tight")

        # save histograms grid to file:
        fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[i], parameters_to_disp=parameters_to_disp)
        # ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
        file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms_" + str(i) + ".pdf")
        fig.savefig(file_path, bbox_inches="tight")
