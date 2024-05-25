#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 4 with required number of MPI processes):
mpiexec -n 4 python3 -m mpi4py typical_example.py

(Here, one of the processes will be used as collector,
one as solvers pool, remaining processes will be used as samplers.
Additional MPI processes will be spawned by solvers pool.)
"""

import os

import numpy as np
import scipy.stats
import solver_examples.solver_spec_examples
from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage

# solver example, takes 2 parameters, returns 1 observation:
solver_spec = solver_examples.solver_spec_examples.SolverSpecExample2()

# configuration (specification of basic settings of the sampling framework):
conf = surrDAMH.Configuration(output_dir="out_typical_example", no_parameters=2, no_observations=1, min_snapshots_initial=4)

# choice of surrogate model:
updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
# updater = surrDAMH.surrogates.RBFInterpolationUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
# updater = surrDAMH.surrogates.KDTreeUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations, no_nearest_neighbors=5)
# updater = surrDAMH.surrogates.PyTorchNNOngoingUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations, hidden_layer_sizes=(20, 10))

# Gaussian prior distribution:
prior = surrDAMH.distributions.FromScipy(scipy.stats.multivariate_normal(mean=[-1.0, 1.0], cov=np.eye(2)))

# likelihood (additive Gaussian noise):
observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=[-2, 2])
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=0.01)

# sampling process stages:
list_of_stages = []
# during MH stage, initial surrogate model is constructed:
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=400))
# during DAMH-SMU stage, surrogate model is further updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=1.0, time_limit=20, surrogate_model_updates=True))
# during DAMH stage, surrogate model is used but not updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=1.0, time_limit=20, surrogate_model_updates=False))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec)
sam.run()

# post processing:
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.get_summary()

    # save histograms grid to file:
    fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[1, 2])
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")

    # save plot of chains to file:
    fig, _ = samples.plot_chains(stages_to_disp=[2])
    file_path = os.path.join(conf.output_dir, "post_processing_output", "chains.pdf")
    fig.savefig(file_path, bbox_inches="tight")

    # save plot of chain cummulative averages to file:
    fig, _ = samples.plot_chains(average=True, stages_to_disp=[2])
    file_path = os.path.join(conf.output_dir, "post_processing_output", "averages.pdf")
    fig.savefig(file_path, bbox_inches="tight")
