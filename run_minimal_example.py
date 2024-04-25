#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpirun -n 6 python3 -m mpi4py run_minimal_example.py
"""

import os

import scipy.stats
from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage

D = surrDAMH.distributions.Normal(mean=[1.0, -1.0], sd=0.001)
# D = surrDAMH.distributions.FromScipy(scipy.stats.multivariate_normal(mean=[0.0, 0.0], cov=0.001))  # type: ignore
conf = surrDAMH.Configuration(output_dir="minimal_example", no_parameters=2, no_observations=1, no_solvers=2,
                              pickled_observations=True, num_snapshots_initial=5, min_snapshots_to_update=10, max_sampler_isend_requests=100,
                              initial_sample_type="user_specified", initial_samples_distribution=D)
solver_spec = surrDAMH.solver_specification.SolverSpecExample1()
# updater = surrDAMH.surrogates.RBFInterpolationUpdater(conf.no_parameters, conf.no_observations)
updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
prior = surrDAMH.distributions.Normal(mean=[0.0, 0.0], sd=1.0)
# prior = scipy.stats.multivariate_normal(mean=[0.0, 0.0], cov=1.0)  # type: ignore

observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=[-2, 2])
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)
# likelihood = scipy.stats.multivariate_normal(mean=observations, cov=1.0)  # type: ignore

list_of_stages = []
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd=0.5, max_evaluations=1000, is_adaptive=False))
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd=0.5, max_evaluations=1000, is_adaptive=True))
# list_of_stages.append(Stage(algorithm_type="MH", proposal_sd=0.5, max_evaluations=500, is_adaptive=False))
# list_of_stages.append(Stage(algorithm_type="MH", proposal_sd=0.5, max_evaluations=500, is_adaptive=False))
# list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=0.5, max_evaluations=600, surrogate_is_updated=True))
# list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=0.5, max_evaluations=2000, surrogate_is_updated=False))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood, solver_spec=solver_spec, list_of_stages=list_of_stages)
sam.run()

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)
    fig, axes = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[1])
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")
