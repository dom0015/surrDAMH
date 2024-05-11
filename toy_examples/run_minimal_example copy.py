#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpirun -n 6 python3 -m mpi4py run_minimal_example.py
"""

import os

import scipy.stats
from mpi4py import MPI

import solver_examples.solver_examples
import solver_examples.solver_spec_examples
import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage
import solver_examples

dist_norm = surrDAMH.distributions.Normal(mean=[1.0, -1.0], sd=0.001)
# D = surrDAMH.distributions.FromScipy(scipy.stats.multivariate_normal(mean=[0.0, 0.0], cov=0.001))  # type: ignore
conf = surrDAMH.Configuration(output_dir="minimal_example", no_parameters=2, no_observations=1, no_solvers=2, use_collector=True, use_solvers_pool=False,
                              pickled_observations=True, min_snapshots_initial=5, min_snapshots_to_update=1, max_sampler_isend_requests=100,
                              initial_sample_type="user_specified", initial_samples_distribution=dist_norm, max_collected_snapshots_per_loop=10000)
solver_spec = solver_examples.solver_spec_examples.SolverSpecExample1()
# updater = surrDAMH.surrogates.RBFInterpolationUpdater(conf.no_parameters, conf.no_observations)  # , neighbors=1000)
updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
prior = surrDAMH.distributions.Normal(mean=[0.0, 0.0], sd=1.0)
# prior = scipy.stats.multivariate_normal(mean=[0.0, 0.0], cov=1.0)  # type: ignore

observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=[-2, 2])
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)
# likelihood = scipy.stats.multivariate_normal(mean=observations, cov=1.0)  # type: ignore

list_of_stages = []
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=500, adaptive=False))
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=500, adaptive=True))
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=500, adaptive=False))
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5, max_evaluations=500, surrogate_model_updates=True))
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=500, adaptive=True, use_only_surrogate=True))
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=500, adaptive=False))
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5, max_evaluations=500, surrogate_model_updates=True))

solver_instance = surrDAMH.solvers.get_solver_from_spec(solver_spec)
solver_spec = None

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec, solver_instance=solver_instance)
sam.run()

# post processing:
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.print_summary()

    # save histograms grid to file:
    fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[1])
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")
