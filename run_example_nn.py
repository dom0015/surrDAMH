#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpirun -n 6 python3 -m mpi4py run_example_nn.py
"""

import numpy as np
from mpi4py import MPI
import surrDAMH
# import surrDAMH.post_processing as post
import os
from surrDAMH.priors.independent_components import Uniform, Beta, Lognormal, Normal
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage

# basic configuration
conf = surrDAMH.Configuration(output_dir="output_dir", no_parameters=4, no_observations=1,
                              use_collector=True, initial_sample_type="prior", transform_before_surrogate=True, save_raw_data=True,
                              num_snapshots_initial=5, min_snapshots_to_update=5)

# solver spacification
solver_spec = surrDAMH.solver_specification.SolverSpecNonlinearGeneric(no_parameters=conf.no_parameters, no_observations=conf.no_observations, sleep=1e-2)

# surrogate model updater
# updater = surrDAMH.surrogates.RBFInterpolationUpdater(conf.no_parameters, conf.no_observations, kernel="thin_plate_spline")
# updater = surrDAMH.surrogates.NNSklearnUpdater(conf.no_parameters, conf.no_observations, hidden_layer_sizes=(20, 20), activation='relu')
updater = surrDAMH.surrogates.NNSklearnOngoingUpdater(conf.no_parameters, conf.no_observations, hidden_layer_sizes=(20, 20),
                                                      activation='relu', solver='adam', learning_rate_init=1e-3, iterations_batch=100)

# prior distribution
list_of_components = [Normal(0, 2), Normal(0, 2), Uniform(-3, 3), Beta(2, 2), Lognormal(0, 1)]
list_of_components = list_of_components[0:conf.no_parameters]
prior = surrDAMH.priors.PriorIndependentComponents(list_of_components)

# observations and likelihood
observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec, np.ones(conf.no_parameters,))
noise_sd = np.abs(observations)*0.1  # cannot be zero
likelihood = surrDAMH.likelihoods.LikelihoodNormal(conf.no_observations, observations, sd=noise_sd)

# stages of sampling process
list_of_stages = []
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd=0.2, max_evaluations=100))
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=0.2, max_evaluations=100, surrogate_is_updated=True))
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=0.2, max_evaluations=500, surrogate_is_updated=False))

# run the sampling process
sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood, solver_spec=solver_spec, list_of_stages=list_of_stages)
sam.run()

# save histograms grid
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)
    fig, axes = samples.plot_hist_grid(bins1d=20, bins2d=20, stages_to_disp=[1, 2])
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")
