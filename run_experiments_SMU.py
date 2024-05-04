#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpirun -n 6 python3 -m mpi4py run_experiments_SMU.py
"""

import numpy as np
from mpi4py import MPI
import surrDAMH
# import surrDAMH.post_processing as post
import os
from surrDAMH.distributions.independent_components import Uniform, Beta, Lognormal, Normal
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage
import matplotlib.pyplot as plt


conf = surrDAMH.Configuration(output_dir="output_SMU_1", no_parameters=2, no_observations=1,
                              use_collector=True, initial_sample_type="prior", transform_before_surrogate=True, save_raw_data=True)

solver_spec = surrDAMH.solver_specification.SolverSpecExample1()

updater = surrDAMH.surrogates.NNSklearnUpdater(conf.no_parameters, conf.no_observations, hidden_layer_sizes=(40, 40), activation='relu')

# prior = surrDAMH.priors.PriorNormal(conf.no_parameters, mean=[5.0, 3.0], cov=[[4, -2], [-2, 4]])
list_of_components = [Normal(0, 4), Normal(0, 4), Normal(0, 4), Normal(0, 4), Uniform(-4, 8),
                      Uniform(-4, 8), Beta(2, 2), Uniform(3, 5), Lognormal(0, 1), Normal(0, 2)]
list_of_components = list_of_components[0:conf.no_parameters]
prior = surrDAMH.distributions.PriorIndependentComponents(list_of_components)
# prior = surrDAMH.priors.PriorNormal(conf.no_parameters, 0.0, 1.0)

observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec, [-2, 2])  # , -2, 2])
noise_sd = 1.0  # np.abs(observations)*0.1  # TODO: cannot be zero
likelihood = surrDAMH.likelihoods.LikelihoodNormal(conf.no_observations, observations, sd=noise_sd)

list_of_stages = []
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd=0.5, max_evaluations=25, surrogate_model_updates=True))
no_test_stages = 5
for i in range(no_test_stages):
    if i > 0:
        list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=0.5, max_evaluations=25, surrogate_model_updates=True))
    list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=0.5, max_evaluations=1000, surrogate_model_updates=False, is_excluded=True))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood, solver_spec=solver_spec, list_of_stages=list_of_stages)
data_for_analysis = sam.run()


comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)
    fig, axes = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=list(range(1, no_test_stages*2)))
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")

