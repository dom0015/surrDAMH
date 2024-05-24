#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 4 with required number of MPI processes):
mpiexec -n 4 python3 -m mpi4py typical_example.py
"""

import os

from mpi4py import MPI

import solver_examples.solver_examples
import solver_examples.solver_spec_examples
import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage
import solver_examples

"""This solver example illustrates potential problems with posteriors with separated regions of higher probability,
see chains.pdf and observations.pdf."""

# solver example, takes 2 parameters, returns 1 observation:
solver_spec = solver_examples.solver_spec_examples.SolverSpecExample1()

# configuration (specification of basic settings of the sampling framework):
conf = surrDAMH.Configuration(output_dir="out_typical_example", no_parameters=2, no_observations=1)

# polynominal surrogate model updater:
updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)

# Gaussian prior distribution:
prior = surrDAMH.distributions.Normal(mean=[0.0, 0.0], sd=1.0)

# likelihood (additive Gaussian noise):
observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=[-2, 2])
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)

# sampling process stages:
list_of_stages = []
# during MH stage, initial surrogate model is constructed:
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=1.0, max_evaluations=500))
# during DAMH-SMU stage, surrogate model is further updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=1.0, max_evaluations=500, surrogate_model_updates=True))
# during DAMH stage, surrogate model is used but not updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=1.0, max_evaluations=500, surrogate_model_updates=False))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec)
sam.run()

# post processing:
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.print_summary()

    # save histograms grid to file:
    fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30)
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")

    # save plot of chains to file:
    fig, _ = samples.plot_chains(stages_to_disp=[1, 2])
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "chains.pdf")
    fig.savefig(file_path, bbox_inches="tight")

    # save plot of chain cummulative averages to file:
    fig, _ = samples.plot_chains(average=True, stages_to_disp=[1, 2])
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "averages.pdf")
    fig.savefig(file_path, bbox_inches="tight")
