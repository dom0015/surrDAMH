#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpirun -n 6 python3 -m mpi4py run_surrDAMH.py output_dir examples/simple.json
"""

from mpi4py import MPI
import sys
import surrDAMH
import surrDAMH.post_processing as post
import os
from surrDAMH.priors.independent_components import Uniform, Beta, Lognormal, Normal
from surrDAMH.modules.tools import ensure_dir
import numpy as np


output_dir = sys.argv[1]
conf_file_path = sys.argv[2]

conf = surrDAMH.Configuration(output_dir=output_dir, use_surrogate=True)
conf.set_from_file(conf_file_path)
conf.no_parameters = 3
conf.no_observations = 2

updater = surrDAMH.surrogates.PolynomialSklearnUpdater(conf.no_parameters, conf.no_observations, max_degree=5)
# prior = surrDAMH.priors.PriorNormal(conf.no_parameters, mean=[5.0, 3.0], cov=[[4, -2], [-2, 4]])
list_of_components = [Beta(3, 5), Normal(3, 5), Uniform(3, 5)]
prior = surrDAMH.priors.PriorIndependentComponents(list_of_components)

solver_spec = surrDAMH.solver_specification.SolverSpecGeneric(no_parameters=conf.no_parameters, no_observations=conf.no_observations)

# artificial observations and noise std
observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec, [4, 4])
noise_sd = np.abs(observations)*0.1  # TODO: cannot be zero
likelihood = surrDAMH.likelihoods.LikelihoodNormal(conf.no_observations, observations, sd=noise_sd)

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood, solver_spec=solver_spec)
sam.run()


comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = post.Samples(conf.no_parameters, output_dir)
    fig, axes = samples.plot_hist_grid()
    file_path = ensure_dir(os.path.join(output_dir, "post_processing_output", "histograms.pdf"))
    fig.savefig(file_path, bbox_inches="tight")


"""
postupně přesunout data z configuration, něco asi zbyde

Prior   prior (e.g. Gaussian, i.e. prior_std, prior_mean)
int     no_parameters
(sampler)

Likel.  likelihood (e.g. additive gaussian noise, i.e. noise_std)
np      observations -> int no_observations
(sampler)

Stages:
list of Stage
    type
    max_samples
    time_limit
    proposal_std --- prefferably Proposal instance
    surrogate_is_updated (bool)
    excluded (bool)
    adaptive (bool)
    use_only_surrogate (bool)
    target_rate (float)
    sample_limit (int)
(sampler)

Surrogate:
Trainer instance
(collector)

Solver: (TODO parent) - instance vytváří až spawnované procesy
str     solver_module_path
str     solver_module_name
str     solver_init_name
dict    solver_parameters
(solver pool)

Other:
no_solvers
problem_name / output_dir

"""

"""
prior po skupinách parametrů
transform bere i více samplů najednou (np array) ? není třeba
sjednotit sample, parameters, atd. vs. snapshots
seřadit parametry funkcí - nutné, volitelné, =None, ...
check seed!!
imports within the package
"""
