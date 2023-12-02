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
from surrDAMH.stages import Stage
import numpy as np


output_dir = sys.argv[1]
conf_dict_path = sys.argv[2]

conf = surrDAMH.Configuration(output_dir=output_dir, no_parameters=3, no_observations=2, use_surrogate=True)
conf.set_from_dict(conf_dict_path=conf_dict_path)

updater = surrDAMH.surrogates.PolynomialSklearnUpdater(conf.no_parameters, conf.no_observations, max_degree=5)
# prior = surrDAMH.priors.PriorNormal(conf.no_parameters, mean=[5.0, 3.0], cov=[[4, -2], [-2, 4]])
list_of_components = [Uniform(-2, 2), Normal(0, 2), Beta(2, 2), Uniform(3, 5), Lognormal(0, 1)]
list_of_components = list_of_components[0:conf.no_parameters]
prior = surrDAMH.priors.PriorIndependentComponents(list_of_components)

# solver_spec = surrDAMH.solver_specification.SolverSpecGeneric(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
solver_spec = surrDAMH.solver_specification.SolverSpecExample1()

# artificial observations and noise std
observations = 2.0  # surrDAMH.solvers.calculate_artificial_observations(solver_spec, [4, 4])
print("OBSERVATIONS:", observations)
noise_sd = 1.0  # np.abs(observations)*0.1  # TODO: cannot be zero
likelihood = surrDAMH.likelihoods.LikelihoodNormal(conf.no_observations, observations, sd=noise_sd)
list_of_stages = []
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd=1.0, max_evaluations=200))
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=None, max_evaluations=200, surrogate_is_updated=True))
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=None, max_evaluations=1000, surrogate_is_updated=False))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood, solver_spec=solver_spec, list_of_stages=list_of_stages)
sam.run()


comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = post.Samples(conf.no_parameters, output_dir)
    fig, axes = samples.plot_hist_grid(bins1d=40, bins2d=30)
    file_path = ensure_dir(os.path.join(output_dir, "post_processing_output", "histograms.pdf"))
    fig.savefig(file_path, bbox_inches="tight")


"""
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
"""

"""
každou stage ukládat do jiné složky
prior po skupinách parametrů
transform bere i více samplů najednou (np array) ? není třeba
sjednotit sample, parameters, atd. vs. snapshots
seřadit parametry funkcí - nutné, volitelné, =None, ...
check seed!!
imports within the package
readme
# TODO: prior, likelihood, etc. set_from_dict as another option?
# the dictionary can be also created on start and saved to yaml file
#  = everything needed to reproduce the sampling process, maybe also seed? (no)
#  - output dir and yaml path should not be in this file
# přidat možnost navázat na samplování
#  - může být na základě yaml file
#  - nutné uložit setting surrogate modelu
# taky může být možnost začít samplovat "od začátku", ale s existujícím (už naučeným) surrogate modelem
# in stage: if "use_only_surrogate", then "surrogate_is_updated" should be False (or implement this?)
# in classes_SAMPLER: is G_initial_sample used?
"""
