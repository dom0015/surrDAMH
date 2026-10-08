#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
python3 one_process_only.py
"""

from solver_examples.solver_examples import Solver_illustrative_local

import surrDAMH

"""
Solver is local (i.e. solvers pool is not used),
no surrogate model is used (i.e. no collector is used
and samples are generated using the basic MH algorithm).
"""

solver_instance = Solver_illustrative_local()
conf = surrDAMH.Configuration(output_dir="out_one_process_only",
                              use_collector=False, use_solvers_pool=False)
prior = surrDAMH.distributions.PriorIndependentComponents([
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
])
# dimension-free: mean=5.0 is a bare scalar, but solver_instance declares no_observations=1,
# so Problem can broadcast the likelihood to the right size.
likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=1.0)

# sampling process stages:
list_of_stages = []
list_of_stages.append(surrDAMH.stages.Stage(algorithm="MH", proposal=surrDAMH.proposals.RandomWalk(scale=0.5), max_evaluations=500))

problem = surrDAMH.Problem(prior, likelihood, solver=solver_instance)
run = problem.run_sampling(conf, list_of_stages)
