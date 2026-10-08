#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 2 with required number of MPI processes):
mpiexec -n 2 python3 -m mpi4py minimal_example.py

(Here all of the processes will be used as samplers.
Additional MPI process will be spawned by solvers pool.)
"""

from solver_examples.solver_spec_examples import SolverSpecExample1

import surrDAMH
from surrDAMH.proposals import RandomWalk

"""
Minimal example without surrogate model (i.e. no collector is used and samples are
generated using the basic MH algorithm).
"""

solver_spec = SolverSpecExample1()
conf = surrDAMH.Configuration(output_dir="out_minimal_example", use_collector=False, no_solvers=1)
prior = surrDAMH.distributions.PriorIndependentComponents([
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
])
# mean=[5.0] (not the bare scalar 5.0): solver is a SolverSpec here, not a Solver instance, so
# Problem cannot read no_observations off it -- the length of `mean` supplies it instead.
likelihood = surrDAMH.distributions.Normal(mean=[5.0], sd=1.0)

# sampling process stages:
list_of_stages = []
list_of_stages.append(surrDAMH.stages.Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=500))

problem = surrDAMH.Problem(prior, likelihood, solver=solver_spec)
run = problem.run_sampling(conf, list_of_stages)
