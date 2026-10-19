#!/usr/bin/env python3
"""
Mixture proposal (C2): a small fixed random walk mixed into an adaptive one, then a plain stage. Run with:
    mpiexec -n 4 python3 -m mpi4py mixture_proposal_example.py
(4 samplers, no collector; the solver runs inside the samplers.)
"""
from solver_examples.solver_examples import Solver_illustrative_local

import surrDAMH
from surrDAMH.proposals import Mixture, RandomWalk
from surrDAMH.stages import Stage

solver = Solver_illustrative_local()                                   # declares no_parameters=2, no_observations=1
prior = surrDAMH.distributions.Normal(mean=0.0, sd=1.0)              # dimension taken from the solver
likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=1.0)         # observed data (1 value) and noise sd
problem = surrDAMH.Problem(prior, likelihood, solver=solver)

conf = surrDAMH.Configuration(output_dir="out_mixture_example", use_solvers_pool=False, use_collector=False)
stages = [
    # every iteration: with probability 0.1 a small fixed step, with 0.9 the adaptive random walk (the only adaptive component)
    Stage(proposal=Mixture([RandomWalk(scale=0.05), RandomWalk()], probabilities=[0.1, 0.9]), max_evaluations=1000, burn_in=True),
    Stage(proposal=RandomWalk(), max_evaluations=1000),               # starts from the mixture's tuned scale (carry-over)
]
run = problem.run_sampling(conf, stages)
