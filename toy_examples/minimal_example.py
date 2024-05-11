#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpirun -n 2 python3 -m mpi4py minimal_example.py
"""

from solver_examples.solver_spec_examples import SolverSpecExample1

import surrDAMH

solver_spec = SolverSpecExample1()
conf = surrDAMH.Configuration(output_dir="out_minimal_example", no_parameters=2, no_observations=1,
                              use_collector=False)
prior = surrDAMH.distributions.Normal(mean=[0.0, 0.0], sd=1.0)
likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=1.0)

# sampling process stages:
list_of_stages = []
list_of_stages.append(surrDAMH.stages.Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=500))

sam = surrDAMH.SamplingFramework(conf, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec)
sam.run()
