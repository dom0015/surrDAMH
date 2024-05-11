#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
python3 one_process_only.py
"""

from solver_examples.solver_examples import Solver_illustrative_local

import surrDAMH

solver_instance = Solver_illustrative_local()
conf = surrDAMH.Configuration(output_dir="out_one_process_only", no_parameters=2, no_observations=1,
                              use_collector=False, use_solvers_pool=False)
prior = surrDAMH.distributions.Normal(mean=[0.0, 0.0], sd=1.0)
likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=1.0)

# sampling process stages:
list_of_stages = []
list_of_stages.append(surrDAMH.stages.Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=500))

sam = surrDAMH.SamplingFramework(conf, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_instance=solver_instance)
sam.run()
