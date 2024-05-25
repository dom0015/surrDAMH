#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# import pickle
from typing import List

# import yaml
from mpi4py import MPI
# from scipy.stats import rv_continuous

import surrDAMH.process_COLLECTOR
import surrDAMH.process_SAMPLER
import surrDAMH.process_SOLVER
from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.solver_specification import SolverSpec
from surrDAMH.solvers import Solver, get_solver_from_spec
from surrDAMH.stages import Stage
from surrDAMH.surrogates.parent import Evaluator, Updater


def identity(sample):
    return sample


class SamplingFramework:
    """
    Created on each MPI rank (except spawned solvers). 
    Forward model solver (mapping from parameters fo observations) must be specified,
    if solvers pool is used, solver must be specified using solver_spec,
    if solvers pool is not used, solver can be specified using solver_spec or solver_instance.

    Args:
        conf (Configuration),
        prior (Distribution),
        likelihood (Distribution),
        solver_spec (SolverSpec),
        solver_instance (Solver): only if solvers pool is not used,
        list_of_stages (List[Stage]),
        surrogate_updater (Updater | None)
    """

    def __init__(self, conf: Configuration, prior: Distribution, likelihood: Distribution,
                 list_of_stages: List[Stage], solver_spec: SolverSpec | None = None, solver_instance: Solver | None = None,
                 surrogate_updater: Updater | None = None, surrogate_evaluator: Evaluator | None = None):
        self.conf = conf
        self.prior = prior
        self.likelihood = likelihood
        self.solver_spec = solver_spec
        self.list_of_stages = list_of_stages
        self.surrogate_updater = surrogate_updater
        self.surrogate_evaluator = surrogate_evaluator
        self.solver_instance = solver_instance

    def run(self):
        comm_world = MPI.COMM_WORLD
        rank_world = comm_world.Get_rank()

        # if rank_world == 0:  # serialize to file
        #     ensure_dir(self.conf.output_dir)
        #     with open(os.path.join(self.conf.output_dir, "sampling_framework.yaml"), 'w') as f:
        #         yaml.dump(self, f)

        # check if prior has the "transform" method:
        if not hasattr(self.prior, "transform"):
            self.prior.transform = identity

        if rank_world == self.conf.rank_solvers_pool:
            assert self.solver_spec is not None, "solver_spec must be given"
            optional_output = surrDAMH.process_SOLVER.run_SOLVER(self.conf, self.solver_spec)
        elif rank_world == self.conf.rank_collector:
            assert self.surrogate_updater is not None
            optional_output = surrDAMH.process_COLLECTOR.run_COLLECTOR(self.conf, surrogate_updater=self.surrogate_updater)
        else:
            if self.conf.use_solvers_pool is False:
                if self.solver_instance is None:
                    assert self.solver_spec is not None, "either solver_spec or solver_instance must be given"
                    solver_output_dir = ensure_dir(os.path.join(self.conf.output_dir, "solver_output", "rank{}".format(rank_world)))
                    self.solver_instance = get_solver_from_spec(self.solver_spec, solver_id=rank_world, solver_output_dir=solver_output_dir)
            else:
                self.solver_instance = None
            optional_output = surrDAMH.process_SAMPLER.run_SAMPLER(
                self.conf, self.prior, self.likelihood, self.list_of_stages, solver_instance=self.solver_instance, surrogate_evaluator=self.surrogate_evaluator)

        comm_world.Barrier()

        return optional_output
