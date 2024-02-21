#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mpi4py import MPI
import surrDAMH.process_SAMPLER
import surrDAMH.process_SOLVER
import surrDAMH.process_COLLECTOR
from surrDAMH.surrogates.parent import Updater
from surrDAMH.surrogates.parent import Evaluator
from surrDAMH.configuration import Configuration
from surrDAMH.priors.parent import Prior
from surrDAMH.likelihoods.parent import Likelihood
from surrDAMH.solver_specification import SolverSpec
from typing import List
from surrDAMH.stages import Stage
from surrDAMH.modules.tools import ensure_dir
import os
import pickle
import yaml


class SamplingFramework:
    """
    Created on each MPI rank (except spawned solvers).
    Allows user-specified surrogate model.
    """

    def __init__(self, configuration: Configuration, prior: Prior, likelihood: Likelihood,
                 solver_spec: SolverSpec, list_of_stages: List[Stage], surrogate_updater: Updater = None, surrogate_evaluator: Evaluator = None):
        self.configuration = configuration
        self.surrogate_updater = surrogate_updater
        self.surrogate_evaluator = surrogate_evaluator
        self.prior = prior
        self.likelihood = likelihood
        self.solver_spec = solver_spec
        self.list_of_stages = list_of_stages

    def run(self):
        comm_world = MPI.COMM_WORLD
        rank_world = comm_world.Get_rank()

        if rank_world == 0:  # serialize to file
            ensure_dir(self.configuration.output_dir)
            with open(os.path.join(self.configuration.output_dir, "sampling_framework.pkl"), 'wb') as f:
                pickle.dump(self, f)
            with open(os.path.join(self.configuration.output_dir, "sampling_framework.yaml"), 'w') as f:
                yaml.dump(self, f)

        """
        pickle deserialize:
        with open(os.path.join(conf.output_dir, "sampling_framework.pkl"), 'rb') as f:
            reconstructed_sam = pickle.load(f)
        """
        if rank_world == self.configuration.no_samplers:
            optional_output = surrDAMH.process_SOLVER.run_SOLVER(self.configuration, self.prior, self.solver_spec)
        elif rank_world == self.configuration.no_samplers+1:
            optional_output = surrDAMH.process_COLLECTOR.run_COLLECTOR(self.configuration, surrogate_updater=self.surrogate_updater)
        else:
            optional_output = surrDAMH.process_SAMPLER.run_SAMPLER(
                self.configuration, self.prior, self.likelihood, self.list_of_stages, surrogate_evaluator=self.surrogate_evaluator)

        comm_world.Barrier()
        print("RANK", rank_world, "terminated.")

        return optional_output
