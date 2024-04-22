#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from typing import List

import yaml
from mpi4py import MPI
from scipy.stats import rv_continuous

import surrDAMH.process_COLLECTOR
import surrDAMH.process_SAMPLER
import surrDAMH.process_SOLVER
from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.solver_specification import SolverSpec
from surrDAMH.stages import Stage
from surrDAMH.surrogates.parent import Evaluator, Updater

import cProfile


def profile(filename=None, comm=MPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.rank)
                pr.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator


def identity(sample):
    return sample


class SamplingFramework:
    """
    Created on each MPI rank (except spawned solvers).
    Allows user-specified surrogate model.
    """

    def __init__(self, conf: Configuration, prior: Distribution, likelihood: Distribution,
                 solver_spec: SolverSpec, list_of_stages: List[Stage],
                 surrogate_updater: Updater | None = None, surrogate_evaluator: Evaluator | None = None):
        self.conf = conf
        self.surrogate_updater = surrogate_updater
        self.surrogate_evaluator = surrogate_evaluator
        self.prior = prior
        self.likelihood = likelihood
        self.solver_spec = solver_spec
        self.list_of_stages = list_of_stages

    @profile(filename="profile_out3ad")
    def run(self):
        comm_world = MPI.COMM_WORLD
        rank_world = comm_world.Get_rank()

        if rank_world == 0:  # serialize to file
            ensure_dir(self.conf.output_dir)
            with open(os.path.join(self.conf.output_dir, "sampling_framework.pkl"), 'wb') as f:
                pickle.dump(self, f)
            with open(os.path.join(self.conf.output_dir, "sampling_framework.yaml"), 'w') as f:
                yaml.dump(self, f)

        # check if prior has the "transform" method:
        if not hasattr(self.prior, "transform"):
            self.prior.transform = identity

        """
        pickle deserialize:
        with open(os.path.join(conf.output_dir, "sampling_framework.pkl"), 'rb') as f:
            reconstructed_sam = pickle.load(f)
        """
        if rank_world == self.conf.no_samplers:
            optional_output = surrDAMH.process_SOLVER.run_SOLVER(self.conf, self.prior, self.solver_spec)
        elif rank_world == self.conf.no_samplers+1:
            assert self.surrogate_updater is not None
            optional_output = surrDAMH.process_COLLECTOR.run_COLLECTOR(self.conf, surrogate_updater=self.surrogate_updater)
        else:
            optional_output = surrDAMH.process_SAMPLER.run_SAMPLER(
                self.conf, self.prior, self.likelihood, self.list_of_stages, surrogate_evaluator=self.surrogate_evaluator)

        comm_world.Barrier()

        return optional_output
