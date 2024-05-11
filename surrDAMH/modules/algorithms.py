#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

import csv
import os
import time
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import numpy.typing as npt

from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules.communication import (CommEvaluator_sampler,
                                            CommSnapshot_sampler, Communicator)
from surrDAMH.modules.proposals import (GaussRandomWalk,
                                        GaussRandomWalk_adaptive, Proposal)
from surrDAMH.stages import Stage


@dataclass
class Sample:
    parameters: npt.NDArray
    observations: npt.NDArray | None = None  # G(parameters)
    posterior: float | None = None  # logarithm of posterior
    solver_tag: int = 0

    # copy method:
    def copy(self):
        if self.observations is None:
            new_instance = type(self)(parameters=self.parameters.copy(),
                                      posterior=self.posterior, solver_tag=self.solver_tag)
        else:
            new_instance = type(self)(parameters=self.parameters.copy(),
                                      observations=self.observations.copy(),
                                      posterior=self.posterior, solver_tag=self.solver_tag)
        return new_instance


class Algorithm_PARENT:
    def __init__(self, stage: Stage, proposal: Proposal, initial_sample: Sample, rank_world: int,
                 conf: Configuration, prior: Distribution, likelihood: Distribution,
                 commSolver: Communicator, commSnapshot: CommSnapshot_sampler | None = None,
                 commEvaluator: CommEvaluator_sampler | None = None, seed: int = 0) -> None:
        self.stage = stage
        self.proposal = proposal
        # self.current = Sample(parameters=initial_sample)
        self.current = initial_sample
        self.rank_world = rank_world
        self.conf = conf
        self.prior = prior
        self.likelihood = likelihood
        self.commSolver = commSolver
        self.commSnapshot = commSnapshot
        self.commEvaluator = commEvaluator
        self.seed = seed
        if self.commSnapshot is not None and stage.send_snapshots_to_collector:
            self.send_to_collector = self._send_to_collector
        else:
            self.send_to_collector = self._empty_function
        self.no_accepted = 0
        self.no_prerejected = 0
        self.no_rejected = 0
        self.no_rejected_current = 0
        self.proposed: Sample
        self._generator = np.random.RandomState(seed)
        self.monitor = Monitor(output_dir=self.conf.output_dir, stage=stage, basename="rank" + str(self.rank_world).zfill(4) + ".csv")
        self.prepare()

    def prepare(self) -> None:
        self.time_start = time.time()
        if self.current.observations is None:
            self.commSolver.set_parameters(self.current.parameters)
            result = self.commSolver.get_observations()
            if isinstance(result, tuple):
                self.current.observations, self.current.solver_tag = result
            else:
                self.current.observations = result
        self.current.posterior = self.calculate_log_posterior(self.current.parameters, self.current.observations, self.current.solver_tag)

    def request_observations(self) -> None:
        self.commSolver.set_parameters(self.proposed.parameters)
        result = self.commSolver.get_observations()
        if isinstance(result, tuple):
            self.proposed.observations, self.proposed.solver_tag = result
        else:
            self.proposed.observations = result
        self.proposed.posterior = self.calculate_log_posterior(self.proposed.parameters, self.proposed.observations, self.proposed.solver_tag)

    def if_accepted(self) -> None:
        self.current_sample_to_file()
        self.send_to_collector(sample=self.current, weight=self.no_rejected_current+1)
        self.no_accepted += 1
        self.no_rejected_current = 0
        # self.current = deepcopy(self.proposed)
        self.current = self.proposed.copy()
        # self.current = Sample(parameters=self.proposed.parameters.copy(), observations=self.proposed.observations.copy(),
        #                      posterior=self.proposed.posterior, solver_tag=self.proposed.solver_tag)
        self.raw_data_to_file(type="accepted", tag=self.current.solver_tag, observations=self.current.observations)

    def if_rejected(self):
        self.no_rejected += 1
        self.no_rejected_current += 1
        if not self.current.solver_tag < 0:
            self.send_to_collector(sample=self.proposed, weight=0)
        self.raw_data_to_file(type="rejected", tag=self.current.solver_tag, observations=self.proposed.observations)

    def calculate_log_posterior(self, parameters: npt.NDArray, observation: npt.NDArray, solver_tag: int = 0) -> float:
        if solver_tag < 0:
            return -np.inf
        log_likelihood = self.likelihood.logpdf(observation)
        log_prior = self.prior.logpdf(parameters)
        res = log_likelihood + log_prior
        return res

    def sample_acceptance_log(self, log_acceptance_probability):
        temp = self._generator.uniform(0.0, 1.0)
        if np.log(temp) < log_acceptance_probability:
            return True  # accepted
        else:
            return False  # rejected

    def current_sample_to_file(self):
        if self.conf.transform_before_saving:
            row: List[Any] = [1+self.no_rejected_current] + list(self.prior.transform(self.current.parameters))
        else:
            row: List[Any] = [1+self.no_rejected_current] + list(self.current.parameters)
        row.append(self.current.posterior)
        self.monitor(data_name="samples", row=row, condition=self.stage.save_to_file)

    def raw_data_to_file(self, type: str, tag, observations):
        if self.conf.save_snapshots_to_file:
            if self.conf.transform_before_saving:
                row = [type] + list(self.prior.transform(self.proposed.parameters))
            else:
                row = [type] + list(self.proposed.parameters)
            row += [tag]
            row += list(observations.flatten())
            self.monitor(data_name="raw_data", row=row)

    def _send_to_collector(self, sample, weight):
        parameters = sample.parameters.copy()
        observations = sample.observations.copy()
        if self.conf.transform_before_surrogate:
            parameters = self.prior.transform(parameters)
        assert self.commSnapshot is not None
        self.commSnapshot.send_to_collector([parameters, observations, weight])

    def _empty_function(self, **kw):
        return

    def finalize(self):
        self.current_sample_to_file()
        self.monitor(data_name="notes", row=["accepted", "rejected", "pre-rejected", "sum", "seed"], condition=self.stage.save_to_file)
        no_all = self.no_accepted + self.no_rejected + self.no_prerejected
        notes = [self.no_accepted, self.no_rejected, self.no_prerejected, no_all, self.seed]
        self.monitor(data_name="notes", row=notes, condition=self.stage.save_to_file)
        self.monitor.close_files()


class Algorithm_MH(Algorithm_PARENT):  # initiated by SAMPLERs
    def run(self):
        max_steps = min(self.stage.max_samples, self.stage.max_evaluations)
        for i in range(max_steps):
            parameters = self.proposal.propose_sample(self.current.parameters)
            self.proposed = Sample(parameters=parameters)
            self.request_observations()
            log_acceptance_probability_exact = self.proposal.get_log_acceptance_probability(self.proposed.posterior, self.current.posterior)
            acceptance_probability = min(1.0, np.exp(log_acceptance_probability_exact))
            self.proposal.adapt(proposed_sample=self.proposed.parameters, acceptance_probability=acceptance_probability)
            if self.sample_acceptance_log(log_acceptance_probability_exact):
                self.if_accepted()
            else:
                self.if_rejected()
            if time.time() - self.time_start > self.stage.time_limit:
                print("SAMPLER at rank", self.rank_world, "time limit ", self.stage.time_limit, " reached - loop", i, flush=True)
                break
        self.finalize()


class Algorithm_DAMH(Algorithm_PARENT):  # initiated by SAMPLERs
    def run(self):
        # calculate approximate observations and posterior for current (initial) sample:
        assert self.commEvaluator is not None
        self.surrogate_evaluator = self.commEvaluator.evaluator
        if self.surrogate_evaluator is None:
            self.surrogate_evaluator = self.commEvaluator.get_evaluator()
            self.commEvaluator.request_evaluator()
        observation_approx_current = self.get_approximate_observations(self.current.parameters)
        log_posterior_approx_current = self.calculate_log_posterior(self.current.parameters, observation_approx_current)

        for i in range(self.stage.max_samples):
            surrogate_evaluator_changed = False  # TODO: check this
            parameters = self.proposal.propose_sample(self.current.parameters)
            self.proposed = Sample(parameters=parameters)
            if self.stage.surrogate_model_updates:
                if self.commEvaluator.evaluator_is_available():
                    self.surrogate_evaluator = self.commEvaluator.get_evaluator()
                    self.commEvaluator.request_evaluator()
                    surrogate_evaluator_changed = True
            # if the surrogate model changed, it it necessary to recalculate
            # approximate observation for current sample
            if surrogate_evaluator_changed:
                observation_approx_current, observation_approx_proposed = self.get_approximate_observations(self.current.parameters, self.proposed.parameters)
                log_posterior_approx_current = self.calculate_log_posterior(self.current.parameters, observation_approx_current)
            else:
                observation_approx_proposed = self.get_approximate_observations(self.proposed.parameters)
            log_posterior_approx_proposed = self.calculate_log_posterior(self.proposed.parameters, observation_approx_proposed)
            log_posterior_ratio_approx = self.proposal.get_log_acceptance_probability(log_posterior_approx_proposed, log_posterior_approx_current)
            if self.sample_acceptance_log(log_posterior_ratio_approx):
                self.request_observations()
                log_acceptance_probability_exact = self.proposal.get_log_acceptance_probability(self.proposed.posterior, self.current.posterior)
                row = [i] + [self.proposed.posterior] + [log_posterior_approx_proposed]
                if self.sample_acceptance_log(log_acceptance_probability_exact - log_posterior_ratio_approx):
                    self.monitor(data_name="accepted", row=row, condition=self.stage.save_to_file)
                    self.if_accepted()
                else:
                    self.monitor(data_name="rejected", row=row, condition=self.stage.save_to_file)
                    self.if_rejected()
            else:
                self.no_prerejected += 1
                self.no_rejected_current += 1
                self.raw_data_to_file(type="prerejected", tag=0, observations=observation_approx_proposed)
            if time.time() - self.time_start > self.stage.time_limit:
                # print("SAMPLER at rank", self.rank_world, "time limit ", self.stage.time_limit, " reached - loop", i, flush=True)
                break
            if (self.no_rejected + self.no_accepted) >= self.stage.max_evaluations:
                # print("SAMPLER at RANK", self.rank_world, "evaluations limit ", self.stage.max_evaluations, " reached - loop", i, flush=True)
                break
        self.finalize()

    def get_approximate_observations(self, parameters0: npt.NDArray, parameters1: npt.NDArray | None = None):
        if self.conf.transform_before_surrogate:
            par0_tr = self.prior.transform(parameters0.copy())
            argument = [par0_tr]
            if parameters1 is not None:
                par1_tr = self.prior.transform(parameters1.copy())
                argument.append(par1_tr)
        else:
            argument = [parameters0.copy()]
            if parameters1 is not None:
                argument.append(parameters1.copy())
        res = self.surrogate_evaluator(np.array(argument))
        if parameters1 is None:
            return res
        else:
            return res[0, :], res[1, :]


class Writer:
    def __init__(self, dirname, basename):
        path = os.path.join(dirname, basename)
        os.makedirs(dirname, exist_ok=True)
        self.__file = open(path, 'w')
        self.__writer = csv.writer(self.__file)

    def writerow(self, row):
        self.__writer.writerow(row)

    def close_file(self):
        self.__file.close()


class Monitor:
    def __init__(self, output_dir, stage, basename):
        self.output_dir = output_dir
        self.stage = stage
        self.basename = basename
        self.writers = dict()

    def add_writer(self, data_name: str):
        dirname = os.path.join(self.output_dir, "sampling_output", data_name, self.stage.name)
        writer = Writer(dirname=dirname, basename=self.basename)
        self.writers[data_name] = writer

    def __call__(self, data_name, row, condition=True):
        if condition:
            if data_name not in self.writers.keys():
                self.add_writer(data_name)
            self.writers[data_name].writerow(row)
        else:
            pass

    def close_files(self):
        for writer in self.writers.values():
            writer.close_file()
