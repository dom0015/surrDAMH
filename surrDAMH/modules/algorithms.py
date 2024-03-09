#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

from typing import Any
from mpi4py import MPI
import numpy as np
import os
import csv
import time
from surrDAMH.priors.parent import Prior
from surrDAMH.likelihoods.parent import Likelihood

ANALYZE = False


class Algorithm_PARENT:
    def __init__(self, proposal, commSolver, seed=0, initial_sample=None, commSurrogate=None,
                 conf=None, stage=None, prior: Prior = None, likelihood: Likelihood = None):
        self.prior = prior
        self.stage = stage
        self.likelihood = likelihood
        self.proposal = proposal
        self.commSolver = commSolver
        self.seed = seed
        self.current_sample = initial_sample
        self.G_current_sample = None
        self.conf = conf
        if self.current_sample is None:
            self.current_sample = self.prior.mean
        self.commSurrogate = commSurrogate
        if conf.use_collector and stage.surrogate_is_updated:
            self._send_to_collector = self._send_to_collector__
        else:
            self._send_to_collector = self._empty_function
        self._generator = np.random.RandomState(seed)
        self.no_accepted = 0
        self.no_prerejected = 0
        self.no_rejected = 0
        if proposal.is_symmetric:
            self.is_accepted_sample = self.__acceptance_log_symmetric
        else:
            # TODO: not implemented
            return
        self.rank_world = MPI.COMM_WORLD.Get_rank()
        self.monitor = Monitor(output_dir=self.conf.output_dir, stage=stage, basename="rank" + str(self.rank_world).zfill(4) + ".csv")

    def prepare(self):
        self.time_start = time.time()
        if self.G_current_sample is None:
            self.commSolver.send_parameters(self.current_sample)
            self.convergence_tag, self.G_current_sample = self.commSolver.recv_observations()
        if ANALYZE:
            print(self.rank_world, "EXACT CURRENT", flush=True)
        self.log_posterior_exact_current = self.calculate_log_posterior(self.current_sample, self.G_current_sample, self.convergence_tag)
        self.no_rejected_current = 0
        self.pre_posterior_current_sample = 0

    def request_observations(self):
        self.commSolver.send_parameters(self.proposed_sample)
        self.convergence_tag, self.G_proposed_sample = self.commSolver.recv_observations()
        if ANALYZE:
            print(self.rank_world, "EXACT PROPOSED", flush=True)
        self.log_posterior_exact_proposed = self.calculate_log_posterior(self.proposed_sample, self.G_proposed_sample, self.convergence_tag)
        self.log_posterior_ratio_exact = self.log_posterior_exact_proposed - self.log_posterior_exact_current

    def if_accepted(self):
        self.current_sample_to_file()
        self._send_to_collector(sample=self.current_sample, observation=self.G_current_sample, weight=self.no_rejected_current+1)
        self.no_accepted += 1
        self.no_rejected_current = 0
        self.current_sample = self.proposed_sample
        self.G_current_sample = self.G_proposed_sample
        self.log_posterior_exact_current = self.log_posterior_exact_proposed
        self.raw_data_to_file(type="accepted", tag=self.convergence_tag, observations=self.G_current_sample)

    def if_rejected(self):
        self.no_rejected += 1
        self.no_rejected_current += 1
        if not self.convergence_tag < 0:
            self._send_to_collector(sample=self.proposed_sample, observation=self.G_proposed_sample, weight=0)
        self.raw_data_to_file(type="rejected", tag=self.convergence_tag, observations=self.G_proposed_sample)

    def finalize(self):
        self.current_sample_to_file()
        self.monitor(data_name="notes", row=["accepted", "rejected", "pre-rejected", "sum", "seed"], condition=self.stage.is_saved)
        no_all = self.no_accepted + self.no_rejected + self.no_prerejected
        notes = [self.no_accepted, self.no_rejected, self.no_prerejected, no_all, self.seed]
        self.monitor(data_name="notes", row=notes, condition=self.stage.is_saved)
        self.monitor.close_files()

    def __acceptance_log_symmetric(self, log_ratio):
        temp = self._generator.uniform(0.0, 1.0)
        temp = np.log(temp)
        self.monitor(data_name="acceptance", row=[temp, log_ratio, temp < log_ratio])
        if temp < log_ratio:  # accepted
            return True
        else:
            return False

    def current_sample_to_file(self):
        if self.conf.transform_before_saving:
            row = [1+self.no_rejected_current] + list(self.prior.transform(self.current_sample))
        else:
            row = [1+self.no_rejected_current] + list(self.current_sample)
        row.append(self.log_posterior_exact_current)
        row.append(self.pre_posterior_current_sample)
        self.monitor(data_name="samples", row=row, condition=self.stage.is_saved)

    def raw_data_to_file(self, type: str, tag, observations):
        if self.conf.save_raw_data:
            if self.conf.transform_before_saving:
                row = [type] + list(self.prior.transform(self.proposed_sample))
            else:
                row = [type] + list(self.proposed_sample)
            row.append(tag)
            row.append(observations.flatten())
            self.monitor(data_name="raw_data", row=row)

    def _send_to_collector__(self, sample, observation, weight):
        sample = sample.copy()
        observation = observation.copy()
        if self.conf.transform_before_surrogate:
            sample = self.prior.transform(sample)
        self.commSurrogate.send_to_collector([sample, observation, weight])

    def _empty_function(self, **kw):
        return

    def calculate_log_posterior(self, sample, observation, convergence_tag=0):
        if convergence_tag < 0:
            return -np.inf
        log_likelihood = self.likelihood.calculate_log_likelihood(observation)
        log_prior = self.prior.calculate_log_prior(sample)
        if ANALYZE:
            print(self.rank_world, "LOG LIKELIHOOD, LOG PRIOR:", log_likelihood, log_prior, flush=True)
        res = log_likelihood + log_prior
        return res


class Algorithm_MH(Algorithm_PARENT):  # initiated by SAMPLERs
    def run(self):
        max_steps = min(self.stage.max_samples, self.stage.max_evaluations)
        self.prepare()
        for i in range(max_steps):
            self.proposed_sample = self.proposal.propose_sample(self.current_sample)
            self.request_observations()
            if self.is_accepted_sample(self.log_posterior_ratio_exact):
                self.if_accepted()
            else:
                self.if_rejected()
            if time.time() - self.time_start > self.stage.time_limit:
                print("SAMPLER at RANK", self.rank_world, "time limit ", self.stage.time_limit, " reached - loop", i)
                break
        self.finalize()


class Algorithm_MH_adaptive(Algorithm_PARENT):  # initiated by SAMPLERs
    def run(self):
        max_steps = min(self.stage.max_samples, self.stage.max_evaluations)
        self.target_rate = self.stage.adaptive_target_rate  # target acceptance rate
        if self.target_rate is None:
            self.target_rate = 0.25
        self.corr_limit = self.stage.adaptive_corr_limit  # maximal alowed correlation of proposal distribution
        if self.corr_limit is None:
            self.corr_limit = 0.3
        self.sample_limit = self.stage.adaptive_sample_limit  # minimal number of accepted/rejected samples to evaluate acceptance rate
        if self.sample_limit is None:
            self.sample_limit = 10
        self.prepare()
        samples = np.empty((0, self.conf.no_parameters))
        fweights = np.empty((0,), dtype=int)
        samples = np.vstack((samples, self.current_sample))
        fweights = np.append(fweights, 1)
        # idx_accepted = np.empty((0,),dtype=bool)
        counter_accepted = 0
        counter_rejected = 0
        init_flag = True
        coef = 1
        # find initial proposal SD:
        if self.proposal.proposal_std.ndim == 1:
            initial_SD = self.proposal.proposal_std
        else:
            initial_SD = np.sqrt(np.diag(self.proposal.proposal_std))
        COV = initial_SD
        for i in range(max_steps):
            self.proposed_sample = self.proposal.propose_sample(self.current_sample)
            self.request_observations()
            if self.is_accepted_sample(self.log_posterior_ratio_exact):
                self.if_accepted()
                # idx_accepted = np.append(idx_accepted,True)
                fweights = np.append(fweights, 1)
                samples = np.vstack((samples, self.current_sample))
                counter_accepted += 1
            else:
                self.if_rejected()
                # idx_accepted = np.append(idx_accepted,False)
                fweights[-1] += 1
                counter_rejected += 1
            if counter_rejected >= self.sample_limit and counter_accepted >= self.sample_limit:
                current_rate = counter_accepted/(counter_accepted+counter_rejected)
                # print("ACCEPTED:", counter_accepted, "REJECTED", counter_rejected, "-> RATE", current_rate)
                COV = np.cov(samples, fweights=fweights, rowvar=False)
                SD = np.sqrt(np.diag(COV))
                CORR = COV/SD.reshape((self.conf.no_parameters, 1))
                CORR = CORR/SD.reshape((1, self.conf.no_parameters))
                # print(COV)
                # print(CORR)
                # correction of covariance matrix (maximal alowed correlation):
                CORR[CORR < -self.corr_limit] = -self.corr_limit
                CORR[CORR > self.corr_limit] = self.corr_limit
                np.fill_diagonal(CORR, 1)
                COV = CORR*SD.reshape((self.conf.no_parameters, 1))
                COV = COV*SD.reshape((1, self.conf.no_parameters))
                # print(CORR)
                if init_flag:
                    init_flag = False
                    coef = np.mean(initial_SD/SD)
                ratio = current_rate/self.target_rate
                if ratio > 1.2:  # acceptance rate is too high:
                    coef = coef*min(ratio**(2/self.conf.no_parameters), 2.0)
                    self.proposal.set_covariance(coef*COV)
                    # print("COVARIANCE CHANGED (rate too high):", ratio, self.Proposal.proposal_std)
                elif (1/ratio) > 1.2:  # acceptance rate is too low:
                    coef = coef*max(ratio**(2/self.conf.no_parameters), 0.5)
                    self.proposal.set_covariance(coef*COV)
                #     print("COVARIANCE CHANGED (rate too low):", ratio, self.Proposal.proposal_std)
                # else:
                #     print("COVARIANCE NOT CHANGED:", ratio)
                # print("RANK", MPI.COMM_WORLD.Get_rank(), "acceptance rate:", counter_accepted, "/",
                #       counter_rejected+counter_accepted, "=", np.round(current_rate, 4), "coef:", coef)
                counter_accepted = 0
                counter_rejected = 0

            if time.time() - self.time_start > self.stage.time_limit:
                print("SAMPLER at RANK", self.rank_world, "time limit ", self.stage.time_limit, " reached - loop", i)
                break
        print("RANK", self.rank_world, "FINAL COV", coef*COV)
        self.finalize()


class Algorithm_DAMH(Algorithm_PARENT):  # initiated by SAMPLERs
    def run(self):
        self.prepare()
        if self.conf.transform_before_surrogate:
            sample = self.prior.transform(self.current_sample.copy())
            self.commSurrogate.send_parameters(sample)
        else:
            self.commSurrogate.send_parameters(self.current_sample)
        tag, observation_approx_current = self.commSurrogate.recv_observations()
        self.log_posterior_approx_current = self.calculate_log_posterior(self.current_sample, observation_approx_current)
        for i in range(self.stage.max_samples):
            self.proposed_sample = self.proposal.propose_sample(self.current_sample)
            # it is necessary to recalculate GS_current_cample,
            # because the surrogate model may have changed
            if self.conf.transform_before_surrogate:
                c_sample = self.prior.transform(self.current_sample.copy())
                p_sample = self.prior.transform(self.proposed_sample.copy())
                self.commSurrogate.send_parameters(np.array([c_sample, p_sample]))
            else:
                self.commSurrogate.send_parameters(np.array([self.current_sample, self.proposed_sample]))
            tag, tmp = self.commSurrogate.recv_observations()
            observation_approx_current = tmp[0, :]
            if ANALYZE:
                print(self.rank_world, "APPROX CURRENT", flush=True)
            self.log_posterior_approx_current = self.calculate_log_posterior(self.current_sample, observation_approx_current)
            observation_approx_proposed = tmp[1, :]
            if ANALYZE:
                print(self.rank_world, "APPROX PROPOSED", flush=True)
            log_posterior_approx_proposed = self.calculate_log_posterior(self.proposed_sample, observation_approx_proposed)
            log_posterior_ratio_approx = log_posterior_approx_proposed - self.log_posterior_approx_current

            if self.is_accepted_sample(log_posterior_ratio_approx):
                self.request_observations()
                # if self.rank_world == 0:
                # print("SURROGATE", observation_approx_proposed[:10], flush=True)
                # print("EXACT", self.G_proposed_sample[:10], flush=True)
                # print("DIFF", np.mean(np.abs(observation_approx_proposed-self.G_proposed_sample)), flush=True)
                row = [log_posterior_approx_proposed, self.log_posterior_approx_current, self.log_posterior_exact_proposed, self.log_posterior_exact_current]
                self.monitor(data_name="acceptance", row=row)
                row = [i] + [self.log_posterior_exact_proposed] + [log_posterior_approx_proposed]
                if self.is_accepted_sample(self.log_posterior_ratio_exact - log_posterior_ratio_approx):
                    self.monitor(data_name="accepted", row=row, condition=self.stage.is_saved)
                    self.if_accepted()
                else:
                    self.monitor(data_name="rejected", row=row, condition=self.stage.is_saved)
                    self.if_rejected()
            else:
                self.no_prerejected += 1
                self.no_rejected_current += 1
                self.raw_data_to_file(type="prerejected", tag=0, observations=observation_approx_proposed)
            if time.time() - self.time_start > self.stage.time_limit:
                print("SAMPLER at RANK", self.rank_world, "time limit ", self.stage.time_limit, " reached - loop", i)
                break
            if (self.no_rejected + self.no_accepted) >= self.stage.max_evaluations:
                print("SAMPLER at RANK", self.rank_world, "evaluations limit ", self.stage.max_evaluations, " reached - loop", i)
                break
        self.finalize()


class Proposal_GaussRandomWalk:  # initiated by SAMPLERs
    def __init__(self, no_parameters, proposal_std=1.0, seed=0):
        self.no_parameters = no_parameters
        self._generator = np.random.RandomState(seed)
        self.set_covariance(proposal_std)
        self.is_symmetric = True
        self.is_exponential = True

    def set_covariance(self, proposal_sd=1.0):
        # prior std is scalar/vector/covariance matrix:
        if np.isscalar(proposal_sd):
            self.sd = np.full((self.no_parameters,), proposal_sd)
        else:
            self.sd = np.array(proposal_sd)
        if self.sd.ndim == 1:  # proposal - normal uncorrelated
            self.propose_sample = self._propose_sample_uncorrelated
        else:  # proposal - normal correlated
            self.propose_sample = self._propose_sample_multivariate

    def _propose_sample_uncorrelated(self, current_sample):
        sample = self._generator.normal(current_sample, self.sd)
        return sample

    def _propose_sample_multivariate(self, current_sample):
        sample = self._generator.multivariate_normal(current_sample, self.sd)
        return sample


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
