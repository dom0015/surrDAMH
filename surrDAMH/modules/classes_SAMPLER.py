#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import os
import csv
import time
from surrDAMH.priors.parent import Prior
from surrDAMH.likelihoods.parent import Likelihood


class Algorithm_PARENT:
    def __init__(self, Proposal, Solver, seed=0, initial_sample=None, G_initial_sample=None,
                 Surrogate=None, conf=None, stage=None, prior: Prior = None, likelihood: Likelihood = None):
        self.conf = conf
        self.prior = prior
        self.likelihood = likelihood
        self.Proposal = Proposal
        self.Solver = Solver
        self.max_samples = stage.max_samples
        self.max_evaluations = stage.max_evaluations
        self.seed = seed
        self.output_dir = conf.output_dir
        self.current_sample = initial_sample
        self.is_saved = stage.is_saved
        self.name = stage.name
        self.rank_world = MPI.COMM_WORLD.Get_rank()
        self.csv_filename = "rank" + str(self.rank_world).zfill(4) + ".csv"
        if self.current_sample is None:
            self.current_sample = self.prior.mean
        self.G_current_sample = G_initial_sample
        self.Surrogate = Surrogate
        if Surrogate is None:
            self._send_to_surrogate = self._empty_function
        elif stage.surrogate_is_updated:
            self._send_to_surrogate = self._send_to_surrogate__
        else:
            self._send_to_surrogate = self._empty_function
        self.save_raw_data = conf.save_raw_data
        self.transform_before_saving = conf.transform_before_saving
        if self.transform_before_saving:
            self.transform = self.prior.transform
        self.time_limit = stage.time_limit
        self._generator = np.random.RandomState(seed)
        self.no_accepted = 0
        self.no_prerejected = 0
        self.no_rejected = 0
        if Proposal.is_symmetric:
            self.is_accepted_sample = self._acceptance_log_symmetric
            self.compute_posterior = self.calculate_log_posterior
        else:
            # not implemented
            return

    def prepare(self):
        self.time_start = time.time()
        if self.is_saved:
            # saves [no. sample posterior pre_posterior]:
            filename_G = os.path.join(self.output_dir, "sampling_output", "samples", self.name, self.csv_filename)
            os.makedirs(os.path.dirname(filename_G), exist_ok=True)
            self._file_G = open(filename_G, 'w')
            self._writer_G = csv.writer(self._file_G)
            self._write_to_file = self._write_to_file__
        else:
            self._write_to_file = self._empty_function
        if self.save_raw_data:
            filename = os.path.join(self.output_dir, "sampling_output", "raw_data", self.name, self.csv_filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self._file_raw = open(filename, 'w')
            self.writer_raw = csv.writer(self._file_raw)
            """ temporary (4 lines, monitoring log_ratio): """
            # filename = os.path.join(self.output_dir, "temp", self.Problem.name, "log_ratio", self.name + ".csv")
            # os.makedirs(os.path.dirname(filename), exist_ok=True)
            # self.__file_temp = open(filename, 'w')
            # self.writer_temp = csv.writer(self.__file_temp)
        if self.G_current_sample is None:
            self.Solver.send_parameters(self.current_sample)
            self.convergence_tag, self.G_current_sample = self.Solver.recv_observations()
        self.posterior_current_sample = self.compute_posterior(self.current_sample, self.G_current_sample, self.convergence_tag)
        self.no_rejected_current = 0
        self.pre_posterior_current_sample = 0

    def request_observations(self):
        self.Solver.send_parameters(self.proposed_sample)
        self.convergence_tag, self.G_proposed_sample = self.Solver.recv_observations()
        self.posterior_proposed_sample = self.compute_posterior(self.proposed_sample, self.G_proposed_sample, self.convergence_tag)
        self.log_ratio = self.posterior_proposed_sample - self.posterior_current_sample
        """ temporary: (2 lines) """
        # row = ["log_ratio", self.log_ratio, self.posterior_proposed_sample, self.posterior_current_sample]
        # self.writer_temp.writerow(row)

    def if_accepted(self):
        self._write_to_file()
        self._send_to_surrogate(sample=self.current_sample.copy(), G_sample=self.G_current_sample.copy(), weight=self.no_rejected_current+1)
        self.no_accepted += 1
        self.no_rejected_current = 0
        self.current_sample = self.proposed_sample
        self.G_current_sample = self.G_proposed_sample
        self.posterior_current_sample = self.posterior_proposed_sample
        if self.save_raw_data:
            if self.transform_before_saving:
                row = ['accepted'] + list(self.transform(self.proposed_sample)) + [self.convergence_tag] + list(self.G_current_sample.flatten())
            else:
                row = ['accepted'] + list(self.proposed_sample) + [self.convergence_tag] + list(self.G_current_sample.flatten())
            self.writer_raw.writerow(row)

    def if_rejected(self):
        self.no_rejected += 1
        self.no_rejected_current += 1
        if self.convergence_tag > 0:
            self._send_to_surrogate(sample=self.proposed_sample.copy(), G_sample=self.G_proposed_sample.copy(), weight=0)
        if self.save_raw_data:
            if self.transform_before_saving:
                row = ['rejected'] + list(self.transform(self.proposed_sample)) + [self.convergence_tag] + list(self.G_proposed_sample.flatten())
            else:
                row = ['rejected'] + list(self.proposed_sample) + [self.convergence_tag] + list(self.G_proposed_sample.flatten())
            self.writer_raw.writerow(row)

    def close_files(self):
        self._write_to_file()
        if self.is_saved:
            self._file_G.close()
            filename_notes = os.path.join(self.output_dir, "sampling_output", "notes", self.name, self.csv_filename)
            os.makedirs(os.path.dirname(filename_notes), exist_ok=True)
            labels = ["accepted", "rejected", "pre-rejected", "sum", "seed"]
            no_all = self.no_accepted + self.no_rejected + self.no_prerejected
            notes = [self.no_accepted, self.no_rejected, self.no_prerejected, no_all, self.seed]
            file_notes = open(filename_notes, 'w')
            writer_notes = csv.writer(file_notes)
            writer_notes.writerow(labels)
            writer_notes.writerow(notes)
            file_notes.close()
        if self.save_raw_data:
            self._file_raw.close()
            """ temporary: """
            # self.__file_temp.close()

    def _acceptance_log_symmetric(self, log_ratio):
        temp = self._generator.uniform(0.0, 1.0)
        temp = np.log(temp)
        if temp < log_ratio:  # accepted
            return True
        else:
            return False

    def _write_to_file__(self):
        if self.transform_before_saving:
            row = [1+self.no_rejected_current] + list(self.transform(self.current_sample))
        else:
            row = [1+self.no_rejected_current] + list(self.current_sample)
        row.append(self.posterior_current_sample)
        row.append(self.pre_posterior_current_sample)
        self._writer_G.writerow(row)

    def _send_to_surrogate__(self, sample, G_sample, weight):
        # snapshot = Snapshot(sample=sample, G_sample=G_sample, weight=weight)
        self.Surrogate.send_to_data_collector([sample, G_sample, weight])

    def _empty_function(self, **kw):
        return

    def calculate_log_posterior(self, sample, G_sample, convergence_tag=0):
        if convergence_tag < 0:
            return -np.inf
        return self.likelihood.calculate_log_likelihood(G_sample) + self.prior.calculate_log_prior(sample)


class Algorithm_MH(Algorithm_PARENT):  # initiated by SAMPLERs
    def __init__(self, Proposal, Solver, seed=0, initial_sample=None, G_initial_sample=None,
                 Surrogate=None, conf=None, stage=None, prior=None, likelihood=None):
        super().__init__(Proposal, Solver, seed, initial_sample, G_initial_sample,
                         Surrogate, conf, stage, prior, likelihood)
        self.max_samples = min(self.max_samples, self.max_evaluations)

    def run(self):
        self.prepare()
        for i in range(self.max_samples):
            self.proposed_sample = self.Proposal.propose_sample(self.current_sample)
            self.request_observations()
            if self.is_accepted_sample(self.log_ratio):
                self.if_accepted()
            else:
                self.if_rejected()
            if time.time() - self.time_start > self.time_limit:
                print("SAMPLER at RANK", self.rank_world, "time limit ", self.time_limit, " reached - loop", i)
                break
        self.close_files()


class Algorithm_MH_adaptive(Algorithm_PARENT):  # initiated by SAMPLERs
    def __init__(self, Proposal, Solver, seed=0, initial_sample=None, G_initial_sample=None,
                 Surrogate=None, conf=None, stage=None, prior=None, likelihood=None):
        super().__init__(Proposal, Solver, seed, initial_sample, G_initial_sample,
                         Surrogate, conf, stage, prior, likelihood)
        self.max_samples = min(self.max_samples, self.max_evaluations)
        self.target_rate = stage.adaptive_target_rate  # target acceptance rate
        if self.target_rate is None:
            self.target_rate = 0.25
        self.corr_limit = stage.adaptive_corr_limit  # maximal alowed correlation of proposal distribution
        if self.corr_limit is None:
            self.corr_limit = 0.3
        self.sample_limit = stage.adaptive_sample_limit  # minimal number of accepted/rejected samples to evaluate acceptance rate
        if self.sample_limit is None:
            self.sample_limit = 10

    def run(self):
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
        if self.Proposal.proposal_std.ndim == 1:
            initial_SD = self.Proposal.proposal_std
        else:
            initial_SD = np.sqrt(np.diag(self.Proposal.proposal_std))
        COV = initial_SD
        for i in range(self.max_samples):
            self.proposed_sample = self.Proposal.propose_sample(self.current_sample)
            self.request_observations()
            if self.is_accepted_sample(self.log_ratio):
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
                    self.Proposal.set_covariance(coef*COV)
                    # print("COVARIANCE CHANGED (rate too high):", ratio, self.Proposal.proposal_std)
                elif (1/ratio) > 1.2:  # acceptance rate is too low:
                    coef = coef*max(ratio**(2/self.conf.no_parameters), 0.5)
                    self.Proposal.set_covariance(coef*COV)
                #     print("COVARIANCE CHANGED (rate too low):", ratio, self.Proposal.proposal_std)
                # else:
                #     print("COVARIANCE NOT CHANGED:", ratio)
                # print("RANK", MPI.COMM_WORLD.Get_rank(), "acceptance rate:", counter_accepted, "/",
                #       counter_rejected+counter_accepted, "=", np.round(current_rate, 4), "coef:", coef)
                counter_accepted = 0
                counter_rejected = 0

            if time.time() - self.time_start > self.time_limit:
                print("SAMPLER at RANK", self.rank_world, "time limit ", self.time_limit, " reached - loop", i)
                break
        print("RANK", self.rank_world, "FINAL COV", coef*COV)
        self.close_files()


class Algorithm_DAMH(Algorithm_PARENT):  # initiated by SAMPLERs
    def __init__(self, Proposal, Solver, seed=0, initial_sample=None, G_initial_sample=None,
                 Surrogate=None, conf=None, stage=None, prior=None, likelihood=None):
        super().__init__(Proposal, Solver, seed, initial_sample, G_initial_sample,
                         Surrogate, conf, stage, prior, likelihood)

    def run(self):
        self.prepare()
        if self.is_saved:
            # posterior (vs approximated posterior) in rejected samples:
            filename = os.path.join(self.output_dir, "sampling_output", "DAMH_rejected", self.name, self.csv_filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.__file_rejected = open(filename, 'w')
            self.__writer_rejected = csv.writer(self.__file_rejected)
            # posterior (vs approximated posterior) in accepted samples:
            filename = os.path.join(self.output_dir, "sampling_output", "DAMH_accepted", self.name, self.csv_filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.__file_accepted = open(filename, 'w')
            self.__writer_accepted = csv.writer(self.__file_accepted)
        self.Surrogate.send_parameters(self.current_sample)
        tag, GS_current_sample = self.Surrogate.recv_observations()
        self.pre_posterior_current_sample = self.compute_posterior(self.current_sample, GS_current_sample)
        for i in range(self.max_samples):
            self.proposed_sample = self.Proposal.propose_sample(self.current_sample)
            # it is necessary to recalculate GS_current_cample,
            # because the surrogate model may have changed
            self.Surrogate.send_parameters(np.array([self.current_sample, self.proposed_sample]))
            tag, tmp = self.Surrogate.recv_observations()
            GS_current_sample = tmp[0, :]
            # TO DO: do not recalculate posterior if GS_current_sample did not change
            self.pre_posterior_current_sample = self.compute_posterior(self.current_sample, GS_current_sample)
            GS_proposed_sample = tmp[1, :]

            """ temporary_ (5 lines) """
            # self.Solver.send_parameters(self.proposed_sample)
            # tag, G_proposed_sample = self.Solver.recv_observations()
            # v = (GS_proposed_sample-G_proposed_sample)/G_proposed_sample
            # if np.abs(v)>1:
            #     print("!!!!!!!! TEMP !!!!!!!!", GS_proposed_sample, G_proposed_sample, v)

            pre_posterior_proposed_sample = self.compute_posterior(self.proposed_sample, GS_proposed_sample)
            pre_log_ratio = pre_posterior_proposed_sample - self.pre_posterior_current_sample
            """ temporary: (2 lines) """
            # row = ["pre_log_ratio", pre_log_ratio, pre_posterior_proposed_sample, self.pre_posterior_current_sample]
            # self.writer_temp.writerow(row)

            if self.is_accepted_sample(pre_log_ratio):
                self.request_observations()
                if self.is_accepted_sample(self.log_ratio - pre_log_ratio):
                    if self.is_saved:
                        row = [i]
                        row.append(self.posterior_proposed_sample)
                        row.append(pre_posterior_proposed_sample)
                        self.__writer_accepted.writerow(row)
                    self.if_accepted()
                else:
                    if self.is_saved:
                        row = [i]
                        row.append(self.posterior_proposed_sample)
                        row.append(pre_posterior_proposed_sample)
                        self.__writer_rejected.writerow(row)
                    self.if_rejected()
            else:
                self.no_prerejected += 1
                self.no_rejected_current += 1
                if self.save_raw_data:
                    if self.transform_before_saving:
                        row = ['prerejected'] + list(self.transform(self.proposed_sample)) + [0] + list(GS_proposed_sample.flatten())
                    else:
                        row = ['prerejected'] + list(self.proposed_sample) + [0] + list(GS_proposed_sample.flatten())
                    self.writer_raw.writerow(row)
            if time.time() - self.time_start > self.time_limit:
                print("SAMPLER at RANK", self.rank_world, "time limit ", self.time_limit, " reached - loop", i)
                break
            if (self.no_rejected + self.no_accepted) >= self.max_evaluations:
                print("SAMPLER at RANK", self.rank_world, "evaluations limit ", self.max_evaluations, " reached - loop", i)
                break
        self.close_files()
        self.__file_rejected.close()
        self.__file_accepted.close()


class Proposal_GaussRandomWalk:  # initiated by SAMPLERs
    def __init__(self, no_parameters, proposal_std=1.0, seed=0):
        self.no_parameters = no_parameters
        self.__generator = np.random.RandomState(seed)
        self.set_covariance(proposal_std)
        self.is_symmetric = True
        self.is_exponential = True

    def set_covariance(self, proposal_sd=1.0):
        # prior std is scalar/vector/covariance matrix:
        if np.isscalar(proposal_sd):
            self.proposal_std = np.full((self.no_parameters,), proposal_sd)
        else:
            self.proposal_std = np.array(proposal_sd)
        if self.proposal_std.ndim == 1:  # proposal - normal uncorrelated
            self.propose_sample = self._propose_sample_uncorrelated
        else:  # proposal - normal correlated
            self.propose_sample = self.__propose_sample_multivariate

    def _propose_sample_uncorrelated(self, current_sample):
        sample = self.__sample_uncorrelated(self.__generator, current_sample, self.proposal_std)
        return sample

    def __propose_sample_multivariate(self, current_sample):
        sample = self.__sample_multivariate(self.__generator, current_sample, self.proposal_std)
        return sample

    def __sample_uncorrelated(self, generator, var_mean, var_std):
        sample = generator.normal(var_mean, var_std)
        return sample

    def __sample_multivariate(self, generator, var_mean, var_cov):
        sample = generator.multivariate_normal(var_mean, var_cov)
        return sample


# class Problem_Gauss:  # initiated by SAMPLERs
#     def __init__(self, no_parameters, prior_mean=0.0, prior_std=1.0, noise_std=1.0, no_observations=None, observations=None):
#         self.no_parameters = no_parameters
#         if np.isscalar(prior_mean):
#             self.prior_mean = np.full((no_parameters,), prior_mean)
#         else:
#             self.prior_mean = np.array(prior_mean)

#         # prior std is scalar/vector/covariance matrix:
#         if np.isscalar(prior_std):
#             self.prior_std = np.full((no_parameters,), prior_std)
#         else:
#             self.prior_std = np.array(prior_std)
#         # if self.prior_std.ndim == 1:  # prior - normal uncorrelated
#         #     self.get_log_prior = self._get_log_prior_uncorrelated
#         # else:  # prior - normal correlated
#         #     self.get_log_prior = self.__get_log_prior_multivariate

#         self.observations = observations
#         if no_observations is None:
#             no_observations = len(observations)
#         self.no_observations = no_observations
#         self.noise_mean = np.zeros((no_observations,))

#         # # noise std is scalar/vector/covariance matrix:
#         # if np.isscalar(noise_std):is_adaptive=True))
#         #     self.noise_std = np.full((no_observations,), noise_std)
#         # else:
#         #     self.noise_std = np.array(noise_std)
#         # if self.noise_std.ndim == 1:  # noise - normal uncorrelated
#         #     self.get_log_likelihood = self._get_log_likelihood_uncorrelated
#         # else:  # noise - normal correlated
#         #     self.get_log_likelihood = self.__get_log_likelihood_multivariate

#         self.is_exponential = True
#         # self.__generator = np.random.RandomState(seed)

#     # def _get_log_likelihood_uncorrelated(self, G_sample):
#     #     v = self.observations - G_sample
#     #     invCv = v/(self.noise_std**2)
#     #     return -0.5*np.sum(v*invCv)

#     # def __get_log_likelihood_multivariate(self, G_sample):
#     #     v = self.observations - G_sample.ravel()
#     #     invCv = np.linalg.solve(self.noise_std, v)
#     #     return -0.5*np.dot(v, invCv)

#     # def _get_log_prior_uncorrelated(self, sample):
#     #     v = sample - self.prior_mean
#     #     invCv = v/(self.prior_std**2)
#     #     return -0.5*np.dot(v, invCv)

#     # def __get_log_prior_multivariate(self, sample):
#     #     v = sample - self.prior_mean
#     #     invCv = np.linalg.solve(self.prior_std, v)
#     #     return -0.5*np.dot(v, invCv)

#     # def get_log_posterior(self, sample, G_sample, convergence_tag=0):
#     #     if convergence_tag < 0:
#     #         return -np.inf
#     #     return self.get_log_likelihood(G_sample) + self.get_log_prior(sample)

# class Snapshot:
#     def __init__(self, sample=None, G_sample=None, weight=None):
#         self.sample = sample
#         self.G_sample = G_sample
#         self.weight = weight

#     def print(self):
#         print("W:", self.weight, "S:", self.sample, "G:", self.G_sample)
