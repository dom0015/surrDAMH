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

class Algorithm_PARENT:
    def __init__(self, Problem, Proposal, Solver, max_samples, name, seed=0, initial_sample=None, G_initial_sample=None, Surrogate=None, is_saved=True, save_raw_data=False, surrogate_is_updated=True, time_limit=float('inf')):
        self.Problem = Problem
        self.Proposal = Proposal
        self.Solver = Solver
        self.max_samples = max_samples
        self.name = name
        self.seed = seed
        self.current_sample = initial_sample
        if self.current_sample is None:
            self.current_sample = self.Problem.prior_mean.copy()
        self.G_current_sample = G_initial_sample
        self.Surrogate = Surrogate
        if Surrogate is None:
            self.__send_to_surrogate = self._empty_function
        elif surrogate_is_updated is True:
            self.__send_to_surrogate = self.__send_to_surrogate__
        else:
            self.__send_to_surrogate = self._empty_function
        self.is_saved = is_saved
        self.save_raw_data = save_raw_data
        self.time_limit = time_limit
        self.__generator = np.random.RandomState(seed)
        self.no_accepted = 0
        self.no_prerejected = 0
        self.no_rejected = 0
        if Problem.is_exponential and Proposal.is_exponential:
            if Proposal.is_symmetric:
                self.is_accepted_sample = self._acceptance_log_symmetric
                self.compute_posterior = self.Problem.get_log_posterior
            else:
                # not implemented
                return
        else:
            # not implemented
            return
                
    def prepare(self):
        self.time_start = time.time()
        if self.is_saved:
            # saves [no. sample posterior pre_posterior]:
            filename_G = "saved_samples/" + self.Problem.name + "/data/" + self.name + ".csv"
            os.makedirs(os.path.dirname(filename_G), exist_ok=True)
            self.__file_G = open(filename_G, 'w')
            self.__writer_G = csv.writer(self.__file_G)
            self.__write_to_file = self.__write_to_file__
        else:
            self.__write_to_file = self._empty_function
        if self.save_raw_data:
            filename = "saved_samples/" + self.Problem.name + "/raw_data/" + self.name + ".csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.__file_raw = open(filename, 'w')
            self.writer_raw = csv.writer(self.__file_raw)
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
        
    def if_accepted(self):
        self.__write_to_file()
        self.__send_to_surrogate(sample=self.current_sample.copy(), G_sample=self.G_current_sample.copy(), weight=self.no_rejected_current+1)    
        self.no_accepted += 1
        self.no_rejected_current = 0    
        self.current_sample = self.proposed_sample
        self.G_current_sample = self.G_proposed_sample
        self.posterior_current_sample = self.posterior_proposed_sample
        if self.save_raw_data:
            row = ['accepted'] + list(self.proposed_sample) + [self.convergence_tag] + list(self.G_current_sample)
            self.writer_raw.writerow(row)
        
    def if_rejected(self):
        self.no_rejected += 1
        self.no_rejected_current += 1
        if self.convergence_tag>0:
            self.__send_to_surrogate(sample=self.proposed_sample.copy(), G_sample=self.G_proposed_sample.copy(), weight=0)
        if self.save_raw_data:
            row = ['rejected'] + list(self.proposed_sample) + [self.convergence_tag] + list(self.G_proposed_sample)
            self.writer_raw.writerow(row)
        
    def close_files(self):
        self.__write_to_file()
        if self.is_saved:
            self.__file_G.close()
            filename_notes = "saved_samples/" + self.Problem.name + "/notes/" + self.name + ".csv"
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
            self.__file_raw.close()

    def _acceptance_log_symmetric(self,log_ratio):
        temp = self.__generator.uniform(0.0,1.0)
        temp = np.log(temp)
        if temp<log_ratio: # accepted
            return True
        else:
            return False
        
    def __write_to_file__(self):
        row = [1+self.no_rejected_current] + list(self.current_sample)
        row.append(self.posterior_current_sample)
        row.append(self.pre_posterior_current_sample)
        self.__writer_G.writerow(row)
    
    def __send_to_surrogate__(self, sample, G_sample, weight):
        snapshot = Snapshot(sample=sample, G_sample=G_sample, weight=weight)
        self.Surrogate.send_to_data_collector(snapshot)
        
    def _empty_function(self,**kw):
        return
    
class Algorithm_MH(Algorithm_PARENT): # initiated by SAMPLERs
    def __init__(self, Problem, Proposal, Solver, max_samples, name, seed=0, initial_sample=None, G_initial_sample=None, Surrogate=None, is_saved=True, save_raw_data=False, surrogate_is_updated=True, time_limit=float('inf')):
        super().__init__(Problem, Proposal, Solver, max_samples, name, seed, initial_sample, G_initial_sample, Surrogate, is_saved, save_raw_data, surrogate_is_updated, time_limit)

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
                print("SAMPLER at RANK", MPI.COMM_WORLD.Get_rank(), "time limit reached - loop",i)
                break
        self.close_files()
        
class Algorithm_DAMH(Algorithm_PARENT): # initiated by SAMPLERs
    def __init__(self, Problem, Proposal, Solver, max_samples, name, seed=0, initial_sample=None, G_initial_sample=None, Surrogate=None, is_saved=True, save_raw_data=False, surrogate_is_updated=True, time_limit=float('inf')):
        super().__init__(Problem, Proposal, Solver, max_samples, name, seed, initial_sample, G_initial_sample, Surrogate, is_saved, save_raw_data, surrogate_is_updated, time_limit)

    def run(self):
        self.prepare()
        if self.is_saved:
            # posterior (vs approximated posterior) in rejected samples:
            filename = "saved_samples/" + self.Problem.name + "/DAMH_rejected/" + self.name + ".csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.__file_rejected = open(filename, 'w')
            self.__writer_rejected = csv.writer(self.__file_rejected)
            # posterior (vs approximated posterior) in accepted samples:
            filename = "saved_samples/" + self.Problem.name + "/DAMH_accepted/" + self.name + ".csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.__file_accepted = open(filename, 'w')
            self.__writer_accepted = csv.writer(self.__file_accepted)
        self.Surrogate.send_parameters(self.current_sample)
        GS_current_sample = self.Surrogate.recv_observations()
        self.pre_posterior_current_sample = self.compute_posterior(self.current_sample, GS_current_sample)
        for i in range(self.max_samples):
            self.proposed_sample = self.Proposal.propose_sample(self.current_sample)
            # it is necessary to recalculate GS_current_cample,
            # because the surrogate model may have changed
            self.Surrogate.send_parameters(np.array([self.current_sample,self.proposed_sample]))
            tmp = self.Surrogate.recv_observations()
            GS_current_sample = tmp[0,:]
            # TO DO: do not recalculate posterior if GS_current_sample did not change
            self.pre_posterior_current_sample = self.compute_posterior(self.current_sample, GS_current_sample)
            GS_proposed_sample = tmp[1,:]
            pre_posterior_proposed_sample = self.compute_posterior(self.proposed_sample, GS_proposed_sample)
            pre_log_ratio = pre_posterior_proposed_sample - self.pre_posterior_current_sample
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
                    row = ['prerejected'] + list(self.proposed_sample) + list(GS_proposed_sample)
                    self.writer_raw.writerow(row)
            if time.time() - self.time_start > self.time_limit:
                print("SAMPLER at RANK", MPI.COMM_WORLD.Get_rank(), "time limit reached - loop",i)
                break
        self.close_files()
        self.__file_rejected.close()
        self.__file_accepted.close()

class Proposal_GaussRandomWalk: # initiated by SAMPLERs
    def __init__(self, no_parameters, proposal_std=1.0, seed=0):
        self.no_parameters = no_parameters
        self.__generator = np.random.RandomState(seed)
        self.set_covariance(proposal_std)
        self.is_symmetric = True
        self.is_exponential = True
        
    def set_covariance(self, proposal_std=1.0):
        # prior std is scalar/vector/covariance matrix:
        if np.isscalar(proposal_std):
            self.proposal_std = np.full((self.no_parameters,),proposal_std)
        else:
            self.proposal_std = np.array(proposal_std)
        if self.proposal_std.ndim == 1: # proposal - normal uncorrelated
            self.propose_sample = self._propose_sample_uncorrelated
        else: # proposal - normal correlated
            self.propose_sample = self.__propose_sample_multivariate

    def _propose_sample_uncorrelated(self, current_sample):
        sample = self.__sample_uncorrelated(self.__generator, current_sample, self.proposal_std)
        return sample
    
    def __propose_sample_multivariate(self, current_sample):
        sample = self.__sample_multivariate(self.__generator, current_sample, self.proposal_std)
        return sample
    
    def __sample_uncorrelated(self,generator,var_mean,var_std):
        sample = generator.normal(var_mean,var_std)
        return sample
    
    def __sample_multivariate(self,generator,var_mean,var_cov):
        sample = generator.multivariate_normal(var_mean,var_cov)
        return sample
    
class Problem_Gauss: # initiated by SAMPLERs
    def __init__(self, no_parameters, prior_mean=0.0, prior_std=1.0, noise_std=1.0, no_observations=None, observations=None, seed=0, name='default_problem_name'):
        self.no_parameters = no_parameters
        if np.isscalar(prior_mean):
            self.prior_mean = np.full((no_parameters,),prior_mean)
        else:
            self.prior_mean = np.array(prior_mean)
            
        # prior std is scalar/vector/covariance matrix:
        if np.isscalar(prior_std):
            self.prior_std = np.full((no_parameters,),prior_std)
        else:
            self.prior_std = np.array(prior_std)
        if self.prior_std.ndim == 1: # prior - normal uncorrelated
            self.get_log_prior = self._get_log_prior_uncorrelated
        else: # prior - normal correlated
            self.get_log_prior = self.__get_log_prior_multivariate
            
        self.observations = observations
        if no_observations == None:
            no_observations = len(observations)
        self.no_observations = no_observations
        self.noise_mean = np.zeros((no_observations,))
        
        # noise std is scalar/vector/covariance matrix:
        if np.isscalar(noise_std):
            self.noise_std = np.full((no_observations,),noise_std)
        else:
            self.noise_std = np.array(noise_std)
        if self.noise_std.ndim == 1: # noise - normal uncorrelated
            self.get_log_likelihood = self._get_log_likelihood_uncorrelated
        else: # noise - normal correlated
            self.get_log_likelihood = self.__get_log_likelihood_multivariate
        
        self.name = name
        self.is_exponential = True
        self.__generator = np.random.RandomState(seed)
    
    def _get_log_likelihood_uncorrelated(self, G_sample):
        v = self.observations - G_sample
        invCv = v/(self.noise_std**2)
        return -0.5*np.sum(v*invCv)
    
    def __get_log_likelihood_multivariate(self, G_sample):
        v = self.observations - G_sample.ravel()
        invCv = np.linalg.solve(self.noise_std,v)
        return -0.5*np.dot(v,invCv)

    def _get_log_prior_uncorrelated(self, sample):
        v = sample - self.prior_mean
        invCv = v/(self.prior_std**2)
        return -0.5*np.dot(v,invCv)
    
    def __get_log_prior_multivariate(self, sample):
        v = sample - self.prior_mean
        invCv = np.linalg.solve(self.prior_std,v)
        return -0.5*np.dot(v,invCv)
    
    def get_log_posterior(self, sample, G_sample, convergence_tag=0):
        if convergence_tag<0:
            return -np.inf
        return self.get_log_likelihood(G_sample) + self.get_log_prior(sample)
    
class Snapshot:
    def __init__(self, sample=None, G_sample=None, weight=None):
        self.sample = sample
        self.G_sample = G_sample
        self.weight = weight
        
    def print(self):
        print("W:", self.weight, "S:", self.sample, "G:", self.G_sample)