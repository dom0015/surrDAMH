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
import sys

class Algorithm_MH: # initiated by SAMPLERs
    def __init__(self, Problem, Proposal, Solver, max_samples, name, seed=0, initial_sample=None, G_initial_sample=None, Surrogate=None, is_saved=True):
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
        elif Surrogate.is_updated is True:
            self.__send_to_surrogate = self.__send_to_surrogate__
        else:
            self.__send_to_surrogate = self._empty_function
        self.is_saved = is_saved
        self.__generator = np.random.RandomState(seed)
        self.no_accepted = 0
        self.no_prerejected = 0
        self.no_rejected = 0
        if Problem.is_exponential and Proposal.is_exponential:
            if Proposal.is_symmetric:
                self.__is_accepted = self._acceptance_log_symmetric
                self.__get_posterior = self.Problem.get_log_posterior
            else:
                # not implemented
                return
        else:
            # not implemented
            return
    
    def run(self):
        print("RANK", MPI.COMM_WORLD.Get_rank(), "SAMPLER MH starts")
        if self.is_saved:
            filename = self.Problem.name + "/" + self.name + ".csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.__file = open(filename, 'w')
            self.__writer = csv.writer(self.__file)
            self.__write_to_file = self.__write_to_file__
        else:
            self.__write_to_file = self._empty_function
        if self.G_current_sample is None:
            self.Solver.send_request(self.current_sample)
            self.G_current_sample = self.Solver.get_solution()
        self.posterior_current_sample = self.__get_posterior(self.current_sample, self.G_current_sample)
        self.no_rejected_current = 0
        for i in range(self.max_samples):
            self.proposed_sample = self.Proposal.propose_sample(self.current_sample)
            self.Solver.send_request(self.proposed_sample)
            G_proposed_sample = self.Solver.get_solution()
            self.posterior_proposed_sample = self.__get_posterior(self.proposed_sample, G_proposed_sample)
            if self.__is_accepted(self.posterior_proposed_sample - self.posterior_current_sample):
                self.__write_to_file()
                self.__send_to_surrogate(sample=self.current_sample.copy(), G_sample=self.G_current_sample.copy(), weight=self.no_rejected_current+1)    
                self.current_sample = self.proposed_sample
                self.G_current_sample = G_proposed_sample
                self.posterior_current_sample = self.posterior_proposed_sample
                self.no_accepted += 1
                self.no_rejected_current = 0       
            else:
                self.no_rejected += 1
                self.no_rejected_current += 1
                self.__send_to_surrogate(sample=self.proposed_sample, G_sample=G_proposed_sample, weight=0)
                
        self.__write_to_file()
        if self.is_saved:
            self.__file.close()
            filename_notes = self.Problem.name + "/notes/" + self.name + ".csv"
            os.makedirs(os.path.dirname(filename_notes), exist_ok=True)
            notes = [self.no_accepted, self.no_rejected, self.seed]
            file_notes = open(filename_notes, 'w')
            writer_notes = csv.writer(file_notes)
            writer_notes.writerow(notes)
            file_notes.close()
        print("RANK", MPI.COMM_WORLD.Get_rank(), "SAMPLER MH finishes")

    def _acceptance_log_symmetric(self,log_ratio):
        temp = self.__generator.uniform(0.0,1.0)
        #print(self.posterior_proposed_sample, self.posterior_current_sample, temp, np.log(temp))
        temp = np.log(temp)
        if temp<log_ratio: # accepted
            return True
        else:
            return False
        
    def __write_to_file__(self):
        row = [1+self.no_rejected_current]
        for j in range(self.Problem.no_parameters):
            row.append(self.current_sample[j])
        self.__writer.writerow(row)
    
    def __send_to_surrogate__(self, sample, G_sample, weight):
        snapshot = Snapshot(sample=sample, G_sample=G_sample, weight=weight)
        self.Surrogate.send_to_data_collector(snapshot)
        
    def _empty_function(self,**kw):
        return
    
class Algorithm_DAMH: # initiated by SAMPLERs
    def __init__(self, Problem, Proposal, Solver, max_samples, name, seed=0, initial_sample=None, G_initial_sample=None, Surrogate=None, is_saved=True):
        self.Problem = Problem
        self.Proposal = Proposal
        self.Solver = Solver
        self.max_samples = max_samples
        self.name = name
        self.seed = seed
        self.current_sample = initial_sample
        if self.current_sample is None:
            self.current_sample = self.Problem.prior_mean
        self.G_current_sample = G_initial_sample
        self.Surrogate = Surrogate
        if Surrogate is None:
            self.__send_to_surrogate = self._empty_function
        elif Surrogate.is_updated is True:
            self.__send_to_surrogate = self.__send_to_surrogate__
        else:
            self.__send_to_surrogate = self._empty_function
        self.is_saved = is_saved
        self.__generator = np.random.RandomState(seed)
        self.no_accepted = 0
        self.no_prerejected = 0
        self.no_rejected = 0
        if Problem.is_exponential and Proposal.is_exponential:
            if Proposal.is_symmetric:
                self.__is_accepted = self._acceptance_log_symmetric
                self.__get_posterior = self.Problem.get_log_posterior
            else:
                # not implemented
                return
        else:
            # not implemented
            return
    
    def run(self):
        print("RANK", MPI.COMM_WORLD.Get_rank(), "SAMPLER DAMH starts")
        if self.is_saved:
            filename = self.Problem.name + "/" + self.name + ".csv"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.__file = open(filename, 'w')
            self.__writer = csv.writer(self.__file)
            self.__write_to_file = self.__write_to_file__
        else:
            self.__write_to_file = self._empty_function
        if self.G_current_sample is None:
            self.Solver.send_request(self.current_sample.copy())
            self.G_current_sample = self.Solver.get_solution()
        self.posterior_current_sample = self.__get_posterior(self.current_sample, self.G_current_sample)
        self.Surrogate.send_request(self.current_sample.copy())
        GS_current_sample = self.Surrogate.get_solution()
        pre_posterior_current_sample = self.__get_posterior(self.current_sample, GS_current_sample)
        self.no_rejected_current = 0
        for i in range(self.max_samples):
            self.proposed_sample = self.Proposal.propose_sample(self.current_sample)
            self.Surrogate.send_request(self.proposed_sample.copy())
            GS_proposed_sample = self.Surrogate.get_solution()
            pre_posterior_proposed_sample = self.__get_posterior(self.proposed_sample, GS_proposed_sample)
            pre_log_ratio = pre_posterior_proposed_sample - pre_posterior_current_sample
            if self.__is_accepted(pre_log_ratio):
                self.Solver.send_request(self.proposed_sample.copy())
                G_proposed_sample = self.Solver.get_solution()
                self.posterior_proposed_sample = self.__get_posterior(self.proposed_sample, G_proposed_sample)
                log_ratio = self.posterior_proposed_sample - self.posterior_current_sample
#                print("log ratio:", log_ratio, pre_log_ratio)
                if self.__is_accepted(log_ratio - pre_log_ratio):
                    self.__write_to_file()
                    self.__send_to_surrogate(sample=self.proposed_sample, G_sample=G_proposed_sample, weight=self.no_rejected_current+1)
                    self.no_accepted += 1
                    self.no_rejected_current = 0           
                    self.current_sample = self.proposed_sample
                    self.posterior_current_sample = self.posterior_proposed_sample
                    pre_posterior_current_sample = pre_posterior_proposed_sample
                else:
                    self.no_rejected += 1
                    self.no_rejected_current += 1
            else:
                self.no_prerejected += 1
                self.no_rejected_current += 1
       
        self.__write_to_file()
        if self.is_saved:
            self.__file.close()
            filename_notes = self.Problem.name + "/notes/" + self.name + ".csv"
            os.makedirs(os.path.dirname(filename_notes), exist_ok=True)
            notes = [self.no_accepted, self.no_rejected, self.seed]
            file_notes = open(filename_notes, 'w')
            writer_notes = csv.writer(file_notes)
            writer_notes.writerow(notes)
            file_notes.close()
        print("RANK", MPI.COMM_WORLD.Get_rank(), "SAMPLER DAMH finishes")

    def _acceptance_log_symmetric(self,log_ratio):
        temp = self.__generator.uniform(0.0,1.0)
        #print(self.posterior_proposed_sample, self.posterior_current_sample, temp, np.log(temp))
        temp = np.log(temp)
        if temp<log_ratio: # accepted
            return True
        else:
            return False
        
    def __write_to_file__(self):
        row = [1+self.no_rejected_current]
        for j in range(self.Problem.no_parameters):
            row.append(self.current_sample[j])
        self.__writer.writerow(row)
    
    def __send_to_surrogate__(self, sample, G_sample, weight):
        snapshot = Snapshot(sample=sample, G_sample=G_sample, weight=weight)
        self.Surrogate.send_to_data_collector(snapshot)
        
    def _empty_function(self,**kw):
        return

class Proposal_GaussRandomWalk: # initiated by SAMPLERs
    def __init__(self, no_parameters, proposal_std=1.0, proposal_cov=None, seed=0):
        self.__generator = np.random.RandomState(seed)
        self.proposal_cov = proposal_cov
        if self.proposal_cov is None:
            self.propose_sample = self._propose_sample_uncorrelated
            if np.isscalar(proposal_std):
                self.proposal_std = np.full((no_parameters,),proposal_std)
            else:
                self.proposal_std = np.array(proposal_std)
        else:
            self.proposal_std = None
            self.propose_sample = self.__propose_sample_multivariate
        self.is_symmetric = True
        self.is_exponential = True

    def _propose_sample_uncorrelated(self, current_sample):
        sample = self.__sample_uncorrelated(self.__generator, current_sample, self.proposal_std)
        return sample
    
    def __propose_sample_multivariate(self, current_sample):
        sample = self.__sample_multivariate(self.__generator, current_sample, self.proposal_cov)
        return sample
    
    def __sample_uncorrelated(self,generator,var_mean,var_std):
        sample = generator.normal(var_mean,var_std)
        return sample
    
    def __sample_multivariate(self,generator,var_mean,var_cov):
        sample = generator.multivariate_normal(var_mean,var_cov)
        return sample
    
class Problem_Gauss: # initiated by SAMPLERs
    def __init__(self, no_parameters, prior_mean=0.0, prior_std=1.0, prior_cov=None, noise_std=1.0, noise_cov=None, no_observations=None, observations=None, seed=0, name='default_problem_name'):
        self.no_parameters = no_parameters
        if np.isscalar(prior_mean):
            self.prior_mean = np.full((no_parameters,),prior_mean)
        else:
            self.prior_mean = np.array(prior_mean)
        self.prior_cov = prior_cov
        if self.prior_cov is None:
            self.get_log_prior = self._get_log_prior_uncorrelated
            if np.isscalar(prior_std):
                self.prior_std = np.full((no_parameters,),prior_std)
            else:
                self.prior_std = np.array(prior_std)
        else:
            self.prior_std = None
            self.get_log_prior = self.__get_log_prior_multivariate
        self.observations = observations
        if no_observations == None:
            no_observations = len(observations)
        self.no_observations = no_observations
        self.noise_mean = np.zeros((no_observations,))
        self.noise_cov = noise_cov
        if self.noise_cov is None:
            self.get_log_likelihood = self._get_log_likelihood_uncorrelated
            if np.isscalar(noise_std):
                self.noise_std = np.full((no_observations,),noise_std)
            else:
                self.noise_std = np.array(noise_std)
        else:
            self.noise_std = None
            self.get_log_likelihood = self.__get_log_likelihood_multivariate
        self.name = name
        self.is_exponential = True
        self.__generator = np.random.RandomState(seed)
    
    def _get_log_likelihood_uncorrelated(self, G_sample):
        v = self.observations - G_sample
        invCv = v/self.noise_std
        return -0.5*np.sum(v*invCv)
    
    def __get_log_likelihood_multivariate(self, G_sample):
        v = self.observations - G_sample
        invCv = np.linalg.solve(self.noise_cov,v)
        return -0.5*np.dot(v,invCv)

    def _get_log_prior_uncorrelated(self, sample):
        v = sample - self.prior_mean
        invCv = v/self.prior_std
        return -0.5*np.dot(v,invCv)
    
    def __get_log_prior_multivariate(self, sample):
        v = sample - self.prior_mean
        invCv = np.linalg.solve(self.prior_cov,v)
        return -0.5*np.dot(v,invCv)
    
    def get_log_posterior(self, sample, G_sample):
        return self.get_log_likelihood(G_sample) + self.get_log_prior(sample)
    
    def sample_prior(self,generator=None):
        if generator is None:
            generator = self.__generator
        if self.prior_cov is None:
            sample = self.__sample_uncorrelated(generator, self.prior_mean, self.prior_std)
        else:
            sample = self.__sample_multivariate(generator, self.prior_mean, self.prior_cov)
        return sample
    
    def sample_noise(self,generator=None):
        if generator is None:
            generator = self.__generator
        if self.noise_cov is None:
            sample = self.__sample_uncorrelated(generator, self.noise_mean, self.noise_std)
        else:
            sample = self.__sample_multivariate(generator, self.noise_mean, self.noise_cov)
        return sample
    
    def __sample_uncorrelated(self,generator,var_mean,var_std):
        sample = generator.normal(var_mean,var_std)
        return sample
    
    def __sample_multivariate(self,generator,var_mean,var_cov):
        sample = generator.multivariate_normal(var_mean,var_cov)
        return sample
    
class Snapshot: # initiated by SAMPLERs (if surrogate is updated) and
    def __init__(self, sample=None, G_sample=None, weight=None):
        self.sample = sample
        self.G_sample = G_sample
        self.weight = weight
        
    def print(self):
        print("W:", self.weight, "S:", self.sample, "G:", self.G_sample)
       
class Solver_MPI_parent: # initiated by full SOLVER
    def __init__(self, no_parameters, no_observations, maxprocs=1):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_requests = 1
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['process_CHILD.py'], maxprocs=maxprocs)
        self.tag = 0
        self.received_data = np.zeros(self.no_observations)
    
    def send_request(self, data_par):
        self.tag += 1
        self.data_par = data_par.copy()
        self.comm.Send(self.data_par, dest=0, tag=self.tag)
        print('DEBUG - PARENT Send request FROM', self.comm.Get_rank(), '(', MPI.COMM_WORLD.Get_rank(), ')', 'TO child:', 0, "TAG:", self.tag)
        
    def get_solution(self):
        self.comm.Recv(self.received_data, source=0, tag=self.tag)
        print('DEBUG - PARENT Recv solution FROM child', 0, 'TO:', self.comm.Get_rank(), '(', MPI.COMM_WORLD.Get_rank(), ')', "TAG:", self.tag)
        return self.received_data.copy()
    
    def is_solved(self):
        # check the parent-child communicator if there is an incoming message
        tmp = self.comm.Iprobe(source=0, tag=self.tag)
        if tmp:
            return True
        else:
            return False
    
    def terminate(self):
        self.comm.Send(np.empty((1,self.no_parameters)), dest=0, tag=0)
        print("Solver spawned by rank", MPI.COMM_WORLD.Get_rank(), "will be disconnected.")
        self.comm.Disconnect()
    
class Solver_MPI_collector_MPI: # initiated by SAMPLERs
    def __init__(self, no_parameters, no_observations, rank_solver, is_updated=False, rank_data_collector=None):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_requests = 1
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.rank_solver = rank_solver
        self.is_updated = is_updated
        self.rank_data_collector = rank_data_collector
        self.tag_solver = 0
        self.tag_collector = 0
        self.received_data = np.zeros(self.no_observations)
        self.terminated_solver = None
        self.terminated_collector = True
        if not rank_data_collector is None:
            self.terminated_collector = False
        
    def send_request(self, sent_data):
        self.tag_solver += 1
        self.comm.Send(sent_data, dest=self.rank_solver, tag=self.tag_solver)
        print('DEBUG - Solver_MPI_collector_MPI (', self.rank, ') Send request FROM', self.rank, 'TO:', self.rank_solver, "TAG:", self.tag_solver)
    
    def get_solution(self, ):
        self.comm.Recv(self.received_data, source=self.rank_solver, tag=self.tag_solver)
        print('DEBUG - Solver_MPI_collector_MPI (', self.rank, ') Recv solution FROM', self.rank_solver, 'TO:', self.rank, "TAG:", self.tag_solver)
        return self.received_data.copy()
    
    def send_to_data_collector(self, sent_snapshot):
        # needed only if self.is_updated == True
        self.tag_collector += 1
        print('debug - Solver_MPI_collector_MPI (', self.rank, ') - sent_snapshot', self.rank_data_collector, self.comm.Get_size(), self.rank)
        self.comm.send(sent_snapshot, dest=self.rank_data_collector, tag=self.tag_collector)
        
    def is_solved(self):
        # check COMM_WORLD if there is an incoming message from the solver
        tmp = self.comm.Iprobe(source=self.rank_solver, tag=self.tag_solver)
        if tmp:
            return True
        else:
            return False
        
    def terminate(self, ):
        # assume that terminate may be called multiple times
        if not self.terminated_solver:
            sent_data = np.zeros(self.no_parameters)
            print('debug - Solver_MPI_collector_MPI terminate',self.rank_solver, self.comm.Get_size(), self.rank)
            self.comm.Send(sent_data, dest=self.rank_solver, tag=0)
            self.terminated_solver = True
        if not self.terminated_collector:
            snapshot = Snapshot()
            self.comm.send(snapshot, dest=self.rank_data_collector, tag=0)
            self.terminated_collector = True

class Solver_local_collector_MPI: # initiated by SAMPLERs
    # local solver (evaluated on SAMPLERs)
    # with external data COLLECTOR (separate MPI process)
    def __init__(self, no_parameters, no_observations, local_solver_instance, is_updated=False, rank_data_collector=None):
        self.max_requests = 1
        self.comm = MPI.COMM_WORLD
        self.local_solver_instance = local_solver_instance
        self.is_updated = is_updated
        self.rank_data_collector = rank_data_collector
        self.solver_data = [None] * 2       # double buffer
        self.solver_data_idx = 0            # idx of current buffer 0/1
        self.tag_terminate = 0
        self.tag_ready_to_receive = 1
        self.tag_sent_data = 2
        self.terminated = None
        self.terminated_data = True
        if not rank_data_collector is None:
            self.terminated_data = False
        self.computation_in_progress = False # TO DO: remove?
        self.empty_buffer = np.zeros(1)
        self.req = self.comm.irecv(source=self.rank_data_collector, tag=self.tag_sent_data)
        self.comm.Isend(self.empty_buffer, dest=self.rank_data_collector, tag=self.tag_ready_to_receive)
        self.list_snapshots_to_send = []
        self.status = MPI.Status()

    def send_request(self, received_parameters):
        self.received_parameters = received_parameters # TO DO: copy?
        self.computation_in_progress = True

    def get_solution(self, ):
        print("SOLVER_DATA taken", self.solver_data_idx)
        calculated_observations = self.local_solver_instance.apply(self.solver_data[self.solver_data_idx],self.received_parameters)
        self.computation_in_progress = False
        return calculated_observations

    def send_to_data_collector(self, snapshot_to_send):
        # sends list of snapshots to data COLLECTOR
        # needed only if is_updated == True
        self.list_snapshots_to_send.append(snapshot_to_send)
        tmp = self.comm.Iprobe(source=self.rank_data_collector, tag=self.tag_ready_to_receive)
        if tmp: # if the collector is ready to receive new snapshots
            tmp = np.zeros(1) # TO DO: useless pointer / cancel message
            self.comm.Recv(tmp,source=self.rank_data_collector, tag=self.tag_ready_to_receive)
            print('DEBUG - Solver_local_collector_MPI (', self.comm.Get_rank(), ') - sent_snapshots to', self.rank_data_collector)
            self.comm.isend(self.list_snapshots_to_send.copy(), dest=self.rank_data_collector, tag=self.tag_sent_data)
            self.list_snapshots_to_send = []
            # TO DO: isend
        self.receive_update_if_ready()

    def receive_update_if_ready(self):
        # receive updated solver data (e.g. for updated surrogate model)
        # check COMM_WORLD if there is an incoming message from the data COLLECTOR
        # TO DO: when to check for updates
        # TO DO: avoid copying of received data
        r = self.req.test(status=self.status)
        if r[0]:
            self.solver_data[1 - self.solver_data_idx] = r[1]
            self.solver_data_idx = 1 - self.solver_data_idx
            self.req = self.comm.irecv(source=self.rank_data_collector, tag=self.tag_sent_data)
            self.comm.Isend(self.empty_buffer, dest=self.rank_data_collector, tag=self.tag_ready_to_receive)
            
    def is_solved(self):
        if self.computation_in_progress:
            return False
        else:
            return True

    def terminate(self, ):
        # assume that terminate may be called multiple times
        if not self.terminated:
            self.terminated = True
        if not self.terminated_data:
            snapshot = Snapshot()
            self.comm.send(snapshot, dest=self.rank_data_collector, tag=self.tag_terminate)
            self.terminated_data = True
