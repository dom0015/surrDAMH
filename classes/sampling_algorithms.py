#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

import numpy as np
import os
import csv
from mpi4py import MPI
import sys

class Proposal_GaussRandomWalk:
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
    
class Problem_Gauss:
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
        return -0.5*np.dot(v,invCv)
    
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
        
class Algorithm_MH:
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
        self.no_rejected_current = 0
        for i in range(self.max_samples):
            self.proposed_sample = self.Proposal.propose_sample(self.current_sample)
            self.Solver.send_request(self.proposed_sample.copy())
            G_proposed_sample = self.Solver.get_solution()
#            if any(self.proposed_sample != G_proposed_sample):
#                print(self.Solver.comm.Get_rank(), self.proposed_sample, G_proposed_sample, "TAG=", self.Solver.tag)
            self.posterior_proposed_sample = self.__get_posterior(self.proposed_sample, G_proposed_sample)
            if self.__is_accepted(self.posterior_proposed_sample - self.posterior_current_sample):
                self.__write_to_file()
                self.__send_to_surrogate(sample=self.current_sample, G_sample=self.G_current_sample, weight=self.no_rejected_current+1)    
                self.current_sample = self.proposed_sample
                self.G_current_sample = G_proposed_sample
                self.posterior_current_sample = self.posterior_proposed_sample
                self.no_accepted += 1
                self.no_rejected_current = 0       
            else:
                self.no_rejected += 1
                self.no_rejected_current += 1
                self.__send_to_surrogate(sample=self.proposed_sample, G_sample=G_proposed_sample, weight=0)
                
        f = getattr(self.Solver,"terminate",None)
        if callable(f):
            self.Solver.terminate()
            
        f = getattr(self.Surrogate,"terminate",None)
        if callable(f):
            self.Surrogate.terminate()
                
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
            #print(notes)

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
        self.Surrogate.send_snapshot(snapshot)
        
    def _empty_function(self,**kw):
        return
    
class Algorithm_DAMH:
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
    
        f = getattr(self.Solver,"terminate",None)
        if callable(f):
            self.Solver.terminate()
            
        f = getattr(self.Surrogate,"terminate",None)
        if callable(f):
            self.Surrogate.terminate()
                
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
            #print(notes)

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
        self.Surrogate.send_snapshot(snapshot)
        
    def _empty_function(self,**kw):
        return
    
class Snapshot:
    def __init__(self, sample=None, G_sample=None, weight=None):
        self.sample = sample
        self.G_sample = G_sample
        self.weight = weight
        
    def print(self):
        print("W:", self.weight, "S:", self.sample, "G:", self.G_sample)
    
class Solver_local_linela4:
    def __init__(self, f=-0.1, L=1.0, M12=0.25, M23=0.5, M34=0.75):
        self.f = f
        self.L = L
        self.M12 = M12
        self.M23 = M23
        self.M34 = M34
        self.no_parameters = 4
        self.no_observations = 1
        
    def get_solution(self, data_par):
        k1 = data_par[0]
        k2 = data_par[1]
        k3 = data_par[2]
        k4 = data_par[3]
        C4 = (self.f*self.L)/k4
        C3 = C4*k4/k3
        C2 = C3*k3/k2
        C1 = C2*k2/k1
        D1 = 0
        D2 = -self.f/k1*self.M12*self.M12/2 + C1*self.M12 + D1 + self.f/k2*self.M12*self.M12/2 - C2*self.M12
        D3 = -self.f/k2*self.M23*self.M23/2 + C2*self.M23 + D2 + self.f/k3*self.M23*self.M23/2 - C3*self.M23
        D4 = -self.f/k3*self.M34*self.M34/2 + C3*self.M34 + D3 + self.f/k4*self.M34*self.M34/2 - C4*self.M34
        uL = -self.f/k4*self.L*self.L/2 + C4*self.L + D4
        return uL
    
class Solver_local_linela2:
    def __init__(self, f=-0.1, L=1.0, M=0.5):
        self.f = f
        self.L = L
        self.M = M
        self.no_parameters = 2
        self.no_observations = 1
        
    def get_solution(self, data_par):
        k1 = data_par[0]
        k2 = data_par[1]
        D1 = (self.f*self.L)/k2
        C1 = D1*k2/k1
        D2 = -self.f/(2*k1)*(self.M*self.M)+C1*self.M+self.f/(2*k2)*(self.M*self.M)-D1*self.M
        uL = -self.f/(2*k2)*(self.L*self.L)+D1*self.L+D2
        return uL
    
class Solver_local_himmelblau:
    def __init__(self, ):
        self.no_parameters = 2
        self.no_observations = 1
        
    def get_solution(self, data_par):
        x1 = data_par[0]
        x2 = data_par[1]
        y = (x1*x1 + x2 - 11)**2 + (x1 + x2*x2 - 7)**2
        return y
    
class Solver_local_2to2:
    def __init__(self, ):
        self.no_parameters = 2
        self.no_observations = 2
        self.request_solved = True
        self.max_requests = 1
    
    def send_request(self, data_par):
        self.data_par = data_par
        
    def get_solution(self, ):
        x1 = self.data_par[0,0]
        x2 = self.data_par[0,1]
        y1 = (x1*x1 + x2 - 11)**2 + (x1 + x2*x2 - 7)**2
        y2 = x1 + x2
        return np.array([[y1,y2]])
    
    def terminate(self):
        print("Terminate function is empty.")

    
class Solver_external_2to2:
    def __init__(self, maxprocs=1):
        self.no_parameters = 2
        self.no_observations = 2
        self.request_solved = True
        self.max_requests = 1
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['launcher.py'], maxprocs=maxprocs)
        self.tag = 1
    
    def send_request(self, data_par):
        self.data_par = data_par.copy()
        self.comm.Send(self.data_par, dest=0, tag=self.tag)
        
    def get_solution(self):
        received_data = np.empty((1,self.no_observations))
        self.comm.Recv(received_data, source=0, tag=self.tag)
        return received_data
    
    def terminate(self):
        self.comm.Send(np.empty((1,self.no_parameters)), dest=0, tag=0)
        print("Solver spawned by rank", MPI.COMM_WORLD.Get_rank(), "will be disconnected.")
        self.comm.Disconnect()
    
class Solver_linker:
    def __init__(self, no_parameters, no_observations, rank_full_solver, is_updated=False, rank_data_collector=None):
        self.no_parameters = 2
        self.no_observations = 2
        self.comm = MPI.COMM_WORLD
        self.rank_full_solver = rank_full_solver
        self.is_updated = is_updated
        self.rank_data_collector = rank_data_collector
        self.tag = 0
        self.tag_data = 0
        self.received_data = np.zeros(no_parameters)
        self.terminated = None
        self.terminated_data = True
        if not rank_data_collector is None:
            self.terminated_data = False
        
    def send_request(self, sent_data):
        self.tag += 1
        print('debug - rank', self.rank_full_solver, self.comm.Get_size(), self.comm.Get_rank())
        self.comm.Send(sent_data, dest=self.rank_full_solver, tag=self.tag)
#        print("Request", self.tag, sent_data)
    
    def get_solution(self, ):
        self.comm.Recv(self.received_data, source=self.rank_full_solver, tag=self.tag)
#        print("Solution", self.tag, self.received_data)
        return self.received_data
    
    def send_snapshot(self, sent_snapshot):
        # needed only if is_updated == True
        self.tag_data += 1
        print('debug - sent_snapshot', self.rank_data_collector, self.comm.Get_size(), self.comm.Get_rank())
        self.comm.send(sent_snapshot, dest=self.rank_data_collector, tag=self.tag_data)
        
    def terminate(self, ):
        # assume that terminate may be called multiple times
        if not self.terminated:
            sent_data = np.zeros(self.no_parameters)
            print('debug - terminate',self.rank_full_solver, self.comm.Get_size(), self.comm.Get_rank())
            self.comm.Send(sent_data, dest=self.rank_full_solver, tag=0)
            self.terminated = True
        if not self.terminated_data:
            snapshot = Snapshot()
            self.comm.send(snapshot, dest=self.rank_data_collector, tag=0)
            self.terminated_data = True
            
class Surrogate_col:
    def __init__(self, no_parameters, no_observations, is_updated=True, max_degree=5):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.is_updated = is_updated
        self.max_degree = max_degree
        self.alldata_par = np.empty((0,self.no_parameters))
        self.alldata_obs = np.empty((0,self.no_observations))
        self.alldata_wei = np.empty((0,1))
        self.update_finished = True
        
    def add_data(self,snapshots):
        L = len(snapshots)
        newdata_par = np.empty((L,self.no_parameters))
        newdata_obs = np.empty((L,self.no_observations))
        newdata_wei = np.empty((L,1))
        for i in range(L):
            newdata_par[i,:] = snapshots[i].sample
            newdata_obs[i,:] = snapshots[i].G_sample
            newdata_wei[i,:] = 1 #snapshots[i].weight TEMP!
        self.alldata_par = np.vstack((self.alldata_par, newdata_par))
        self.alldata_obs = np.vstack((self.alldata_obs, newdata_obs))
        self.alldata_wei = np.vstack((self.alldata_wei, newdata_wei))
    
    def calculate(self):
        no_snapshots, no_parameters = self.alldata_par.shape
        degree = int(np.floor(np.log(no_snapshots)/np.log(no_parameters)))
#        print("degree",degree)
        if degree == 0:
            degree=1
        if degree>self.max_degree:
            degree=self.max_degree
        poly = self.generate_polynomials_degree(no_parameters,degree)
        N = poly.shape[0]
        H = self.hermite_poly_normalized(degree)
        coefs = np.ones((no_snapshots,N))
        for i in range(N):
            for j in range(no_parameters):
                H_row = H[int(poly[i,j]),:]#.reshape((1,degree+1))
                par_col = self.alldata_par[:,j]#.reshape((no_snapshots,1))
                coefs[:,i] *= self.poly_eval(H_row,par_col)
        coefs_wei = (coefs * self.alldata_wei).transpose()
        A = np.matmul(coefs_wei,coefs)
        RHS = np.matmul(coefs_wei,self.alldata_obs)
        c = np.matmul(np.linalg.pinv(A),RHS)
        SOL = [c, H, poly, degree]
#        print("C",c,"H",H,"poly",poly,"degree",degree,"wei",alldata_wei)
        return SOL, no_snapshots

    def apply(self, SOL, newdata_par):
        c, H, poly, degree = SOL
        N = poly.shape[0]
        no_newdata = newdata_par.shape[0]
        no_observations = c.shape[1]
        newdata_surrogate = np.zeros((no_newdata,no_observations))
        phi = np.ones((no_newdata,N))
        for i in range(N):
            for j in range(self.no_parameters):
                H_row = H[int(poly[i,j]),:]#.reshape((1,degree+1))
                par_col = newdata_par[:,j]#.reshape((1,1))
                phi[:,i] *= self.poly_eval(H_row,par_col)#.reshape((1,))
#        for k in range(no_newdata):
#            newdata_surrogate[k,:] = np.matmul(phi[k,:],c)
        newdata_surrogate = np.matmul(phi,c)
        return newdata_surrogate
    
    def generate_polynomials_degree(self,dim,degree):
        poly = np.zeros([1,dim])
        
        if degree==0:
            return poly
        
        if degree>0:
            poly = np.vstack((poly,np.eye(dim)))
            
        if degree>1:
            temp1 = np.eye(dim) # const
            temp = np.eye(dim)
            for i in range(degree-1):
                polynew = np.zeros([temp1.shape[0]*temp.shape[0],dim])
                idx = 0
                for j in range(temp1.shape[0]):
                    for k in range(temp.shape[0]):
                        polynew[idx] = temp1[j,:]+temp[k,:]
                        idx += 1
                temp = np.unique(polynew,axis=0)
                poly = np.vstack((poly,temp))
        return poly
    
    def hermite_poly_normalized(self,degree):
        n = degree + 1
        H = np.zeros((n,n))
        H[0,0] = 1
        if degree==0:
            return H
        H[1,1] = 1
        diff = np.arange(1,n)
        for i in range(2,n):
            H[i,1:] += H[i-1,:-1]
            H[i,:-1] -= diff*H[i-1,1:]
        for i in range(n):
            H[i,:] = np.divide(H[i,:],np.sqrt(np.math.factorial(i)))
        return H
    
    def poly_eval(self, p, grid):
#        print("p shape",p.shape)
#        print("g shape",grid.shape, grid.size)
        n = p.size
        values = np.zeros(grid.size)
        temp = np.ones(grid.shape)
        values += p[0]
        for i in range(1,n):
            temp = temp*grid
            values += temp*p[i]
#        print("values:",values)
        return values