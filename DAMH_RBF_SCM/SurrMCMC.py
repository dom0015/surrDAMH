#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:17:05 2019

@author: simona
"""

from mpi4py import MPI

import multiprocessing as mp
import numpy as np

import process_master
import process_worker
import SamplingFramework
import ModelGauss
import MHsampler
from lhs_norm import lhs_norm
from scm import scm
from rbf import rbf
from Stage import Stage

class SurrMCMC:
    
    def __init__(self, idx_groups=None, no_chains=1, seed=0, processes_per_solver=1):
        self.idx_groups = idx_groups
        self.no_chains = no_chains
        self.seed = seed
        self.processes_per_solver = processes_per_solver
        self.stages = []

    def setForwardModel(self, no_parameters=None, no_observations=None, solver=None):
        if solver == None:
            self.no_parameters = 4
            self.no_observations = 1
            from Examples import Examples
            self.solver = Examples.linela4
        else:
            self.no_parameters = no_parameters
            self.no_observations = no_observations
            self.solver = solver
            
    def setInverseProblem(self, type_Bayes='Gauss', priorMean=[10], priorStd=[1.5], noiseStd=[2*0.0001], artificial_real_parameters=[8], noisy_observation=None):
        if len(priorMean)!=self.no_parameters:
            self.priorMean = np.ones(self.no_parameters)*priorMean
        else:
            self.priorMean = np.array(priorMean)
        if len(priorStd)!=self.no_parameters:
            self.priorStd = np.ones(self.no_parameters)*priorStd
        else:
            self.priorStd = np.array(priorStd)
        if len(noiseStd)!=self.no_observations:
            self.noiseStd = np.ones(self.no_observations)*noiseStd
        else:
            self.noiseStd = np.array(noiseStd)
        if len(artificial_real_parameters)!=self.no_parameters:
            self.artificial_real_parameters = np.ones(self.no_parameters)*artificial_real_parameters
        else:
            self.artificial_real_parameters = np.array(artificial_real_parameters)
        self.noisy_observation = noisy_observation
            
    def addStage(self, type_sampling='MH', limit_time=10, limit_samples=100, name_stage='default_name', type_proposal='Gauss', proposalStd=0.5):
        if name_stage == 'default_name':
            name_stage += str(len(self.stages))
        s=Stage(type_sampling, limit_time, limit_samples, name_stage, type_proposal, proposalStd)
        self.stages.append(s)
        
    def setSurrogate(self, type_surrogate='RBF', min_snapshots=5, max_snapshots=100, max_degree=2, initial_iteration=None, no_keep=0, expensive=0, type_kernel=0, type_solver=0):
        self.min_snapshots = min_snapshots
        self.max_snapshots = max_snapshots
        if type_surrogate == 'SCM':
            self.surrogate = scm(max_degree)
        else:
            self.surrogate = rbf(initial_iteration, no_keep, expensive, type_kernel, type_solver)
        
    def run(self):
        comm_world = MPI.COMM_WORLD
        group_world = comm_world.Get_group()
        rank_world = comm_world.Get_rank()
        
        if self.idx_groups == None:
            size_world = comm_world.Get_size()
            self.idx_groups = [0]
            no_solvers = int(np.floor((size_world-1)/self.processes_per_solver))
            temp = np.repeat(1+np.arange(no_solvers),self.processes_per_solver)
            self.idx_groups.extend(temp)
            self.idx_groups.extend([no_solvers]*(size_world-1-len(temp)))
            print("idx_groups: ", self.idx_groups)
        else:
            no_solvers = max(self.idx_groups)
        group_leader_ids = []
        for i in range(no_solvers):
            group_leader_ids.append(self.idx_groups.index(i+1))
            
        comm_world.Barrier()
        
        if rank_world == 0:
            group_local = MPI.Group.Incl(group_world,[])
            comm_local = comm_world.Create(group_local)
        else:
            id_group = self.idx_groups[rank_world];
            group_local_ids = [j for j, e in enumerate(self.idx_groups) if e == id_group]
            group_local = MPI.Group.Incl(group_world,group_local_ids)
            comm_local = comm_world.Create(group_local)
            print('Rank_world:',rank_world,'id_group:',id_group,'local rank:',group_local.Get_rank(),'local size:',group_local.Get_size(),'group_local_ids:',group_local_ids,)
        
        comm_world.Barrier()
        
        if rank_world == 0:
            Model = ModelGauss.ModelGauss(self.no_parameters, self.no_observations, self.priorMean, self.priorStd, self.noiseStd, self.noisy_observation)
            MHsampler.MHsampler.Model = Model
            
            if Model.observation == None:
                comm_world.Send(np.append(0,self.artificial_real_parameters), dest=group_leader_ids[0], tag=1)
                received_data = np.zeros(self.no_observations+self.no_parameters+1)
                comm_world.Recv(received_data, source=group_leader_ids[0],tag=1)
                Model.SetNoisyObservation(received_data[self.no_parameters+1:])
                print('artificial_observation_without_noise:',received_data[1:])
                print('artificial observation with noise:',Model.observation)
            
            SF = SamplingFramework.SamplingFramework(self.no_chains, self.stages, self.max_snapshots, self.min_snapshots, no_solvers)
            MHsampler.MHsampler.SF = SF
            MHsampler.MHsampler.seed = self.seed
            initial_samples = lhs_norm(Model,SF.no_chains,self.seed)
            print("initial samples:",initial_samples)
            G_data = np.zeros([0,self.no_observations+self.no_parameters])
            if __name__ == 'SurrMCMC':
                jobs = []
                for i in range(SF.no_chains + SF.no_helpers):
                    p = mp.Process(target=process_worker.worker, args=(i,comm_world,Model,SF,self.surrogate,no_solvers,initial_samples,group_leader_ids))
                    p.start()
                    jobs.append(p)
            process_master.master(comm_world,Model,SF,G_data,no_solvers,group_leader_ids)
            if __name__ == 'SurrMCMC':
                for i in range(SF.no_chains + SF.no_helpers):
                    print(jobs[i])
                    jobs[i].join()
                    print("Join job",i)
        ## FINISH SAMPLING PROCESS
            print('Master finished')
            SF.Finalize()
        
        if rank_world in group_leader_ids:
            print("Hello, I am group leader with local rank", comm_local.Get_rank(), "and global rank", rank_world )
            finish = 0
            status = MPI.Status()
            received_data = np.zeros(self.no_parameters + 1)
            sent_data = np.zeros(self.no_parameters + self.no_observations + 1)#+rank_world
            while finish == 0:
                comm_world.Recv(received_data, source=0, tag=MPI.ANY_TAG, status=status)
        #        print("Solver",rank_world,"received data",received_data)
                num_child = received_data[0]
                data_par = received_data[1:]
                tag = status.Get_tag()
                if tag == 150:
                    print("BYE BYE from rank_world", rank_world)
                    finish = 1
                else:
                    result = self.solver(data_par)
                    sent_data[0] = num_child
                    sent_data[1:5] = data_par
                    sent_data[5] = result
                    comm_world.Send(sent_data, dest=0, tag=1)
        #            print("Solver",rank_world, "Send data",sent_data)
        
        comm_world.Barrier()
        print("MPI process", rank_world, "finished.")
