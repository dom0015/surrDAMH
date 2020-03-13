#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

from mpi4py import MPI
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

from configuration import Configuration
C = Configuration()
rank_full_solver = C.rank_full_solver
rank_surr_collector = C.rank_surr_collector
no_parameters = C.no_parameters
no_observations = C.no_observations

import classes_SAMPLER as cS
import classes_communication as cCOMM

seed0 = max(1000,size_world)*rank_world
print("PROCESS SAMPLER, seeds:", [seed0, seed0+1, seed0+2], "RANK:", rank_world)

my_Sol = cCOMM.Solver_MPI_collector_MPI(no_parameters=no_parameters, 
                              no_observations=no_observations, 
                              rank_solver=rank_full_solver) # only knows the MPI rank to communicate with
my_Surr_Solver = C.surr_solver_init(**C.surr_solver_parameters)
my_Surr = cCOMM.Solver_local_collector_MPI(no_parameters=no_parameters, 
                                        no_observations=no_observations, 
                                        local_solver_instance=my_Surr_Solver, 
                                        is_updated=True, 
                                        rank_collector=rank_surr_collector)
my_Prob = cS.Problem_Gauss(no_parameters=no_parameters,
                           noise_std=[2.0, 0.1],
                           prior_mean=0.0, 
                           prior_std=1.0,
                           no_observations=no_observations, 
                           observations=[66.4, 2],
                           seed=seed0,
                           name='my_problem2')
my_Prop = cS.Proposal_GaussRandomWalk(no_parameters=no_parameters,
                                      proposal_std=0.8,
                                      seed=seed0+1)
# TO DO: time limit
my_Alg = cS.Algorithm_MH(my_Prob, my_Prop, my_Sol,
                         Surrogate = my_Surr,
                         initial_sample=my_Prob.prior_mean,
                         max_samples=10,
                         name='my_MH_alg' + str(rank_world),
                         seed=seed0+2)
my_Alg1 = cS.Algorithm_DAMH(my_Prob, my_Prop, my_Sol,
                            Surrogate = my_Surr,
                            initial_sample=my_Prob.prior_mean,
                            max_samples=10,
                            name='my_DAMH_alg' + str(rank_world),
                            seed=seed0+3)
my_Alg.run()
print("SAMPLER MH:", rank_world, "- acc/rej/prerej:", my_Alg.no_accepted, my_Alg.no_rejected, my_Alg.no_prerejected, my_Alg.no_accepted/(my_Alg.no_accepted+my_Alg.no_rejected)*100, '%')
my_Alg1.run()

f = getattr(my_Sol,"terminate",None)
if callable(f):
    my_Sol.terminate()
    
f = getattr(my_Surr,"terminate",None)
if callable(f):
    my_Surr.terminate()

comm_world.Barrier()
print("MPI process", rank_world, "(SAMPLER) terminated.")