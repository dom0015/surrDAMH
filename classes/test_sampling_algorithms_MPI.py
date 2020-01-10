#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

from mpi4py import MPI
import sampling_algorithms as sa
import numpy as np

from configuration import Configuration
C = Configuration()
no_solvers = C.no_solvers
no_algorithms = C.no_algorithms

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

#tmp = np.random.RandomState(44+rank_world)
#print(tmp.uniform(), tmp.uniform(), tmp.uniform())

seed0 = max(1000,size_world)*rank_world
print([seed0, seed0+1, seed0+2])

#my_Sol = sa.Solver_local_2to2()
my_Sol = sa.Solver_linker(no_parameters=2, no_observations=2, rank_full_solver=no_algorithms)
my_Surr = sa.Solver_linker(no_parameters=2, no_observations=2, rank_full_solver=no_algorithms+1)#, is_updated=True) #, rank_data_collector=no_algorithms+2)
my_Prob = sa.Problem_Gauss(no_parameters=my_Sol.no_parameters,
                           noise_std=[2.0, 0.1],
                           prior_mean=0.0, 
                           prior_std=1.5,
                           no_observations=my_Sol.no_observations, 
                           observations=[66.4, 2],
                           seed=seed0,
                           name='my_problem')
my_Prop = sa.Proposal_GaussRandomWalk(no_parameters=my_Sol.no_parameters,
                                      proposal_std=0.8,
                                      seed=seed0+1)
my_Alg = sa.Algorithm_MH(my_Prob, my_Prop, my_Sol,
                         Surrogate = None, # my_Surr,
                         initial_sample=my_Prob.prior_mean,
                         max_samples=10,
                         name='my_MH_alg' + str(rank_world),
                         seed=seed0+2)
my_Alg.run()
print("ALG:", rank_world, size_world, "- acc/rej/prerej:", my_Alg.no_accepted, my_Alg.no_rejected, my_Alg.no_prerejected, my_Alg.no_accepted/(my_Alg.no_accepted+my_Alg.no_rejected)*100, '%')

comm_world.Barrier()
print("MPI process", rank_world, "(MH algorithm) terminated.")