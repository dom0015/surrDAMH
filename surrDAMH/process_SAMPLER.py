#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

from mpi4py import MPI
from modules import classes_SAMPLER as cS
from modules import classes_communication
from modules import lhs_normal as LHS
from configuration import Configuration

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

no_samplers, problem_name = comm_world.recv(source=MPI.ANY_SOURCE)
comm_world.Barrier()
C = Configuration(no_samplers, problem_name)
seed0 = max(1000,size_world)*rank_world # TO DO seeds

my_Sol = classes_communication.Solver_MPI_collector_MPI(no_parameters=C.no_parameters, 
                              no_observations=C.no_observations, 
                              rank_solver=C.solver_parent_rank) # only knows the MPI rank to communicate with
my_Surr_Solver = C.surr_solver_init(**C.surr_solver_parameters)
my_Surr = classes_communication.Solver_local_collector_MPI(no_parameters=C.no_parameters, 
                                        no_observations=C.no_observations, 
                                        local_solver_instance=my_Surr_Solver,
                                        rank_collector=C.rank_surr_collector)
my_Prob = cS.Problem_Gauss(no_parameters=C.no_parameters,
                           noise_std=C.problem_parameters['noise_std'],
                           prior_mean=C.problem_parameters['prior_mean'], 
                           prior_std=C.problem_parameters['prior_std'],
                           no_observations=C.no_observations,
                           observations=C.problem_parameters['observations'],
                           seed=seed0,
                           name=C.problem_name)
my_Prop = cS.Proposal_GaussRandomWalk(no_parameters=C.no_parameters, seed=seed0+1)

initial_samples = LHS.lhs_normal(C.no_parameters,C.problem_parameters['prior_mean'],C.problem_parameters['prior_std'],C.no_samplers,0)
initial_sample = initial_samples[rank_world]
for i,d in enumerate(C.list_alg):
    # TO DO: adaptive proposal if needed
    my_Prop.set_covariance(proposal_std = d['proposal_std'])
    seed = seed0 + 2 + i
    if d['type'] == 'MH':
        my_Alg = cS.Algorithm_MH(my_Prob, my_Prop, my_Sol,
                         Surrogate = my_Surr,
                         surrogate_is_updated = d['surrogate_is_updated'],
                         initial_sample=initial_sample,
                         max_samples=d['max_samples'],
                         time_limit=d['time_limit'],
                         save_raw_data=True,
                         name='alg' + str(i) + 'MH_rank' + str(rank_world),
                         seed=seed)
    else:
        my_Alg = cS.Algorithm_DAMH(my_Prob, my_Prop, my_Sol,
                        Surrogate = my_Surr,
                        surrogate_is_updated = d['surrogate_is_updated'],
                        initial_sample=initial_sample,
                        max_samples=d['max_samples'],
                        time_limit=d['time_limit'],
                        save_raw_data=True,
                        name='alg' + str(i) + 'DAMH_rank' + str(rank_world),
                        seed=seed)
    print('--- SAMPLER ' + my_Alg.name + ' starts ---')
    my_Alg.run()
    initial_sample = my_Alg.current_sample
    print('--- SAMPLER ' + my_Alg.name + ' --- acc/rej/prerej:', my_Alg.no_accepted, my_Alg.no_rejected, my_Alg.no_prerejected)

f = getattr(my_Sol,"terminate",None)
if callable(f):
    my_Sol.terminate()

f = getattr(my_Surr,"terminate",None)
if callable(f):
    my_Surr.terminate()

comm_world.Barrier()
print("RANK", rank_world, "(SAMPLER) terminated.")
