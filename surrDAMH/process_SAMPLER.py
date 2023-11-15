#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

from mpi4py import MPI
import sys
from modules import classes_SAMPLER as cS
from modules import classes_communication
from modules import lhs_normal as LHS
from configuration import Configuration

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

comm_sampler = comm_world.Split(color=0, key=rank_world)

no_samplers, problem_path = comm_world.recv(source=MPI.ANY_SOURCE, tag=100)
comm_world.Barrier() # barrier probably not necessary when send/recv paired by tag=100
output_dir = sys.argv[1]
# data = None
# data = comm_world.bcast(data,root=MPI.ANY_SOURCE)
# no_samplers, problem_name = data
# print(rank_world,size_world,no_samplers,problem_name)
C = Configuration(no_samplers, problem_path)
seed0 = max(1000,size_world)*rank_world # TO DO seeds

my_Sol = classes_communication.Solver_MPI_collector_MPI(no_parameters=C.no_parameters, 
                              no_observations=C.no_observations, 
                              rank_solver=C.solver_parent_rank,
                              pickled_observations=C.pickled_observations) # only knows the MPI rank to communicate with
if C.use_surrogate:
    my_Surr_Solver = C.surr_solver_init(**C.surr_solver_parameters)
    my_Surr = classes_communication.Solver_local_collector_MPI(no_parameters=C.no_parameters, 
                                            no_observations=C.no_observations, 
                                            local_solver_instance=my_Surr_Solver,
                                            rank_collector=C.rank_surr_collector)
else:
    my_Surr = None
my_Prob = cS.Problem_Gauss(no_parameters=C.no_parameters,
                           noise_std=C.problem_parameters['noise_std'],
                           prior_mean=C.problem_parameters['prior_mean'], 
                           prior_std=C.problem_parameters['prior_std'],
                           no_observations=C.no_observations,
                           observations=C.problem_parameters['observations'],
                           seed=seed0,
                           name=C.problem_name)
my_Prop = cS.Proposal_GaussRandomWalk(no_parameters=C.no_parameters, seed=seed0+1)

if C.initial_sample_type == "lhs":
    initial_samples = LHS.lhs_normal(C.no_parameters,C.problem_parameters['prior_mean'],C.problem_parameters['prior_std'],C.no_samplers,0)
    initial_sample = initial_samples[rank_world]
else:
    initial_sample = None # will be set to prior mean
for i,d in enumerate(C.list_alg):
    if ("proposal_std" in d.keys()) and (d["proposal_std"] is not None):
        my_Prop.set_covariance(proposal_std = d['proposal_std'])
    seed = seed0 + 2 + i
    if d['type'] == 'MH':
        if "use_only_surrogate" in d.keys() and d["use_only_surrogate"]==True:
            Solver = my_Surr
            Surrogate = None
            surrogate_is_updated = False
        else:
            Solver = my_Sol
            Surrogate = my_Surr
            surrogate_is_updated = d['surrogate_is_updated']
        if "adaptive" in d.keys() and d["adaptive"]==True:
            target_rate = None
            if "target_rate" in d.keys():
                target_rate = d["target_rate"]
            corr_limit = None
            if "corr_limit" in d.keys():
                corr_limit = d["corr_limit"]
            sample_limit = None
            if "sample_limit" in d.keys():
                sample_limit = d["sample_limit"]
            my_Alg = cS.Algorithm_MH_adaptive(my_Prob, my_Prop,
                             Solver = Solver,
                             Surrogate = Surrogate,
                             surrogate_is_updated = surrogate_is_updated,
                             initial_sample=initial_sample,
                             max_samples=d['max_samples'],
                             max_evaluations=d['max_evaluations'],
                             time_limit=d['time_limit'],
                             target_rate=target_rate,
                             corr_limit=corr_limit,
                             sample_limit=sample_limit,
                             save_raw_data=C.save_raw_data,
                             transform_before_saving=C.transform_before_saving,
                             name='alg' + str(i).zfill(4) + 'MH_adaptive_rank' + str(rank_world),
                             seed=seed, output_dir=output_dir)
        else:
            my_Alg = cS.Algorithm_MH(my_Prob, my_Prop,
                             Solver = Solver,
                             Surrogate = Surrogate,
                             surrogate_is_updated = surrogate_is_updated,
                             initial_sample=initial_sample,
                             max_samples=d['max_samples'],
                             max_evaluations=d['max_evaluations'],
                             time_limit=d['time_limit'],
                             save_raw_data=C.save_raw_data,
                             transform_before_saving=C.transform_before_saving,
                             name='alg' + str(i).zfill(4) + 'MH_rank' + str(rank_world),
                             seed=seed, output_dir=output_dir)
    elif d['type'] == 'DAMH':
        my_Alg = cS.Algorithm_DAMH(my_Prob, my_Prop, my_Sol,
                        Surrogate = my_Surr,
                        surrogate_is_updated = d['surrogate_is_updated'],
                        initial_sample=initial_sample,
                        max_samples=d['max_samples'],
                        max_evaluations=d['max_evaluations'],
                        time_limit=d['time_limit'],
                        save_raw_data=C.save_raw_data,
                        transform_before_saving=C.transform_before_saving,
                        name='alg' + str(i).zfill(4) + 'DAMH_rank' + str(rank_world),
                        seed=seed, output_dir=output_dir)
    print('--- SAMPLER ' + my_Alg.name + ' starts ---')
    #print("RANK", rank_world, "INITIAL SAMPLE", initial_sample)
    my_Alg.run()
    #print("RANK", rank_world, "LAST SAMPLE", my_Alg.current_sample)
    if "adaptive" in d.keys() and d["adaptive"]==True:
        sendbuf = my_Prop.proposal_std
        recvbuf = sendbuf.copy()
        comm_sampler.Allreduce(sendbuf, recvbuf)
        my_Prop.set_covariance(proposal_std = recvbuf/no_samplers)
    if "excluded" in d.keys() and d["excluded"]==True:
        pass
    else:
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
