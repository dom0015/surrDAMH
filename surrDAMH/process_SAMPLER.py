#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

from mpi4py import MPI
from surrDAMH.modules import classes_SAMPLER as cS
from surrDAMH.modules import classes_communication
from surrDAMH.modules import lhs_normal as lhs
from surrDAMH.configuration import Configuration
from surrDAMH.priors.parent import Prior
from surrDAMH.likelihoods.parent import Likelihood


def run_SAMPLER(conf: Configuration, prior: Prior, likelihood: Likelihood):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    size_world = comm_world.Get_size()

    comm_sampler = comm_world.Split(color=0, key=rank_world)

    seed0 = max(1000, size_world)*rank_world  # TO DO seeds

    my_Sol = classes_communication.Solver_MPI_collector_MPI(no_parameters=conf.no_parameters,
                                                            no_observations=conf.no_observations,
                                                            rank_solver=conf.solver_parent_rank,
                                                            pickled_observations=conf.pickled_observations)  # only knows the MPI rank to communicate with
    if conf.use_surrogate:
        my_Surr = classes_communication.Solver_local_collector_MPI(rank_collector=conf.rank_collector)
    else:
        my_Surr = None
    # my_Prob = cS.Problem_Gauss(no_parameters=conf.no_parameters,
    #                            noise_std=conf.problem_parameters['noise_std'],
    #                            #    prior_mean=conf.problem_parameters['prior_mean'],
    #                            #    prior_std=conf.problem_parameters['prior_std'],
    #                            no_observations=conf.no_observations,
    #                            observations=conf.problem_parameters['observations'],
    #                            seed=seed0)
    my_Prop = cS.Proposal_GaussRandomWalk(no_parameters=conf.no_parameters, seed=seed0+1)

    if conf.initial_sample_type == "lhs":
        initial_samples = lhs.lhs_normal(conf.no_parameters, prior.mean, prior.sd_approximation, conf.no_samplers, 0)
        initial_sample = initial_samples[rank_world]
    else:
        initial_sample = None  # will be set to prior mean
    for i, d in enumerate(conf.list_alg):
        if ("proposal_std" in d.keys()) and (d["proposal_std"] is not None):
            my_Prop.set_covariance(proposal_std=d['proposal_std'])
        seed = seed0 + 2 + i
        if d['type'] == 'MH':
            if "use_only_surrogate" in d.keys() and d["use_only_surrogate"]:
                Solver = my_Surr
                Surrogate = None
                surrogate_is_updated = False
            else:
                Solver = my_Sol
                Surrogate = my_Surr
                surrogate_is_updated = d['surrogate_is_updated']
            if "adaptive" in d.keys() and d["adaptive"]:
                target_rate = None
                if "target_rate" in d.keys():
                    target_rate = d["target_rate"]
                corr_limit = None
                if "corr_limit" in d.keys():
                    corr_limit = d["corr_limit"]
                sample_limit = None
                if "sample_limit" in d.keys():
                    sample_limit = d["sample_limit"]
                my_Alg = cS.Algorithm_MH_adaptive(my_Prop,
                                                  Solver=Solver,
                                                  Surrogate=Surrogate,
                                                  conf=conf,
                                                  prior=prior,
                                                  likelihood=likelihood,
                                                  surrogate_is_updated=surrogate_is_updated,
                                                  initial_sample=initial_sample,
                                                  max_samples=d['max_samples'],
                                                  max_evaluations=d['max_evaluations'],
                                                  time_limit=d['time_limit'],
                                                  target_rate=target_rate,
                                                  corr_limit=corr_limit,
                                                  sample_limit=sample_limit,
                                                  save_raw_data=conf.save_raw_data,
                                                  transform_before_saving=conf.transform_before_saving,
                                                  name='alg' + str(i).zfill(4) + 'MH_adaptive_rank' + str(rank_world),
                                                  seed=seed, output_dir=conf.output_dir)
            else:
                my_Alg = cS.Algorithm_MH(my_Prop,
                                         Solver=Solver,
                                         Surrogate=Surrogate,
                                         conf=conf,
                                         prior=prior,
                                         likelihood=likelihood,
                                         surrogate_is_updated=surrogate_is_updated,
                                         initial_sample=initial_sample,
                                         max_samples=d['max_samples'],
                                         max_evaluations=d['max_evaluations'],
                                         time_limit=d['time_limit'],
                                         save_raw_data=conf.save_raw_data,
                                         transform_before_saving=conf.transform_before_saving,
                                         name='alg' + str(i).zfill(4) + 'MH_rank' + str(rank_world),
                                         seed=seed, output_dir=conf.output_dir)
        elif d['type'] == 'DAMH':
            my_Alg = cS.Algorithm_DAMH(my_Prop, my_Sol,
                                       Surrogate=my_Surr,
                                       conf=conf,
                                       prior=prior,
                                       likelihood=likelihood,
                                       surrogate_is_updated=d['surrogate_is_updated'],
                                       initial_sample=initial_sample,
                                       max_samples=d['max_samples'],
                                       max_evaluations=d['max_evaluations'],
                                       time_limit=d['time_limit'],
                                       save_raw_data=conf.save_raw_data,
                                       transform_before_saving=conf.transform_before_saving,
                                       name='alg' + str(i).zfill(4) + 'DAMH_rank' + str(rank_world),
                                       seed=seed, output_dir=conf.output_dir)
        print('--- SAMPLER ' + my_Alg.name + ' starts ---')
        # print("RANK", rank_world, "INITIAL SAMPLE", initial_sample)
        my_Alg.run()
        # print("RANK", rank_world, "LAST SAMPLE", my_Alg.current_sample)
        if "adaptive" in d.keys() and d["adaptive"]:
            sendbuf = my_Prop.proposal_std
            recvbuf = sendbuf.copy()
            comm_sampler.Allreduce(sendbuf, recvbuf)
            my_Prop.set_covariance(proposal_std=recvbuf/conf.no_samplers)
        if "excluded" in d.keys() and d["excluded"]:
            pass
        else:
            initial_sample = my_Alg.current_sample
        print('--- SAMPLER ' + my_Alg.name + ' --- acc/rej/prerej:', my_Alg.no_accepted, my_Alg.no_rejected, my_Alg.no_prerejected, flush=True)

    f = getattr(my_Sol, "terminate", None)
    if callable(f):
        my_Sol.terminate()

    f = getattr(my_Surr, "terminate", None)
    if callable(f):
        my_Surr.terminate()

    comm_world.Barrier()
    print("RANK", rank_world, "(SAMPLER) terminated.")
