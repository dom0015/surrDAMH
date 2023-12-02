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
from typing import List
from surrDAMH.stages import Stage


def run_SAMPLER(conf: Configuration, prior: Prior, likelihood: Likelihood, list_of_stages: List[Stage]):
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
    my_Prop = cS.Proposal_GaussRandomWalk(no_parameters=conf.no_parameters, seed=seed0+1)

    if conf.initial_sample_type == "lhs":
        initial_samples = lhs.lhs_normal(conf.no_parameters, prior.mean, prior.sd_approximation, conf.no_samplers, 0)
        initial_sample = initial_samples[rank_world]
    else:
        initial_sample = prior.mean
    # for i, d in enumerate(conf.list_alg):
    for i, stage in enumerate(list_of_stages):
        if stage.proposal_sd is not None:  # if None, result of adaptive stage is used
            my_Prop.set_covariance(proposal_sd=stage.proposal_sd)
        seed = seed0 + 2 + i
        # TODO: nedelat nastaveni typu promenna = stage.promenna, poslat celou stage
        if stage.algorithm_type == 'MH':
            if stage.use_only_surrogate:
                Solver = my_Surr
                Surrogate = None
                surrogate_is_updated = False
            else:
                Solver = my_Sol
                Surrogate = my_Surr
                surrogate_is_updated = stage.surrogate_is_updated
            if stage.is_adaptive:
                my_Alg = cS.Algorithm_MH_adaptive(Proposal=my_Prop,
                                                  Solver=Solver,
                                                  Surrogate=Surrogate,
                                                  conf=conf,
                                                  stage=stage,
                                                  prior=prior,
                                                  likelihood=likelihood,
                                                  initial_sample=initial_sample,
                                                  name='alg' + str(i).zfill(4) + 'MH_adaptive_rank' + str(rank_world),
                                                  seed=seed)
            else:
                my_Alg = cS.Algorithm_MH(Proposal=my_Prop,
                                         Solver=Solver,
                                         Surrogate=Surrogate,
                                         conf=conf,
                                         stage=stage,
                                         prior=prior,
                                         likelihood=likelihood,
                                         initial_sample=initial_sample,
                                         name='alg' + str(i).zfill(4) + 'MH_rank' + str(rank_world),
                                         seed=seed)
        elif stage.algorithm_type == 'DAMH':
            my_Alg = cS.Algorithm_DAMH(Proposal=my_Prop,
                                       Solver=my_Sol,
                                       Surrogate=my_Surr,
                                       conf=conf,
                                       stage=stage,
                                       prior=prior,
                                       likelihood=likelihood,
                                       initial_sample=initial_sample,
                                       name='alg' + str(i).zfill(4) + 'DAMH_rank' + str(rank_world),
                                       seed=seed)
        print('--- SAMPLER ' + my_Alg.name + ' starts ---')
        # print("RANK", rank_world, "INITIAL SAMPLE", initial_sample)
        my_Alg.run()
        # print("RANK", rank_world, "LAST SAMPLE", my_Alg.current_sample)
        if stage.is_adaptive:
            sendbuf = my_Prop.proposal_std
            recvbuf = sendbuf.copy()
            comm_sampler.Allreduce(sendbuf, recvbuf)
            my_Prop.set_covariance(proposal_sd=recvbuf/conf.no_samplers)
        if not stage.is_excluded:
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
