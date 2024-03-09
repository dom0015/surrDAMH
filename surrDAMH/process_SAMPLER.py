#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

from mpi4py import MPI
from surrDAMH.modules import algorithms as alg
from surrDAMH.modules import communication
from surrDAMH.modules import lhs_normal as lhs
from surrDAMH.configuration import Configuration
from surrDAMH.priors.parent import Prior
from surrDAMH.likelihoods.parent import Likelihood
from surrDAMH.surrogates.parent import Evaluator
from typing import List
from surrDAMH.stages import Stage
import numpy as np


def run_SAMPLER(conf: Configuration, prior: Prior, likelihood: Likelihood, list_of_stages: List[Stage], surrogate_evaluator: Evaluator = None):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    size_world = comm_world.Get_size()
    comm_sampler = comm_world.Split(color=0, key=rank_world)

    commSolver = communication.SolverMPI(conf=conf)  # only knows the MPI rank to communicate with
    commSurrogate = communication.SurrogateLocal_CollectorMPI(conf=conf, evaluator=surrogate_evaluator)

    if conf.initial_sample_type == "lhs":
        initial_samples = lhs.lhs_normal(conf.no_parameters, prior.mean, prior.sd_approximation, conf.no_samplers, 0)
        initial_sample = initial_samples[rank_world]
    elif conf.initial_sample_type == "user_specified":
        initial_sample = np.random.normal(conf.initial_sample, prior.sd_approximation*0.01)
        print("INITIAL SAMPLE - rank", rank_world, "=", initial_sample, flush=True)
    else:
        initial_sample = np.random.normal(prior.mean, prior.sd_approximation*0.01)
        # prior.mean TODO: problems with RBF surrogate - singular matrix

    seed0 = max(1000, size_world)*rank_world  # TO DO check seeds
    my_Prop = alg.Proposal_GaussRandomWalk(no_parameters=conf.no_parameters, seed=seed0+1)
    for i, stage in enumerate(list_of_stages):
        if stage.proposal_sd is not None:  # if None, result of adaptive stage is used
            my_Prop.set_covariance(proposal_sd=stage.proposal_sd)
        seed = seed0 + 2 + i
        if stage.algorithm_type == 'MH':
            if stage.use_only_surrogate:
                Solver = commSurrogate
                Surrogate = None
            else:
                Solver = commSolver
                Surrogate = commSurrogate
            if stage.is_adaptive:
                stage.name = 'alg' + str(i).zfill(4) + '_MH_adaptive'
                my_Alg = alg.Algorithm_MH_adaptive(proposal=my_Prop,
                                                   commSolver=Solver,
                                                   commSurrogate=Surrogate,
                                                   conf=conf,
                                                   stage=stage,
                                                   prior=prior,
                                                   likelihood=likelihood,
                                                   initial_sample=initial_sample,
                                                   seed=seed)
            else:
                stage.name = 'alg' + str(i).zfill(4) + 'MH'
                my_Alg = alg.Algorithm_MH(proposal=my_Prop,
                                          commSolver=Solver,
                                          commSurrogate=Surrogate,
                                          conf=conf,
                                          stage=stage,
                                          prior=prior,
                                          likelihood=likelihood,
                                          initial_sample=initial_sample,
                                          seed=seed)
        elif stage.algorithm_type == 'DAMH':
            stage.name = 'alg' + str(i).zfill(4) + 'DAMH'
            my_Alg = alg.Algorithm_DAMH(proposal=my_Prop,
                                        commSolver=commSolver,
                                        commSurrogate=commSurrogate,
                                        conf=conf,
                                        stage=stage,
                                        prior=prior,
                                        likelihood=likelihood,
                                        initial_sample=initial_sample,
                                        seed=seed)
        # print('--- SAMPLER ' + my_Alg.stage.name + ' starts ---')
        my_Alg.run()
        if stage.is_adaptive:
            sendbuf = my_Prop.sd
            recvbuf = sendbuf.copy()
            comm_sampler.Allreduce(sendbuf, recvbuf)
            my_Prop.set_covariance(proposal_sd=recvbuf/conf.no_samplers)
        if not stage.is_excluded:
            initial_sample = my_Alg.current_sample
        print('--- SAMPLER ' + my_Alg.stage.name + ' --- acc/rej/prerej:', my_Alg.no_accepted, my_Alg.no_rejected, my_Alg.no_prerejected, flush=True)

    f = getattr(commSolver, "terminate", None)
    if callable(f):
        commSolver.terminate()

    f = getattr(commSurrogate, "terminate", None)
    if callable(f):
        commSurrogate.terminate()

    comm_world.Barrier()
    comm_world.Barrier()
    commSurrogate.tag = 0
    return []
