#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

from mpi4py import MPI
from surrDAMH.modules import algorithms as alg
from surrDAMH.modules import proposals
from surrDAMH.modules import communication
from surrDAMH.modules import lhs_normal as lhs
from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution
from surrDAMH.surrogates.parent import Evaluator
from typing import List
from surrDAMH.stages import Stage


def run_SAMPLER(conf: Configuration, prior: Distribution, likelihood: Distribution, list_of_stages: List[Stage], surrogate_evaluator: Evaluator | None = None):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    size_world = comm_world.Get_size()
    comm_sampler = comm_world.Split(color=0, key=rank_world)
    seed0 = max(1000, size_world)*rank_world

    commSolver = communication.SolverMPI(conf=conf)
    if conf.use_collector:
        assert conf.rank_collector is not None
        commEvaluator = communication.CommEvaluator_sampler(rank_collector=conf.rank_collector,
                                                            max_buffer_size=conf.max_buffer_size)
        commEvaluator.request_evaluator()
        commSnapshot = communication.CommSnapshot_sampler(rank_collector=conf.rank_collector,
                                                          max_sampler_isend_requests=conf.max_sampler_isend_requests)
    else:
        commEvaluator = None
        commSnapshot = None

    if conf.initial_sample_type == "lhs":
        initial_samples = lhs.lhs_normal(loc=prior.mean, scale=conf.lhs_scale, n=conf.no_samplers, seed=0)
        initial_sample = alg.Sample(parameters=initial_samples[rank_world])
    elif conf.initial_sample_type == "user_specified":
        initial_sample = alg.Sample(parameters=conf.initial_samples_distribution.rvs())
    else:
        initial_sample = alg.Sample(parameters=prior.rvs())
    print("Sampler at rank", rank_world, "- initial sample:", initial_sample, flush=True)

    my_Prop = proposals.GaussRandomWalk(no_parameters=conf.no_parameters, seed=seed0+1)
    # my_Prop = proposals.GaussRandomWalk_adaptive(no_parameters=conf.no_parameters, seed=seed0+1)
    no_stages = len(list_of_stages)
    for i, stage in enumerate(list_of_stages):
        if stage.proposal_sd is not None:  # if None, result of adaptive stage is used
            my_Prop.set_covariance(sd_or_cov=stage.proposal_sd)
        seed = seed0 + 2 + i
        if stage.send_snapshots_to_collector:
            commSnapshot_stage = commSnapshot
        else:
            commSnapshot_stage = None
        if stage.use_only_surrogate:
            commSolver_stage = commSnapshot
            # TODO: !!! requires methods set_parameters, get_observations
            # implement tool: Evaluator as Solver (maybe to the parent class)
        else:
            commSolver_stage = commSolver

        if stage.algorithm_type == 'DAMH' or stage.use_only_surrogate:
            # the stage evaluates surrogate model
            assert i > 0  # initial stage cannot use surrogate model
            assert commEvaluator is not None
            commEvaluator_stage = commEvaluator
            # if previous stage did not request updates, wait for surrogate model update
            # if list_of_stages[i-1].surrogate_model_updates is False:
            #     commEvaluator.request_evaluator()
            #     commEvaluator.get_evaluator()  # wait for evaluator
            #     if stage.surrogate_model_updates:  # DAMH-SMU
            #         commEvaluator.request_evaluator()
            # else:  # previous stage was DAMH-SMU
            #     if not stage.surrogate_model_updates:  # this stage is not DAMH-SMU
            #         commEvaluator.get_evaluator()
        else:
            commEvaluator_stage = None
        if stage.algorithm_type == 'MH':
            if stage.is_adaptive:
                stage.name = 'alg' + str(i).zfill(4) + '_MH_adaptive'
                my_Alg = alg.Algorithm_MH_adaptive(proposal=my_Prop,
                                                   commSolver=commSolver_stage,
                                                   commSnapshot=commSnapshot_stage,
                                                   commEvaluator=commEvaluator_stage,
                                                   conf=conf,
                                                   stage=stage,
                                                   prior=prior,
                                                   likelihood=likelihood,
                                                   initial_sample=initial_sample,
                                                   rank_world=rank_world,
                                                   seed=seed)
            else:
                stage.name = 'alg' + str(i).zfill(4) + 'MH'
                my_Alg = alg.Algorithm_MH(proposal=my_Prop,
                                          commSolver=commSolver_stage,
                                          commSnapshot=commSnapshot_stage,
                                          commEvaluator=commEvaluator_stage,
                                          conf=conf,
                                          stage=stage,
                                          prior=prior,
                                          likelihood=likelihood,
                                          initial_sample=initial_sample,
                                          rank_world=rank_world,
                                          seed=seed)
        elif stage.algorithm_type == 'DAMH':
            stage.name = 'alg' + str(i).zfill(4) + 'DAMH'
            my_Alg = alg.Algorithm_DAMH(proposal=my_Prop,
                                        commSolver=commSolver_stage,
                                        commSnapshot=commSnapshot_stage,
                                        commEvaluator=commEvaluator_stage,
                                        conf=conf,
                                        stage=stage,
                                        prior=prior,
                                        likelihood=likelihood,
                                        initial_sample=initial_sample,
                                        rank_world=rank_world,
                                        seed=seed)
        # print('--- SAMPLER ' + my_Alg.stage.name + ' starts ---')
        my_Alg.run()
        if stage.is_adaptive:
            sendbuf = my_Prop.sd
            recvbuf = sendbuf.copy()
            comm_sampler.Allreduce(sendbuf, recvbuf)
            my_Prop.set_covariance(sd_or_cov=recvbuf/conf.no_samplers)
        if not stage.is_excluded:
            initial_sample = my_Alg.current
        following_DAMH = [list_of_stages[j].algorithm_type == "DAMH" for j in range(i+1, no_stages)]
        following_onlySurr = [list_of_stages[j].use_only_surrogate for j in range(i+1, no_stages)]
        stages_will_use_surrogate = following_DAMH or following_onlySurr
        if commSnapshot is not None:
            if any(stages_will_use_surrogate):
                pass
            else:
                print("Requesting evaluator.", flush=True)
                commEvaluator.get_evaluator_and_terminate()
                commSnapshot.terminate()
                commSnapshot = None
        print('Stage', my_Alg.stage.name, 'at MPI rank', rank_world, 'finished - acc/rej/prerej samples:',
              my_Alg.no_accepted, my_Alg.no_rejected, my_Alg.no_prerejected, flush=True)
        comm_sampler.Barrier()
        # print("Barrier after stage", my_Alg.stage.name, "- rank", rank_world, flush=True)

    f = getattr(commSolver, "terminate", None)
    if callable(f):
        commSolver.terminate()

    f = getattr(commSnapshot, "terminate", None)
    if callable(f):
        commSnapshot.terminate()

    # f = getattr(commEvaluator, "get_evaluator_and_terminate", None)
    # if callable(f):
    #     commEvaluator.terminate()

    comm_world.Barrier()
    comm_world.Barrier()
    # commSurrogate.tag = 0
    return []
