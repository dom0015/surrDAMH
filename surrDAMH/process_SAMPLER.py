#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

from typing import List

from mpi4py import MPI

from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules import algorithms as alg
from surrDAMH.modules import communication
from surrDAMH.modules import lhs_normal as lhs
from surrDAMH.modules import proposals
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.parent import Evaluator


def run_SAMPLER(conf: Configuration, prior: Distribution, likelihood: Distribution, list_of_stages: List[Stage],
                solver_instance: Solver | None, surrogate_evaluator: Evaluator | None):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    comm_sampler = comm_world.Split(color=0, key=rank_world)

    if conf.use_solvers_pool:
        commSolver = communication.SolverMPI(conf=conf)  # communication among solvers
    else:
        commSolver = solver_instance

    # initialization of communicators between sampler and collector:
    if conf.use_collector:
        assert conf.rank_collector is not None, "use_collector is True, but rank_collector is None"
        commEvaluator = communication.CommEvaluator_sampler(rank_collector=conf.rank_collector,
                                                            max_buffer_size=conf.max_buffer_size)
        commEvaluator.request_evaluator()
        commSnapshot = communication.CommSnapshot_sampler(rank_collector=conf.rank_collector,
                                                          max_sampler_isend_requests=conf.max_sampler_isend_requests)
    else:
        commEvaluator = None
        commSnapshot = None

    # choice of initial sample:
    if conf.initial_sample_type == "lhs":
        initial_samples = lhs.lhs_normal(loc=prior.mean, scale=conf.lhs_scale, n=conf.no_samplers, seed=0)
        initial_sample = alg.Sample(parameters=initial_samples[rank_world])
    elif conf.initial_sample_type == "user_specified":
        assert conf.initial_samples_distribution is not None, "if initial_sample_type == 'user_specified', initial_samples_distribution must be specified"
        initial_sample = alg.Sample(parameters=conf.initial_samples_distribution.rvs())
    else:
        initial_sample = alg.Sample(parameters=prior.rvs())
    print("Sampler at rank", rank_world, "- initial sample:", initial_sample.parameters, flush=True)

    # TODO: train method for surrogate updater
    # TODO: examples with simple visualization
    # TODO: readme
    # TODO: join MH and MH_adaptive
    # TODO: user_specified proposal, předávání dat mezi nimi?
    # TODO: use pre-collected snapshots

    proposal_cov_adaptive = None
    no_stages = len(list_of_stages)
    for i, stage in enumerate(list_of_stages):
        seed0 = 10*(no_stages*rank_world + i)

        # choice of proposal distribution for this stage:
        if stage.adaptive:
            my_Prop = proposals.GaussRandomWalk_adaptive(no_parameters=conf.no_parameters, seed=seed0+1)
        else:
            my_Prop = proposals.GaussRandomWalk(no_parameters=conf.no_parameters, seed=seed0+1)
        if stage.proposal_sd_or_cov is None:  # if None, result of adaptive stage is used
            assert proposal_cov_adaptive is not None, f"proposal sd/cov not specified for stage {i}"
            my_Prop.set_covariance(sd_or_cov=proposal_cov_adaptive)
        else:
            my_Prop.set_covariance(sd_or_cov=stage.proposal_sd_or_cov)

        # choice of communicators for this stage:
        if stage.send_snapshots_to_collector:
            commSnapshot_stage = commSnapshot
        else:
            commSnapshot_stage = None
        if stage.algorithm_type == 'DAMH':  # or stage.use_only_surrogate:
            # the stage evaluates surrogate model
            assert i > 0, "initial stage cannot use surrogate model"
            assert commEvaluator is not None
            commEvaluator_stage = commEvaluator
        else:
            commEvaluator_stage = None
        if stage.use_only_surrogate:
            assert commEvaluator is not None, "use_only_surrogate is True but no surrogate has been constructed yet"
            assert commEvaluator.evaluator is not None, "use_only_surrogate is True but no surrogate has been constructed yet"
            commSolver_stage = commEvaluator.evaluator.as_solver()
            # TODO: !!! test this
        else:
            commSolver_stage = commSolver

        # choice of algorithm for this stage:
        if stage.algorithm_type == 'MH':
            alg_class = alg.Algorithm_MH
            if stage.adaptive:
                stage.name = 'alg' + str(i).zfill(4) + '_MH-adaptive'
            else:
                stage.name = 'alg' + str(i).zfill(4) + '_MH'
        elif stage.algorithm_type == 'DAMH':
            alg_class = alg.Algorithm_DAMH
            if stage.surrogate_model_updates:
                stage.name = 'alg' + str(i).zfill(4) + '_DAMH-SMU'
            else:
                stage.name = 'alg' + str(i).zfill(4) + '_DAMH'

        # run sampling algorithm:
        alg_instance = alg_class(proposal=my_Prop,
                                 commSolver=commSolver_stage,
                                 commSnapshot=commSnapshot_stage,
                                 commEvaluator=commEvaluator_stage,
                                 conf=conf,
                                 stage=stage,
                                 prior=prior,
                                 likelihood=likelihood,
                                 initial_sample=initial_sample,
                                 rank_world=rank_world,
                                 seed=seed0+2)
        alg_instance.run()

        # set mean proposal covariance for next stage:
        if stage.adaptive:
            sendbuf = my_Prop.sd
            recvbuf = sendbuf.copy()
            comm_sampler.Allreduce(sendbuf, recvbuf)
            proposal_cov_adaptive = recvbuf/conf.no_samplers

        # set initial sample for next stage:
        if not stage.is_excluded:
            initial_sample = alg_instance.current

        # terminate communicators between sampler and collector if they will not be used later:
        following_DAMH = [list_of_stages[j].algorithm_type == "DAMH" for j in range(i+1, no_stages)]
        following_onlySurr = [list_of_stages[j].use_only_surrogate for j in range(i+1, no_stages)]
        stages_will_use_surrogate = following_DAMH or following_onlySurr
        if commSnapshot is not None:
            if any(stages_will_use_surrogate):
                pass
            else:
                assert commEvaluator is not None
                commEvaluator.get_evaluator_and_terminate()
                commSnapshot.terminate()
                commSnapshot = None

        # stage finished, wait for all samplers:
        print('Stage', alg_instance.stage.name, 'at MPI rank', rank_world, 'finished - acc/rej/prerej samples:',
              alg_instance.no_accepted, alg_instance.no_rejected, alg_instance.no_prerejected, flush=True)
        comm_sampler.Barrier()
    f = getattr(commSolver, "terminate", None)
    if callable(f):
        commSolver.terminate()
    f = getattr(commSnapshot, "terminate", None)
    if callable(f):
        assert commSnapshot is not None
        commSnapshot.terminate()
    comm_world.Barrier()
    comm_world.Barrier()
    return []
