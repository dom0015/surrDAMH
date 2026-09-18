#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

from typing import Any, List, cast

import numpy as np
from mpi4py import MPI

from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution, rvs_with_generator
from surrDAMH.modules.algorithm_interfaces_local import LocalEvaluatorProvider
from surrDAMH.modules.algorithm_interfaces_mpi import (MpiEvaluatorProvider,
                                                       MpiSnapshotSink,
                                                       MpiSolverPoolObservationProvider)
from surrDAMH.modules.communication import recv_initial_surrogate_availability
from surrDAMH.modules import algorithms as alg
from surrDAMH.modules import lhs_normal as lhs
from surrDAMH.modules.proposal_builder import build_proposal
from surrDAMH.modules.proposals import as_covariance_matrix
from surrDAMH.modules.seeds import initial_sample_seed, stage_seed0
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage, stage_name
from surrDAMH.surrogates.parent import Evaluator
from surrDAMH.modules.continuation import save_last_sample


def run_SAMPLER(conf: Configuration, prior: Distribution, likelihood: Distribution, list_of_stages: List[Stage],
                solver_instance: Solver | None, surrogate_evaluator: Evaluator | None):
    comm_world = MPI.COMM_WORLD
    rank_world = comm_world.Get_rank()
    comm_sampler = comm_world.Split(color=0, key=rank_world)

    if conf.use_solvers_pool:
        commSolver = MpiSolverPoolObservationProvider(conf=conf)  # communication among solvers
    else:
        commSolver = solver_instance

    # initialization of communicators between sampler and collector:
    if conf.use_collector:
        assert conf.rank_collector is not None, "use_collector is True, but rank_collector is None"
        commEvaluator = MpiEvaluatorProvider.from_rank_collector(
            rank_collector=conf.rank_collector,
            max_buffer_size=conf.max_buffer_size,
            request_initial_evaluator=True,
        )
        commSnapshot = MpiSnapshotSink.from_rank_collector(
            rank_collector=conf.rank_collector,
            max_sampler_isend_requests=conf.max_sampler_isend_requests,
        )
    else:
        commEvaluator = LocalEvaluatorProvider(surrogate_evaluator) if surrogate_evaluator is not None else None
        commSnapshot = None

    no_stages = len(list_of_stages)

    # choice of initial sample:
    initial_sample_generator = np.random.default_rng(initial_sample_seed(no_stages, rank_world))
    if conf.initial_sample_type == "lhs":
        initial_samples = lhs.lhs_normal(loc=prior.mean, scale=conf.lhs_scale, n=conf.no_samplers, seed=0)
        initial_sample = alg.Sample(parameters=initial_samples[rank_world])
    elif conf.initial_sample_type == "user_specified":
        assert conf.initial_samples_distribution is not None, "if initial_sample_type == 'user_specified', initial_samples_distribution must be specified"
        initial_sample = alg.Sample(parameters=rvs_with_generator(conf.initial_samples_distribution,
                                                                  initial_sample_generator))
    elif conf.initial_sample_type == "continued":
        assert conf.continued_samples is not None, "continued_samples must be populated by Configuration when initial_sample_type == 'continued'"
        initial_sample = alg.Sample(parameters=conf.continued_samples[rank_world])
    else:
        initial_sample = alg.Sample(parameters=rvs_with_generator(prior, initial_sample_generator))
    print("Sampler at rank", rank_world, "- initial sample:", initial_sample.parameters, flush=True)

    proposal_cov_adaptive = None
    # A30: True once a previous stage has written this chain's current state to its samples
    # file, so the next stage must not count that state again in its first row.
    initial_sample_is_carried_over = False

    first_stage = list_of_stages[0]
    first_stage_needs_surrogate = (first_stage.algorithm_type == "DAMH"
                                   or str(first_stage.proposal_type).startswith("Hamiltonian"))
    if conf.use_collector:
        # start-up handshake (WS8, finding 2.2), counterpart of the send in process_COLLECTOR:
        # every sampler consumes exactly one message here, whether or not it needs a surrogate.
        assert conf.rank_collector is not None
        collector_can_provide_evaluator = recv_initial_surrogate_availability(conf.rank_collector)
        if first_stage_needs_surrogate and not collector_can_provide_evaluator:
            raise RuntimeError(
                "The first stage requires a surrogate model (DAMH algorithm or Hamiltonian proposal), but the"
                " collector cannot provide one: it has no pretrained surrogate and fewer than"
                f" min_snapshots_initial={conf.min_snapshots_initial} preloaded snapshots, while new snapshots"
                " can only be produced by the samplers themselves - the run would deadlock. Start with an MH"
                " stage, pass initial_snapshots, use a pretrained/restored surrogate updater, or lower"
                " min_snapshots_initial.")

    for i, stage in enumerate(list_of_stages):
        seed0 = stage_seed0(no_stages, rank_world, i)

        # choice of proposal distribution for this stage:
        my_Prop = build_proposal(stage=stage, conf=conf, prior=prior, seed=seed0+1,
                                 prev_cov=proposal_cov_adaptive, stage_index=i)

        # choice of communicators for this stage:
        if stage.send_snapshots_to_collector:
            commSnapshot_stage = commSnapshot
        else:
            commSnapshot_stage = None
        if stage.algorithm_type == 'DAMH':  # or stage.use_only_surrogate:
            # the stage evaluates surrogate model
            # assert i > 0, "initial stage cannot use surrogate model"
            assert commEvaluator is not None
            commEvaluator_stage = commEvaluator
        else:
            commEvaluator_stage = None
        if stage.proposal_type == "Hamiltonian" or stage.proposal_type == "HamiltonianInfinite":  # TODO
            # TODO this is a temporary workaround, we need the exact model for gradients until surrogate supports them
            assert commEvaluator is not None
            commEvaluator_stage = commEvaluator
        if stage.use_only_surrogate:
            assert commEvaluator is not None, "use_only_surrogate is True but no surrogate has been constructed yet"
            assert commEvaluator.evaluator is not None, "use_only_surrogate is True but no surrogate has been constructed yet"
            commSolver_stage = commEvaluator.evaluator.as_solver()
        else:
            commSolver_stage = commSolver

        # choice of algorithm for this stage (stage_name raises for an unknown algorithm_type):
        stage.name = stage_name(stage, i)
        alg_class = alg.Algorithm_MH if stage.algorithm_type == 'MH' else alg.Algorithm_DAMH

        # run sampling algorithm:
        alg_instance = alg_class(proposal=my_Prop,
                                 observation_provider=cast(Any, commSolver_stage),
                                 snapshot_collector=cast(Any, commSnapshot_stage),
                                 evaluator_provider=cast(Any, commEvaluator_stage),
                                 conf=conf,
                                 stage=stage,
                                 prior=prior,
                                 likelihood=likelihood,
                                 initial_sample=initial_sample,
                                 rank_world=rank_world,
                                 seed=seed0+2,
                                 initial_sample_is_carried_over=initial_sample_is_carried_over)
        alg_instance.run()

        # set mean proposal covariance for next stage:
        if stage.adaptive:
            # G2: normalise to a 2-D covariance matrix on EVERY rank before the reduction, so
            # ranks whose adapt() counts straddled the adaptation period cannot disagree on
            # the buffer shape (finding 2.5). runner_local does the same normalisation so that
            # a local run still reproduces chain 0 of an MPI run.
            sendbuf = as_covariance_matrix(my_Prop.sd_or_cov, conf.no_parameters)
            recvbuf = np.empty_like(sendbuf)
            comm_sampler.Allreduce(sendbuf, recvbuf)
            proposal_cov_adaptive = recvbuf/conf.no_samplers
            print('Stage', alg_instance.stage.name, 'at MPI rank', rank_world, 'prop_cov', my_Prop.sd_or_cov)

        # set initial sample for next stage (after a use_only_surrogate stage its surrogate
        # observations are dropped so that the next stage re-evaluates it exactly, finding A11):
        if not stage.is_excluded:
            initial_sample = alg.sample_carried_to_next_stage(alg_instance.current, stage)
        # A30: whichever sample the next stage starts from -- this stage's final state, or (with
        # is_excluded) the state this stage started from -- was written by this stage's samples
        # file iff it wrote one at all; runner_local sets the same flag the same way.
        initial_sample_is_carried_over = stage.save_to_file
        # persist this stage's last sample so another experiment can continue from it
        save_last_sample(conf, stage.name, rank_world, alg_instance.current.parameters)

        # terminate communicators between sampler and collector if they will not be used later:
        following_DAMH = [list_of_stages[j].algorithm_type == "DAMH" for j in range(i+1, no_stages)]
        following_onlySurr = [list_of_stages[j].use_only_surrogate for j in range(i+1, no_stages)]
        following_hamiltonian = [list_of_stages[j].proposal_type in ("Hamiltonian", "HamiltonianInfinite") for j in range(i+1, no_stages)]
        stages_will_use_surrogate = any(following_DAMH) or any(following_onlySurr) or any(following_hamiltonian)
        if commSnapshot is not None:
            if stages_will_use_surrogate:
                pass
            else:
                assert commEvaluator is not None
                commEvaluator.get_evaluator_and_terminate()
                commSnapshot.terminate()
                commSnapshot = None

        # stage finished, wait for all samplers:
        print('Stage', alg_instance.stage.name, 'at MPI rank', rank_world, 'finished - acc/rej/prerej samples:',
              alg_instance.counter_accepted, alg_instance.counter_rejected, alg_instance.counter_prerejected, flush=True)
        comm_sampler.Barrier()
    f = getattr(commSolver, "terminate", None)
    if callable(f):
        f()
    f = getattr(commSnapshot, "terminate", None)
    if callable(f):
        f()
    comm_world.Barrier()
    comm_world.Barrier()
    return []
