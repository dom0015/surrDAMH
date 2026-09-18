#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone, single-chain sampling runner (no MPI orchestration).

``run_local()`` performs the same per-stage loop as
``surrDAMH.process_SAMPLER.run_SAMPLER``, but wires the in-process adapters from
``surrDAMH.modules.algorithm_interfaces_local`` instead of the MPI ones:

- the forward model is called directly (``LocalSolverAdapter``),
- snapshots are consumed in process (``LocalSurrogateManager``) instead of being
  sent to a collector process,
- surrogate evaluators are handed over in memory.

Single-chain semantics: the local chain corresponds to ``rank_world = 0`` of an
MPI run, and the proposal / algorithm seeds are computed with the same formula
(``seed0 = 10*(no_stages*rank_world + i)`` with ``rank_world = 0``), so a local
run reproduces chain 0 of an MPI run with the same configuration and the same
initial sample.

Continuation works in both directions: every stage's last sample is written to
``sampling_output/last_sample/<stage>/rank0000.npz`` like ``run_SAMPLER`` does, and
``initial_sample_type="continued"`` reads chain 0 of the source run.

Not covered here (MPI / collector territory, unchanged):
- multi-chain reductions (adaptive covariances are taken from the single chain),
- solver pools and spawned solver processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, cast

import numpy as np

from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution, rvs_with_generator
from surrDAMH.modules import algorithms as alg
from surrDAMH.modules import lhs_normal as lhs
from surrDAMH.modules.algorithm_interfaces_local import (LocalEvaluatorProvider,
                                                         LocalSolverAdapter,
                                                         LocalSurrogateManager)
from surrDAMH.modules.continuation import save_last_sample
from surrDAMH.modules.manifest import (build_run_manifest, finalize_run_manifest,
                                       write_run_manifest)
from surrDAMH.modules.proposal_builder import build_proposal
from surrDAMH.modules.proposals import as_covariance_matrix
from surrDAMH.modules.seeds import initial_sample_seed, stage_seed0
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage, stage_name
from surrDAMH.surrogates.parent import (Evaluator, Updater,
                                        apply_output_normalization_from_likelihood)

LOCAL_RANK_WORLD = 0  # a local run behaves like chain 0 of an MPI run


@dataclass
class StageResult:
    """Outcome of one sampling stage of a local run."""

    name: str
    stage: Stage
    current: alg.Sample  # last state of the chain in this stage
    counter_accepted: int
    counter_rejected: int
    counter_prerejected: int


@dataclass
class SamplingResult:
    """Outcome of a local sampling run (one chain, all stages in order)."""

    stage_results: List[StageResult] = field(default_factory=list)

    @property
    def final_sample(self) -> alg.Sample | None:
        """Last state of the chain after the last stage."""
        if not self.stage_results:
            return None
        return self.stage_results[-1].current

    @property
    def by_stage_name(self) -> dict[str, StageResult]:
        """Stage results keyed by the stage name assigned by the runner."""
        return {result.name: result for result in self.stage_results}

    def __len__(self) -> int:
        return len(self.stage_results)

    def __getitem__(self, index: int) -> StageResult:
        return self.stage_results[index]


def _get_initial_sample(conf: Configuration, prior: Distribution, no_stages: int) -> alg.Sample:
    """Initial sample of the local chain; mirrors ``run_SAMPLER`` for rank 0.

    The ``rvs``-based branches draw from the same per-rank generator as ``run_SAMPLER``
    (``initial_sample_seed(no_stages, 0)``, G4), so chain-0 reproduction also holds for
    ``initial_sample_type="prior"``/``"user_specified"``.
    """
    generator = np.random.default_rng(initial_sample_seed(no_stages, LOCAL_RANK_WORLD))
    if conf.initial_sample_type == "lhs":
        initial_samples = lhs.lhs_normal(loc=prior.mean, scale=conf.lhs_scale, n=conf.no_samplers, seed=0)
        return alg.Sample(parameters=initial_samples[LOCAL_RANK_WORLD])
    if conf.initial_sample_type == "user_specified":
        assert conf.initial_samples_distribution is not None, "if initial_sample_type == 'user_specified', initial_samples_distribution must be specified"
        return alg.Sample(parameters=rvs_with_generator(conf.initial_samples_distribution, generator))
    if conf.initial_sample_type == "continued":
        assert conf.continued_samples is not None, "continued_samples must be populated by Configuration when initial_sample_type == 'continued'"
        return alg.Sample(parameters=conf.continued_samples[LOCAL_RANK_WORLD])
    return alg.Sample(parameters=rvs_with_generator(prior, generator))


def run_local(conf: Configuration, prior: Distribution, likelihood: Distribution, stages: List[Stage],
              solver: Solver, updater: Updater | None = None,
              evaluator: Evaluator | None = None) -> SamplingResult:
    """
    Run all sampling stages of a single chain in one process (no MPI roles).

    Args:
        conf: configuration; ``use_collector`` and ``use_solvers_pool`` must be
            ``False``, because the local runner replaces both roles.
        prior: prior distribution.
        likelihood: likelihood.
        stages: sampling stages, executed in the given order.
        solver: forward model, called directly in this process.
        updater: optional surrogate updater; if given, snapshots are used to
            train the surrogate in process (``LocalSurrogateManager``) and the
            resulting evaluator is handed to DAMH stages.
        evaluator: optional fixed surrogate evaluator, used when no ``updater``
            is given (read-only, never retrained).

    Returns:
        ``SamplingResult`` with one ``StageResult`` per stage.
    """
    if conf.use_collector or conf.use_solvers_pool:
        raise ValueError("run_local() requires a configuration with use_collector=False and use_solvers_pool=False "
                         "(the local runner replaces the collector and the solvers pool)")
    if updater is not None and evaluator is not None:
        raise ValueError("run_local() accepts either 'updater' (surrogate is trained in process) or 'evaluator' "
                         "(fixed surrogate), not both")

    # effective settings, once (WS5); same block as SamplingFramework.run() prints on rank 0
    print(conf.describe(), flush=True)
    for i, stage in enumerate(stages):
        print(stage.describe(i), flush=True)

    observation_provider = LocalSolverAdapter(solver)

    surrogate_manager: LocalSurrogateManager | None = None
    evaluator_provider: LocalSurrogateManager | LocalEvaluatorProvider | None = None
    if updater is not None:
        updater.set_use_gradients(conf.use_surrogate_gradients)
        # WS6: same hook as SamplingFramework -- an updater configured with
        # output_normalization="likelihood" takes its statistics from the likelihood here,
        # before any snapshot is added.
        apply_output_normalization_from_likelihood(updater, likelihood)
        surrogate_manager = LocalSurrogateManager(
            updater=updater,
            min_snapshots_initial=conf.min_snapshots_initial,
            min_snapshots_to_update=conf.min_snapshots_to_update,
            initial_snapshots=updater.get_initial_snapshots(),
        )
        evaluator_provider = surrogate_manager
    elif evaluator is not None:
        evaluator_provider = LocalEvaluatorProvider(evaluator)

    no_stages = len(stages)
    initial_sample = _get_initial_sample(conf, prior, no_stages)
    print("Local sampler - initial sample:", initial_sample.parameters, flush=True)

    result = SamplingResult()
    proposal_cov_adaptive = None
    # A30 (see AlgorithmBase's docstring); set exactly as in process_SAMPLER
    initial_sample_is_carried_over = False

    for i, stage in enumerate(stages):
        seed0 = stage_seed0(no_stages, LOCAL_RANK_WORLD, i)

        proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=seed0+1,
                                  prev_cov=proposal_cov_adaptive, stage_index=i)

        # choice of services for this stage (mirrors process_SAMPLER):
        snapshot_collector_stage = surrogate_manager if stage.send_snapshots_to_collector else None
        evaluator_provider_stage = None
        if stage.algorithm_type == "DAMH" or stage.proposal_type in ("Hamiltonian", "HamiltonianInfinite"):
            assert evaluator_provider is not None, ("stage requires a surrogate model; pass 'updater' or 'evaluator' "
                                                    "to run_local()")
            evaluator_provider_stage = evaluator_provider
        if stage.use_only_surrogate:
            assert evaluator_provider is not None, "use_only_surrogate is True but no surrogate has been constructed yet"
            assert evaluator_provider.evaluator is not None, "use_only_surrogate is True but no surrogate has been constructed yet"
            observation_provider_stage: Any = LocalSolverAdapter(evaluator_provider.evaluator.as_solver())
        else:
            observation_provider_stage = observation_provider

        stage.name = stage_name(stage, i)
        alg_class = alg.Algorithm_MH if stage.algorithm_type == "MH" else alg.Algorithm_DAMH

        alg_instance = alg_class(proposal=proposal,
                                 observation_provider=cast(Any, observation_provider_stage),
                                 snapshot_collector=cast(Any, snapshot_collector_stage),
                                 evaluator_provider=cast(Any, evaluator_provider_stage),
                                 conf=conf,
                                 stage=stage,
                                 prior=prior,
                                 likelihood=likelihood,
                                 initial_sample=initial_sample,
                                 rank_world=LOCAL_RANK_WORLD,
                                 seed=seed0+2,
                                 initial_sample_is_carried_over=initial_sample_is_carried_over)
        alg_instance.run()

        # proposal covariance for the next stage (single chain, no reduction needed, but the
        # same G2 normalisation as process_SAMPLER so that the carried covariance has the same
        # shape -- and hence the same RNG stream in the next stage -- as MPI chain 0):
        if stage.adaptive:
            proposal_cov_adaptive = as_covariance_matrix(proposal.sd_or_cov, conf.no_parameters)

        # initial sample for the next stage (after a use_only_surrogate stage its surrogate
        # observations are dropped so that the next stage re-evaluates it exactly, finding A11):
        if not stage.is_excluded:
            initial_sample = alg.sample_carried_to_next_stage(alg_instance.current, stage)
        # A30: the sample handed to the next stage was written by this stage's samples file iff
        # this stage wrote one (same rule as process_SAMPLER, including the is_excluded case)
        initial_sample_is_carried_over = stage.save_to_file
        # persist this stage's last sample so another run (local or MPI) can continue from it,
        # exactly as run_SAMPLER does for every rank:
        save_last_sample(conf, stage.name, LOCAL_RANK_WORLD, alg_instance.current.parameters)

        assert stage.name is not None
        result.stage_results.append(StageResult(name=stage.name,
                                                stage=stage,
                                                current=alg_instance.current,
                                                counter_accepted=alg_instance.counter_accepted,
                                                counter_rejected=alg_instance.counter_rejected,
                                                counter_prerejected=alg_instance.counter_prerejected))
        print("Stage", stage.name, "finished - acc/rej/prerej samples:", alg_instance.counter_accepted,
              alg_instance.counter_rejected, alg_instance.counter_prerejected, flush=True)

    if evaluator_provider is not None:
        evaluator_provider.close()

    # run manifest (WS4, library_notes/09_improvement_plan.md §0 principle 4): written after the
    # stage loop so stage.name (assigned above) is already correct; must never abort a run, so
    # any failure here is a printed warning, not an exception.
    try:
        manifest = build_run_manifest(
            conf, stages, prior, likelihood, runner="local",
            solver_instance=solver, surrogate_updater=updater, surrogate_evaluator=evaluator,
            mpi_layout=None, use_surrogate_gradients_requested=conf.use_surrogate_gradients)
        write_run_manifest(conf.output_dir, manifest)
        stage_counters = [{"name": r.name, "counter_accepted": r.counter_accepted,
                           "counter_rejected": r.counter_rejected,
                           "counter_prerejected": r.counter_prerejected} for r in result.stage_results]
        finalize_run_manifest(conf.output_dir, extra={"stage_counters": stage_counters})
    except Exception as exc:
        print(f"WARNING: failed to write/finalize run manifest: {exc}", flush=True)

    return result


__all__ = ["run_local", "SamplingResult", "StageResult"]
