#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 items 11, 13, 14, 15 (library_notes/07_testing_plan.md).

Item 12 no longer exists: the state-dependent surrogate shift it was about was removed from
the library (finding 1.1, 2026-09-18), so there is nothing left to test.

Uses ``surrDAMH.modules.algorithm_interfaces_local`` (LocalSolverAdapter,
LocalEvaluatorProvider, InMemorySnapshotSink) to build ``Algorithm_MH`` / ``Algorithm_DAMH``
instances directly, without any MPI machinery.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import pytest

from surrDAMH.modules.algorithm_interfaces_local import (
    InMemorySnapshotSink,
    LocalEvaluatorProvider,
    LocalSolverAdapter,
)
from surrDAMH.modules.algorithms import (SOLVER_TAG_NONFINITE_PROPOSAL, Algorithm_DAMH,
                                         Algorithm_MH, Sample)
from surrDAMH.modules.proposals import (GaussRandomWalk, GaussRandomWalk_adaptive, Hamiltonian,
                                        Proposal)
from surrDAMH.proposals import PCN, RandomWalk
from surrDAMH.proposals import Hamiltonian as HamiltonianSpec
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.parent import Evaluator


@dataclass
class MinimalAlgorithmConfig:
    """Minimal stand-in satisfying ``AlgorithmConfiguration``
    (surrDAMH/modules/algorithm_interfaces.py) without importing mpi4py-flavoured
    ``surrDAMH.configuration.Configuration``."""

    output_dir: str
    # no_parameters/no_observations fix the width of the format-v2 CSV column blocks
    no_parameters: int = 2
    no_observations: int = 2
    transform_before_saving: bool = False
    save_snapshots_to_file: bool = False
    transform_before_surrogate: bool = False
    use_surrogate_gradients: bool = False


class FixedLinearEvaluator(Evaluator):
    """A deterministic, non-updating surrogate: G~(x) = A' @ x."""

    def __init__(self, A: np.ndarray) -> None:
        self.A = np.asarray(A, dtype=float)

    def __call__(self, datapoints: np.ndarray) -> np.ndarray:
        return datapoints @ self.A.T


class FailingSolver(Solver):
    """Always returns a negative solver_tag (zero-filled observations)."""

    def __init__(self, no_observations: int) -> None:
        self.no_observations = no_observations

    def set_parameters(self, parameters):
        self.parameters = parameters

    def get_observations(self):
        return np.zeros(self.no_observations), -1


class FailSecondCallSolver(Solver):
    """Returns a valid observation the first time (for the initial sample), then a failing
    (tag=-1) observation on every later call (for a proposed sample)."""

    def __init__(self, A: np.ndarray) -> None:
        self.A = np.asarray(A, dtype=float)
        self._calls = 0

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        self._calls += 1
        if self._calls == 1:
            return self.A @ self.parameters, 0
        return np.zeros(self.A.shape[0]), -1


@pytest.mark.unit
@pytest.mark.parametrize("subchain_length", [1, 5, 20])
def test_damh_telescoping_invariant_fixed_surrogate(linear_gaussian_problem, tmp_path, subchain_length):
    """
    Item 11: with a FIXED (non-updating) surrogate, ``correction_log_ratio`` returned by
    ``_propose_new_sample_using_subchain`` telescopes exactly to
    ``subchain_current.log_likelihood_approx - initial.log_likelihood_approx``
    (the accumulated surrogate log-likelihood ratios of the accepted sub-chain steps), for
    subchain_length in {1, 5, 20}. See the derivation comment above
    ``Algorithm_DAMH.run`` in modules/algorithms.py.
    """
    problem = linear_gaussian_problem
    wrong_but_fixed_A = 1.3 * problem.A  # deliberately wrong surrogate, but fixed during the sub-chain

    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path))
    stage = Stage(
        algorithm="DAMH",
        proposal=RandomWalk(scale=0.5),
        subchain_length=subchain_length,
        surrogate_model_updates=False,  # keep the surrogate fixed for the whole sub-chain
        max_samples=1,
        max_evaluations=1,
        name="test_damh_telescoping",  # normally set by process_SAMPLER; not exercised here
    )
    proposal = GaussRandomWalk(no_parameters=problem.no_parameters, sd_or_cov=0.5, seed=3)
    initial_sample = Sample(parameters=np.array([0.1, -0.2]))
    evaluator_provider = LocalEvaluatorProvider(evaluator=FixedLinearEvaluator(wrong_but_fixed_A))

    algo = Algorithm_DAMH(
        stage=stage,
        proposal=proposal,
        initial_sample=initial_sample,
        rank_world=0,
        conf=conf,
        prior=problem.prior,
        likelihood=problem.likelihood,
        observation_provider=LocalSolverAdapter(problem.solver),
        snapshot_collector=None,
        evaluator_provider=evaluator_provider,
        seed=1,
    )

    initial_log_likelihood_approx = algo.current.log_likelihood_approx
    assert initial_log_likelihood_approx is not None

    (subchain_current, counter_subchain_accepted, correction_log_ratio,
     subchain_log_acceptance_probabilities) = algo._propose_new_sample_using_subchain()

    expected = subchain_current.log_likelihood_approx - initial_log_likelihood_approx
    assert correction_log_ratio == pytest.approx(expected, abs=1e-9)
    # one log acceptance probability per sub-chain step, in order (2026-09-20)
    assert len(subchain_log_acceptance_probabilities) == subchain_length


@pytest.mark.unit
def test_damh_adaptive_proposal_sees_prerejected_iterations(linear_gaussian_problem, tmp_path):
    """
    Recommendation 1 of library_notes/16_adaptivity_options_research_2026-09-20.md (2026-09-20):
    ``Algorithm_DAMH.run`` calls ``Proposal.adapt`` on EVERY outer iteration. On a pre-rejected
    iteration (the sub-chain never moved, so no exact evaluation happened) it passes
    ``log_acceptance_probability=-inf`` -- acceptance probability 0 -- and the current state as
    ``proposed_sample``. Before, pre-rejected iterations were skipped, so the adapted rate was the
    conditional second-stage acceptance (-> 1 as the surrogate improves) and the proposal scale
    diverged (`15` §3.3; prototype `16` §4.2: 17/18 -> 0/18 divergences). Checked to fail with
    the ``adapt`` call in the pre-rejected branch removed (then ``len(calls) < n_outer``).
    """
    problem = linear_gaussian_problem
    calls: list[dict] = []

    class RecordingAdaptive(GaussRandomWalk_adaptive):
        def adapt(self, proposed_sample, log_acceptance_probability, current_sample,
                  subchain_log_acceptance_probabilities=None):
            calls.append(dict(
                proposed=np.array(proposed_sample, dtype=float, copy=True),
                log_acceptance_probability=float(log_acceptance_probability),
                current=np.array(current_sample, dtype=float, copy=True),
                subchain=None if subchain_log_acceptance_probabilities is None
                else list(subchain_log_acceptance_probabilities),
            ))
            super().adapt(proposed_sample=proposed_sample,
                          log_acceptance_probability=log_acceptance_probability,
                          current_sample=current_sample,
                          subchain_log_acceptance_probabilities=subchain_log_acceptance_probabilities)

    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path))
    subchain_length = 3
    stage = Stage(
        algorithm="DAMH",
        # deliberately wide, so the surrogate sub-chain rejects often
        proposal=RandomWalk(scale=2.0, adaptive=True),
        subchain_length=subchain_length,
        surrogate_model_updates=False,
        max_samples=400,
        max_evaluations=400,
        name="test_damh_adapt_prerejected",
    )
    proposal = RecordingAdaptive(no_parameters=problem.no_parameters, sd_or_cov=2.0, seed=3)
    initial_sample = Sample(parameters=np.array([0.1, -0.2]))
    evaluator_provider = LocalEvaluatorProvider(evaluator=FixedLinearEvaluator(1.3 * problem.A))

    algo = Algorithm_DAMH(
        stage=stage,
        proposal=proposal,
        initial_sample=initial_sample,
        rank_world=0,
        conf=conf,
        prior=problem.prior,
        likelihood=problem.likelihood,
        observation_provider=LocalSolverAdapter(problem.solver),
        snapshot_collector=None,
        evaluator_provider=evaluator_provider,
        seed=1,
    )
    algo.run()

    n_outer = algo.counter_accepted + algo.counter_rejected + algo.counter_prerejected
    assert algo.counter_prerejected > 0, "the test needs pre-rejected iterations to be meaningful"
    assert algo.counter_accepted + algo.counter_rejected > 0
    # one adapt() call per outer iteration, pre-rejected ones scored as acceptance probability 0
    assert len(calls) == n_outer
    prerejected_calls = [c for c in calls if c["log_acceptance_probability"] == -np.inf]
    assert len(prerejected_calls) == algo.counter_prerejected
    assert all(np.isfinite(c["log_acceptance_probability"]) for c in calls
               if c["log_acceptance_probability"] != -np.inf)
    # the adaptive proposal's own counter reflects it: one Robbins-Monro step per outer iteration
    assert proposal.n == n_outer
    assert np.all(np.isfinite(proposal.sd_or_cov))

    # 2026-09-20: adapt() is called AFTER the outer decision and is handed the POST-decision
    # chain state, plus the sub-chain's own per-step log acceptance probabilities.
    for call in calls:
        assert call["subchain"] is not None and len(call["subchain"]) == subchain_length
    accepted_calls = [c for c in calls if np.allclose(c["current"], c["proposed"])]
    assert accepted_calls, "no iteration was accepted, so the post-decision state is not exercised"
    # the last call's post-decision state must be the chain's final state
    np.testing.assert_allclose(calls[-1]["current"], algo.current.parameters)
    # a pre-rejected iteration passes the current state as BOTH proposal and post-decision state
    for call in prerejected_calls:
        np.testing.assert_allclose(call["current"], call["proposed"])


@pytest.mark.unit
def test_initial_sample_with_negative_solver_tag_raises_runtime_error(linear_gaussian_problem, tmp_path):
    """
    Item 13 (initial failure): after WS0 item B4, an initial sample whose solver returns a
    negative tag (non-finite log-likelihood) raises RuntimeError at construction
    (``AlgorithmBase._prepare_run``), fail-fast instead of silently getting stuck for the
    whole stage.
    """
    problem = linear_gaussian_problem
    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path))
    stage = Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_samples=1, max_evaluations=1, name="test_initial_failure")
    proposal = GaussRandomWalk(no_parameters=problem.no_parameters, sd_or_cov=0.5, seed=3)
    initial_sample = Sample(parameters=np.array([0.1, -0.2]))
    rank_world = 2

    with pytest.raises(RuntimeError, match=r"rank 2.*solver_tag=-1"):
        Algorithm_MH(
            stage=stage,
            proposal=proposal,
            initial_sample=initial_sample,
            rank_world=rank_world,
            conf=conf,
            prior=problem.prior,
            likelihood=problem.likelihood,
            observation_provider=LocalSolverAdapter(FailingSolver(problem.no_observations)),
            snapshot_collector=None,
            evaluator_provider=None,
            seed=1,
        )


@pytest.mark.unit
def test_failed_proposed_sample_not_sent_to_snapshot_collector(linear_gaussian_problem, tmp_path):
    """
    Item 13 (mid-chain failure): a failed *proposed* sample (solver_tag < 0) must not reach
    the snapshot collector, even though it is rejected -- WS0 item C2's fix to
    ``_handle_rejection`` (tests ``self.proposed.solver_tag``, not ``self.current.solver_tag``).
    A normal (tag=0) rejection, by contrast, IS still forwarded (with weight=0) as a
    regression guard: this is what a pre-C2 build would have gotten backwards.
    """
    problem = linear_gaussian_problem
    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path))
    stage = Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_samples=1, max_evaluations=1)
    proposal = GaussRandomWalk(no_parameters=problem.no_parameters, sd_or_cov=0.5, seed=3)
    initial_sample = Sample(parameters=np.array([0.1, -0.2]))
    sink = InMemorySnapshotSink()

    algo = Algorithm_MH(
        stage=stage,
        proposal=proposal,
        initial_sample=initial_sample,
        rank_world=0,
        conf=conf,
        prior=problem.prior,
        likelihood=problem.likelihood,
        observation_provider=LocalSolverAdapter(FailSecondCallSolver(problem.A)),
        snapshot_collector=sink,
        evaluator_provider=None,
        seed=1,
    )

    # Simulate one MH step whose PROPOSED sample fails (FailSecondCallSolver fails on its 2nd call).
    algo.proposed = algo._propose_new_sample(algo.current.parameters)
    algo._evaluate_proposed_sample()
    assert algo.proposed.solver_tag < 0

    algo._handle_rejection()
    assert len(sink) == 0, "a failed proposed sample (solver_tag < 0) must not reach the snapshot collector"

    # Positive control: a normal (tag=0) rejection IS forwarded (regression guard for C2).
    algo.proposed = Sample(parameters=algo.current.parameters.copy(), observations=problem.A @ algo.current.parameters,
                           log_likelihood=-np.inf, log_prior=0.0, solver_tag=0)
    algo._handle_rejection()
    assert len(sink) == 1, "an ordinary (tag=0) rejection is expected to still be forwarded to the collector"


@pytest.mark.unit
def test_mh_weight_column_bookkeeping(linear_gaussian_problem, tmp_path):
    """
    Item 14: after a short MH run, the sum of the ``samples`` CSV's weight column (column 0,
    written by ``AlgorithmBase._record_current_sample``) equals
    ``counter_accepted + counter_rejected + counter_prerejected + 1`` (the final row written
    by ``_finalize_run``). Runs against the real CSV output (``SamplingOutputMonitor`` needs
    no MPI, only a local ``output_dir``).

    This is the FIRST stage of a run (``initial_sample_is_carried_over`` defaults to False),
    so the ``+1`` survives A30 unchanged: the initial state is counted here because no
    previous stage wrote it. The ``+0`` case is covered by
    ``tests/test_runner_local.py::test_two_stage_weights_count_each_state_once``.
    """
    problem = linear_gaussian_problem
    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path))
    stage = Stage(algorithm="MH", proposal=RandomWalk(scale=0.3), max_samples=30, max_evaluations=30,
                  save_to_file=True, name="test_weight_bookkeeping")
    proposal = GaussRandomWalk(no_parameters=problem.no_parameters, sd_or_cov=0.3, seed=3)
    initial_sample = Sample(parameters=np.array([0.1, -0.2]))

    algo = Algorithm_MH(
        stage=stage,
        proposal=proposal,
        initial_sample=initial_sample,
        rank_world=0,
        conf=conf,
        prior=problem.prior,
        likelihood=problem.likelihood,
        observation_provider=LocalSolverAdapter(problem.solver),
        snapshot_collector=None,
        evaluator_provider=None,
        seed=1,
    )
    algo.run()

    csv_path = os.path.join(tmp_path, "sampling_output", "samples", stage.name, "rank0000.csv")
    with open(csv_path, newline="") as f:
        header, *rows = list(csv.reader(f))
    # format v2: the first row is the header
    assert header == ["multiplicity", "par_0", "par_1", "log_posterior"]
    weight_sum = sum(int(row[0]) for row in rows)

    expected = algo.counter_accepted + algo.counter_rejected + algo.counter_prerejected + 1
    assert weight_sum == expected
    assert algo.counter_prerejected == 0  # MH has no pre-rejection concept
    assert algo.initial_sample_is_carried_over is False  # A30: first stage keeps the +1


@pytest.mark.unit
def test_mh_weight_column_drops_the_plus_one_for_a_carried_over_initial_state(linear_gaussian_problem, tmp_path):
    """
    A30 (decided 2026-09-17), algorithm-level half: with
    ``initial_sample_is_carried_over=True`` the first row of the stage counts only the
    rejections spent in this stage, so the per-stage invariant becomes
    ``sum(weight) == accepted + rejected + prerejected`` (no ``+1``) and the state is not
    counted twice across a stage boundary. Everything else -- the chain, the rows, the other
    weights -- is unchanged, which is checked by running the same stage both ways.
    """
    problem = linear_gaussian_problem

    def run_stage(output_dir, carried_over: bool):
        conf = MinimalAlgorithmConfig(output_dir=str(output_dir))
        stage = Stage(algorithm="MH", proposal=RandomWalk(scale=0.3), max_samples=30, max_evaluations=30,
                      save_to_file=True, name="test_a30")
        algo = Algorithm_MH(
            stage=stage,
            proposal=GaussRandomWalk(no_parameters=problem.no_parameters, sd_or_cov=0.3, seed=3),
            initial_sample=Sample(parameters=np.array([0.1, -0.2])),
            rank_world=0,
            conf=conf,
            prior=problem.prior,
            likelihood=problem.likelihood,
            observation_provider=LocalSolverAdapter(problem.solver),
            snapshot_collector=None,
            evaluator_provider=None,
            seed=1,
            initial_sample_is_carried_over=carried_over,
        )
        algo.run()
        csv_path = os.path.join(output_dir, "sampling_output", "samples", stage.name, "rank0000.csv")
        with open(csv_path, newline="") as f:
            header, *rows = list(csv.reader(f))  # format v2: drop the header row
        assert header[0] == "multiplicity"
        return algo, rows

    algo_fresh, rows_fresh = run_stage(tmp_path / "fresh", carried_over=False)
    algo_carried, rows_carried = run_stage(tmp_path / "carried", carried_over=True)

    total_iterations = algo_carried.counter_accepted + algo_carried.counter_rejected + algo_carried.counter_prerejected
    assert sum(int(row[0]) for row in rows_fresh) == total_iterations + 1
    assert sum(int(row[0]) for row in rows_carried) == total_iterations

    # only the FIRST row's weight differs; the chain itself is untouched
    assert int(rows_fresh[0][0]) == int(rows_carried[0][0]) + 1
    assert [row[1:] for row in rows_fresh] == [row[1:] for row in rows_carried]
    assert [row[0] for row in rows_fresh[1:]] == [row[0] for row in rows_carried[1:]]


@pytest.mark.unit
def test_artificial_acceptance_multiplicator_field_removed():
    """
    Item 15: ``artificial_acceptance_multiplicator`` was REMOVED entirely from ``Stage``
    (WS0 author decision B5, not just deprecated/warned-on) -- so the plan's original item 15
    ("emits a warning when != 1") is moot; there is no such field or warning to test. This
    documents the removal decision instead.
    """
    stage = Stage()
    assert not hasattr(stage, "artificial_acceptance_multiplicator")


@pytest.mark.unit
def test_csv_writers_are_line_buffered(tmp_path):
    """
    G6 (decided 2026-09-17): ``modules.monitoring.CsvWriter`` opens its file with
    ``buffering=1`` (line buffering), so a completed row is on disk before the writer is
    closed -- a still-running or killed run can be inspected and post-processed. No content
    change; this test only checks the flush timing.
    """
    from surrDAMH.modules.monitoring import CsvWriter

    path = tmp_path / "out" / "rank0000.csv"
    writer = CsvWriter(dirname=str(tmp_path / "out"), basename="rank0000.csv")
    try:
        writer.writerow([1, 0.5, -2.0])
        with open(path, newline="") as f:
            assert list(csv.reader(f)) == [["1", "0.5", "-2.0"]]
        writer.writerow([2, 0.25, -1.0])
        with open(path, newline="") as f:
            assert list(csv.reader(f)) == [["1", "0.5", "-2.0"], ["2", "0.25", "-1.0"]]
        assert writer._file.line_buffering is True
    finally:
        writer.close_file()


# ---------------------------------------------------------------------------
# WS7: an MH stage with a gradient-based proposal may refresh its surrogate
# ---------------------------------------------------------------------------

class LinearGradientEvaluator(Evaluator):
    """Differentiable surrogate ``G~(x) = A' @ x`` (so a Hamiltonian proposal can run on it)."""

    def __init__(self, A: np.ndarray) -> None:
        A = np.asarray(A, dtype=float)
        super().__init__(no_parameters=A.shape[1], no_observations=A.shape[0])
        self.A = A

    def __call__(self, datapoints: np.ndarray) -> np.ndarray:
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.A.T

    def jacobian(self, datapoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(datapoints, dtype=float).reshape(-1)
        return self.A, self.A @ x


class RetrainingEvaluatorProvider(LocalEvaluatorProvider):
    """
    Stand-in for a collector that always has a newer surrogate ready: every
    ``request_evaluator()`` makes one more evaluator available, so an algorithm that polls
    once per iteration picks up a new one every time (and one that does not poll never does).

    It starts with nothing delivered yet but the first evaluator pending, which is the state
    the MPI provider is in at the start of a stage (``request_initial_evaluator=True``, then
    one outstanding request at all times).
    """

    def __init__(self, factors: list[float], A: np.ndarray) -> None:
        self.A = np.asarray(A, dtype=float)
        self.factors = list(factors)
        self.served = 0
        super().__init__(evaluator=None)
        self.set_evaluator(LinearGradientEvaluator(self.A))

    def request_evaluator(self) -> None:
        super().request_evaluator()
        if self.served < len(self.factors):
            self.set_evaluator(LinearGradientEvaluator(self.factors[self.served] * self.A))
            self.served += 1


def _run_hamiltonian_mh_stage(problem, tmp_path, name, **stage_kwargs):
    """One short MH stage with a Hamiltonian proposal on a surrogate that keeps improving."""
    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path), use_surrogate_gradients=True)
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(), max_evaluations=6,
                  name=name, **stage_kwargs)
    proposal = Hamiltonian(no_parameters=problem.no_parameters, step_size=0.1, num_steps=3,
                           sd_or_cov=1.0, seed=4)
    provider = RetrainingEvaluatorProvider(factors=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6], A=problem.A)
    algo = Algorithm_MH(
        stage=stage,
        proposal=proposal,
        initial_sample=Sample(parameters=np.array([0.1, -0.2])),
        rank_world=0,
        conf=conf,
        prior=problem.prior,
        likelihood=problem.likelihood,
        observation_provider=LocalSolverAdapter(problem.solver),
        snapshot_collector=None,
        evaluator_provider=provider,
        seed=1,
    )
    first_evaluator = algo.surrogate_evaluator
    algo.run()
    return algo, provider, first_evaluator


@pytest.mark.unit
def test_mh_hamiltonian_stage_refreshes_the_gradient_surrogate_when_opted_in(linear_gaussian_problem, tmp_path):
    """
    WS7 item 2: ``surrogate_model_updates=True`` on an MH stage with a Hamiltonian proposal
    makes it poll for a newer evaluator once per iteration and install the one it gets.
    """
    stage_probe = Stage(algorithm="MH", proposal=HamiltonianSpec(),
                        surrogate_model_updates=True, max_evaluations=6)
    assert stage_probe.surrogate_model_updates is True  # the opt-in survives __post_init__

    algo, _provider, first_evaluator = _run_hamiltonian_mh_stage(
        linear_gaussian_problem, tmp_path, "ws7_opt_in", surrogate_model_updates=True)

    assert algo.counter_evaluator_refreshes == 6, algo.counter_evaluator_refreshes
    assert algo.surrogate_evaluator is not first_evaluator
    # the proposal really uses the newest surrogate: its gradient matches the last evaluator
    assert np.allclose(algo.surrogate_evaluator.A, 1.6 * linear_gaussian_problem.A)
    assert algo.counter_accepted + algo.counter_rejected == 6


@pytest.mark.unit
def test_mh_hamiltonian_stage_keeps_one_evaluator_when_not_opted_in(linear_gaussian_problem, tmp_path):
    """WS7 item 2, default path: without the opt-in the stage keeps the evaluator it fetched at
    stage start for its whole run - the behaviour of every MH stage before WS7."""
    algo, _provider, first_evaluator = _run_hamiltonian_mh_stage(
        linear_gaussian_problem, tmp_path, "ws7_default")

    assert algo.stage.surrogate_model_updates is False
    assert algo.counter_evaluator_refreshes == 0
    assert algo.surrogate_evaluator is first_evaluator


@pytest.mark.unit
def test_mh_stage_without_gradients_never_polls_even_if_the_flag_is_forced(linear_gaussian_problem, tmp_path):
    """Defence in depth: even with ``surrogate_model_updates`` forced back on after
    ``__post_init__`` refused it, a non-gradient MH proposal does not touch the evaluator."""
    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path), use_surrogate_gradients=True)
    stage = Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=5, name="ws7_forced")
    stage.surrogate_model_updates = True  # bypasses the __post_init__ guard on purpose
    provider = RetrainingEvaluatorProvider(factors=[1.1, 1.2, 1.3, 1.4, 1.5], A=linear_gaussian_problem.A)
    algo = Algorithm_MH(
        stage=stage,
        proposal=GaussRandomWalk(no_parameters=linear_gaussian_problem.no_parameters, sd_or_cov=0.5, seed=3),
        initial_sample=Sample(parameters=np.array([0.1, -0.2])),
        rank_world=0,
        conf=conf,
        prior=linear_gaussian_problem.prior,
        likelihood=linear_gaussian_problem.likelihood,
        observation_provider=LocalSolverAdapter(linear_gaussian_problem.solver),
        snapshot_collector=None,
        evaluator_provider=provider,
        seed=1,
    )
    first_evaluator = algo.surrogate_evaluator
    algo.run()
    assert algo.counter_evaluator_refreshes == 0
    assert algo.surrogate_evaluator is first_evaluator


# ---------------------------------------------------------------------------
# 2026-09-20: adapt-then-freeze through run_sampling_local (the preliminary-stage -> fixed-stage
# workflow the hand-over exists for), for the adaptive random walk and adaptive pCN.
# ---------------------------------------------------------------------------

# d=2 here, where the 0.234 target is the high-dimensional limit rather than the optimum
# (the optimum is ~0.44 at d=1, ~0.35 at d=2), so the frozen stage is only required to land in
# a widened band around it instead of within the 0.03 the d=10 unit tests use.
FROZEN_ACCEPTANCE_BAND = (0.15, 0.35)


def _run_adapt_then_freeze(problem, tmp_path, stages):
    """``Problem.run_sampling_local`` on the linear-Gaussian fixture; returns
    ``(stage_results, output_dir)``."""
    from surrDAMH.configuration import Configuration
    from surrDAMH.core import Problem

    conf = Configuration(no_parameters=problem.no_parameters, no_observations=problem.no_observations,
                         output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False)
    run = Problem(problem.prior, problem.likelihood, solver=problem.solver).run_sampling_local(conf, stages)
    return run.stage_results, tmp_path


def _assert_frozen_stage_is_correct(problem, output_dir, stage_result, label):
    """Frozen-stage acceptance inside the band, posterior mean within 4 batch-means SE."""
    import helpers_statistics as stats

    acceptance = stage_result.counter_accepted / (stage_result.counter_accepted
                                                  + stage_result.counter_rejected)
    chain = stats.read_weighted_samples(output_dir, stage_result.name, problem.no_parameters)
    statistics = stats.chain_statistics(chain)
    print(f"[{label}] frozen acceptance {acceptance:.3f}, mean "
          f"{np.array2string(statistics.mean, precision=4)} vs closed form "
          f"{np.array2string(problem.posterior_mean, precision=4)} "
          f"(max |dev|/SE {stats.max_deviation_in_se(statistics.mean, problem.posterior_mean, statistics.se_mean):.2f})")
    low, high = FROZEN_ACCEPTANCE_BAND
    assert low < acceptance < high, (
        f"{label}: frozen-stage acceptance {acceptance:.4f} outside {FROZEN_ACCEPTANCE_BAND}")
    stats.assert_close(f"{label} posterior mean vs closed form", statistics.mean,
                       problem.posterior_mean, statistics.se_mean)


@pytest.mark.unit
def test_adaptive_random_walk_hands_a_working_covariance_to_the_next_stage(linear_gaussian_problem,
                                                                          tmp_path):
    """
    The hand-over the rewrite exists for: an adaptive RWMH stage (3000 evaluations) followed by
    a fixed MH stage with ``RandomWalk(adaptive=False)`` (no scale of its own) must produce a
    frozen stage that keeps the target acceptance and samples the closed-form posterior. Under
    the old rule the frozen stage's acceptance was unpredictable (0.07-0.44 in the prototype
    study, `16` §4.1).
    """
    problem = linear_gaussian_problem
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5, adaptive=True),
                    max_samples=3000, max_evaluations=3000),
              # adaptive=False pins the frozen consumer stage (since 2026-09-21 a proposal that
              # leaves its step None would otherwise keep adapting: "adapt unless pinned")
              Stage(algorithm="MH", proposal=RandomWalk(adaptive=False),
                    max_samples=6000, max_evaluations=6000)]
    result, output_dir = _run_adapt_then_freeze(problem, tmp_path, stages)

    adaptive_stage_result, frozen_stage_result = result[0], result[1]
    assert adaptive_stage_result.name == "alg0000_MH-adaptive"
    _assert_frozen_stage_is_correct(problem, output_dir, frozen_stage_result, "adaptive RW")

    # the per-period adaptation trace was written and is readable through read_run
    from surrDAMH.modules.run_data import read_run
    run = read_run(str(output_dir), load_raw_data=False)
    adaptive_stats = run.adaptive_stats[run.stage_index("alg0000_MH-adaptive")]
    assert adaptive_stats is not None
    assert list(adaptive_stats.columns) == ["n", "mean_acceptance_probability", "log_sigma",
                                            "trace_C_over_d", "shrinkage_delta", "rank_world"]
    assert len(adaptive_stats) == 3000 // 10
    assert run.adaptive_stats[run.stage_index("alg0001_MH")] is None  # non-adaptive stage


@pytest.mark.unit
def test_adaptive_pcn_hands_its_beta_to_the_next_stage(linear_gaussian_problem, tmp_path):
    """The same hand-over for pCN: an adaptive pCN stage followed by a fixed pCN stage with
    ``PCN(adaptive=False)`` (no beta of its own) must get the adapted beta and keep the target
    acceptance."""
    problem = linear_gaussian_problem
    stages = [Stage(algorithm="MH", proposal=PCN(adaptive=True),
                    max_samples=3000, max_evaluations=3000),
              Stage(algorithm="MH", proposal=PCN(adaptive=False),
                    max_samples=6000, max_evaluations=6000)]
    result, output_dir = _run_adapt_then_freeze(problem, tmp_path, stages)

    _assert_frozen_stage_is_correct(problem, output_dir, result[1], "adaptive pCN")

    from surrDAMH.modules.run_data import read_run
    run = read_run(str(output_dir), load_raw_data=False)
    adaptive_stats = run.adaptive_stats[0]
    assert adaptive_stats is not None
    assert list(adaptive_stats.columns) == ["n", "mean_acceptance_probability", "beta", "rank_world"]
    # the adapted beta really moved away from the 0.5 default it started at
    assert abs(float(adaptive_stats["beta"].iloc[-1]) - 0.5) > 0.05


# ---------------------------------------------------------------------------
# 2026-09-20: non-finite proposals and non-finite surrogates
#
# Failure being guarded against: an adaptive Hamiltonian's early (by-design) step-size
# overshoot sent a leapfrog trajectory to inf; the library still solved the full model
# there and sent the outlier to the collector, whose neural network then trained itself
# into NaN and published a NaN surrogate that the sampler installed.
# ---------------------------------------------------------------------------

class CountingLinearSolver(Solver):
    """``G(x) = A @ x`` that counts how often it was actually called."""

    def __init__(self, A: np.ndarray) -> None:
        self.A = np.asarray(A, dtype=float)
        self.calls = 0

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        self.calls += 1
        return self.A @ self.parameters


class StubProposal(Proposal):
    """
    Proposal returning a fixed sequence of samples, cycling: ``propose_sample`` ignores the
    current state. Records every ``adapt()`` call and every acceptance-ratio evaluation.
    """

    def __init__(self, samples: list[np.ndarray]) -> None:
        super().__init__()
        self.no_parameters = len(samples[0])
        self._generator = np.random.RandomState(0)
        self.samples = samples
        self.proposal_calls = 0
        self.acceptance_calls = 0
        self.adapt_log_acceptance_probabilities: list[float] = []

    def propose_sample(self, current_sample):
        sample = self.samples[self.proposal_calls % len(self.samples)]
        self.proposal_calls += 1
        return np.array(sample, dtype=float)

    def get_log_acceptance_probability(self, log_likelihood_proposed, log_likelihood_current,
                                       log_prior_proposed, log_prior_current):
        self.acceptance_calls += 1
        return (log_likelihood_proposed - log_likelihood_current,
                log_prior_proposed - log_prior_current)

    def adapt(self, proposed_sample, log_acceptance_probability, current_sample,
              subchain_log_acceptance_probabilities=None):
        self.adapt_log_acceptance_probabilities.append(log_acceptance_probability)


@pytest.mark.unit
def test_non_finite_mh_proposal_is_rejected_without_solver_call_or_snapshot(linear_gaussian_problem,
                                                                           tmp_path):
    """
    Item A: a non-finite proposal in an MH stage is rejected without evaluating the full model
    and without reaching the collector; the acceptance ratio is not even formed (a Hamiltonian's
    momentum term would be NaN), ``adapt()`` is fed ``-inf`` (= acceptance probability 0, the
    divergent-transition signal the dual averaging needs) and one RuntimeWarning is emitted.
    """
    problem = linear_gaussian_problem
    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path))
    stage = Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_samples=1, max_evaluations=1,
                  name="nonfinite_mh")
    proposal = StubProposal([np.array([np.inf, np.nan])])
    solver = CountingLinearSolver(problem.A)
    sink = InMemorySnapshotSink()

    algo = Algorithm_MH(
        stage=stage,
        proposal=proposal,
        initial_sample=Sample(parameters=np.array([0.1, -0.2])),
        rank_world=0,
        conf=conf,
        prior=problem.prior,
        likelihood=problem.likelihood,
        observation_provider=LocalSolverAdapter(solver),
        snapshot_collector=sink,
        evaluator_provider=None,
        seed=1,
    )
    initial_parameters = algo.current.parameters.copy()
    solver_calls_after_initial_sample = solver.calls  # the initial sample IS evaluated

    with pytest.warns(RuntimeWarning, match="produced a NON-FINITE sample"):
        algo.run()

    assert solver.calls == solver_calls_after_initial_sample, "the full model was called for a non-finite proposal"
    assert len(sink) == 0, "a non-finite proposal must not reach the surrogate collector"
    assert algo.proposed.solver_tag == SOLVER_TAG_NONFINITE_PROPOSAL
    assert algo.proposed.log_likelihood == -np.inf
    assert algo.counter_nonfinite_proposals == 1
    assert algo.counter_rejected == 1
    assert algo.counter_accepted == 0
    assert proposal.acceptance_calls == 0, "get_log_acceptance_probability must be skipped"
    assert proposal.adapt_log_acceptance_probabilities == [-np.inf]
    np.testing.assert_array_equal(algo.current.parameters, initial_parameters)
    assert np.all(np.isfinite(algo.current.parameters))


class RecordingLinearEvaluator(FixedLinearEvaluator):
    """``FixedLinearEvaluator`` that remembers every batch it was asked to evaluate."""

    def __init__(self, A: np.ndarray) -> None:
        super().__init__(A)
        self.calls: list[np.ndarray] = []

    def __call__(self, datapoints: np.ndarray) -> np.ndarray:
        self.calls.append(np.asarray(datapoints, dtype=float).copy())
        return super().__call__(datapoints)


@pytest.mark.unit
def test_non_finite_damh_subchain_proposal_skips_the_surrogate(linear_gaussian_problem, tmp_path):
    """
    Item A for DAMH: a non-finite sub-chain proposal is not handed to the surrogate, is scored
    as ``-inf`` in the per-step list the adaptation consumes, and leaves the sub-chain where it
    was -- so the sub-chain end state (the outer proposal) is always finite.
    """
    problem = linear_gaussian_problem
    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path))
    stage = Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), subchain_length=4,
                  surrogate_model_updates=False, max_samples=1, max_evaluations=1,
                  name="nonfinite_damh")
    # alternating: non-finite, finite, non-finite, finite
    proposal = StubProposal([np.array([np.inf, 0.0]), np.array([0.15, -0.25]),
                             np.array([np.nan, np.nan]), np.array([0.2, -0.3])])
    evaluator = RecordingLinearEvaluator(1.3 * problem.A)

    algo = Algorithm_DAMH(
        stage=stage,
        proposal=proposal,
        initial_sample=Sample(parameters=np.array([0.1, -0.2])),
        rank_world=0,
        conf=conf,
        prior=problem.prior,
        likelihood=problem.likelihood,
        observation_provider=LocalSolverAdapter(problem.solver),
        snapshot_collector=None,
        evaluator_provider=LocalEvaluatorProvider(evaluator=evaluator),
        seed=1,
    )
    calls_before_subchain = len(evaluator.calls)

    with pytest.warns(RuntimeWarning, match="produced a NON-FINITE sample"):
        (subchain_current, _counter_accepted, _correction,
         subchain_log_acceptance_probabilities) = algo._propose_new_sample_using_subchain()

    assert len(subchain_log_acceptance_probabilities) == 4
    assert subchain_log_acceptance_probabilities[0] == -np.inf
    assert subchain_log_acceptance_probabilities[2] == -np.inf
    assert np.isfinite(subchain_log_acceptance_probabilities[1])
    assert np.isfinite(subchain_log_acceptance_probabilities[3])
    assert algo.counter_nonfinite_proposals == 2
    assert np.all(np.isfinite(subchain_current.parameters)), "the sub-chain end state must be finite"
    # the surrogate was queried only for the two finite proposals
    for batch in evaluator.calls[calls_before_subchain:]:
        assert np.all(np.isfinite(batch)), "the surrogate was called on a non-finite proposal"
    assert len(evaluator.calls) - calls_before_subchain == 2


class NonFiniteGradientEvaluator(LinearGradientEvaluator):
    """A surrogate whose weights went NaN: every prediction and every gradient is NaN."""

    def __call__(self, datapoints: np.ndarray) -> np.ndarray:
        return np.full_like(super().__call__(datapoints), np.nan)

    def jacobian(self, datapoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        jacobian, evaluation = super().jacobian(datapoints)
        return np.full_like(jacobian, np.nan), np.full_like(evaluation, np.nan)


def _hamiltonian_algorithm_with_provider(problem, tmp_path, provider, name):
    conf = MinimalAlgorithmConfig(output_dir=str(tmp_path), use_surrogate_gradients=True)
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(), surrogate_model_updates=True,
                  max_evaluations=4, name=name)
    proposal = Hamiltonian(no_parameters=problem.no_parameters, step_size=0.1, num_steps=3,
                           sd_or_cov=1.0, seed=4)
    return Algorithm_MH(
        stage=stage,
        proposal=proposal,
        initial_sample=Sample(parameters=np.array([0.1, -0.2])),
        rank_world=0,
        conf=conf,
        prior=problem.prior,
        likelihood=problem.likelihood,
        observation_provider=LocalSolverAdapter(problem.solver),
        snapshot_collector=None,
        evaluator_provider=provider,
        seed=1,
    )


@pytest.mark.unit
def test_refresh_keeps_the_previous_evaluator_when_the_new_one_is_non_finite(linear_gaussian_problem,
                                                                            tmp_path):
    """
    Item D: an evaluator that predicts NaN at the current chain state is refused; the sampler
    keeps the one it has, a new request is still posted, and a later finite evaluator is
    installed normally.
    """
    problem = linear_gaussian_problem
    provider = LocalEvaluatorProvider(evaluator=LinearGradientEvaluator(problem.A))
    algo = _hamiltonian_algorithm_with_provider(problem, tmp_path, provider, "nonfinite_evaluator")
    first_evaluator = algo.surrogate_evaluator

    provider.set_evaluator(NonFiniteGradientEvaluator(problem.A))
    with pytest.warns(RuntimeWarning, match="surrogate evaluator received from the collector"):
        assert algo._refresh_surrogate_evaluator_if_needed() is False

    assert algo.surrogate_evaluator is first_evaluator
    assert algo.counter_evaluator_refreshes == 0
    assert algo.counter_nonfinite_evaluators == 1
    # the gradients the proposal uses are still the finite ones
    assert np.all(np.isfinite(algo._compute_surrogate_log_likelihood_gradient(algo.current.parameters)))

    # positive control: a finite evaluator is installed exactly as before
    healthy_evaluator = LinearGradientEvaluator(1.5 * problem.A)
    provider.set_evaluator(healthy_evaluator)
    assert algo._refresh_surrogate_evaluator_if_needed() is True
    assert algo.surrogate_evaluator is healthy_evaluator
    assert algo.counter_evaluator_refreshes == 1
    assert algo.counter_nonfinite_evaluators == 1
