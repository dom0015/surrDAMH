#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation rules for the surrogate versions received during a stage (``Stage.installation``,
``surrDAMH.installation``; 2026-10-23, paper §6.5, def:randomised and def:averaged of
``pdf_reports/adaptive_nn_ergodicity_paper/main.tex``).

Pinned here: the two specs and their ``probability``/``gain``; the ``Stage`` checks, ``*`` and
``describe()``; the manifest records the spec and ``installation_seed = seed0 + 6`` (shifted in a
continuation). Randomised rule: exactly one install-stream uniform per outer iteration ``i >= 1``,
installs only on coin success with a pending (newest) version, replayed from
``RandomState(seed0 + 6)``; the ``installations`` CSV and the two ``notes`` columns; a refused
non-finite version is dropped; the MH stage with a gradient surrogate; the deferred sub-chain
adaptation (no version ever arrives -> the adaptive random walk proposes exactly as the fixed one);
the closed-form posterior with the polynomial updater; the in-process MPI sampler == the local
runner. Averaged rule: ``AveragedInstallation(1, 1.0)`` installs the mean of the RECEIVED versions
(the stage-start network replaced by the first one, unlike ``polyak_averaging=True``); general gains;
the weight box (model part only); a local run.
The linear-Gaussian problem is the one of ``tests/unit/test_safety_kernel.py``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import surrDAMH
from helpers_statistics import (FixedInitialSample, assert_matches_closed_form, chain_statistics,
                                read_weighted_samples)
from surrDAMH.installation import AveragedInstallation, RandomisedInstallation
from surrDAMH.modules.algorithm_interfaces import EvaluatorProvider
from surrDAMH.modules.algorithm_interfaces_local import LocalEvaluatorProvider, LocalSolverAdapter
from surrDAMH.modules.algorithms import (INSTALLATION_NOTES_COLUMNS, AlgorithmBase, Algorithm_DAMH, Sample)
from surrDAMH.modules.error_model import ERROR_MODEL_STATE_KEYS
from surrDAMH.modules.proposals import GaussRandomWalk
from surrDAMH.modules.seeds import INSTALLATION_SEED_OFFSET, SEED_FORMULA
from surrDAMH.proposals import Hamiltonian, RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import POSTERIOR_AFFECTING_FIELDS, Stage
from surrDAMH.surrogates import PolynomialSklearnUpdater
from surrDAMH.surrogates.parent import Evaluator

pytestmark = pytest.mark.unit

A = np.array([[1.0, 0.5], [0.0, 1.0]])
OBSERVED = A @ np.array([0.7, -0.4])
OBS_SD = np.array([0.3, 0.3])
A_SURROGATE = np.array([[1.1, 0.4], [0.05, 0.9]])


class LinearSolver(Solver):
    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return A @ self.parameters


class StateEvaluator(Evaluator):
    """``G~(x) = M x + b``, averageable; ``extra`` entries (e.g. error-model keys) ride along in the state."""

    def __init__(self, M, b=(0.0, 0.0), extra=None, version=None):
        super().__init__(no_parameters=2, no_observations=2)
        self.M = np.asarray(M, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.extra = {} if extra is None else {k: np.asarray(v, dtype=float) for k, v in extra.items()}
        self.version = version

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.M.T + self.b

    def jacobian(self, datapoints):
        x = np.asarray(datapoints, dtype=float).reshape(-1)
        return self.M, self.M @ x + self.b

    def averageable_state(self):
        return {"M": self.M.copy(), "b": self.b.copy(), **{k: v.copy() for k, v in self.extra.items()}}

    def with_averageable_state(self, state):
        extra = {k: v for k, v in state.items() if k not in ("M", "b")}
        return StateEvaluator(state["M"], state["b"], extra)


def _version(v):
    """Version ``v`` of the surrogate: a slightly different linear map (finite)."""
    return StateEvaluator(A_SURROGATE * (1.0 + 0.01 * v), version=v)


class ScheduleProvider(EvaluatorProvider):
    """
    A new version becomes available at the polls (1-based) where ``available(poll)`` is True; the
    k-th one handed out is ``make(k)``. ``handed_out`` records every evaluator returned by
    ``get_evaluator`` after the initial one.
    """

    def __init__(self, make=_version, available=lambda poll: True):
        self.make = make
        self.available = available
        self.evaluator = make(0)
        self.polls = 0
        self.handed_out: list = []
        self._pending = None

    def request_evaluator(self):
        return None

    def announce_waiting(self):
        return None

    def close(self):
        return None

    def evaluator_is_available(self):
        self.polls += 1
        if self._pending is None and self.available(self.polls):
            self._pending = self.make(len(self.handed_out) + 1)
        return self._pending is not None

    def get_evaluator(self):
        if self._pending is not None:
            self.evaluator = self._pending
            self.handed_out.append(self._pending)
            self._pending = None
        return self.evaluator


def _problem(**kwargs):
    prior = surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(2)])
    return surrDAMH.Problem(prior, surrDAMH.distributions.Normal(mean=OBSERVED, sd=OBS_SD), solver=LinearSolver(),
                            **kwargs)


def _conf(output_dir, **kwargs):
    return surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedInitialSample([0.3, -0.2]),
                                  save_snapshots_to_file=True, **kwargs)


def _csv(output_dir, kind, stage_name):
    return pd.read_csv(os.path.join(str(output_dir), "sampling_output", kind, stage_name, "rank0000.csv"))


def _manifest(output_dir):
    with open(os.path.join(str(output_dir), "sampling_output", "run_manifest.json")) as f:
        return json.load(f)


def _posterior():
    cov = np.linalg.inv(np.eye(2) + A.T @ np.diag(1.0 / OBS_SD ** 2) @ A)
    return cov @ (A.T @ np.diag(1.0 / OBS_SD ** 2) @ OBSERVED), cov


@pytest.fixture
def instances(monkeypatch):
    """Every algorithm instance (= one stage of the chain), in order, captured at _finalize_run."""
    captured: list = []
    original_finalize = AlgorithmBase._finalize_run

    def finalize(self):
        captured.append(self)
        original_finalize(self)

    monkeypatch.setattr(AlgorithmBase, "_finalize_run", finalize)
    return captured


def _use_provider(monkeypatch, provider):
    """Make run_sampling_local(surrogate_evaluator=...) use ``provider`` (and so the MPI sampler in process)."""
    import surrDAMH.process_SAMPLER as process_SAMPLER
    import surrDAMH.runner_local as runner_local
    monkeypatch.setattr(runner_local, "LocalEvaluatorProvider", lambda evaluator=None: provider)
    monkeypatch.setattr(process_SAMPLER, "LocalEvaluatorProvider", lambda evaluator=None: provider)


def _direct_damh(tmp_path, provider, stage, proposal=None, exact_proposal=None):
    problem = _problem()
    conf = _conf(tmp_path)
    conf.resolve_problem_sizes(2, 2)
    return Algorithm_DAMH(stage=stage, proposal=proposal or GaussRandomWalk(2, sd_or_cov=0.5, seed=1),
                          initial_sample=Sample(np.array([0.3, -0.2])), rank_world=0, conf=conf,
                          prior=problem.prior, likelihood=problem.likelihood,
                          observation_provider=LocalSolverAdapter(LinearSolver()),
                          evaluator_provider=provider, seed=2, exact_proposal=exact_proposal)


# ---------------------------------------------------------------------------------------------
# The specs
# ---------------------------------------------------------------------------------------------

def test_randomised_probability():
    rule = RandomisedInstallation(n_lambda=10, a_lambda=0.5)
    assert (rule.n_lambda, rule.a_lambda) == (10.0, 0.5) and isinstance(rule.n_lambda, float)
    assert [rule.probability(n) for n in (-3, 0, 1, 5, 10)] == [1.0] * 5
    assert rule.probability(40) == pytest.approx(0.5) and rule.probability(1000) == pytest.approx(0.1)
    assert RandomisedInstallation(1, 1.0).probability(4) == pytest.approx(0.25)


@pytest.mark.parametrize("kwargs", [dict(n_lambda=0.5, a_lambda=0.5), dict(n_lambda=np.inf, a_lambda=0.5),
                                    dict(n_lambda=np.nan, a_lambda=0.5), dict(n_lambda=True, a_lambda=0.5),
                                    dict(n_lambda=10, a_lambda=0.0), dict(n_lambda=10, a_lambda=1.5),
                                    dict(n_lambda="10", a_lambda=0.5)])
def test_randomised_invalid(kwargs):
    with pytest.raises(ValueError, match="RandomisedInstallation"):
        RandomisedInstallation(**kwargs)


def test_averaged_gain():
    rule = AveragedInstallation()
    assert (rule.j0, rule.a_zeta, rule.weight_box) == (1, 1.0, None)
    assert [rule.gain(j) for j in (1, 2, 4)] == [1.0, 0.5, 0.25]
    rule = AveragedInstallation(j0=4, a_zeta=0.5, weight_box=3)
    assert rule.weight_box == 3.0 and isinstance(rule.weight_box, float)
    assert [rule.gain(j) for j in (1, 4)] == [1.0, 1.0] and rule.gain(16) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="j must be >= 1"):
        rule.gain(0)


@pytest.mark.parametrize("kwargs", [dict(j0=0), dict(j0=1.5), dict(j0=True), dict(a_zeta=0.0), dict(a_zeta=1.2),
                                    dict(weight_box=0.0), dict(weight_box=-1.0), dict(weight_box=np.inf)])
def test_averaged_invalid(kwargs):
    with pytest.raises(ValueError, match="AveragedInstallation"):
        AveragedInstallation(**kwargs)


def test_specs_are_frozen():
    with pytest.raises(Exception):
        RandomisedInstallation(10, 0.5).n_lambda = 3.0


# ---------------------------------------------------------------------------------------------
# Stage checks
# ---------------------------------------------------------------------------------------------

def test_stage_field_membership_and_describe():
    assert "installation" in POSTERIOR_AFFECTING_FIELDS
    assert Stage(algorithm="DAMH", max_evaluations=5).installation is None
    text = Stage(algorithm="DAMH", max_evaluations=5, installation=RandomisedInstallation(50, 0.5)).describe()
    assert "installation*=RandomisedInstallation(n_lambda=50.0, a_lambda=0.5)" in text
    assert "probability lambda_n = min(1, (n/50)^-0.5)" in text
    text = Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=5,
                 installation=AveragedInstallation(2, 0.6, 4.0)).describe()
    assert "zeta_j = min(1, (j/2)^-0.6), each version clipped to [-4, 4]" in text
    text = Stage(algorithm="DAMH", proposal=Hamiltonian(step_size=0.1, num_steps=3), max_evaluations=5,
                 installation=AveragedInstallation()).describe()
    assert "fixed proposal density only" in text


def test_stage_refuses_a_wrong_type_and_polyak():
    with pytest.raises(TypeError, match="Stage.installation must be None"):
        Stage(algorithm="DAMH", max_evaluations=5, installation="randomised")
    with pytest.raises(ValueError, match="installation and polyak_averaging=True conflict"):
        Stage(algorithm="DAMH", max_evaluations=5, polyak_averaging=True, installation=RandomisedInstallation(5, 1))


@pytest.mark.parametrize("kwargs", [dict(algorithm="MH"),
                                    dict(algorithm="DAMH", surrogate_model_updates=False),
                                    dict(algorithm="MH", proposal=Hamiltonian(step_size=0.1, num_steps=3))])
def test_stage_without_versions_resets_with_a_warning(kwargs, capsys):
    stage = Stage(max_evaluations=5, installation=RandomisedInstallation(5, 1.0), **kwargs)
    assert stage.installation is None
    assert "only used by a stage that receives surrogate versions" in capsys.readouterr().out


def test_mh_gradient_stage_with_updates_keeps_the_rule():
    stage = Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1, num_steps=3), surrogate_model_updates=True,
                  max_evaluations=5, installation=RandomisedInstallation(5, 1.0))
    assert stage.installation == RandomisedInstallation(5, 1.0)


def test_averaged_needs_fixed_sub_chain_parameters():
    with pytest.raises(ValueError, match="needs fixed sub-chain parameters, but the proposal RandomWalk adapts"):
        Stage(algorithm="DAMH", max_evaluations=5, installation=AveragedInstallation())
    with pytest.raises(ValueError, match="exact step's random walk adapts"):
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=5, exact_step_probability=0.1,
              installation=AveragedInstallation())
    Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=5, exact_step_probability=0.1,
          exact_step_proposal="prior", installation=AveragedInstallation())
    Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=5, exact_step_probability=0.1,
          exact_step_scale=1.0, installation=AveragedInstallation())
    # the randomised rule accepts adaptive proposals (their parameters change only at installs)
    Stage(algorithm="DAMH", max_evaluations=5, exact_step_probability=0.1, installation=RandomisedInstallation(5, 1))


# ---------------------------------------------------------------------------------------------
# Randomised rule: the coin stream
# ---------------------------------------------------------------------------------------------

def _replay(seed, rule, no_iterations, available_at):
    """Installs [(iteration, version)] the rule must make; ``available_at(i)``: a version arrives at the poll of iteration i."""
    generator = np.random.RandomState(seed)
    installs, pending, received = [], None, 0
    for i in range(no_iterations):
        if available_at(i):
            received += 1
            pending = received
        if i >= 1 and generator.uniform() < rule.probability(i - 1) and pending is not None:
            installs.append((i, pending))
            pending = None
    return installs, received, generator


@pytest.mark.parametrize("every", [1, 7])
def test_randomised_rule_replays_the_install_stream(tmp_path, monkeypatch, instances, every):
    rule = RandomisedInstallation(n_lambda=20, a_lambda=0.5)
    provider = ScheduleProvider(available=lambda poll: poll % every == 0)
    _use_provider(monkeypatch, provider)
    out = tmp_path / "run"
    run = _problem().run_sampling_local(_conf(out), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=50),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), subchain_length=2, max_samples=600,
              exact_step_probability=0.2, installation=rule)], surrogate_evaluator=_version(0))
    result, algo = run.stage_results[1], instances[1]
    no_iterations = len(_csv(out, "subchain_stats", result.name))
    assert no_iterations == 600 and provider.polls == 600  # one poll per outer iteration, exact steps included
    # stage 1 of 2, rank 0: seed0 = 10, installation seed 16
    assert algo.installation_seed == 10 + INSTALLATION_SEED_OFFSET == 16
    expected, received, generator = _replay(16, rule, no_iterations, lambda i: (i + 1) % every == 0)
    log = _csv(out, "installations", result.name)
    assert list(log.columns) == ["iteration", "version", "collector_version"]  # T1 (2026-10-23): third column
    assert list(zip(log["iteration"], log["version"])) == expected
    # exactly one uniform per outer iteration i >= 1: the next draws agree
    assert algo._generator_install.uniform() == generator.uniform()
    notes = _csv(out, "notes", result.name)
    assert list(notes.columns[-2:]) == list(INSTALLATION_NOTES_COLUMNS)
    assert int(notes["installations"].iloc[0]) == len(expected) == result.counter_installations
    assert int(notes["versions_received"].iloc[0]) == received == len(provider.handed_out) \
        == result.counter_versions_received
    # the installed evaluator is the version of the last install
    assert algo.surrogate_evaluator.version == expected[-1][1]
    if every == 1:
        lambdas = np.array([rule.probability(i - 1) for i in range(1, no_iterations)])
        print(f"\n[randomised] installs {len(expected)} vs sum lambda {lambdas.sum():.1f}")
        assert abs(len(expected) - lambdas.sum()) < 4 * np.sqrt(np.sum(lambdas * (1 - lambdas)))
    # the chain state carries the installed surrogate's terms unless a re-scoring is still pending
    if not algo._surrogate_rescore_pending:
        np.testing.assert_allclose(algo.current.observations_approx,
                                   algo.surrogate_evaluator(algo.current.parameters[None, :])[0])


def test_installs_only_with_a_pending_finite_version(tmp_path):
    nan = StateEvaluator(np.full((2, 2), np.nan), version="nan")
    versions = {0: _version(0), 1: nan, 2: _version(2)}  # handed out in this order, at polls 1 and 3
    provider = ScheduleProvider(make=versions.__getitem__, available=lambda poll: poll in (1, 3))
    rule = RandomisedInstallation(n_lambda=10, a_lambda=1.0)  # lambda = 1 for the first 10 iterations
    stage = Stage(algorithm="DAMH", max_evaluations=10, installation=rule, name="direct")
    algo = _direct_damh(tmp_path, provider, stage)
    start = algo.surrogate_evaluator
    assert algo._randomised_installation_step(0) is False  # receives the NaN version, draws nothing
    with pytest.warns(RuntimeWarning, match="NON-FINITE"):
        assert algo._randomised_installation_step(1) is False  # coin succeeds, NaN version dropped
    assert algo._pending_candidate is None and algo.counter_nonfinite_evaluators == 1
    assert algo.surrogate_evaluator is start
    assert algo._randomised_installation_step(2) is True  # receives version 2 (2nd received), installs it
    assert algo.surrogate_evaluator.version == 2 and algo._surrogate_rescore_pending
    assert algo._randomised_installation_step(3) is False  # nothing pending: no install, one uniform drawn
    assert (algo.counter_installations, algo.counter_versions_received) == (1, 2)
    replay = np.random.RandomState(algo.installation_seed)
    for _ in range(3):
        replay.uniform()
    assert algo._generator_install.uniform() == replay.uniform()
    algo._finalize_run()
    log = _csv(tmp_path, "installations", "direct")
    assert list(zip(log["iteration"], log["version"])) == [(2, 2)]


def test_installation_seed_default_follows_the_algorithm_seed(tmp_path):
    stage = Stage(algorithm="DAMH", max_evaluations=10, installation=RandomisedInstallation(5, 1.0), name="direct")
    algo = _direct_damh(tmp_path, ScheduleProvider(), stage)
    assert algo.installation_seed == 2 - 2 + INSTALLATION_SEED_OFFSET


class _SpyProposal(GaussRandomWalk):
    """A fixed random walk that records the deferred-adaptation calls."""

    def __init__(self):
        super().__init__(no_parameters=2, sd_or_cov=0.5, seed=1)
        self.deferred_calls: list = []
        self.applied = 0

    def set_deferred_adaptation(self, deferred):
        self.deferred_calls.append(deferred)

    def apply_pending_adaptation(self):
        self.applied += 1
        return False


def test_deferred_adaptation_is_switched_on_and_applied_at_every_install(tmp_path):
    proposal, exact_proposal = _SpyProposal(), _SpyProposal()
    stage = Stage(algorithm="DAMH", max_samples=300, exact_step_probability=0.3,
                  installation=RandomisedInstallation(n_lambda=10, a_lambda=0.5), name="direct")
    algo = _direct_damh(tmp_path, ScheduleProvider(available=lambda poll: poll % 3 == 0), stage, proposal=proposal,
                        exact_proposal=exact_proposal)
    assert proposal.deferred_calls == [True] and exact_proposal.deferred_calls == [True]
    algo.run()
    assert algo.counter_installations > 3
    assert proposal.applied == exact_proposal.applied == algo.counter_installations
    # no rule: never switched on
    plain = _SpyProposal()
    _direct_damh(tmp_path / "plain", ScheduleProvider(), Stage(algorithm="DAMH", max_samples=5, name="direct"),
                 proposal=plain)
    assert plain.deferred_calls == []


def test_without_versions_the_adaptive_walk_proposes_as_the_fixed_one(tmp_path):
    """
    Deferred sub-chain adaptation: under the randomised rule with a provider that never delivers a
    version, the adaptive random walk's live parameters never change, so the chain is the one of the
    fixed random walk with the same initial scale and seed (sd 0.5: diag(0.25) has the exact
    Cholesky factor diag(0.5), so both draw current + 0.5 z from the same normal stream). Needs the
    deferred-adaptation API of modules/proposals.py.
    """
    def run(name, proposal, rule):
        out = tmp_path / name
        _problem().run_sampling_local(_conf(out), [
            Stage(algorithm="DAMH", proposal=proposal, max_samples=400, subchain_length=3, installation=rule)],
            surrogate_evaluator=StateEvaluator(A_SURROGATE))
        return out

    rule = RandomisedInstallation(n_lambda=10, a_lambda=0.5)
    adaptive = run("adaptive", RandomWalk(scale=0.5, adaptive=True), rule)
    fixed = run("fixed", RandomWalk(scale=0.5), rule)
    free = run("free", RandomWalk(scale=0.5, adaptive=True), None)
    stage = "alg0000_DAMH-SMU"
    for kind in ("samples", "raw_data", "subchain_stats"):
        a = open(adaptive / "sampling_output" / kind / stage / "rank0000.csv", "rb").read()
        b = open(fixed / "sampling_output" / kind / stage / "rank0000.csv", "rb").read()
        assert a == b, kind
    # control: without the rule the adaptation changes the proposals
    assert open(free / "sampling_output" / "raw_data" / stage / "rank0000.csv", "rb").read() != \
        open(fixed / "sampling_output" / "raw_data" / stage / "rank0000.csv", "rb").read()
    # the recursion did run under the rule (adaptive_stats rows), it was only not installed
    assert len(_csv(adaptive, "adaptive_stats", stage)) > 0


def test_mh_gradient_stage_draws_one_coin_per_iteration(tmp_path, monkeypatch, instances):
    rule = RandomisedInstallation(n_lambda=5, a_lambda=1.0)
    provider = ScheduleProvider(available=lambda poll: poll % 2 == 0)
    _use_provider(monkeypatch, provider)
    out = tmp_path / "mh"
    run = _problem().run_sampling_local(_conf(out), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=50),
        Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.2, num_steps=3), max_evaluations=200,
              surrogate_model_updates=True, installation=rule)], surrogate_evaluator=_version(0))
    result, algo = run.stage_results[1], instances[1]
    assert provider.polls == 200
    expected, received, generator = _replay(16, rule, 200, lambda i: (i + 1) % 2 == 0)
    log = _csv(out, "installations", result.name)
    assert list(zip(log["iteration"], log["version"])) == expected
    assert algo._generator_install.uniform() == generator.uniform()
    assert result.counter_installations == len(expected) and result.counter_versions_received == received
    assert algo.counter_evaluator_refreshes == len(expected)


# ---------------------------------------------------------------------------------------------
# Randomised rule: real updater, manifest, MPI sampler parity
# ---------------------------------------------------------------------------------------------

def test_randomised_rule_samples_the_posterior_with_the_polynomial_updater(tmp_path):
    out = tmp_path / "poly"
    run = _problem().run_sampling_local(_conf(out, min_snapshots_to_update=100), [
        Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=500, burn_in=True),
        Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=15_000, subchain_length=1,
              exact_step_probability=0.1, exact_step_proposal="prior",
              installation=RandomisedInstallation(n_lambda=50, a_lambda=0.5))],
        surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=2))
    result = run.stage_results[1]
    log = _csv(out, "installations", result.name)
    notes = _csv(out, "notes", result.name)
    print(f"\n[randomised poly] received {result.counter_versions_received}, installed {result.counter_installations}")
    assert 0 < result.counter_installations == len(log) == int(notes["installations"].iloc[0])
    assert result.counter_versions_received == int(notes["versions_received"].iloc[0]) >= result.counter_installations
    assert log["version"].is_monotonic_increasing and log["iteration"].is_monotonic_increasing
    mean, cov = _posterior()
    statistics = chain_statistics(read_weighted_samples(out, result.name, 2))
    print(f"[randomised poly] {statistics.no_samples} samples: mean {statistics.mean} vs {mean}, "
          f"cov {statistics.cov.ravel()} vs {cov.ravel()}")
    assert_matches_closed_form(statistics, mean, cov, "randomised installation, polynomial updater")


def test_manifest_records_the_spec_and_the_installation_seed_and_a_continuation_shifts_it(tmp_path):
    assert INSTALLATION_SEED_OFFSET == 6 and "installation_seed = seed0+6" in SEED_FORMULA
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=50),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=30,
                    installation=RandomisedInstallation(n_lambda=5, a_lambda=0.5))]
    run = _problem().run_sampling_local(_conf(tmp_path / "a"), stages,
                                        surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2,
                                                                                   max_degree=1))
    manifest = _manifest(tmp_path / "a")
    assert manifest["stages"][1]["installation"] == {"type": "RandomisedInstallation", "n_lambda": 5.0,
                                                     "a_lambda": 0.5}
    assert manifest["stages"][0]["installation"] is None
    per_rank = manifest["seeds"]["per_rank"]
    assert [e["installation_seed"] for e in per_rank] == [6, 16]
    assert all(e["installation_seed"] == e["seed0"] + 6 for e in per_rank)
    data = surrDAMH.read_run(str(tmp_path / "a"))  # the round trip still reads
    assert list(data.notes[1].columns[-2:]) == list(INSTALLATION_NOTES_COLUMNS)
    run.continue_sampling_local(surrDAMH.Configuration(str(tmp_path / "b"), use_collector=False,
                                                       use_solvers_pool=False),
                                [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=20,
                                       installation=AveragedInstallation(weight_box=5.0))],
                                surrogate_evaluator=StateEvaluator(A_SURROGATE))
    continued = _manifest(tmp_path / "b")
    assert continued["seeds"]["per_rank"][0]["seed0"] >= 1_000_000
    assert continued["seeds"]["per_rank"][0]["installation_seed"] == continued["seeds"]["per_rank"][0]["seed0"] + 6
    assert continued["stages"][0]["installation"] == {"type": "AveragedInstallation", "j0": 1, "a_zeta": 1.0,
                                                      "weight_box": 5.0}


def test_local_runner_reproduces_the_mpi_sampler_with_the_rule_and_the_prior_step(tmp_path, monkeypatch):
    from surrDAMH.process_SAMPLER import run_SAMPLER

    def stages():
        return [Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=50),
                Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=300, subchain_length=2,
                      exact_step_probability=0.3, exact_step_proposal="prior",
                      installation=RandomisedInstallation(n_lambda=10, a_lambda=0.5))]

    _use_provider(monkeypatch, ScheduleProvider(available=lambda poll: poll % 4 == 0))
    problem = _problem()
    conf_mpi = _conf(tmp_path / "mpi")
    conf_mpi.resolve_problem_sizes(problem.no_parameters, problem.no_observations)
    run_SAMPLER(conf_mpi, problem.prior, problem.likelihood, stages(), solver_instance=problem.solver_instance,
                surrogate_evaluator=_version(0))
    _use_provider(monkeypatch, ScheduleProvider(available=lambda poll: poll % 4 == 0))
    run = _problem().run_sampling_local(_conf(tmp_path / "local"), stages(), surrogate_evaluator=_version(0))
    assert run.stage_results[1].counter_installations > 0 and run.stage_results[1].counter_exact_steps > 0
    for kind in ("samples", "raw_data", "subchain_stats", "notes", "installations"):
        for stage in ("alg0000_MH", "alg0001_DAMH-SMU"):
            path = os.path.join("sampling_output", kind, stage, "rank0000.csv")
            if kind in ("subchain_stats", "installations") and stage == "alg0000_MH":
                continue
            a = open(tmp_path / "mpi" / path, "rb").read()
            b = open(tmp_path / "local" / path, "rb").read()
            assert a == b, path


# ---------------------------------------------------------------------------------------------
# Averaged rule
# ---------------------------------------------------------------------------------------------

def _averaged_stage(rule, **kwargs):
    return Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=10, installation=rule,
                 name="direct", **kwargs)


def _candidates(n, low=-1.0, high=1.0, seed=3, extra=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        e = None
        if extra:
            e = {ERROR_MODEL_STATE_KEYS[0]: rng.uniform(-3, 3, 2), ERROR_MODEL_STATE_KEYS[1]: np.diag(rng.uniform(1, 3, 2))}
        out.append(StateEvaluator(rng.uniform(low, high, (2, 2)), rng.uniform(low, high, 2), extra=e))
    return out


def test_averaged_rule_one_one_is_the_mean_of_the_received_versions(tmp_path):
    # paper def:averaged with j0 = 1, a_zeta = 1: zeta_j = 1/j over the RECEIVED versions, the
    # stage-start network (theta-bar_0) replaced by the first received one (zeta_1 = 1); the legacy
    # polyak_averaging=True keeps the stage-start network in its mean instead
    candidates = _candidates(6)
    rule = _direct_damh(tmp_path / "rule", LocalEvaluatorProvider(candidates[0]),
                        _averaged_stage(AveragedInstallation(j0=1, a_zeta=1.0)))
    legacy = _direct_damh(tmp_path / "legacy", LocalEvaluatorProvider(candidates[0]),
                          Stage(algorithm="DAMH", max_evaluations=10, polyak_averaging=True, name="direct"))
    for j, candidate in enumerate(candidates[1:], start=1):
        rule._install_evaluator(candidate)
        legacy._install_evaluator(candidate)
        assert rule.counter_polyak_updates == legacy.counter_polyak_updates == j + 1
        for name in ("M", "b"):
            mean_received = np.mean([c.averageable_state()[name] for c in candidates[1:j + 1]], axis=0)
            np.testing.assert_allclose(rule._polyak_average[name], mean_received, rtol=1e-14, atol=1e-15)
            np.testing.assert_allclose(rule.surrogate_evaluator.averageable_state()[name], mean_received,
                                       rtol=1e-14, atol=1e-15)
            if j == 1:  # the first received version is installed unchanged
                np.testing.assert_array_equal(rule.surrogate_evaluator.averageable_state()[name],
                                              candidate.averageable_state()[name])
            mean_legacy = np.mean([c.averageable_state()[name] for c in candidates[:j + 1]], axis=0)
            np.testing.assert_allclose(legacy._polyak_average[name], mean_legacy, rtol=1e-14, atol=1e-15)
    assert rule.counter_polyak_updates == 6


def test_averaged_rule_general_gains(tmp_path):
    rule = AveragedInstallation(j0=2, a_zeta=0.6)
    candidates = _candidates(8)
    algo = _direct_damh(tmp_path, LocalEvaluatorProvider(candidates[0]), _averaged_stage(rule))
    expected = candidates[0].averageable_state()
    for k, candidate in enumerate(candidates[1:], start=2):
        algo._install_evaluator(candidate)
        zeta = min(1.0, ((k - 1) / 2) ** -0.6)  # the (k-1)-th received version (paper def:averaged)
        state = candidate.averageable_state()
        expected = {name: expected[name] + zeta * (state[name] - expected[name]) for name in expected}
        assert algo.counter_polyak_updates == k
        for name in expected:
            np.testing.assert_allclose(algo.surrogate_evaluator.averageable_state()[name], expected[name],
                                       rtol=0, atol=1e-15)
    assert algo._polyak_average["M"] is not None and rule.gain(2) == 1.0 and rule.gain(3) < 1.0


def test_weight_box_clips_the_model_part_only(tmp_path):
    omega = 0.5
    candidates = _candidates(4, low=-2.0, high=2.0, extra=True)
    stage = _averaged_stage(AveragedInstallation(j0=1, a_zeta=1.0, weight_box=omega), surrogate_error_model=True)
    algo = _direct_damh(tmp_path, LocalEvaluatorProvider(candidates[0]), stage)
    assert algo.error_model_enabled

    def projected(candidate):
        state = candidate.averageable_state()
        return {name: value if name in ERROR_MODEL_STATE_KEYS else np.clip(value, -omega, omega)
                for name, value in state.items()}

    # the stage-start evaluator (k = 1) is projected too
    first = projected(candidates[0])
    for name in first:
        np.testing.assert_array_equal(algo._polyak_average[name], first[name])
    assert np.max(np.abs(candidates[0].M)) > omega and np.max(np.abs(algo.surrogate_evaluator.M)) <= omega
    expected = first
    for k, candidate in enumerate(candidates[1:], start=2):
        algo._install_evaluator(candidate)
        state = projected(candidate)
        # zeta_j = 1/j for the j-th RECEIVED version, j = k - 1 (the stage-start one is average_0)
        expected = {name: expected[name] + (state[name] - expected[name]) / (k - 1) for name in expected}
    for name in expected:
        np.testing.assert_allclose(algo._polyak_average[name], expected[name], rtol=0, atol=1e-15)
    assert np.max(np.abs(algo._polyak_average["M"])) <= omega
    # error-model entries are averaged unclipped (over the received versions: zeta_1 = 1 drops the start)
    bias_mean = np.mean([c.extra[ERROR_MODEL_STATE_KEYS[0]] for c in candidates[1:]], axis=0)
    np.testing.assert_allclose(algo._polyak_average[ERROR_MODEL_STATE_KEYS[0]], bias_mean, rtol=1e-14)
    assert np.max(np.abs(bias_mean)) > omega or np.max(np.abs(algo._polyak_average[ERROR_MODEL_STATE_KEYS[1]])) > omega


def test_averaged_rule_local_run(tmp_path, monkeypatch, instances):
    provider = ScheduleProvider(available=lambda poll: poll % 3 == 0)
    _use_provider(monkeypatch, provider)
    rule = AveragedInstallation(j0=3, a_zeta=0.7, weight_box=2.0)
    out = tmp_path / "avg"
    run = _problem().run_sampling_local(_conf(out), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=50),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), subchain_length=2, max_samples=300,
              exact_step_probability=0.2, exact_step_proposal="prior", installation=rule)],
        surrogate_evaluator=_version(0))
    result, algo = run.stage_results[1], instances[1]
    log = _csv(out, "installations", result.name)
    notes = _csv(out, "notes", result.name)
    received = provider.handed_out
    assert len(received) == result.counter_versions_received == int(notes["versions_received"].iloc[0]) > 10
    # every received version is installed (all finite), one per poll; k counts the stage-start one
    assert result.counter_installations == len(received) == len(log) == int(notes["installations"].iloc[0])
    assert algo.counter_polyak_updates == len(received) + 1 == int(notes["polyak_updates"].iloc[0])
    assert list(log["version"]) == list(range(1, len(received) + 1))
    expected = _version(0).averageable_state()
    for k, candidate in enumerate(received, start=2):
        state = candidate.averageable_state()
        zeta = rule.gain(k - 1)  # the (k-1)-th received version (paper def:averaged)
        expected = {name: expected[name] + zeta * (state[name] - expected[name]) for name in expected}
    for name in expected:
        np.testing.assert_allclose(algo._polyak_average[name], expected[name], rtol=0, atol=1e-14)
    # no coin under the averaged rule: the install stream is untouched
    assert algo._generator_install.uniform() == np.random.RandomState(16).uniform()
