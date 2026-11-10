#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3 (2026-10-09): sampler-side Polyak averaging of the received surrogate evaluators
(``Stage.polyak_averaging``; note 18 §6 N1, note 20 §6, note 25 S3).

Pinned here: the running-mean arithmetic and its 1/k step; ``PyTorchNNEvaluator``'s state API
(prediction / jacobian / vjp of the averaged network); pass-through of non-averageable evaluators
with one note; flag off is byte-identical to the code before S3 with the NETWORK updater; flag on
installs the running mean of what the manager produced, restarts per stage and samples the
closed-form posterior; local runner == in-process MPI sampler; a Hamiltonian DAMH stage uses the
averaged network's gradients; a refused (non-finite) candidate advances nothing; Auto sets it.
The linear-Gaussian problem is the one of ``tests/unit/test_safety_kernel.py``.
"""

from __future__ import annotations

import glob
import hashlib
import os

import numpy as np
import pandas as pd
import pytest
import torch

import surrDAMH
from helpers_statistics import (FixedInitialSample, assert_matches_closed_form, chain_statistics,
                                read_weighted_samples)
from surrDAMH.auto import AUTO_POLYAK_AVERAGING, plan_auto
from surrDAMH.modules.algorithm_interfaces import EvaluatorProvider
from surrDAMH.modules.algorithm_interfaces_local import LocalEvaluatorProvider, LocalSolverAdapter
from surrDAMH.modules.algorithms import AlgorithmBase, Algorithm_DAMH, Sample
from surrDAMH.modules.error_model import ERROR_MODEL_STATE_KEYS
from surrDAMH.modules.proposals import GaussRandomWalk
from surrDAMH.proposals import Hamiltonian, RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates import NeuralNetworkUpdater, PolynomialSklearnUpdater
from surrDAMH.surrogates.parent import Evaluator
from surrDAMH.surrogates.torch_perceptron_minibatches import PyTorchMLP, PyTorchNNEvaluator

pytestmark = pytest.mark.unit

A = np.array([[1.0, 0.5], [0.0, 1.0]])
OBSERVED = A @ np.array([0.7, -0.4])
OBS_SD = np.array([0.3, 0.3])


class LinearSolver(Solver):
    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return A @ self.parameters


class LinearStateEvaluator(Evaluator):
    """``G~(x) = M x + b`` whose parameters ``{"M", "b"}`` can be averaged."""

    def __init__(self, M, b=(0.0, 0.0)):
        super().__init__(no_parameters=2, no_observations=2)
        self.M = np.asarray(M, dtype=float)
        self.b = np.asarray(b, dtype=float)

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.M.T + self.b

    def jacobian(self, datapoints):
        x = np.asarray(datapoints, dtype=float).reshape(-1)
        return self.M, self.M @ x + self.b

    def averageable_state(self):
        return {"M": self.M.copy(), "b": self.b.copy()}

    def with_averageable_state(self, state):
        return LinearStateEvaluator(state["M"], state["b"])


class PlainLinearEvaluator(Evaluator):
    """Not averageable (base ``averageable_state() -> None``)."""

    def __init__(self, M=A):
        super().__init__(no_parameters=2, no_observations=2)
        self.M = np.asarray(M, dtype=float)

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.M.T


class SequenceProvider(EvaluatorProvider):
    """Hands out a fixed sequence: a new evaluator becomes available on every ``every``-th poll."""

    def __init__(self, evaluators, every=5):
        self.sequence = list(evaluators)
        self.index = 0
        self.evaluator = self.sequence[0]
        self.every = every
        self.polls = 0
        self._pending = False

    def request_evaluator(self):
        return None

    def announce_waiting(self):
        return None

    def evaluator_is_available(self):
        self.polls += 1
        if not self._pending and self.polls % self.every == 0 and self.index + 1 < len(self.sequence):
            self._pending = True
        return self._pending

    def close(self):
        return None

    def get_evaluator(self):
        if self._pending:
            self.index += 1
            self.evaluator = self.sequence[self.index]
            self._pending = False
        return self.evaluator


def _problem():
    prior = surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(2)])
    return surrDAMH.Problem(prior, surrDAMH.distributions.Normal(mean=OBSERVED, sd=OBS_SD), solver=LinearSolver())


def _conf(output_dir, **kwargs):
    kwargs.setdefault("min_snapshots_initial", 50)
    kwargs.setdefault("min_snapshots_to_update", 40)
    return surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedInitialSample([0.3, -0.2]),
                                  save_snapshots_to_file=True, **kwargs)


def _net(seed=0):
    return NeuralNetworkUpdater(no_parameters=2, no_observations=2, hidden_layer_sizes=(16, 16), seed=seed,
                                iterations_batch=30, learning_rate=1e-2)


def _nn_evaluator(seed, hidden=(8, 8), dtype=torch.float32):
    model = PyTorchMLP(2, 2, hidden, "silu", seed=seed)
    evaluator = PyTorchNNEvaluator(2, 2, model, output_mean=np.array([0.1, -0.2]), output_scale=np.array([0.3, 0.3]))
    if dtype == torch.float64:
        evaluator.model.double()
    return evaluator


def _csv(output_dir, kind, stage_name):
    return pd.read_csv(os.path.join(str(output_dir), "sampling_output", kind, stage_name, "rank0000.csv"))


def _posterior():
    cov = np.linalg.inv(np.eye(2) + A.T @ np.diag(1.0 / OBS_SD ** 2) @ A)
    return cov @ (A.T @ np.diag(1.0 / OBS_SD ** 2) @ OBSERVED), cov


def _direct_damh(tmp_path, provider, stage=None, proposal=None, **kwargs):
    problem = _problem()
    conf = _conf(tmp_path, **kwargs)
    conf.resolve_problem_sizes(2, 2)
    if stage is None:
        stage = Stage(algorithm="DAMH", max_evaluations=10, polyak_averaging=True, name="direct")
    return Algorithm_DAMH(stage=stage, proposal=proposal or GaussRandomWalk(2, seed=1),
                          initial_sample=Sample(np.array([0.3, -0.2])), rank_world=0, conf=conf,
                          prior=problem.prior, likelihood=problem.likelihood,
                          observation_provider=LocalSolverAdapter(LinearSolver()),
                          evaluator_provider=provider, seed=2)


@pytest.fixture
def installs(monkeypatch):
    """Record, per algorithm instance (= stage of the chain), every committed candidate and the instance."""
    records: dict[int, dict] = {}
    original_commit = AlgorithmBase._commit_evaluator
    original_finalize = AlgorithmBase._finalize_run

    def commit(self, candidate, evaluator, pending):
        entry = records.setdefault(id(self), {"instance": self, "candidates": []})
        entry["candidates"].append(candidate)
        original_commit(self, candidate, evaluator, pending)

    def finalize(self):
        records.setdefault(id(self), {"instance": self, "candidates": []})
        original_finalize(self)

    monkeypatch.setattr(AlgorithmBase, "_commit_evaluator", commit)
    monkeypatch.setattr(AlgorithmBase, "_finalize_run", finalize)
    return records


def _mean_state(evaluators):
    # the error-model entries (S1) are averaged only by stages with surrogate_error_model=True
    states = [e.averageable_state() for e in evaluators]
    return {name: np.mean([s[name] for s in states], axis=0) for name in states[0]
            if name not in ERROR_MODEL_STATE_KEYS}


# ---------------------------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------------------------

def test_installed_state_is_the_running_mean_and_moves_like_one_over_k(tmp_path):
    rng = np.random.default_rng(3)
    candidates = [LinearStateEvaluator(rng.uniform(-1, 1, (2, 2)), rng.uniform(-1, 1, 2)) for _ in range(5)]
    algo = _direct_damh(tmp_path, LocalEvaluatorProvider(candidates[0]))
    assert algo.counter_polyak_updates == 1
    installed = [algo.surrogate_evaluator.averageable_state()]
    for k, candidate in enumerate(candidates[1:], start=2):
        returned = algo._install_evaluator(candidate)
        assert returned is algo.surrogate_evaluator and returned is not candidate
        assert algo.counter_polyak_updates == k
        installed.append(returned.averageable_state())
        expected = _mean_state(candidates[:k])
        for name in expected:
            np.testing.assert_allclose(installed[-1][name], expected[name], rtol=0, atol=1e-15)
    # step between consecutive installs: (state_k - mean_{k-1}) / k, bounded by diam / k (diam <= 2 here)
    for k in range(2, 6):
        step = max(np.max(np.abs(installed[k - 1][n] - installed[k - 2][n])) for n in installed[0])
        assert 0 < step * k <= 2.0
        expected_step = max(np.max(np.abs(candidates[k - 1].averageable_state()[n]
                                          - _mean_state(candidates[:k - 1])[n])) for n in installed[0]) / k
        assert step == pytest.approx(expected_step, rel=1e-12)
    # the candidates themselves are untouched
    np.testing.assert_array_equal(candidates[0].M, LinearStateEvaluator(candidates[0].M).M)


def test_flag_off_installs_the_candidate_itself(tmp_path):
    candidate = LinearStateEvaluator(A)
    stage = Stage(algorithm="DAMH", max_evaluations=10, name="direct")
    algo = _direct_damh(tmp_path, LocalEvaluatorProvider(candidate), stage=stage)
    assert algo.surrogate_evaluator is candidate and algo.counter_polyak_updates == 0
    other = LinearStateEvaluator(2 * A)
    assert algo._install_evaluator(other) is other and algo.counter_polyak_updates == 0


def test_polyak_averaging_field_type_and_star():
    assert Stage(algorithm="DAMH", max_evaluations=5).polyak_averaging is False
    stage = Stage(algorithm="DAMH", max_evaluations=5, polyak_averaging=np.bool_(True))
    assert stage.polyak_averaging is True
    assert "polyak_averaging*=True" in stage.describe()
    with pytest.raises(TypeError, match="polyak_averaging must be True or False"):
        Stage(algorithm="DAMH", max_evaluations=5, polyak_averaging=1)


# ---------------------------------------------------------------------------------------------
# PyTorchNNEvaluator state API
# ---------------------------------------------------------------------------------------------

def test_network_with_averageable_state_is_the_averaged_network():
    dtype = torch.float32  # the dtype PyTorchNNEvaluator.__call__ works in
    first, second = _nn_evaluator(1, dtype=dtype), _nn_evaluator(2, dtype=dtype)
    before = {k: v.clone() for k, v in first.model.state_dict().items()}
    state1, state2 = first.averageable_state(), second.averageable_state()
    assert all(v.dtype == np.float64 for v in state1.values())
    average = {name: 0.5 * (state1[name] + state2[name]) for name in state1}
    averaged = first.with_averageable_state(average)
    assert type(averaged) is PyTorchNNEvaluator and averaged is not first and averaged.model is not first.model
    assert next(averaged.model.parameters()).dtype == dtype
    assert (averaged.no_parameters, averaged.no_observations, averaged.use_gradients) == (2, 2, True)
    np.testing.assert_array_equal(averaged.output_mean, first.output_mean)
    np.testing.assert_array_equal(averaged.output_scale, first.output_scale)
    # the original is not modified
    for name, tensor in first.model.state_dict().items():
        assert torch.equal(tensor, before[name])

    # manual reference: average the two state dicts by hand into a third network
    manual_model = PyTorchMLP(2, 2, (8, 8), "silu", seed=None)
    s1, s2 = first.model.state_dict(), second.model.state_dict()
    manual_model.load_state_dict({name: ((s1[name].double() + s2[name].double()) / 2).to(s1[name].dtype) for name in s1})
    manual = PyTorchNNEvaluator(2, 2, manual_model, output_mean=first.output_mean, output_scale=first.output_scale)

    points = np.random.default_rng(0).normal(size=(20, 2))
    np.testing.assert_allclose(averaged(points), manual(points), rtol=0, atol=1e-10)
    assert not np.allclose(averaged(points), first(points))
    vector = np.array([0.7, -1.3])
    for x in points[:5]:
        jac_a, ev_a = averaged.jacobian(x)
        jac_m, ev_m = manual.jacobian(x)
        np.testing.assert_allclose(jac_a, jac_m, rtol=0, atol=1e-10)
        np.testing.assert_allclose(ev_a, ev_m, rtol=0, atol=1e-10)
        g_a, _ = averaged.vjp(x, vector)
        g_m, _ = manual.vjp(x, vector)
        np.testing.assert_allclose(g_a, g_m, rtol=0, atol=1e-10)
    # averaging a state with itself reproduces the network
    np.testing.assert_array_equal(first.with_averageable_state(state1)(points), first(points))


def test_network_with_averageable_state_keeps_a_float64_model_float64():
    evaluator = _nn_evaluator(1, dtype=torch.float64)
    state = evaluator.averageable_state()
    shifted = evaluator.with_averageable_state({name: value + 1e-12 for name, value in state.items()})
    assert all(t.dtype == torch.float64 for t in shifted.model.state_dict().values())
    for name, value in shifted.averageable_state().items():
        np.testing.assert_array_equal(value, state[name] + 1e-12)  # no float32 round trip


def test_network_with_averageable_state_rejects_a_wrong_layout():
    evaluator = _nn_evaluator(1)
    state = evaluator.averageable_state()
    with pytest.raises(KeyError, match="not in the model's state_dict"):
        evaluator.with_averageable_state({**state, "extra": np.zeros(1)})
    name = next(iter(state))
    with pytest.raises(ValueError, match="has shape"):
        evaluator.with_averageable_state({**state, name: np.zeros((1, 1, 1))})


def test_base_evaluator_is_not_averageable():
    evaluator = PlainLinearEvaluator()
    assert evaluator.averageable_state() is None
    with pytest.raises(NotImplementedError):
        evaluator.with_averageable_state({})


# ---------------------------------------------------------------------------------------------
# Non-averageable evaluators pass through
# ---------------------------------------------------------------------------------------------

def test_non_averageable_evaluators_pass_through_with_one_note(tmp_path, capsys):
    def run(name, flag):
        _problem().run_sampling_local(_conf(tmp_path / name, min_snapshots_to_update=20), [
            Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=100),
            Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=200, subchain_length=3,
                  polyak_averaging=flag)],
            surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=2))
    run("off", False)
    capsys.readouterr()
    run("on", True)
    out = capsys.readouterr().out
    assert out.count("polyak_averaging is set, but") == 1 and "exposes no averageable_state()" in out
    assert int(_csv(tmp_path / "on", "notes", "alg0001_DAMH-SMU")["polyak_updates"].iloc[0]) == 0
    for kind in ("samples", "raw_data", "subchain_stats", "notes"):
        a = open(tmp_path / "off" / "sampling_output" / kind / "alg0001_DAMH-SMU" / "rank0000.csv", "rb").read()
        b = open(tmp_path / "on" / "sampling_output" / kind / "alg0001_DAMH-SMU" / "rank0000.csv", "rb").read()
        assert a == b, kind


# ---------------------------------------------------------------------------------------------
# Byte identity, flag off, NETWORK updater. Reference md5s made with the code BEFORE S3
# (2026-10-09, scratchpad s3/s3_ref_run.py, the same two runs as below); the notes column
# "polyak_updates" appended by S3 is checked to be 0 and stripped before hashing.
# ---------------------------------------------------------------------------------------------

REFERENCE_MD5 = {
    "damh/last_sample/alg0000_MH/rank0000.npz": "f1229e06b00e628437c1a21dfa0caf3b",
    "damh/last_sample/alg0001_DAMH-SMU/rank0000.npz": "3c1794e56811b57a48ad0d1205f9d346",
    "damh/notes/alg0000_MH/rank0000.csv": "72a256a1fab5d79e48dab9e7f69fb75d",
    "damh/notes/alg0001_DAMH-SMU/rank0000.csv": "17e2a1ec9c5c6bc967cae33fbf91f90b",
    "damh/raw_data/alg0000_MH/rank0000.csv": "0be185d92f3c413c7ae3dbb64b67d852",
    "damh/raw_data/alg0001_DAMH-SMU/rank0000.csv": "df48d2fba685cfb0646eb576cb82eeb5",
    "damh/samples/alg0000_MH/rank0000.csv": "65eeb706d0921eb2611191dc84660b68",
    "damh/samples/alg0001_DAMH-SMU/rank0000.csv": "c2211b4695237033761a93a41ed0bed3",
    "damh/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "7f2355c2d894386d38b2060b1e3ef9dd",
    "hmc/last_sample/alg0000_MH/rank0000.npz": "61bd523fd8debf84cdcde0e12ba7f6af",
    "hmc/last_sample/alg0001_DAMH-SMU/rank0000.npz": "4188ed9240b00e2f6f34f85f4c92df1d",
    "hmc/notes/alg0000_MH/rank0000.csv": "e2d6ee45cea7e757453442abf5fd57d2",
    "hmc/notes/alg0001_DAMH-SMU/rank0000.csv": "960dc40eb927b69ad391d4014b2bc4e2",
    "hmc/raw_data/alg0000_MH/rank0000.csv": "b0206c82552f5e182fee821a694c0489",
    "hmc/raw_data/alg0001_DAMH-SMU/rank0000.csv": "f40ebdff2217ce0fba5a59167f76e853",
    "hmc/samples/alg0000_MH/rank0000.csv": "efbdd98ce2507cbf74584226efc4bbcd",
    "hmc/samples/alg0001_DAMH-SMU/rank0000.csv": "a289de2b047172ca987e8cd9e37f752c",
    "hmc/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "02ab38632fa94dea4938e039a4e15665",
}


def _stripped_md5(path, kind):
    if path.endswith(".npz"):
        with np.load(path) as z:
            return hashlib.md5(b"".join(np.ascontiguousarray(z[k]).tobytes() for k in sorted(z.files))).hexdigest()
    lines = open(path, "rb").read().split(b"\r\n")
    if kind == "notes":
        assert lines[0].split(b",")[-1] == b"polyak_updates"
        assert all(line.rsplit(b",", 1)[1] == b"0" for line in lines[1:] if line)
        lines = [line.rsplit(b",", 1)[0] if line else line for line in lines]
    return hashlib.md5(b"\r\n".join(lines)).hexdigest()


def _network_reference_runs(root, kwargs):
    """The two runs of scratchpad s3/s3_ref_run.py (same order, configurations and seeds)."""
    _problem().run_sampling_local(_conf(root / "damh"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=200),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=300, subchain_length=5, **kwargs)],
        surrogate_updater=_net())
    _problem().run_sampling_local(_conf(root / "hmc"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=150),
        Stage(algorithm="DAMH", proposal=Hamiltonian(step_size=0.2, num_steps=5), max_evaluations=200,
              subchain_length=1, **kwargs)],
        surrogate_updater=_net())


@pytest.mark.parametrize("kwargs", [{}, {"polyak_averaging": False}], ids=["default", "explicit_false"])
def test_flag_off_is_byte_identical_to_the_code_before_s3_with_the_network(tmp_path, kwargs):
    _network_reference_runs(tmp_path, kwargs)
    produced = {}
    for kind in ("samples", "raw_data", "subchain_stats", "notes", "adaptive_stats", "last_sample"):
        for path in glob.glob(os.path.join(str(tmp_path), "*", "sampling_output", kind, "*", "rank*.*")):
            produced[os.path.relpath(path, str(tmp_path)).replace("/sampling_output", "")] = _stripped_md5(path, kind)
    assert set(produced) == set(REFERENCE_MD5)
    changed = sorted(key for key, expected in REFERENCE_MD5.items() if produced[key] != expected)
    assert not changed, f"changed against the code before S3: {changed}"


# ---------------------------------------------------------------------------------------------
# Flag on, local network run
# ---------------------------------------------------------------------------------------------

def test_flag_on_installs_the_running_mean_restarts_per_stage_and_samples_the_posterior(tmp_path, monkeypatch,
                                                                                      installs):
    import surrDAMH.runner_local as runner_local
    produced: list = []

    class RecordingManager(runner_local.LocalSurrogateManager):
        def _maybe_update_evaluator(self):
            before = self._pending_evaluator
            super()._maybe_update_evaluator()
            if self._pending_evaluator is not None and self._pending_evaluator is not before:
                produced.append(self._pending_evaluator)

    monkeypatch.setattr(runner_local, "LocalSurrogateManager", RecordingManager)
    out = tmp_path / "on"
    run = _problem().run_sampling_local(_conf(out, min_snapshots_to_update=150), [
        Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=500, burn_in=True),
        Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=1500, subchain_length=1, polyak_averaging=True),
        Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=12000, subchain_length=1, polyak_averaging=True)],
        surrogate_updater=_net())
    stages = [entry for entry in installs.values() if entry["instance"].evaluator_provider is not None]
    assert [entry["instance"].stage.name for entry in stages] == ["alg0001_DAMH-SMU", "alg0002_DAMH-SMU"]
    for index, entry in enumerate(stages):
        algo, candidates = entry["instance"], entry["candidates"]
        result = run.stage_results[index + 1]
        # several retrainings were received; k = number of installs = notes column = StageResult
        assert len(candidates) > 2
        assert algo.counter_polyak_updates == len(candidates) == result.counter_polyak_updates
        assert int(_csv(out, "notes", result.name)["polyak_updates"].iloc[0]) == len(candidates)
        # every received candidate is an evaluator the manager produced, each once, in order
        positions = [next(i for i, e in enumerate(produced) if e is c) for c in candidates]
        assert positions == sorted(set(positions))
        # installed = running mean of the received ones (float64 average, float32 network)
        expected = _mean_state(candidates)
        installed = algo.surrogate_evaluator.averageable_state()
        for name in expected:
            np.testing.assert_allclose(algo._polyak_average[name], expected[name], rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(installed[name], expected[name], rtol=1e-6, atol=1e-7)
    # reset: the second stage starts from the raw last evaluator of the first (k = 1), not from its average
    first, second = stages
    assert second["candidates"][0] is first["candidates"][-1]
    print(f"\n[S3 polyak] installs per stage: {[len(s['candidates']) for s in stages]}, "
          f"retrainings produced: {len(produced)}")
    mean, cov = _posterior()
    statistics = chain_statistics(read_weighted_samples(out, "alg0002_DAMH-SMU", 2))
    print(f"[S3 polyak] {statistics.no_samples} samples: mean {statistics.mean} vs {mean} "
          f"(dev/SE {(statistics.mean - mean) / statistics.se_mean}), cov {statistics.cov.ravel()} vs {cov.ravel()} "
          f"(dev/SE {((statistics.cov - cov) / statistics.se_cov).ravel()})")
    assert_matches_closed_form(statistics, mean, cov, "S3 polyak DAMH-SMU network")


def test_new_stage_starts_with_k_equal_one(tmp_path):
    """Directly: a fresh algorithm instance on a provider whose evaluator is cached starts at k = 1."""
    provider = SequenceProvider([LinearStateEvaluator(A), LinearStateEvaluator(2 * A)], every=1)
    first = _direct_damh(tmp_path / "a", provider)
    assert first._refresh_surrogate_evaluator_if_needed() and first.counter_polyak_updates == 2
    np.testing.assert_allclose(first.surrogate_evaluator.M, 1.5 * A)
    second = _direct_damh(tmp_path / "b", provider)
    assert second.counter_polyak_updates == 1
    np.testing.assert_array_equal(second.surrogate_evaluator.M, 2 * A)


# ---------------------------------------------------------------------------------------------
# Local runner == MPI sampler (world of size 1, in process)
# ---------------------------------------------------------------------------------------------

def test_local_runner_reproduces_the_mpi_sampler_with_averaging(tmp_path, monkeypatch):
    import surrDAMH.process_SAMPLER as process_SAMPLER
    import surrDAMH.runner_local as runner_local
    from surrDAMH.process_SAMPLER import run_SAMPLER

    def sequence_factory(evaluator=None):
        return SequenceProvider([_nn_evaluator(seed) for seed in range(6)], every=7)

    monkeypatch.setattr(process_SAMPLER, "LocalEvaluatorProvider", sequence_factory)
    monkeypatch.setattr(runner_local, "LocalEvaluatorProvider", sequence_factory)

    def stages():
        return [Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=50),
                Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=200, subchain_length=3,
                      polyak_averaging=True)]

    problem = _problem()
    conf_mpi = _conf(tmp_path / "mpi")
    conf_mpi.resolve_problem_sizes(problem.no_parameters, problem.no_observations)
    run_SAMPLER(conf_mpi, problem.prior, problem.likelihood, stages(), solver_instance=problem.solver_instance,
                surrogate_evaluator=_nn_evaluator(0))
    run = _problem().run_sampling_local(_conf(tmp_path / "local"), stages(), surrogate_evaluator=_nn_evaluator(0))
    assert run.stage_results[1].counter_polyak_updates == 6
    for kind in ("samples", "raw_data", "subchain_stats", "notes"):
        for stage in ("alg0000_MH", "alg0001_DAMH-SMU"):
            path = os.path.join("sampling_output", kind, stage, "rank0000.csv")
            if kind == "subchain_stats" and stage == "alg0000_MH":
                continue
            a = open(tmp_path / "mpi" / path, "rb").read()
            b = open(tmp_path / "local" / path, "rb").read()
            assert a == b, path


# ---------------------------------------------------------------------------------------------
# Hamiltonian DAMH
# ---------------------------------------------------------------------------------------------

def test_hamiltonian_damh_uses_the_averaged_network_gradients(tmp_path, installs):
    out = tmp_path / "hmc"
    _problem().run_sampling_local(_conf(out, min_snapshots_to_update=30), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=150),
        Stage(algorithm="DAMH", proposal=Hamiltonian(step_size=0.2, num_steps=5), max_evaluations=300,
              subchain_length=1, polyak_averaging=True)],
        surrogate_updater=_net())
    entry = next(e for e in installs.values() if e["instance"].stage.name == "alg0001_DAMH-SMU")
    algo, candidates = entry["instance"], entry["candidates"]
    assert algo.counter_polyak_updates == len(candidates) > 2
    # a manually averaged network (state dicts averaged by hand)
    states = [c.model.state_dict() for c in candidates]
    manual_model = PyTorchMLP(2, 2, (16, 16), "silu", seed=None)
    manual_model.load_state_dict({name: (sum(s[name].double() for s in states) / len(states)).float()
                                  for name in states[0]})
    manual = PyTorchNNEvaluator(2, 2, manual_model, output_mean=candidates[-1].output_mean,
                                output_scale=candidates[-1].output_scale)
    x, v = np.array([0.4, -0.3]), np.array([1.0, -2.0])
    g_installed, e_installed = algo.surrogate_evaluator.vjp(x, v)
    g_manual, e_manual = manual.vjp(x, v)
    np.testing.assert_allclose(g_installed, g_manual, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(e_installed, e_manual, rtol=1e-5, atol=1e-6)
    # and the gradient the proposal sees is the one of the installed (averaged) network
    grad = algo._compute_surrogate_log_likelihood_gradient(x)
    expected, _ = algo.surrogate_evaluator.vjp(x, -algo.likelihood.grad_logpdf(algo.surrogate_evaluator(x[None, :])[0]))
    np.testing.assert_array_equal(grad, expected)
    assert not np.allclose(g_installed, candidates[-1].vjp(x, v)[0])


# ---------------------------------------------------------------------------------------------
# A refused (non-finite) candidate advances nothing
# ---------------------------------------------------------------------------------------------

def test_refused_non_finite_candidate_does_not_advance_k(tmp_path):
    nan = LinearStateEvaluator(np.full((2, 2), np.nan))
    good = [LinearStateEvaluator(A), LinearStateEvaluator(2 * A), LinearStateEvaluator(4 * A)]
    provider = SequenceProvider([good[0], good[1], nan, good[2]], every=1)
    algo = _direct_damh(tmp_path, provider)
    with pytest.warns(RuntimeWarning, match="NON-FINITE"):
        assert algo._refresh_surrogate_evaluator_if_needed()          # good[1]: k = 2
        assert not algo._refresh_surrogate_evaluator_if_needed()      # nan: refused
    assert algo.counter_polyak_updates == 2 and algo.counter_nonfinite_evaluators == 1
    np.testing.assert_allclose(algo.surrogate_evaluator.M, 1.5 * A)
    assert algo._refresh_surrogate_evaluator_if_needed()              # good[2]: k = 3
    assert algo.counter_polyak_updates == 3
    np.testing.assert_allclose(algo.surrogate_evaluator.M, 7.0 / 3.0 * A)
    np.testing.assert_allclose(algo._polyak_average["M"], 7.0 / 3.0 * A)


def test_frozen_stage_newest_evaluator_replaces_the_first(tmp_path):
    """MPI-style provider with get_newest_evaluator: the stale first one is not averaged in (k stays 1)."""

    class NewestProvider(SequenceProvider):
        def get_newest_evaluator(self):
            return self.sequence[-1]

    provider = NewestProvider([LinearStateEvaluator(A), LinearStateEvaluator(3 * A)])
    provider.evaluator = None
    provider.get_evaluator = lambda: provider.sequence[0]
    stage = Stage(algorithm="DAMH", max_evaluations=10, surrogate_model_updates=False, polyak_averaging=True,
                  name="direct")
    algo = _direct_damh(tmp_path, provider, stage=stage)
    assert algo.counter_polyak_updates == 1
    np.testing.assert_array_equal(algo.surrogate_evaluator.M, 3 * A)


# ---------------------------------------------------------------------------------------------
# Auto
# ---------------------------------------------------------------------------------------------

def test_auto_plan_sets_the_flag_on_the_chunks(tmp_path):
    from types import SimpleNamespace
    conf = surrDAMH.Configuration(output_dir=str(tmp_path / "o"), use_collector=False, use_solvers_pool=False)
    plan = plan_auto(SimpleNamespace(no_parameters=2, no_observations=2, solver_instance=None), conf,
                     budget=5_000, time_limit=None, mode="robust", local=True, surrogate_updater=None)
    assert plan.stages[0].polyak_averaging is False
    assert [s.polyak_averaging for s in plan.stages[1:]] == [AUTO_POLYAK_AVERAGING] * 4 == [True] * 4
    assert plan.manifest_entry()["polyak_averaging"] is True
    assert "polyak_averaging=True" in plan.describe()
    small = plan_auto(SimpleNamespace(no_parameters=2, no_observations=2, solver_instance=None), conf,
                      budget=50, time_limit=None, mode="robust", local=True, surrogate_updater=None)
    assert small.manifest_entry()["polyak_averaging"] is None


def test_auto_local_run_with_the_default_network_averages(tmp_path):
    out = tmp_path / "auto"
    conf = surrDAMH.Configuration(output_dir=str(out), use_collector=False, use_solvers_pool=False)
    run = _problem().run_sampling_local_auto(conf, budget=1_000, mode="robust")
    assert run.auto["surrogate"]["default"] is True
    updates = [result.counter_polyak_updates for result in run.stage_results[1:]]
    print(f"\n[S3 auto] polyak_updates per chunk: {updates}")
    assert all(k > 1 for k in updates)
    for result in run.stage_results[1:]:
        assert int(_csv(out, "notes", result.name)["polyak_updates"].iloc[0]) == result.counter_polyak_updates
