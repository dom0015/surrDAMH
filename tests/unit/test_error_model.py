#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1 (2026-10-09): the global Gaussian approximation-error model in the surrogate likelihood
(``surrDAMH.modules.error_model``, ``Stage.surrogate_error_model``; note 22 §4 P3, note 25 S1).

Pinned here: the estimation (Ledoit-Wolf branch, diagonal branch, few-residuals branch); the
corrected likelihood (log-density incl. the relative log-det term, gradient vs finite differences,
sd / cov noise); exactness of DAMH with a deliberately WRONG fixed surrogate (closed-form
posterior with and without the model, higher stage-2 acceptance with it, ``raw_data`` scored with
the corrected likelihood); the Hamiltonian gradient of the corrected potential; re-scoring after an
evaluator refresh (direct and through the S0/S2 "pending" path); S3 averaging of the error model;
flag off byte-identical to the code before S1 with the network AND the polynomial updater; a
non-Gaussian likelihood ignores the flag; the local surrogate manager attaches a model per
retraining; Auto sets the flag. The linear-Gaussian problem is the one of
``tests/unit/test_safety_kernel.py``.
"""

from __future__ import annotations

import glob
import hashlib
import io
import contextlib
import os

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch

import surrDAMH
from helpers_statistics import (FixedInitialSample, assert_matches_closed_form, chain_statistics,
                                read_weighted_samples)
from surrDAMH.auto import AUTO_SURROGATE_ERROR_MODEL, plan_auto
from surrDAMH.modules.algorithm_interfaces import EvaluatorProvider
from surrDAMH.modules.algorithm_interfaces_local import (LocalEvaluatorProvider, LocalSolverAdapter,
                                                         LocalSurrogateManager)
from surrDAMH.modules.algorithms import AlgorithmBase, Algorithm_DAMH, Sample
from surrDAMH.modules.error_model import (ERROR_MODEL_FULL_COV_MAX_OBS, ERROR_MODEL_STATE_KEYS, CorrectedNormal,
                                          ErrorModel, corrected_likelihood, error_model_for_evaluator,
                                          estimate_error_model)
from surrDAMH.modules.proposals import GaussRandomWalk, Proposal
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
#: a deliberately wrong linear surrogate (test_safety_kernel's A') with an offset of ~1 noise sd
A_SURROGATE = np.array([[1.1, 0.4], [0.05, 0.9]])
OFFSET = np.array([0.3, -0.25])


class LinearSolver(Solver):
    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return A @ self.parameters


class AffineEvaluator(Evaluator):
    """Fixed surrogate ``G~(x) = M x + b`` with its Jacobian."""

    def __init__(self, M=A_SURROGATE, b=OFFSET, error_model=None):
        super().__init__(no_parameters=2, no_observations=2)
        self.M = np.asarray(M, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.error_model = error_model

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.M.T + self.b

    def jacobian(self, datapoints):
        x = np.asarray(datapoints, dtype=float).reshape(-1)
        return self.M, self.M @ x + self.b


class SequenceProvider(EvaluatorProvider):
    """Hands out a fixed sequence: a new evaluator becomes available on every ``every``-th poll."""

    def __init__(self, evaluators, every=1):
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


class SequenceProposal(Proposal):
    """Proposes a fixed sequence (ignores the current state); plain MH ratio."""

    def __init__(self, samples):
        super().__init__()
        self.no_parameters = 2
        self._generator = np.random.RandomState(0)
        self.samples = [np.asarray(s, dtype=float) for s in samples]
        self.calls = 0

    def propose_sample(self, current_sample):
        sample = self.samples[self.calls % len(self.samples)]
        self.calls += 1
        return sample.copy()

    def get_log_acceptance_probability(self, log_likelihood_proposed, log_likelihood_current,
                                       log_prior_proposed, log_prior_current):
        return log_likelihood_proposed - log_likelihood_current, log_prior_proposed - log_prior_current


def _prior_residual_model(evaluator, n=500, seed=1):
    """``ErrorModel`` from the residuals ``A u - G~(u)`` on ``n`` prior draws."""
    u = np.random.default_rng(seed).normal(size=(n, 2))
    return estimate_error_model(u @ A.T - evaluator(u))


def _problem(likelihood=None, **kwargs):
    prior = surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(2)])
    if likelihood is None:
        likelihood = surrDAMH.distributions.Normal(mean=OBSERVED, sd=OBS_SD)
    return surrDAMH.Problem(prior, likelihood, solver=LinearSolver(), **kwargs)


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


def _poly(degree=2):
    return PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=degree)


def _csv(output_dir, kind, stage_name):
    return pd.read_csv(os.path.join(str(output_dir), "sampling_output", kind, stage_name, "rank0000.csv"))


def _posterior():
    cov = np.linalg.inv(np.eye(2) + A.T @ np.diag(1.0 / OBS_SD ** 2) @ A)
    return cov @ (A.T @ np.diag(1.0 / OBS_SD ** 2) @ OBSERVED), cov


def _direct_damh(tmp_path, provider, stage=None, proposal=None, likelihood=None, **kwargs):
    problem = _problem(likelihood=likelihood)
    conf = _conf(tmp_path, **kwargs)
    conf.resolve_problem_sizes(2, 2)
    if stage is None:
        stage = Stage(algorithm="DAMH", max_evaluations=10, surrogate_error_model=True, name="direct")
    return Algorithm_DAMH(stage=stage, proposal=proposal or GaussRandomWalk(2, seed=1),
                          initial_sample=Sample(np.array([0.3, -0.2])), rank_world=0, conf=conf,
                          prior=problem.prior, likelihood=problem.likelihood,
                          observation_provider=LocalSolverAdapter(LinearSolver()),
                          evaluator_provider=provider, seed=2)


def _gaussian_logpdf_relative(y, mean_shift, noise_cov, total_cov, g):
    """Hand-written ``log N(y; g + shift, total) - log N(.; ., noise)`` normalisation convention."""
    v = y - mean_shift - g
    return (-0.5 * v @ np.linalg.solve(total_cov, v) - 0.5 * np.linalg.slogdet(total_cov)[1]
            + 0.5 * np.linalg.slogdet(noise_cov)[1])


@pytest.fixture
def installs(monkeypatch):
    """Record, per algorithm instance, every committed candidate and the instance."""
    records: dict[int, dict] = {}
    original_commit = AlgorithmBase._commit_evaluator

    def commit(self, candidate, evaluator, pending):
        entry = records.setdefault(id(self), {"instance": self, "candidates": []})
        entry["candidates"].append(candidate)
        original_commit(self, candidate, evaluator, pending)

    monkeypatch.setattr(AlgorithmBase, "_commit_evaluator", commit)
    return records


# ---------------------------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------------------------

def test_estimation_ledoit_wolf_branch_recovers_mean_and_covariance():
    rng = np.random.default_rng(0)
    mean = np.array([0.5, -1.0, 0.2])
    root = np.array([[1.0, 0.0, 0.0], [0.6, 0.5, 0.0], [-0.2, 0.3, 0.4]])
    cov = root @ root.T
    residuals = mean + rng.normal(size=(20_000, 3)) @ root.T
    model = estimate_error_model(residuals)
    assert model.cov.shape == (3, 3) and not model.is_diagonal and model.no_residuals == 20_000
    np.testing.assert_allclose(model.bias, residuals.mean(axis=0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(model.bias, mean, atol=0.03)
    np.testing.assert_allclose(model.cov, cov, atol=0.03)
    # Ledoit-Wolf shrinks toward tr/m * I: trace of the ML covariance preserved, PSD, symmetric
    ml_cov = np.cov(residuals, rowvar=False, bias=True)
    assert np.trace(model.cov) == pytest.approx(np.trace(ml_cov), rel=1e-12)
    assert np.all(np.linalg.eigvalsh(model.cov) > 0)
    np.testing.assert_array_equal(model.cov, model.cov.T)
    from sklearn.covariance import LedoitWolf
    np.testing.assert_allclose(model.cov, LedoitWolf().fit(residuals).covariance_, rtol=1e-12, atol=1e-15)
    # few residuals with strong correlation: shrinkage visible, still trace preserving and PD
    few = rng.normal(size=(8, 1)) @ np.ones((1, 3)) + 1e-3 * rng.normal(size=(8, 3))
    shrunk = estimate_error_model(few)
    assert np.trace(shrunk.cov) == pytest.approx(np.trace(np.cov(few, rowvar=False, bias=True)), rel=1e-12)
    assert np.all(np.linalg.eigvalsh(shrunk.cov) > 0)
    assert np.linalg.eigvalsh(np.cov(few, rowvar=False, bias=True))[0] < 1e-5  # the ML estimate is near-singular


def test_estimation_diagonal_branch_above_the_threshold():
    m = ERROR_MODEL_FULL_COV_MAX_OBS + 10
    residuals = np.random.default_rng(1).normal(size=(200, m)) * np.linspace(0.1, 2.0, m)
    model = estimate_error_model(residuals)
    assert model.is_diagonal and model.cov.shape == (m,)
    np.testing.assert_allclose(model.cov, residuals.var(axis=0), rtol=1e-12)
    np.testing.assert_array_equal(model.covariance_matrix(), np.diag(model.cov))
    assert model.cov_trace == pytest.approx(float(np.sum(residuals.var(axis=0))))
    # m at the threshold is still full
    assert estimate_error_model(np.random.default_rng(1).normal(size=(200, ERROR_MODEL_FULL_COV_MAX_OBS))).cov.ndim == 2


def test_estimation_with_few_residuals_and_degenerate_input():
    residuals = np.array([[1.0, 2.0], [3.0, 2.0], [2.0, 2.0]])  # n = 3 < max(5, m + 1)
    model = estimate_error_model(residuals)
    np.testing.assert_allclose(model.bias, [2.0, 2.0])
    np.testing.assert_allclose(model.cov, np.diag([2.0 / 3.0, 0.0]))  # var floored at 0, ddof = 0; still (m, m)
    single = estimate_error_model(np.array([[0.5, -0.5]]))
    np.testing.assert_allclose(single.bias, [0.5, -0.5])
    np.testing.assert_array_equal(single.cov, np.zeros((2, 2)))
    # non-finite rows are dropped; nothing finite -> unavailable
    with_nan = estimate_error_model(np.vstack([residuals, [[np.nan, 1.0]]]))
    np.testing.assert_allclose(with_nan.bias, [2.0, 2.0])
    assert estimate_error_model(np.full((3, 2), np.nan)) is None
    assert estimate_error_model(np.empty((0, 2))) is None
    with np.errstate(over="ignore"):
        assert estimate_error_model(np.array([[1e300, 0.0], [-1e300, 0.0], [1e300, 1.0]] * 3)) is None  # overflow
    assert error_model_for_evaluator(AffineEvaluator(), None, None) is None


def test_state_round_trip_and_validation():
    model = ErrorModel(bias=[0.1, -0.2], cov=[[0.3, 0.1], [0.1, 0.2]])
    state = model.as_state()
    assert set(state) == set(ERROR_MODEL_STATE_KEYS) and all(v.dtype == np.float64 for v in state.values())
    state[ERROR_MODEL_STATE_KEYS[0]][0] = 99.0  # copies
    assert model.bias[0] == 0.1
    back = ErrorModel.from_state(model.as_state())
    np.testing.assert_array_equal(back.bias, model.bias)
    np.testing.assert_array_equal(back.cov, model.cov)
    assert model.bias_norm == pytest.approx(np.hypot(0.1, 0.2)) and model.cov_trace == pytest.approx(0.5)
    with pytest.raises(ValueError, match="cov has shape"):
        ErrorModel(bias=[0.0, 0.0], cov=np.eye(3))


# ---------------------------------------------------------------------------------------------
# Corrected likelihood
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("noise", ["sd", "scalar_sd", "cov"])
@pytest.mark.parametrize("model_kind", ["full", "diagonal"])
def test_corrected_likelihood_logpdf_and_gradient(noise, model_kind):
    rng = np.random.default_rng(2)
    m = 3
    y = rng.normal(size=m)
    if noise == "sd":
        likelihood = surrDAMH.distributions.Normal(mean=y, sd=np.array([0.2, 0.3, 0.5]))
        noise_cov = np.diag([0.04, 0.09, 0.25])
    elif noise == "scalar_sd":
        likelihood = surrDAMH.distributions.Normal(mean=y, sd=0.4)
        noise_cov = 0.16 * np.eye(m)
    else:
        noise_cov = np.array([[0.2, 0.05, 0.0], [0.05, 0.3, 0.1], [0.0, 0.1, 0.4]])
        likelihood = surrDAMH.distributions.Normal(mean=y, cov=noise_cov)
    bias = np.array([0.1, -0.3, 0.2])
    if model_kind == "full":
        surr_cov = np.array([[0.05, 0.02, 0.01], [0.02, 0.08, 0.0], [0.01, 0.0, 0.03]])
        model = ErrorModel(bias=bias, cov=surr_cov)
    else:
        surr_cov = np.diag([0.05, 0.08, 0.03])
        model = ErrorModel(bias=bias, cov=np.diag(surr_cov).copy())
    corrected = corrected_likelihood(likelihood, model)
    assert isinstance(corrected, CorrectedNormal) and isinstance(corrected, surrDAMH.distributions.Normal)
    np.testing.assert_allclose(corrected.mean, y - bias)
    total = noise_cov + surr_cov
    expected_offset = -0.5 * np.linalg.slogdet(total)[1] + 0.5 * np.linalg.slogdet(noise_cov)[1]
    assert corrected.log_normalization == pytest.approx(expected_offset, rel=1e-12)
    for _ in range(5):
        g = rng.normal(size=m)
        assert corrected.logpdf(g) == pytest.approx(_gaussian_logpdf_relative(y, bias, noise_cov, total, g), rel=1e-12)
        # gradient (w.r.t. the observation argument) vs central finite differences
        h = 1e-6
        fd = np.array([(corrected.logpdf(g + h * e) - corrected.logpdf(g - h * e)) / (2 * h) for e in np.eye(m)])
        np.testing.assert_allclose(corrected.grad_logpdf(g), fd, rtol=1e-6, atol=1e-7)
    # covariance of the corrected Normal
    np.testing.assert_allclose(np.atleast_2d(corrected.get_covariance()) if corrected.get_covariance().ndim == 2
                               else np.diag(corrected.get_covariance() ** 2), total, rtol=1e-12)
    # a zero error model reproduces the exact likelihood (same normalisation convention)
    zero = corrected_likelihood(likelihood, ErrorModel(bias=np.zeros(m), cov=np.zeros((m, m))))
    g = rng.normal(size=m)
    assert zero.logpdf(g) == pytest.approx(likelihood.logpdf(g), rel=1e-12, abs=1e-12)


def test_corrected_likelihood_propagates_non_finite_surrogate_values():
    """A divergent leapfrog evaluates the surrogate at inf/NaN: the gradient must come back non-finite
    (as with the exact Normal), not raise (found by tests/mpi/test_mpi_auto.py fast mode, 2026-10-09)."""
    likelihood = surrDAMH.distributions.Normal(mean=np.zeros(2), sd=0.3)
    corrected = corrected_likelihood(likelihood, ErrorModel(bias=np.zeros(2), cov=np.array([[0.1, 0.02], [0.02, 0.1]])))
    assert not np.all(np.isfinite(corrected.grad_logpdf(np.array([np.inf, 1.0]))))
    assert not np.isfinite(corrected.logpdf(np.array([np.nan, 1.0])))


def test_corrected_likelihood_needs_a_normal():
    with pytest.raises(TypeError, match="Normal likelihood"):
        corrected_likelihood(surrDAMH.distributions.FromScipy(scipy.stats.multivariate_normal(np.zeros(2))),
                             ErrorModel(bias=np.zeros(2), cov=np.eye(2)))
    with pytest.raises(ValueError, match="observations"):
        corrected_likelihood(surrDAMH.distributions.Normal(mean=np.zeros(3), sd=1.0),
                             ErrorModel(bias=np.zeros(2), cov=np.eye(2)))


# ---------------------------------------------------------------------------------------------
# Evaluator state API (S3 interplay)
# ---------------------------------------------------------------------------------------------

def test_base_evaluator_state_carries_the_error_model():
    plain = AffineEvaluator(error_model=None)
    assert plain.averageable_state() is None
    with pytest.raises(NotImplementedError):
        plain.with_averageable_state({})
    model = ErrorModel(bias=[0.1, 0.2], cov=np.eye(2))
    evaluator = AffineEvaluator(error_model=model)
    state = evaluator.averageable_state()
    assert set(state) == set(ERROR_MODEL_STATE_KEYS)
    new = evaluator.with_averageable_state({k: 2 * v for k, v in state.items()})
    assert new is not evaluator and evaluator.error_model is model
    np.testing.assert_array_equal(new.error_model.bias, [0.2, 0.4])
    np.testing.assert_array_equal(new(np.ones((1, 2))), evaluator(np.ones((1, 2))))
    with pytest.raises(NotImplementedError):
        evaluator.with_averageable_state({**state, "other": np.zeros(1)})


def test_network_state_includes_the_error_model():
    evaluator = PyTorchNNEvaluator(2, 2, PyTorchMLP(2, 2, (8, 8), "silu", seed=1),
                                   output_mean=np.zeros(2), output_scale=np.ones(2))
    plain_keys = set(evaluator.averageable_state())
    assert not plain_keys & set(ERROR_MODEL_STATE_KEYS)
    evaluator.error_model = ErrorModel(bias=[0.1, -0.1], cov=0.5 * np.eye(2))
    state = evaluator.averageable_state()
    assert set(state) == plain_keys | set(ERROR_MODEL_STATE_KEYS)
    shifted = {k: (v + 1.0 if k in ERROR_MODEL_STATE_KEYS else v) for k, v in state.items()}
    new = evaluator.with_averageable_state(shifted)
    np.testing.assert_allclose(new.error_model.bias, [1.1, 0.9])
    np.testing.assert_allclose(evaluator.error_model.bias, [0.1, -0.1])  # original untouched
    x = np.random.default_rng(0).normal(size=(4, 2))
    np.testing.assert_array_equal(new(x), evaluator(x))
    # without error-model entries the new evaluator keeps this one's model
    weights_only = {k: v for k, v in state.items() if k not in ERROR_MODEL_STATE_KEYS}
    assert evaluator.with_averageable_state(weights_only).error_model is evaluator.error_model


# ---------------------------------------------------------------------------------------------
# Sampler exactness with a wrong fixed surrogate
# ---------------------------------------------------------------------------------------------

def _wrong_surrogate_run(output_dir, flag, evaluations=8000, proposal=None, subchain_length=1):
    evaluator = AffineEvaluator()
    evaluator.error_model = _prior_residual_model(evaluator)
    with contextlib.redirect_stdout(io.StringIO()):
        _problem().run_sampling_local(_conf(output_dir), [
            Stage(algorithm="DAMH", proposal=proposal or RandomWalk(scale=0.8), max_evaluations=evaluations,
                  subchain_length=subchain_length, surrogate_error_model=flag)], surrogate_evaluator=evaluator)
    return evaluator


def test_wrong_surrogate_with_the_error_model_is_exact_and_accepts_more(tmp_path):
    mean, cov = _posterior()
    rates = {}
    for flag in (False, True):
        out = tmp_path / str(flag)
        evaluator = _wrong_surrogate_run(out, flag)
        statistics = chain_statistics(read_weighted_samples(out, "alg0000_DAMH-SMU", 2))
        stats = _csv(out, "subchain_stats", "alg0000_DAMH-SMU")
        moved = stats[stats["outer_proposed_changed"] == 1]
        rates[flag] = float(moved["outer_accepted"].mean())
        print(f"\n[S1 wrong surrogate, error model {flag}] {statistics.no_samples} samples: "
              f"mean dev/SE {np.round((statistics.mean - mean) / statistics.se_mean, 2)}, "
              f"cov dev/SE {np.round(((statistics.cov - cov) / statistics.se_cov).ravel(), 2)}, "
              f"outer_acc_given_move {rates[flag]:.3f} ({len(moved)} moves); model bias "
              f"{np.round(evaluator.error_model.bias, 3)}, cov {np.round(evaluator.error_model.cov.ravel(), 4)}")
        assert_matches_closed_form(statistics, mean, cov, f"S1 wrong surrogate, error model {flag}")
        raw = _csv(out, "raw_data", "alg0000_DAMH-SMU")
        likelihood = surrDAMH.distributions.Normal(mean=OBSERVED, sd=OBS_SD)
        used = corrected_likelihood(likelihood, evaluator.error_model) if flag else likelihood
        for state_type in ("prerejected", "accepted"):
            row = raw[raw["state_type"] == state_type].iloc[3]
            g = evaluator(row[["par_0", "par_1"]].to_numpy(dtype=float)[None, :])[0]
            np.testing.assert_allclose(row[["obs_approx_0", "obs_approx_1"]].to_numpy(dtype=float), g, rtol=1e-12)
            if state_type == "prerejected":  # the logged log-likelihood of a pre-rejected row is the surrogate one
                assert row["log_likelihood"] == pytest.approx(used.logpdf(g), rel=1e-12)
    assert rates[True] > rates[False] + 0.1, rates


def test_rescore_uses_the_corrected_likelihood_of_the_installed_evaluator(tmp_path):
    first = AffineEvaluator(error_model=ErrorModel(bias=[0.1, 0.0], cov=0.01 * np.eye(2)))
    second = AffineEvaluator(error_model=ErrorModel(bias=[-0.2, 0.3], cov=np.array([[0.05, 0.01], [0.01, 0.02]])))
    provider = SequenceProvider([first, second], every=1)
    algo = _direct_damh(tmp_path, provider)
    likelihood = algo.likelihood
    g0 = first(algo.current.parameters[None, :])[0]
    assert algo.current.log_likelihood_approx == pytest.approx(
        corrected_likelihood(likelihood, first.error_model).logpdf(g0), rel=1e-14)
    algo._propose_new_sample_using_subchain()  # refreshes to `second` and re-scores self.current
    assert algo.surrogate_evaluator is second
    g = second(algo.current.parameters[None, :])[0]
    expected = corrected_likelihood(likelihood, second.error_model).logpdf(g)
    assert algo.current.log_likelihood_approx == pytest.approx(expected, rel=1e-14)
    assert algo.likelihood_surrogate.logpdf(g) == pytest.approx(expected, rel=1e-14)
    assert algo.current.log_likelihood_approx != pytest.approx(
        corrected_likelihood(likelihood, first.error_model).logpdf(g))


def test_pending_rescore_after_a_sub_chain_without_evaluated_steps(tmp_path):
    """The refresh happens in a sub-chain whose only step is non-finite (S0/item A path): the chain state
    is re-scored with the NEW corrected likelihood at the next evaluated step."""
    first = AffineEvaluator(error_model=ErrorModel(bias=[0.1, 0.0], cov=0.01 * np.eye(2)))
    second = AffineEvaluator(error_model=ErrorModel(bias=[-0.2, 0.3], cov=np.array([[0.05, 0.01], [0.01, 0.02]])))
    provider = SequenceProvider([first, second], every=1)
    stage = Stage(algorithm="DAMH", max_samples=2, subchain_length=1, surrogate_error_model=True, name="direct")
    algo = _direct_damh(tmp_path, provider, stage=stage, proposal=SequenceProposal([[np.nan, np.nan], [0.5, 0.1]]))
    with pytest.warns(RuntimeWarning, match="NON-FINITE"):
        algo._propose_new_sample_using_subchain()
    assert algo.surrogate_evaluator is second and algo._surrogate_rescore_pending
    stale = algo.current.log_likelihood_approx
    algo._propose_new_sample_using_subchain()
    assert not algo._surrogate_rescore_pending
    g = second(algo.current.parameters[None, :])[0]
    assert algo.current.log_likelihood_approx == pytest.approx(
        corrected_likelihood(algo.likelihood, second.error_model).logpdf(g), rel=1e-14)
    assert algo.current.log_likelihood_approx != stale


def test_hamiltonian_gradient_of_the_corrected_potential(tmp_path):
    evaluator = AffineEvaluator(error_model=ErrorModel(bias=[0.2, -0.1], cov=np.array([[0.04, 0.01], [0.01, 0.03]])))
    algo = _direct_damh(tmp_path, LocalEvaluatorProvider(evaluator))
    corrected = algo.likelihood_surrogate
    assert isinstance(corrected, CorrectedNormal)

    def log_likelihood(x):
        return corrected.logpdf(evaluator(x[None, :])[0])

    h = 1e-6
    for x in np.random.default_rng(3).normal(size=(5, 2)):
        gradient = algo._compute_surrogate_log_likelihood_gradient(x)  # grad of -log L~ (vjp path)
        fd = np.array([(log_likelihood(x + h * e) - log_likelihood(x - h * e)) / (2 * h) for e in np.eye(2)])
        np.testing.assert_allclose(gradient, -fd, rtol=1e-6, atol=1e-6)
        # J~^T Sigma_tot^-1 (G~ + eps - y)
        total = np.diag(OBS_SD ** 2) + evaluator.error_model.cov
        expected = evaluator.M.T @ np.linalg.solve(total, evaluator(x[None, :])[0] + evaluator.error_model.bias - OBSERVED)
        np.testing.assert_allclose(gradient, expected, rtol=1e-12, atol=1e-14)


def test_hamiltonian_damh_stage_with_the_error_model_samples_the_posterior(tmp_path):
    out = tmp_path / "hmc"
    _wrong_surrogate_run(out, True, evaluations=4000, proposal=Hamiltonian(step_size=0.2, num_steps=5))
    mean, cov = _posterior()
    statistics = chain_statistics(read_weighted_samples(out, "alg0000_DAMH-SMU", 2))
    print(f"\n[S1 HMC] {statistics.no_samples} samples: mean dev/SE {np.round((statistics.mean - mean) / statistics.se_mean, 2)}")
    assert_matches_closed_form(statistics, mean, cov, "S1 Hamiltonian DAMH with error model")


# ---------------------------------------------------------------------------------------------
# S3 interplay
# ---------------------------------------------------------------------------------------------

def test_polyak_averaging_averages_the_error_model(tmp_path):
    rng = np.random.default_rng(4)
    models = []
    for _ in range(4):
        root = rng.uniform(-0.3, 0.3, (2, 2))
        models.append(ErrorModel(bias=rng.uniform(-1, 1, 2), cov=root @ root.T + 0.01 * np.eye(2)))
    candidates = [AffineEvaluator(error_model=model) for model in models]
    stage = Stage(algorithm="DAMH", max_evaluations=10, surrogate_error_model=True, polyak_averaging=True, name="direct")
    algo = _direct_damh(tmp_path, SequenceProvider(candidates, every=1), stage=stage)
    for k in range(2, 5):
        assert algo._refresh_surrogate_evaluator_if_needed()
        assert algo.counter_polyak_updates == k
        installed = algo.surrogate_evaluator.error_model
        np.testing.assert_allclose(installed.bias, np.mean([m.bias for m in models[:k]], axis=0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(installed.cov, np.mean([m.cov for m in models[:k]], axis=0), rtol=0, atol=1e-15)
        expected = corrected_likelihood(algo.likelihood, installed)
        g = np.array([0.4, -0.2])
        assert algo.likelihood_surrogate.logpdf(g) == pytest.approx(expected.logpdf(g), rel=1e-14)
    # the received candidates are not modified
    for candidate, model in zip(candidates, models):
        assert candidate.error_model is model


def test_polyak_without_the_error_model_flag_is_unchanged_for_error_model_only_evaluators(tmp_path, capsys):
    candidates = [AffineEvaluator(error_model=ErrorModel(bias=[float(i), 0.0], cov=np.eye(2))) for i in range(3)]
    stage = Stage(algorithm="DAMH", max_evaluations=10, polyak_averaging=True, name="direct")
    algo = _direct_damh(tmp_path, SequenceProvider(candidates, every=1), stage=stage)
    assert algo._refresh_surrogate_evaluator_if_needed()
    assert algo.surrogate_evaluator is candidates[1] and algo.counter_polyak_updates == 0
    assert algo.likelihood_surrogate is algo.likelihood
    assert capsys.readouterr().out.count("exposes no averageable_state()") == 1


# ---------------------------------------------------------------------------------------------
# Byte identity, flag off: NETWORK and POLYNOMIAL updater. Reference md5s made with the code BEFORE
# S1 (2026-10-09, scratchpad s1/s1_ref_run.py, the same five runs as below); S1 appends no column
# to these files. Line endings are normalised to "\n" before hashing.
# C1b (2026-10-11, item S): the S2 exact random walk of "damh_s2"/"poly_s2" now reads the chain's shared
# posterior statistics; their 10 stage-1 entries were re-based on the C1b code ("C1b rebase").
# ---------------------------------------------------------------------------------------------

REFERENCE_MD5 = {
    "damh/last_sample/alg0000_MH/rank0000.npz": "f1229e06b00e628437c1a21dfa0caf3b",
    "damh/last_sample/alg0001_DAMH-SMU/rank0000.npz": "3c1794e56811b57a48ad0d1205f9d346",
    "damh/notes/alg0000_MH/rank0000.csv": "75b22009fa7e28745769ef1549a66bdc",
    "damh/notes/alg0001_DAMH-SMU/rank0000.csv": "1e48b3f2bd504bdc26bb6c04f9c0fc95",
    "damh/raw_data/alg0000_MH/rank0000.csv": "ba2017e6a0311fc3c90c5541b06fdb87",
    "damh/raw_data/alg0001_DAMH-SMU/rank0000.csv": "ee527d04bf3a75d78b46cd628a0dc0ef",
    "damh/samples/alg0000_MH/rank0000.csv": "c7e006aa7b771926dc58bb7cb92d1007",
    "damh/samples/alg0001_DAMH-SMU/rank0000.csv": "fa96d47009c6c605b3c8af59649658fa",
    "damh/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "981a38dc7c9835e5c7f985d881569d80",
    "damh_s2/last_sample/alg0000_MH/rank0000.npz": "f1229e06b00e628437c1a21dfa0caf3b",
    "damh_s2/last_sample/alg0001_DAMH-SMU/rank0000.npz": "4e32b80c1952ab4e35693068788f5246",  # C1b rebase (S2 exact step reads the shared statistics)
    "damh_s2/notes/alg0000_MH/rank0000.csv": "75b22009fa7e28745769ef1549a66bdc",
    "damh_s2/notes/alg0001_DAMH-SMU/rank0000.csv": "282fe88312c630dc1dfb5642a4ba1122",  # C1b rebase (S2 exact step reads the shared statistics)
    "damh_s2/raw_data/alg0000_MH/rank0000.csv": "ba2017e6a0311fc3c90c5541b06fdb87",
    "damh_s2/raw_data/alg0001_DAMH-SMU/rank0000.csv": "ae9f473c899dceb19d9f665afda84133",  # C1b rebase (S2 exact step reads the shared statistics)
    "damh_s2/samples/alg0000_MH/rank0000.csv": "c7e006aa7b771926dc58bb7cb92d1007",
    "damh_s2/samples/alg0001_DAMH-SMU/rank0000.csv": "119ed45575c554fece6f6bdf614d3720",  # C1b rebase (S2 exact step reads the shared statistics)
    "damh_s2/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "31a4fc6796756b40abdbda54b8eec0d9",  # C1b rebase (S2 exact step reads the shared statistics)
    "hmc/last_sample/alg0000_MH/rank0000.npz": "61bd523fd8debf84cdcde0e12ba7f6af",
    "hmc/last_sample/alg0001_DAMH-SMU/rank0000.npz": "4188ed9240b00e2f6f34f85f4c92df1d",
    "hmc/notes/alg0000_MH/rank0000.csv": "fea7956b9ed1c1f9d35bb7b61df58f47",
    "hmc/notes/alg0001_DAMH-SMU/rank0000.csv": "d9ffcca5294efb682a5c995ed181a5d4",
    "hmc/raw_data/alg0000_MH/rank0000.csv": "a887e268b665cccbdbe87d0495b091b4",
    "hmc/raw_data/alg0001_DAMH-SMU/rank0000.csv": "e8e82b2611f862a1cc9536bd0779c5e0",
    "hmc/samples/alg0000_MH/rank0000.csv": "382318f7bb471692939f1d9f9d29ccfe",
    "hmc/samples/alg0001_DAMH-SMU/rank0000.csv": "4cab45b09896beff0649f793b265620f",
    "hmc/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "eab51a0096fd6ab644a423e40e8b68ba",
    "poly/last_sample/alg0000_MH/rank0000.npz": "f1229e06b00e628437c1a21dfa0caf3b",
    "poly/last_sample/alg0001_DAMH-SMU/rank0000.npz": "61ef32141102f63146b1d99e72983bcc",
    "poly/notes/alg0000_MH/rank0000.csv": "75b22009fa7e28745769ef1549a66bdc",
    "poly/notes/alg0001_DAMH-SMU/rank0000.csv": "e8ef4e02f3698c6eaef443f6e05b0544",
    "poly/raw_data/alg0000_MH/rank0000.csv": "ba2017e6a0311fc3c90c5541b06fdb87",
    "poly/raw_data/alg0001_DAMH-SMU/rank0000.csv": "9bf7c3dc88b9c210eb22d62026ad7eeb",
    "poly/samples/alg0000_MH/rank0000.csv": "c7e006aa7b771926dc58bb7cb92d1007",
    "poly/samples/alg0001_DAMH-SMU/rank0000.csv": "3cc35944effa827d17fa8c5c63915dab",
    "poly/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "c7a921fd66d109a9d77c8afdd3516890",
    "poly_s2/last_sample/alg0000_MH/rank0000.npz": "f1229e06b00e628437c1a21dfa0caf3b",
    "poly_s2/last_sample/alg0001_DAMH-SMU/rank0000.npz": "a926f259498ceb2f84106779e0c9b2b5",  # C1b rebase (S2 exact step reads the shared statistics)
    "poly_s2/notes/alg0000_MH/rank0000.csv": "75b22009fa7e28745769ef1549a66bdc",
    "poly_s2/notes/alg0001_DAMH-SMU/rank0000.csv": "f32d80b07817af3c255821126499c760",  # C1b rebase (S2 exact step reads the shared statistics)
    "poly_s2/raw_data/alg0000_MH/rank0000.csv": "ba2017e6a0311fc3c90c5541b06fdb87",
    "poly_s2/raw_data/alg0001_DAMH-SMU/rank0000.csv": "bcfd33f4e9409ddbacbc87fae02cefbf",  # C1b rebase (S2 exact step reads the shared statistics)
    "poly_s2/samples/alg0000_MH/rank0000.csv": "c7e006aa7b771926dc58bb7cb92d1007",
    "poly_s2/samples/alg0001_DAMH-SMU/rank0000.csv": "4d282926e6498047467fd6c91aea10f5",  # C1b rebase (S2 exact step reads the shared statistics)
    "poly_s2/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "d573f1901f89aca369b35e6b581b7b68",  # C1b rebase (S2 exact step reads the shared statistics)
}


def _md5(path):
    if path.endswith(".npz"):
        with np.load(path) as z:
            return hashlib.md5(b"".join(np.ascontiguousarray(z[k]).tobytes() for k in sorted(z.files))).hexdigest()
    return hashlib.md5(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def _reference_runs(root, kwargs):
    """The five runs of scratchpad s1/s1_ref_run.py (same order, configurations and seeds)."""
    _problem().run_sampling_local(_conf(root / "damh"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=200),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=300, subchain_length=5, **kwargs)],
        surrogate_updater=_net())
    _problem().run_sampling_local(_conf(root / "hmc"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=150),
        Stage(algorithm="DAMH", proposal=Hamiltonian(step_size=0.2, num_steps=5), max_evaluations=200,
              subchain_length=1, **kwargs)],
        surrogate_updater=_net())
    _problem().run_sampling_local(_conf(root / "damh_s2"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=200),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=300, subchain_length=3,
              exact_step_probability=0.1, audit_prerejected=0.2, polyak_averaging=True, **kwargs)],
        surrogate_updater=_net())
    _problem().run_sampling_local(_conf(root / "poly"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=200),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=300, subchain_length=5, **kwargs)],
        surrogate_updater=_poly())
    _problem().run_sampling_local(_conf(root / "poly_s2"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=200),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=300, subchain_length=3,
              exact_step_probability=0.1, audit_prerejected=0.2, polyak_averaging=True, **kwargs)],
        surrogate_updater=_poly())


@pytest.mark.parametrize("kwargs", [{}, {"surrogate_error_model": False}], ids=["default", "explicit_false"])
def test_flag_off_is_byte_identical_to_the_code_before_s1(tmp_path, kwargs):
    _reference_runs(tmp_path, kwargs)
    produced = {}
    for kind in ("samples", "raw_data", "subchain_stats", "notes", "adaptive_stats", "last_sample"):
        for path in glob.glob(os.path.join(str(tmp_path), "*", "sampling_output", kind, "*", "rank*.*")):
            produced[os.path.relpath(path, str(tmp_path)).replace("/sampling_output", "")] = _md5(path)
    assert set(produced) == set(REFERENCE_MD5)
    changed = sorted(key for key, expected in REFERENCE_MD5.items() if produced[key] != expected)
    assert not changed, f"changed against the code before S1: {changed}"


def test_stage_field_type_and_star():
    assert Stage(algorithm="DAMH", max_evaluations=5).surrogate_error_model is False
    stage = Stage(algorithm="DAMH", max_evaluations=5, surrogate_error_model=np.bool_(True))
    assert stage.surrogate_error_model is True
    assert "surrogate_error_model*=True" in stage.describe()
    with pytest.raises(TypeError, match="surrogate_error_model must be True or False"):
        Stage(algorithm="DAMH", max_evaluations=5, surrogate_error_model=1)


# ---------------------------------------------------------------------------------------------
# Non-Gaussian likelihood; evaluator without a model
# ---------------------------------------------------------------------------------------------

def test_non_gaussian_likelihood_ignores_the_flag_with_a_note(tmp_path, capsys):
    likelihood = surrDAMH.distributions.FromScipy(scipy.stats.multivariate_normal(OBSERVED, np.diag(OBS_SD ** 2)))
    evaluator = AffineEvaluator()
    evaluator.error_model = _prior_residual_model(evaluator)

    def run(name, flag):
        return _problem(likelihood=likelihood).run_sampling_local(_conf(tmp_path / name), [
            Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=300, surrogate_error_model=flag)],
            surrogate_evaluator=evaluator)

    run("off", False)
    capsys.readouterr()
    run_on = run("on", True)
    out = capsys.readouterr().out
    assert out.count("surrogate_error_model is set, but the likelihood is a FromScipy") == 1
    assert run_on.stage_results[0].counter_accepted > 0
    for kind in ("samples", "raw_data", "subchain_stats", "notes"):
        a = open(tmp_path / "off" / "sampling_output" / kind / "alg0000_DAMH-SMU" / "rank0000.csv", "rb").read()
        b = open(tmp_path / "on" / "sampling_output" / kind / "alg0000_DAMH-SMU" / "rank0000.csv", "rb").read()
        assert a == b, kind


def test_evaluator_without_a_model_is_used_uncorrected_with_one_note(tmp_path, capsys):
    algo = _direct_damh(tmp_path, SequenceProvider([AffineEvaluator(), AffineEvaluator()], every=1))
    assert algo.likelihood_surrogate is algo.likelihood
    algo._refresh_surrogate_evaluator_if_needed()
    assert capsys.readouterr().out.count("carries no error_model") == 1


# ---------------------------------------------------------------------------------------------
# Collector path, local: LocalSurrogateManager
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("updater", ["polynomial", "network"])
def test_local_manager_attaches_a_model_after_every_retraining(updater):
    rng = np.random.default_rng(5)
    heldout = rng.normal(size=(30, 2))
    manager = LocalSurrogateManager(updater=_poly(2) if updater == "polynomial" else _net(),
                                    min_snapshots_initial=20, min_snapshots_to_update=15,
                                    error_model_heldout=(heldout, heldout @ A.T + 0.01))
    produced = []
    for x in rng.normal(size=(80, 2)):
        manager.send_to_collector([x, A @ x, 1.0])
        if manager._pending_evaluator is not None and (not produced or manager._pending_evaluator is not produced[-1]):
            produced.append(manager._pending_evaluator)
    assert len(produced) == len(manager.error_model_history) >= 3
    for evaluator, (used, bias_norm, cov_trace) in zip(produced, manager.error_model_history):
        assert evaluator.error_model is not None and evaluator.error_model.cov.shape == (2, 2)
        assert bias_norm == pytest.approx(evaluator.error_model.bias_norm)
        assert cov_trace == pytest.approx(evaluator.error_model.cov_trace)
    # the last one: snapshots of the updater + held-out set, uniform weights
    last = produced[-1]
    parameters, observations, _ = manager.updater.get_training_data_arrays()
    expected = estimate_error_model(np.vstack([observations - last(parameters),
                                               heldout @ A.T + 0.01 - last(heldout)]))
    np.testing.assert_allclose(last.error_model.bias, expected.bias, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(last.error_model.cov, expected.cov, rtol=1e-10, atol=1e-12)
    assert last.error_model.no_residuals == parameters.shape[0] + 30


def test_local_run_with_the_flag_on_installs_the_attached_models(tmp_path, installs):
    out = tmp_path / "poly"
    _problem().run_sampling_local(_conf(out, min_snapshots_to_update=100), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=300),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=1500, subchain_length=2,
              surrogate_error_model=True)],
        surrogate_updater=_poly(1))
    entry = next(e for e in installs.values() if e["instance"].stage.name == "alg0001_DAMH-SMU")
    algo = entry["instance"]
    assert len(entry["candidates"]) > 1 and all(c.error_model is not None for c in entry["candidates"])
    assert isinstance(algo.likelihood_surrogate, CorrectedNormal)
    np.testing.assert_allclose(algo.likelihood_surrogate.mean, OBSERVED - algo.surrogate_evaluator.error_model.bias)


# ---------------------------------------------------------------------------------------------
# Auto
# ---------------------------------------------------------------------------------------------

def test_auto_plan_sets_the_flag_on_the_chunks(tmp_path):
    from types import SimpleNamespace
    conf = surrDAMH.Configuration(output_dir=str(tmp_path / "o"), use_collector=False, use_solvers_pool=False)
    plan = plan_auto(SimpleNamespace(no_parameters=2, no_observations=2, solver_instance=None), conf,
                     budget=5_000, time_limit=None, mode="robust", local=True, surrogate_updater=None)
    assert plan.stages[0].surrogate_error_model is False
    assert [s.surrogate_error_model for s in plan.stages[1:]] == [AUTO_SURROGATE_ERROR_MODEL] * 4 == [True] * 4
    assert plan.manifest_entry()["surrogate_error_model"] is True
    assert "surrogate_error_model=True" in plan.describe()
    small = plan_auto(SimpleNamespace(no_parameters=2, no_observations=2, solver_instance=None), conf,
                      budget=50, time_limit=None, mode="robust", local=True, surrogate_updater=None)
    assert small.manifest_entry()["surrogate_error_model"] is None


def test_auto_local_run_completes_with_the_model_attached(tmp_path, installs):
    out = tmp_path / "auto"
    conf = surrDAMH.Configuration(output_dir=str(out), use_collector=False, use_solvers_pool=False)
    run = _problem().run_sampling_local_auto(conf, budget=1_000, mode="robust")
    assert run.auto["surrogate_error_model"] is True
    chunks = [e for e in installs.values() if e["instance"].stage.algorithm == "DAMH"]
    assert len(chunks) == 4
    for entry in chunks:
        algo = entry["instance"]
        assert algo.error_model_enabled and isinstance(algo.likelihood_surrogate, CorrectedNormal)
        assert all(c.error_model is not None for c in entry["candidates"])
        assert algo.surrogate_evaluator.error_model.no_residuals == 0  # averaged (S3): count not carried
        assert entry["candidates"][-1].error_model.no_residuals > 0
    print(f"\n[S1 auto] last chunk model: bias {np.round(chunks[-1]['instance'].surrogate_evaluator.error_model.bias, 4)}, "
          f"trace {chunks[-1]['instance'].surrogate_evaluator.error_model.cov_trace:.3e}")
