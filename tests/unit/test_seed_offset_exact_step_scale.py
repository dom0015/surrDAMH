#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study-29 prerequisites (2026-10-12):

P1 ``Configuration.seed_offset`` -- an integer added to every seed of the run (all per-stage
streams, the initial-sample seed and the LHS seed; recorded in the manifest ``seeds``). Pinned:
default/explicit 0 are byte-identical; ``seed_offset = 1_000_000`` reproduces, byte for byte, a run
with ``lineage_generation = 1`` (which shifts every seed by the same amount through the already
tested generation offset) -- so every stream, incl. LHS, the exact step and its random walk, is
shifted; validation; manifest entries; a continuation keeps the offset on top of its generation.

P2 ``Stage.exact_step_scale`` -- ``None`` = today's adaptive exact random walk (byte-identical);
a number ``s`` = a NON-adaptive random walk with covariance ``s * 2.38^2/d * Sigma_hat`` from the
shared posterior statistics at stage start (identity with a note while not informative).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import pytest

import surrDAMH
from helpers_statistics import (FixedInitialSample, assert_matches_closed_form, chain_statistics,
                                read_weighted_samples)
from surrDAMH.modules.posterior_statistics import MASS_MIN_STATES, PosteriorStatistics
from surrDAMH.modules.proposal_builder import build_exact_step_proposal
from surrDAMH.modules.proposals import GaussRandomWalk, GaussRandomWalk_adaptive
from surrDAMH.modules.seeds import SEED_FORMULA, conf_seed_offset, run_seed_offset
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import POSTERIOR_AFFECTING_FIELDS as STAGE_FIELDS
from surrDAMH.stages import Stage
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


class LinearGradientEvaluator(Evaluator):
    def __init__(self, matrix=A_SURROGATE):
        matrix = np.asarray(matrix, dtype=float)
        super().__init__(no_parameters=2, no_observations=2)
        self.matrix = matrix

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.matrix.T

    def jacobian(self, datapoints):
        x = np.asarray(datapoints, dtype=float).reshape(-1)
        return self.matrix, self.matrix @ x


def _problem():
    prior = surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(2)])
    return surrDAMH.Problem(prior, surrDAMH.distributions.Normal(mean=OBSERVED, sd=OBS_SD), solver=LinearSolver())


def _conf(output_dir, initial="user_specified", **kwargs):
    extra = {"initial_samples_distribution": FixedInitialSample((0.3, -0.2))} if initial == "user_specified" else {}
    return surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type=initial, save_snapshots_to_file=True, **extra, **kwargs)


def _manifest(output_dir):
    with open(os.path.join(str(output_dir), "sampling_output", "run_manifest.json")) as f:
        return json.load(f)


def _md5s(output_dir):
    out = {}
    so = os.path.join(str(output_dir), "sampling_output")
    for kind in ("samples", "raw_data", "subchain_stats", "notes", "adaptive_stats"):
        for path in sorted(glob.glob(os.path.join(so, kind, "*", "rank*.csv"))):
            out[os.path.relpath(path, so)] = hashlib.md5(open(path, "rb").read()).hexdigest()
    for path in sorted(glob.glob(os.path.join(so, "last_sample", "*", "rank*.npz"))):
        with np.load(path) as z:
            out[os.path.relpath(path, so)] = hashlib.md5(
                b"".join(np.ascontiguousarray(z[k]).tobytes() for k in sorted(z.files))).hexdigest()
    return out


def _all_streams_stages(**exact_kwargs):
    """MH warm-up + DAMH stage with exact steps and audits + a DAMH Hamiltonian stage: every stream
    (proposal, acceptance, exact-step choice, exact random walk, initial sample) is drawn from."""
    return [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=300, burn_in=True),
            Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=200, subchain_length=2,
                  exact_step_probability=0.3, audit_prerejected=0.3, **exact_kwargs),
            Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.4), max_evaluations=100)]


def _run(output_dir, initial="prior", stages=None, **conf_kwargs):
    return _problem().run_sampling_local(_conf(output_dir, initial=initial, **conf_kwargs),
                                         stages if stages is not None else _all_streams_stages(),
                                         surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2,
                                                                                    max_degree=1))


# --------------------------------------------------------------------------------------- P1
def test_run_seed_offset_arithmetic():
    assert run_seed_offset(0, 0) == 0 and run_seed_offset(None, None) == 0
    assert run_seed_offset(2, 7) == 2_000_007
    assert "seed_offset" in SEED_FORMULA

    class C:
        lineage_generation = 1
        seed_offset = 5
    assert conf_seed_offset(C()) == 1_000_005
    assert conf_seed_offset(object()) == 0


@pytest.mark.parametrize("value", [-1, 1.5, True, "3", None])
def test_seed_offset_validation(tmp_path, value):
    with pytest.raises(ValueError, match="seed_offset"):
        _conf(tmp_path, seed_offset=value)


def test_seed_offset_is_marked_and_described(tmp_path):
    from surrDAMH.configuration import POSTERIOR_AFFECTING_FIELDS
    assert "seed_offset" in POSTERIOR_AFFECTING_FIELDS
    conf = _conf(tmp_path, seed_offset=np.int64(300))
    assert conf.seed_offset == 300 and type(conf.seed_offset) is int
    assert "seed_offset*=300" in conf.describe()
    assert conf.requested_posterior_fields()["seed_offset"] == 300


@pytest.mark.parametrize("initial", ["prior", "lhs"])
def test_explicit_zero_is_byte_identical_to_the_default(tmp_path, initial):
    _run(tmp_path / "default", initial=initial)
    _run(tmp_path / "zero", initial=initial, seed_offset=0)
    a, b = _md5s(tmp_path / "default"), _md5s(tmp_path / "zero")
    assert len(a) > 10 and a == b


@pytest.mark.parametrize("initial", ["prior", "lhs"])
def test_offset_shifts_every_stream_like_a_generation(tmp_path, initial):
    """seed_offset = 1_000_000 and lineage_generation = 1 shift every seed (proposal, acceptance,
    exact-step choice, exact random walk, initial sample, LHS) by the same amount, so the two runs
    must agree byte for byte; a run with offset 0 must differ."""
    _run(tmp_path / "offset", initial=initial, seed_offset=1_000_000)
    _run(tmp_path / "generation", initial=initial, lineage_generation=1)
    _run(tmp_path / "plain", initial=initial)
    offset, generation, plain = (_md5s(tmp_path / n) for n in ("offset", "generation", "plain"))
    assert offset == generation
    differing = [k for k in plain if plain[k] != offset.get(k)]
    assert any(k.startswith("samples/") for k in differing)


def test_offset_is_recorded_in_the_manifest(tmp_path):
    _run(tmp_path / "a", initial="lhs", seed_offset=200_000)
    seeds = _manifest(tmp_path / "a")["seeds"]
    assert seeds["seed_offset"] == 200_000 and seeds["lhs_seed"] == 200_000 and seeds["formula"] == SEED_FORMULA
    # 3 stages, rank 0: seed0 = 10*i + offset
    assert [e["seed0"] for e in seeds["per_rank"]] == [200_000, 200_010, 200_020]
    for e in seeds["per_rank"]:
        assert (e["proposal_seed"], e["algorithm_seed"], e["exact_step_seed"], e["exact_proposal_seed"]) == \
            (e["seed0"] + 1, e["seed0"] + 2, e["seed0"] + 4, e["seed0"] + 5)
        assert e["initial_sample_seed"] == 200_003
    assert _manifest(tmp_path / "a")["configuration"]["seed_offset"] == 200_000
    _run(tmp_path / "b")
    assert _manifest(tmp_path / "b")["seeds"]["seed_offset"] == 0


def test_runner_passes_the_shifted_seeds(tmp_path, monkeypatch):
    import surrDAMH.runner_local as runner_local
    seen = {"proposal": [], "exact": []}
    original_build, original_exact = runner_local.build_proposal, runner_local.build_exact_step_proposal

    def build(*args, **kwargs):
        seen["proposal"].append(kwargs["seed"])
        return original_build(*args, **kwargs)

    def build_exact(*args, **kwargs):
        seen["exact"].append(kwargs["seed"])
        return original_exact(*args, **kwargs)
    monkeypatch.setattr(runner_local, "build_proposal", build)
    monkeypatch.setattr(runner_local, "build_exact_step_proposal", build_exact)
    _run(tmp_path, seed_offset=7)
    assert seen["proposal"] == [8, 18, 28] and seen["exact"] == [7 + 10 + 5]


def test_continuation_keeps_the_offset_on_top_of_the_generation(tmp_path):
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=50)]
    run = _run(tmp_path / "a", stages=stages, seed_offset=300)
    run.continue_sampling_local(surrDAMH.Configuration(str(tmp_path / "b"), use_collector=False,
                                                       use_solvers_pool=False, seed_offset=300),
                                [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20)])
    entry = _manifest(tmp_path / "b")["seeds"]["per_rank"][0]
    # generation 1, global stage index 1 of a 2-stage lineage, rank 0
    assert entry["seed0"] == 1_000_000 + 300 + 10


# --------------------------------------------------------------------------------------- P2
@pytest.mark.parametrize("value", [0, 0.0, -1.0, float("inf"), float("nan"), True, "1"])
def test_exact_step_scale_validation(value):
    with pytest.raises(ValueError, match="exact_step_scale"):
        Stage(algorithm="DAMH", exact_step_probability=0.1, exact_step_scale=value, max_evaluations=10)


def test_exact_step_scale_field_basics(capsys):
    assert "exact_step_scale" in STAGE_FIELDS
    stage = Stage(algorithm="DAMH", exact_step_probability=0.1, exact_step_scale=2, max_evaluations=10)
    assert stage.exact_step_scale == 2.0 and type(stage.exact_step_scale) is float
    text = stage.describe()
    assert "exact_step_scale*=2.0" in text and "NOT adaptive" in text
    assert "not used (exact_step_probability=0)" in Stage(algorithm="DAMH", exact_step_scale=1.0,
                                                          max_evaluations=10).describe()
    assert Stage(algorithm="DAMH", max_evaluations=10).exact_step_scale is None
    mh = Stage(algorithm="MH", exact_step_scale=1.0, max_evaluations=10)
    assert mh.exact_step_scale is None and "exact_step_scale=None" in capsys.readouterr().out


class _Conf:
    no_parameters = 3
    use_surrogate_gradients = True


def _statistics(n, seed=0):
    rng = np.random.default_rng(seed)
    stats = PosteriorStatistics(3)
    cov = np.array([[1.0, 0.3, 0.0], [0.3, 0.5, 0.1], [0.0, 0.1, 2.0]])
    for x in rng.multivariate_normal(np.zeros(3), cov, size=n):
        stats.update(x)
    return stats


def _prior3():
    return surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(3)])


def test_builder_fixed_scale_uses_the_statistics_covariance():
    stats = _statistics(500)
    proposal = build_exact_step_proposal(conf=_Conf(), prior=_prior3(), seed=15, statistics=stats, scale=2.0)
    assert type(proposal) is GaussRandomWalk  # not adaptive
    np.testing.assert_allclose(proposal.sd_or_cov, 2.0 * 2.38 ** 2 / 3 * stats.covariance())
    assert proposal.resolution["exact_step_scale"] == 2.0
    assert proposal.resolution["exact_step_covariance_source"] == "covariance (n=500)"
    np.testing.assert_allclose(proposal.resolution["exact_step_covariance_trace"], np.trace(proposal.sd_or_cov))
    # adapt() is the base no-op: the covariance never changes
    before = proposal.sd_or_cov.copy()
    for _ in range(50):
        proposal.adapt(proposed_sample=np.ones(3), log_acceptance_probability=-5.0, current_sample=np.zeros(3))
    np.testing.assert_array_equal(proposal.sd_or_cov, before)
    # the same seed as the adaptive one would get: its draws are those of RandomState(15)
    expected = np.random.RandomState(15).multivariate_normal(np.zeros(3), before)
    np.testing.assert_allclose(proposal.propose_sample(np.zeros(3)), expected)


@pytest.mark.parametrize("n", [0, MASS_MIN_STATES - 1, None])
def test_builder_fixed_scale_falls_back_to_the_identity(capsys, n):
    stats = None if n is None else _statistics(n)
    proposal = build_exact_step_proposal(conf=_Conf(), prior=_prior3(), seed=1, statistics=stats, scale=0.5,
                                         stage_index=4)
    np.testing.assert_allclose(proposal.sd_or_cov, 0.5 * 2.38 ** 2 / 3 * np.eye(3))
    assert proposal.resolution["exact_step_covariance_source"] == "identity"
    assert "identity covariance shape" in capsys.readouterr().out


def test_builder_default_is_the_adaptive_random_walk():
    proposal = build_exact_step_proposal(conf=_Conf(), prior=_prior3(), seed=1, statistics=_statistics(500))
    assert isinstance(proposal, GaussRandomWalk_adaptive) and not hasattr(proposal, "resolution")


def test_explicit_none_is_byte_identical_to_the_default(tmp_path):
    _run(tmp_path / "default")
    _problem().run_sampling_local(_conf(tmp_path / "none", initial="prior"),
                                  _all_streams_stages(exact_step_scale=None),
                                  surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2,
                                                                             max_degree=1))
    assert _md5s(tmp_path / "default") == _md5s(tmp_path / "none")


def _fixed_exact_run(out, scale, evaluations=12_000):
    run = _problem().run_sampling_local(
        _conf(out),
        [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=1000, burn_in=True),
         Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=evaluations, exact_step_probability=1.0,
               exact_step_scale=scale)],
        surrogate_evaluator=LinearGradientEvaluator())
    return run.stage_results[1]


def test_fixed_exact_random_walk_samples_the_posterior_and_its_scale_matters(tmp_path):
    result = _fixed_exact_run(tmp_path / "s1", 1.0)
    resolution = _manifest(tmp_path / "s1")["proposal_resolution"][1]
    assert resolution["exact_step_scale"] == 1.0
    assert resolution["exact_step_covariance_source"].startswith("covariance (n=")
    cov = np.linalg.inv(np.eye(2) + A.T @ np.diag(1.0 / OBS_SD ** 2) @ A)
    mean = cov @ (A.T @ np.diag(1.0 / OBS_SD ** 2) @ OBSERVED)
    statistics = chain_statistics(read_weighted_samples(tmp_path / "s1", result.name, 2))
    assert_matches_closed_form(statistics, mean, cov, "fixed exact random walk, scale 1")
    rate_1 = result.counter_exact_accepted / result.counter_exact_steps
    # 2.38^2/d * Sigma is near-optimal for a Gaussian target (acceptance ~0.35 in 2-D)
    assert 0.2 < rate_1 < 0.55
    small = _fixed_exact_run(tmp_path / "s025", 0.25, 3000)
    large = _fixed_exact_run(tmp_path / "s4", 4.0, 3000)
    rate_small = small.counter_exact_accepted / small.counter_exact_steps
    rate_large = large.counter_exact_accepted / large.counter_exact_steps
    assert rate_small > rate_1 > rate_large
    # the stage's exact random walk is not adapted: the trace stays at the resolved value
    trace = _manifest(tmp_path / "s4")["proposal_resolution"][1]["exact_step_covariance_trace"]
    assert trace == pytest.approx(4.0 / 0.25 * _manifest(tmp_path / "s025")["proposal_resolution"][1][
        "exact_step_covariance_trace"], rel=1e-12)
