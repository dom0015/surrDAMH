#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The exact prior-proposal step of a DAMH stage (``Stage.exact_step_proposal="prior"``, 2026-10-23,
paper §6.5; ``pdf_reports/adaptive_nn_ergodicity_paper/main.tex``, thm:prior-step).

Pinned here: the field checks and its ``*``/``describe()`` notes; ``PriorIndependence`` (shapes,
reproducibility from the seed, the ``(log L(y) - log L(x), 0)`` ratio, ``reseed``); the builder and
both runners' wiring (seed ``seed0 + 5``, ``kind``); the manifest's ``proposal_resolution``;
``p = 1`` and ``p = 0.5`` sample the closed-form posterior; the prior draws of a run are exactly the
stream ``default_rng(seed0 + 5)``; draws outside the S0 prior box are rejected unevaluated.
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
from surrDAMH.modules.algorithms import SOLVER_TAG_OUT_OF_BOUNDS, Algorithm_DAMH
from surrDAMH.modules.prior_proposal import PriorIndependence
from surrDAMH.modules.proposal_builder import build_exact_step_proposal
from surrDAMH.proposals import RandomWalk
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
    """``G(x) = A x``; records every call."""

    def __init__(self):
        self.points: list[np.ndarray] = []

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        self.points.append(self.parameters.copy())
        return A @ self.parameters


class LinearEvaluator(Evaluator):
    """Fixed surrogate ``G~(x) = A' x``."""

    def __init__(self, matrix=A_SURROGATE):
        super().__init__(no_parameters=2, no_observations=2)
        self.matrix = np.asarray(matrix, dtype=float)

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.matrix.T


def _prior():
    return surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(2)])


def _problem(solver=None, **kwargs):
    return surrDAMH.Problem(_prior(), surrDAMH.distributions.Normal(mean=OBSERVED, sd=OBS_SD),
                            solver=solver if solver is not None else LinearSolver(), **kwargs)


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


def _check_posterior(output_dir, stage_name, label):
    mean, cov = _posterior()
    statistics = chain_statistics(read_weighted_samples(output_dir, stage_name, 2))
    print(f"\n[{label}] {statistics.no_samples} samples: mean {statistics.mean} vs {mean} "
          f"(dev/SE {(statistics.mean - mean) / statistics.se_mean}), cov {statistics.cov.ravel()} vs {cov.ravel()}")
    assert_matches_closed_form(statistics, mean, cov, label)


# ---------------------------------------------------------------------------------------------
# Field checks
# ---------------------------------------------------------------------------------------------

def test_default_and_membership():
    stage = Stage(algorithm="DAMH", max_evaluations=10)
    assert stage.exact_step_proposal == "random_walk"
    assert "exact_step_proposal" in POSTERIOR_AFFECTING_FIELDS
    assert "exact_step_proposal*='random_walk'" in stage.describe()


@pytest.mark.parametrize("value", ["Prior", "rw", None, 1, ""])
def test_unknown_value_raises(value):
    with pytest.raises(ValueError, match="Stage.exact_step_proposal must be one of"):
        Stage(algorithm="DAMH", max_evaluations=10, exact_step_probability=0.5, exact_step_proposal=value)


@pytest.mark.parametrize("algorithm", ["DAMH", "MH"])
def test_prior_with_exact_step_scale_raises(algorithm):
    with pytest.raises(ValueError, match="exact_step_proposal='prior' and exact_step_scale conflict"):
        Stage(algorithm=algorithm, max_evaluations=10, exact_step_probability=0.5, exact_step_proposal="prior",
              exact_step_scale=1.0)


def test_mh_stage_resets_with_a_warning(capsys):
    stage = Stage(algorithm="MH", max_evaluations=10, exact_step_proposal="prior")
    assert stage.exact_step_proposal == "random_walk"
    assert "Warning: exact_step_proposal='prior' is only used by a DAMH stage" in capsys.readouterr().out
    Stage(algorithm="MH", max_evaluations=10)
    assert "Warning" not in capsys.readouterr().out


def test_describe_notes():
    used = Stage(algorithm="DAMH", max_evaluations=10, exact_step_probability=0.1, exact_step_proposal="prior")
    text = used.describe()
    assert "exact_step_proposal*='prior'" in text and "proposes y ~ prior" in text
    unused = Stage(algorithm="DAMH", max_evaluations=10, exact_step_proposal="prior")
    assert "exact_step_proposal is not used (exact_step_probability=0)" in unused.describe()
    assert "y ~ prior" not in Stage(algorithm="DAMH", max_evaluations=10, exact_step_probability=0.1).describe()


# ---------------------------------------------------------------------------------------------
# PriorIndependence
# ---------------------------------------------------------------------------------------------

def test_prior_independence_draws_the_prior_stream():
    proposal = PriorIndependence(_prior(), no_parameters=2, seed=15)
    draws = np.array([proposal.propose_sample(np.array([5.0, 5.0])) for _ in range(5)])
    rng = np.random.default_rng(15)
    np.testing.assert_array_equal(draws, np.array([rng.standard_normal(2) for _ in range(5)]))
    assert draws.shape == (5, 2) and draws.dtype == float
    assert proposal.get_log_acceptance_probability(-1.0, -3.5, 7.0, -2.0) == (2.5, 0.0)
    assert proposal.needs_gradients is False and proposal.resolution == {"exact_step_proposal": "prior"}
    proposal.adapt(proposed_sample=draws[0], log_acceptance_probability=0.0, current_sample=draws[0])  # no-op
    proposal.reseed(15)
    np.testing.assert_array_equal(proposal.propose_sample(np.zeros(2)), draws[0])


def test_prior_independence_flattens_a_gaussian_mixture_draw():
    prior = surrDAMH.distributions.GaussianMixture(means=[np.zeros(3), np.full(3, 3.0)])
    assert np.shape(prior.rvs(n=1, generator=np.random.default_rng(0))) == (1, 3)
    proposal = PriorIndependence(prior, no_parameters=3, seed=1)
    sample = proposal.propose_sample(np.zeros(3))
    assert sample.shape == (3,)
    np.testing.assert_array_equal(sample, np.asarray(prior.rvs(n=1, generator=np.random.default_rng(1))).ravel())
    with pytest.raises(ValueError, match="expected 2"):
        PriorIndependence(prior, no_parameters=2, seed=1).propose_sample(np.zeros(2))


def test_builder_kind():
    conf = _conf("unused")
    conf.resolve_problem_sizes(2, 2)
    proposal = build_exact_step_proposal(conf=conf, prior=_prior(), seed=15, kind="prior")
    assert isinstance(proposal, PriorIndependence) and proposal.no_parameters == 2
    with pytest.raises(ValueError, match="takes no scale"):
        build_exact_step_proposal(conf=conf, prior=_prior(), seed=15, kind="prior", scale=1.0)
    with pytest.raises(ValueError, match="kind must be"):
        build_exact_step_proposal(conf=conf, prior=_prior(), seed=15, kind="pcn")
    assert not isinstance(build_exact_step_proposal(conf=conf, prior=_prior(), seed=15), PriorIndependence)


# ---------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------

def _prior_run(tmp_path, probability, evaluations, name="run", **damh_kwargs):
    out = tmp_path / name
    run = _problem().run_sampling_local(
        _conf(out, min_snapshots_to_update=500),
        [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=500, burn_in=True),
         Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=evaluations,
               exact_step_probability=probability, exact_step_proposal="prior", **damh_kwargs)],
        surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1))
    return out, run.stage_results[1]


def test_probability_one_samples_the_posterior_from_the_prior_stream(tmp_path):
    out, result = _prior_run(tmp_path, 1.0, 20_000)
    stats = _csv(out, "subchain_stats", result.name)
    assert len(stats) == result.counter_exact_steps == 20_000 and set(stats["kernel"]) == {1}
    raw = _csv(out, "raw_data", result.name)
    # stage 1 of 2, rank 0: seed0 = 10, the prior draws come from default_rng(seed0 + 5) in order
    rng = np.random.default_rng(15)
    expected = np.array([rng.standard_normal(2) for _ in range(len(raw))])
    np.testing.assert_allclose(raw[["par_0", "par_1"]].to_numpy(), expected, rtol=0, atol=1e-15)
    # acceptance with the likelihood ratio only: the independence sampler accepts rarely here
    rate = result.counter_exact_accepted / 20_000
    print(f"\n[prior step p=1] acceptance {rate:.3f}")
    assert 0.02 < rate < 0.5
    # the stage proposal never adapted (no DA iteration)
    assert not os.path.exists(os.path.join(str(out), "sampling_output", "adaptive_stats", result.name))
    resolution = {entry["name"]: entry for entry in _manifest(out)["proposal_resolution"]}
    assert resolution[result.name]["exact_step_proposal"] == "prior"
    assert "exact_step_proposal" not in resolution["alg0000_MH-adaptive"]
    assert _manifest(out)["stages"][1]["exact_step_proposal"] == "prior"
    _check_posterior(out, result.name, "prior step p=1")


def test_probability_half_samples_the_posterior(tmp_path):
    out, result = _prior_run(tmp_path, 0.5, 20_000)
    stats = _csv(out, "subchain_stats", result.name)
    exact = stats["kernel"] == 1
    assert int(exact.sum()) == result.counter_exact_steps > 0 and int((~exact).sum()) > 0
    assert int(stats.loc[exact, "outer_accepted"].sum()) == result.counter_exact_accepted > 0
    _check_posterior(out, result.name, "prior step p=0.5")


def test_runs_are_reproducible_and_the_default_has_no_resolution_key(tmp_path):
    out_a, _ = _prior_run(tmp_path, 0.5, 500, name="a")
    out_b, result = _prior_run(tmp_path, 0.5, 500, name="b")
    for kind in ("samples", "raw_data", "subchain_stats", "notes"):
        a = open(os.path.join(str(out_a), "sampling_output", kind, result.name, "rank0000.csv"), "rb").read()
        b = open(os.path.join(str(out_b), "sampling_output", kind, result.name, "rank0000.csv"), "rb").read()
        assert a == b, kind
    out_rw = tmp_path / "rw"
    _problem().run_sampling_local(
        _conf(out_rw), [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=100),
                        Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=100, exact_step_probability=0.5)],
        surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1))
    assert all("exact_step_proposal" not in entry for entry in _manifest(out_rw)["proposal_resolution"])


def test_the_runner_builds_the_prior_proposal_on_the_exact_proposal_seed(tmp_path, monkeypatch):
    import surrDAMH.runner_local as runner_local
    built = {}
    original_builder = runner_local.build_exact_step_proposal

    def recording_builder(*args, **kwargs):
        built.update(seed=kwargs["seed"], kind=kwargs["kind"])
        return original_builder(*args, **kwargs)

    seen = {}
    original_init = Algorithm_DAMH.__init__

    def recording_init(self, *args, **kwargs):
        seen["exact_proposal"] = kwargs.get("exact_proposal")
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(runner_local, "build_exact_step_proposal", recording_builder)
    monkeypatch.setattr(Algorithm_DAMH, "__init__", recording_init)
    _problem().run_sampling_local(
        _conf(tmp_path), [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=50),
                          Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=20,
                                exact_step_probability=0.5, exact_step_proposal="prior")],
        surrogate_evaluator=LinearEvaluator())
    assert built == {"seed": 15, "kind": "prior"}
    assert isinstance(seen["exact_proposal"], PriorIndependence)


def test_prior_draws_outside_the_box_are_rejected_unevaluated(tmp_path):
    solver = LinearSolver()
    out = tmp_path / "box"
    result = _problem(solver, prior_bound=0.8).run_sampling_local(
        _conf(out), [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.3), max_samples=2_000,
                           exact_step_probability=1.0, exact_step_proposal="prior")],
        surrogate_evaluator=LinearEvaluator()).stage_results[0]
    assert result.counter_exact_steps == 2_000
    assert result.counter_out_of_bounds > 0
    raw = _csv(out, "raw_data", result.name)
    outside = raw["solver_tag"] == SOLVER_TAG_OUT_OF_BOUNDS
    assert int(outside.sum()) == result.counter_out_of_bounds
    assert (raw.loc[outside, "state_type"] == "rejected").all()
    assert raw.loc[outside, ["obs_0", "obs_1", "obs_approx_0", "obs_approx_1"]].isna().all().all()
    assert (np.abs(raw.loc[outside, ["par_0", "par_1"]].to_numpy()).max(axis=1) > 0.8).all()
    # solver calls: the initial state plus every in-box prior draw, nothing outside the box
    assert len(solver.points) == 1 + 2_000 - result.counter_out_of_bounds
    assert np.all(np.abs(np.array(solver.points)) <= 0.8)
