#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S0 (2026-10-09): bounded prior, ``Problem(prior_bound=R)`` (default 8.0).

The target is the prior truncated to the box ``|u_i - m_i| <= R s_i`` in the internal space; a
proposal outside the box is rejected before ANY evaluation (solver tag -3,
``counter_out_of_bounds``, ``notes`` column ``out_of_bounds``). Pinned here: the box of every
supported prior and the argument checks; MH, DAMH sub-chain and Hamiltonian proposals outside
the box never reach the solver, the surrogate or the collector; the sampled truncated posterior
matches quadrature; ``prior_bound=None`` (and the default bound on a run that never reaches
it) is byte-identical to the code before S0; the start-up check of the initial state; the
manifest entry and the continuation check.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
import scipy.stats

import surrDAMH
import surrDAMH.post_processing as pp
from helpers_statistics import assert_close, chain_statistics, read_weighted_samples
from surrDAMH.distributions import (FromScipy, GaussianMixture, Normal, NormalComponent,
                                    PriorIndependentComponents)
from surrDAMH.modules.algorithm_interfaces_local import LocalEvaluatorProvider, LocalSolverAdapter
from surrDAMH.modules.algorithms import SOLVER_TAG_OUT_OF_BOUNDS, Algorithm_DAMH, Sample
from surrDAMH.modules.proposals import Proposal
from surrDAMH.process_SAMPLER import run_SAMPLER
from surrDAMH.proposals import Hamiltonian, RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates import PolynomialSklearnUpdater
from surrDAMH.surrogates.parent import Evaluator

pytestmark = pytest.mark.unit

A = np.array([[1.0, 0.5], [0.0, 1.0]])
OBS_SD = 0.3
OBSERVED = A @ np.array([0.7, -0.4])


class CountingLinearSolver(Solver):
    """``G(x) = A x`` (the linear-Gaussian problem of ``tests/conftest.py``); records every call."""

    def __init__(self):
        self.no_parameters = 2
        self.no_observations = 2
        self.points: list[np.ndarray] = []

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        self.points.append(self.parameters.copy())
        return A @ self.parameters


class FixedSample:
    def __init__(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def rvs(self):
        return self.parameters.copy()


class RecordingEvaluator(Evaluator):
    """Wraps an evaluator and records every point it is asked for."""

    def __init__(self, inner, log):
        super().__init__(inner.no_parameters, inner.no_observations)
        self.inner = inner
        self.log = log

    def __call__(self, datapoints):
        self.log.append(np.array(datapoints, dtype=float).copy())
        return self.inner(datapoints)


class RecordingPolynomialUpdater(PolynomialSklearnUpdater):
    """Polynomial updater whose evaluators record their inputs (``self.evaluated``)."""

    def __init__(self):
        super().__init__(no_parameters=2, no_observations=2, max_degree=2)
        self.evaluated: list[np.ndarray] = []

    def get_evaluator(self):
        return RecordingEvaluator(super().get_evaluator(), self.evaluated)


class LinearJacobianEvaluator(Evaluator):
    """Exact surrogate ``A x`` with its Jacobian (drives a Hamiltonian proposal); records calls."""

    def __init__(self):
        super().__init__(2, 2)
        self.evaluated: list[np.ndarray] = []

    def __call__(self, datapoints):
        self.evaluated.append(np.array(datapoints, dtype=float).copy())
        return np.atleast_2d(datapoints) @ A.T

    def jacobian(self, datapoints):
        x = np.asarray(datapoints, dtype=float).ravel()
        self.evaluated.append(x[None, :].copy())
        return A.copy(), A @ x


def _prior():
    return PriorIndependentComponents([NormalComponent(mu=0.0, sigma=1.0) for _ in range(2)])


def _problem(solver=None, **kwargs):
    return surrDAMH.Problem(_prior(), Normal(mean=OBSERVED, sd=OBS_SD),
                            solver=solver if solver is not None else CountingLinearSolver(), **kwargs)


def _conf(output_dir, initial=(0.1, -0.1), **kwargs):
    return surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedSample(initial),
                                  save_snapshots_to_file=True, **kwargs)


def _raw_data(output_dir, stage_name):
    return pd.read_csv(os.path.join(str(output_dir), "sampling_output", "raw_data", stage_name, "rank0000.csv"))


def _notes(output_dir, stage_name):
    return pd.read_csv(os.path.join(str(output_dir), "sampling_output", "notes", stage_name, "rank0000.csv"))


def _inside(points, bound):
    points = np.atleast_2d(points)
    return np.all(np.abs(points) <= bound, axis=1)


# ---------------------------------------------------------------------------------------------
# Interface: defaults, box of every supported prior, argument checks
# ---------------------------------------------------------------------------------------------

def test_default_bound_is_8_and_none_is_unbounded():
    problem = _problem()
    assert problem.prior_bound == 8.0
    np.testing.assert_array_equal(problem.prior_box[0], [-8.0, -8.0])
    np.testing.assert_array_equal(problem.prior_box[1], [8.0, 8.0])
    assert problem.prior.internal_box is problem.prior_box
    assert "prior_bound=8" in problem.describe()

    unbounded = _problem(prior_bound=None)
    assert unbounded.prior_bound is None and unbounded.prior_box is None
    assert unbounded.prior.internal_box is None
    assert "prior unbounded (prior_bound=None)" in unbounded.describe()


def test_the_users_prior_object_is_not_modified():
    prior = _prior()
    first = surrDAMH.Problem(prior, Normal(mean=OBSERVED, sd=OBS_SD), solver=CountingLinearSolver(), prior_bound=2.0)
    second = surrDAMH.Problem(prior, Normal(mean=OBSERVED, sd=OBS_SD), solver=CountingLinearSolver(), prior_bound=3.0)
    assert not hasattr(prior, "internal_box")
    np.testing.assert_array_equal(first.prior.internal_box[1], [2.0, 2.0])
    np.testing.assert_array_equal(second.prior.internal_box[1], [3.0, 3.0])


@pytest.mark.parametrize("bound", [0.0, -1.0, np.inf, np.nan, True, "8"])
def test_invalid_bound_raises(bound):
    with pytest.raises(ValueError, match="prior_bound must be a positive finite number"):
        _problem(prior_bound=bound)


def test_normal_prior_box_is_in_the_standardized_space():
    problem = surrDAMH.Problem(Normal(mean=np.array([5.0, -3.0]), sd=np.array([2.0, 10.0])),
                               Normal(mean=OBSERVED, sd=OBS_SD), solver=CountingLinearSolver(), prior_bound=4.0)
    np.testing.assert_array_equal(problem.prior_box[0], [-4.0, -4.0])
    np.testing.assert_array_equal(problem.prior_box[1], [4.0, 4.0])


def test_from_scipy_multivariate_normal_box_and_univariate_raises():
    rv = scipy.stats.multivariate_normal(mean=[1.0, 2.0], cov=np.diag([4.0, 9.0]))
    problem = surrDAMH.Problem(FromScipy(rv), Normal(mean=OBSERVED, sd=OBS_SD), solver=CountingLinearSolver(),
                               prior_bound=8.0)
    np.testing.assert_allclose(problem.prior_box[0], [1.0 - 16.0, 2.0 - 24.0])
    np.testing.assert_allclose(problem.prior_box[1], [1.0 + 16.0, 2.0 + 24.0])

    univariate = FromScipy(scipy.stats.norm(0.0, 1.0))
    with pytest.raises(ValueError, match="prior_bound=None"):
        surrDAMH.Problem(univariate, Normal(mean=OBSERVED, sd=OBS_SD), solver=CountingLinearSolver())
    problem = surrDAMH.Problem(univariate, Normal(mean=OBSERVED, sd=OBS_SD), solver=CountingLinearSolver(),
                               prior_bound=None)
    assert problem.prior_box is None


def test_gaussian_mixture_box_uses_the_mixture_mean_and_sd():
    means = [np.array([-2.0, 0.0]), np.array([2.0, 1.0])]
    covs = [np.diag([1.0, 4.0]), np.diag([1.0, 1.0])]
    prior = GaussianMixture(means, covs, weights=[1.0, 3.0])
    problem = surrDAMH.Problem(prior, Normal(mean=OBSERVED, sd=OBS_SD), solver=CountingLinearSolver(), prior_bound=2.0)
    w = np.array([0.25, 0.75])
    centre = w[0] * means[0] + w[1] * means[1]
    variance = w[0] * (np.diag(covs[0]) + means[0] ** 2) + w[1] * (np.diag(covs[1]) + means[1] ** 2) - centre ** 2
    np.testing.assert_allclose(problem.prior_box[0], centre - 2.0 * np.sqrt(variance))
    np.testing.assert_allclose(problem.prior_box[1], centre + 2.0 * np.sqrt(variance))
    # cross-check the moments against a large sample of the mixture
    draws = prior.rvs(n=200_000, generator=np.random.default_rng(0))
    np.testing.assert_allclose(draws.mean(axis=0), centre, atol=0.02)
    np.testing.assert_allclose(draws.std(axis=0), np.sqrt(variance), atol=0.02)


def test_unknown_prior_class_needs_prior_bound_none():
    class Custom(surrDAMH.Distribution):
        no_parameters = 2

        def logpdf(self, sample):
            return 0.0

    with pytest.raises(ValueError, match="pass prior_bound=None"):
        surrDAMH.Problem(Custom(), Normal(mean=OBSERVED, sd=OBS_SD), solver=CountingLinearSolver())
    assert surrDAMH.Problem(Custom(), Normal(mean=OBSERVED, sd=OBS_SD), solver=CountingLinearSolver(),
                            prior_bound=None).prior_box is None


# ---------------------------------------------------------------------------------------------
# Rejection without evaluation
# ---------------------------------------------------------------------------------------------

def test_mh_out_of_bounds_proposals_are_never_evaluated_nor_collected(tmp_path):
    solver = CountingLinearSolver()
    problem = _problem(solver, prior_bound=0.5)
    updater = PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1)
    run = problem.run_sampling_local(_conf(tmp_path), [Stage(algorithm="MH", proposal=RandomWalk(scale=5.0),
                                                             max_evaluations=200)],
                                     surrogate_updater=updater)
    result = run.stage_results[0]
    oob = result.counter_out_of_bounds
    assert oob > 0
    evaluated = result.counter_accepted + result.counter_rejected - oob
    # one solver call for the initial state, one per evaluated proposal, none for the rest
    assert len(solver.points) == 1 + evaluated
    assert np.all(_inside(np.array(solver.points), 0.5))
    # the collector (local updater) got every state that was left and every evaluated rejection,
    # i.e. nothing with tag -3 (and nothing outside the box)
    assert updater.par.shape[0] == result.counter_accepted + (result.counter_rejected - oob)
    assert np.all(_inside(updater.par, 0.5)) and np.all(np.isfinite(updater.obs))

    raw = _raw_data(tmp_path, "alg0000_MH")
    tagged = raw[raw["solver_tag"] == SOLVER_TAG_OUT_OF_BOUNDS]
    assert len(tagged) == oob
    assert set(tagged["state_type"]) == {"rejected"}
    assert tagged[["obs_0", "obs_1", "obs_approx_0", "obs_approx_1"]].isna().all().all()
    assert not np.any(_inside(tagged[["par_0", "par_1"]].to_numpy(), 0.5))
    assert np.all(np.isneginf(tagged["log_likelihood"]))
    others = raw[raw["solver_tag"] != SOLVER_TAG_OUT_OF_BOUNDS]
    assert np.all(_inside(others[["par_0", "par_1"]].to_numpy(), 0.5))

    notes = _notes(tmp_path, "alg0000_MH")
    assert list(notes.columns) == ["accepted", "rejected", "pre-rejected", "sum", "seed", "out_of_bounds",
                                   "exact_steps", "exact_accepted", "audited", "audit_hidden", "polyak_updates"]
    assert int(notes["out_of_bounds"].iloc[0]) == oob

    # post-processing: the summary sums the counters by name (seed is not summed)
    summary = pp.Samples(2, str(tmp_path), selection=False).get_summary()
    assert int(summary["out_of_bounds"].iloc[0]) == oob
    assert "seed" not in summary.columns
    assert int(summary["sum"].iloc[0]) == result.counter_accepted + result.counter_rejected


def test_damh_subchain_out_of_bounds_proposals_skip_the_surrogate(tmp_path):
    solver = CountingLinearSolver()
    problem = _problem(solver, prior_bound=0.5)
    updater = RecordingPolynomialUpdater()
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.3), max_evaluations=60),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=5.0), max_evaluations=40, subchain_length=3)]
    run = problem.run_sampling_local(_conf(tmp_path), stages, surrogate_updater=updater)
    damh = run.stage_results[1]
    assert damh.name == "alg0001_DAMH-SMU"
    assert damh.counter_out_of_bounds > 0

    # the surrogate was never asked for a point outside the box, neither was the solver
    assert updater.evaluated, "the surrogate was not used at all"
    assert np.all(_inside(np.vstack(updater.evaluated), 0.5))
    assert np.all(_inside(np.array(solver.points), 0.5))
    assert np.all(_inside(updater.par, 0.5)) and np.all(np.isfinite(updater.obs))

    raw = _raw_data(tmp_path, "alg0001_DAMH-SMU")
    tagged = raw[raw["solver_tag"] == SOLVER_TAG_OUT_OF_BOUNDS]
    assert len(tagged) == damh.counter_out_of_bounds  # the outer step itself is never out of bounds
    assert set(tagged["state_type"]) == {"prerejected"}
    assert tagged[["obs_0", "obs_1", "obs_approx_0", "obs_approx_1"]].isna().all().all()
    assert not np.any(_inside(tagged[["par_0", "par_1"]].to_numpy(), 0.5))
    others = raw[raw["solver_tag"] != SOLVER_TAG_OUT_OF_BOUNDS]
    assert np.all(_inside(others[["par_0", "par_1"]].to_numpy(), 0.5))
    assert int(_notes(tmp_path, "alg0001_DAMH-SMU")["out_of_bounds"].iloc[0]) == damh.counter_out_of_bounds
    # the bookkeeping of the outer chain is unchanged: sub-chain rejections are not counted there
    notes = _notes(tmp_path, "alg0001_DAMH-SMU")
    assert int(notes["sum"].iloc[0]) == damh.counter_accepted + damh.counter_rejected + damh.counter_prerejected


def test_hamiltonian_trajectory_ending_outside_is_rejected_unevaluated(tmp_path):
    solver = CountingLinearSolver()
    problem = _problem(solver, prior_bound=1.0)
    evaluator = LinearJacobianEvaluator()
    stage = Stage(algorithm="MH", proposal=Hamiltonian(step_size=3.0, num_steps=3), max_evaluations=40)
    run = problem.run_sampling_local(_conf(tmp_path), [stage], surrogate_evaluator=evaluator)
    result = run.stage_results[0]
    assert result.counter_out_of_bounds > 0
    assert len(solver.points) == 1 + result.counter_accepted + result.counter_rejected - result.counter_out_of_bounds
    assert np.all(_inside(np.array(solver.points), 1.0))
    raw = _raw_data(tmp_path, result.name)
    tagged = raw[raw["solver_tag"] == SOLVER_TAG_OUT_OF_BOUNDS]
    assert len(tagged) == result.counter_out_of_bounds
    assert not np.any(_inside(tagged[["par_0", "par_1"]].to_numpy(), 1.0))
    assert tagged[["obs_0", "obs_1"]].isna().all().all()


# ---------------------------------------------------------------------------------------------
# The truncated posterior is sampled correctly
# ---------------------------------------------------------------------------------------------

def _truncated_posterior_moments(bound, n=1201):
    """Mean and covariance of 1_B(u) N(u; 0, I) N(A u; data, OBS_SD^2 I) by quadrature on a grid."""
    grid = np.linspace(-bound, bound, n)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    residual = points @ A.T - OBSERVED
    log_density = -0.5 * np.sum(points ** 2, axis=1) - 0.5 * np.sum(residual ** 2, axis=1) / OBS_SD ** 2
    weights = np.exp(log_density - log_density.max())
    # trapezoid weights in both directions
    trapezoid = np.ones(n)
    trapezoid[[0, -1]] = 0.5
    weights *= np.outer(trapezoid, trapezoid).ravel()
    weights /= weights.sum()
    mean = weights @ points
    centered = points - mean
    cov = (centered * weights[:, None]).T @ centered
    return mean, cov


def test_truncated_posterior_matches_quadrature(tmp_path):
    problem = _problem(prior_bound=1.0)
    stage = Stage(algorithm="MH", proposal=RandomWalk(scale=0.6), max_evaluations=60_000)
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedSample([0.5, -0.3]))  # no raw_data
    run = problem.run_sampling_local(conf, [stage])
    assert run.stage_results[0].counter_out_of_bounds > 0
    samples = read_weighted_samples(tmp_path, "alg0000_MH", 2)
    assert np.all(np.abs(samples) <= 1.0)
    statistics = chain_statistics(samples)
    mean, cov = _truncated_posterior_moments(1.0)
    deviation_mean = (statistics.mean - mean) / statistics.se_mean
    deviation_cov = (statistics.cov - cov) / statistics.se_cov
    print("truncated posterior: sampled mean", statistics.mean, "quadrature", mean, "dev/SE", deviation_mean)
    print("truncated posterior: sampled cov", statistics.cov.ravel(), "quadrature", cov.ravel(),
          "dev/SE", deviation_cov.ravel())
    assert_close("truncated posterior mean vs quadrature", statistics.mean, mean, statistics.se_mean, 3.0)
    assert_close("truncated posterior covariance vs quadrature", statistics.cov, cov, statistics.se_cov, 3.0)
    # the test is meaningful: the untruncated posterior is clearly different
    untruncated_mean, untruncated_cov = _truncated_posterior_moments(12.0, n=2401)
    assert np.max(np.abs((statistics.mean - untruncated_mean) / statistics.se_mean)) > 5.0 \
        or np.max(np.abs((statistics.cov - untruncated_cov) / statistics.se_cov)) > 5.0


# ---------------------------------------------------------------------------------------------
# Byte identity: prior_bound=None is the code before S0, and so is the default bound on a run
# that never reaches it. Reference md5s made with the code BEFORE S0 (2026-10-09, scratchpad
# s0/s0_ref_run.py, same configuration as below); notes are compared without the appended
# out_of_bounds column, which did not exist then (nor the S2 notes columns and the subchain_stats
# kernel column, stripped as well).
# ---------------------------------------------------------------------------------------------

REFERENCE_MD5 = {
    "mh": {
        "samples/alg0000_MH": "5af1d512c2cd5732b1b3aece573ff8f6",
        "raw_data/alg0000_MH": "75b8dc108e5ad30dd5f79d015d2d715c",
        "notes/alg0000_MH": "836835a3dec5a8fd4a71b7113ec4dbc3",
    },
    "damh": {
        "samples/alg0000_MH": "65eeb706d0921eb2611191dc84660b68",
        "samples/alg0001_DAMH-SMU": "7fe8a3e7bbb9b9f9294a8a6f478eeee3",
        "raw_data/alg0000_MH": "0be185d92f3c413c7ae3dbb64b67d852",
        "raw_data/alg0001_DAMH-SMU": "f1c37b61545dfa1e9334bbc84c598d93",
        "subchain_stats/alg0001_DAMH-SMU": "866b02c29dce63dced0208d9abd862d5",
        "notes/alg0000_MH": "d599bc11355e5cbf254fdeb12079dab2",
        "notes/alg0001_DAMH-SMU": "bd7dd8aa94e5601e00851640ac20c50b",
    },
}


def _md5_of(output_dir, key):
    path = os.path.join(str(output_dir), "sampling_output", key, "rank0000.csv")
    with open(path, "rb") as f:
        content = f.read()
    if key.startswith("notes/"):
        # out_of_bounds (S0) and, since S2 (2026-10-09), exact_steps/exact_accepted/audited/audit_hidden,
        # since S3 (2026-10-09) polyak_updates
        lines = content.split(b"\r\n")
        assert lines[0].endswith(b",out_of_bounds,exact_steps,exact_accepted,audited,audit_hidden,polyak_updates")
        assert all(line.endswith(b",0,0,0,0,0,0") for line in lines[1:] if line)
        content = b"\r\n".join(line.rsplit(b",", 6)[0] if line else line for line in lines)
    if key.startswith("subchain_stats/"):
        # the kernel column appended by S2 (2026-10-09); 0 = delayed-acceptance step on every row here
        lines = content.split(b"\r\n")
        assert lines[0].endswith(b",kernel") and all(line.endswith(b",0") for line in lines[1:] if line)
        content = b"\r\n".join(line.rsplit(b",", 1)[0] if line else line for line in lines)
    return hashlib.md5(content).hexdigest()


@pytest.mark.parametrize("kwargs", [{"prior_bound": None}, {}], ids=["unbounded", "default_bound"])
def test_runs_inside_the_box_are_byte_identical_to_the_code_before_s0(tmp_path, kwargs):
    def problem():
        return _problem(**kwargs)

    def conf(name):
        return _conf(tmp_path / name, initial=(0.3, -0.2))

    problem().run_sampling_local(conf("mh"), [Stage(algorithm="MH", proposal=RandomWalk(scale=0.8),
                                                    max_evaluations=2000)])
    problem().run_sampling_local(conf("damh"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=200),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=300, subchain_length=5)],
        surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=2))
    for run_name, files in REFERENCE_MD5.items():
        for key, expected in files.items():
            assert _md5_of(tmp_path / run_name, key) == expected, f"{run_name}/{key} changed"


# ---------------------------------------------------------------------------------------------
# Initial state, manifest, continuation
# ---------------------------------------------------------------------------------------------

def test_initial_state_outside_the_box_raises_local(tmp_path):
    problem = _problem()
    with pytest.raises(RuntimeError, match=r"chain 0 .*coordinate 0 = 9\.0 is not in \[-8\.0, 8\.0\]"):
        problem.run_sampling_local(_conf(tmp_path, initial=(9.0, 0.0)),
                                   [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=5)])
    # the same state is accepted without a bound
    _problem(prior_bound=None).run_sampling_local(
        _conf(tmp_path / "unbounded", initial=(9.0, 0.0)),
        [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=5)])


def test_initial_state_outside_the_box_raises_in_the_mpi_sampler(tmp_path):
    """``run_SAMPLER`` (world of size 1, as in ``tests/test_runner_local.py``) checks at start-up."""
    problem = _problem()
    conf = _conf(tmp_path, initial=(0.0, -9.0))
    conf.resolve_problem_sizes(problem.no_parameters, problem.no_observations)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=5)]
    with pytest.raises(RuntimeError, match=r"chain 0 \(sampler rank 0\).*coordinate 1 = -9\.0"):
        run_SAMPLER(conf, problem.prior, problem.likelihood, stages, solver_instance=problem.solver_instance,
                    surrogate_evaluator=None)


def _manifest(output_dir):
    with open(os.path.join(str(output_dir), "sampling_output", "run_manifest.json")) as f:
        return json.load(f)


def test_manifest_records_the_bound_and_continuation_checks_it(tmp_path):
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20)]
    run_a = _problem(prior_bound=3.0).run_sampling_local(_conf(tmp_path / "a"), stages)
    entry = _manifest(tmp_path / "a")["problem"]
    assert entry["prior_bound"] == 3.0
    assert entry["prior_box"] == {"lower": [-3.0, -3.0], "upper": [3.0, 3.0]}
    assert _problem(prior_bound=None)._manifest_entry()["prior_box"] is None

    # same Problem: continues
    fresh = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=10)]
    run_a.continue_sampling_local(surrDAMH.Configuration(str(tmp_path / "b"), use_collector=False,
                                                         use_solvers_pool=False), fresh)
    # the run reloaded with a Problem of another bound: refused ...
    loaded = surrDAMH.SamplingRun.load(str(tmp_path / "a"), problem=_problem(prior_bound=8.0))
    with pytest.raises(ValueError, match=r"prior_bound=8\.0.*prior_bound=3\.0"):
        loaded.continue_sampling_local(surrDAMH.Configuration(str(tmp_path / "c"), use_collector=False,
                                                              use_solvers_pool=False),
                                       [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=10)])
    # ... unless the new Problem is given explicitly (recorded as a different problem)
    run_a.continue_sampling_local(surrDAMH.Configuration(str(tmp_path / "d"), use_collector=False,
                                                         use_solvers_pool=False),
                                  [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=10)],
                                  problem=_problem(prior_bound=8.0))
    assert _manifest(tmp_path / "d")["lineage"]["same_problem"] is False
    assert _manifest(tmp_path / "d")["problem"]["prior_bound"] == 8.0


# ---------------------------------------------------------------------------------------------
# DAMH: an evaluator change seen by a sub-chain none of whose steps was evaluated
# ---------------------------------------------------------------------------------------------

class _SequenceProposal(Proposal):
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


class _ScaledLinearEvaluator(Evaluator):
    def __init__(self, factor):
        super().__init__(2, 2)
        self.factor = factor

    def __call__(self, datapoints):
        return self.factor * (np.atleast_2d(datapoints) @ A.T)


def test_damh_rescores_the_chain_state_after_a_subchain_with_no_evaluated_step(tmp_path):
    """
    A new evaluator is installed at the start of a sub-chain whose every step is out of bounds:
    nothing re-scores the chain state with it there, so the NEXT sub-chain must (otherwise its
    first ratio would compare the old surrogate at the state with the new one at the proposal and
    the delayed-acceptance correction would not telescope under one surrogate).
    """
    problem = _problem(prior_bound=1.0)
    conf = _conf(tmp_path)
    conf.resolve_problem_sizes(2, 2)
    old, new = _ScaledLinearEvaluator(1.0), _ScaledLinearEvaluator(3.0)
    provider = LocalEvaluatorProvider(old)
    proposal = _SequenceProposal([[5.0, 5.0], [5.0, -5.0], [0.2, 0.1], [0.25, 0.1]])
    stage = Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=2, subchain_length=2,
                  name="rescore")
    algo = Algorithm_DAMH(stage=stage, proposal=proposal, initial_sample=Sample(parameters=np.array([0.1, -0.1])),
                          rank_world=0, conf=conf, prior=problem.prior, likelihood=problem.likelihood,
                          observation_provider=LocalSolverAdapter(CountingLinearSolver()),
                          evaluator_provider=provider, seed=1)
    state = algo.current.parameters.copy()
    np.testing.assert_allclose(algo.current.observations_approx, old(state[None, :])[0])

    provider.set_evaluator(new)
    algo._propose_new_sample_using_subchain()  # installs `new`; both steps out of bounds
    assert algo.counter_out_of_bounds == 2 and algo.surrogate_evaluator is new
    algo._propose_new_sample_using_subchain()  # no newer evaluator, but the re-scoring is still due
    np.testing.assert_allclose(algo.current.observations_approx, new(state[None, :])[0])
    assert algo._surrogate_rescore_pending is False
