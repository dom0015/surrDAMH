#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1 (2026-10-11, author decision D7): one persistent posterior-covariance estimate per chain
(``modules/posterior_statistics.py``), the Hamiltonian mass from it (``mass="covariance"``), the
trajectory length in time (``integration_time``), the dual-averaging state carried across stages.

Pinned here:
* ``PosteriorStatistics``: Welford == numpy, merge == the statistics of the concatenation, vector
  round trip, ``covariance()`` == the adaptive random walk's estimator before C1 (formula copied
  below), ``precision()`` symmetric positive definite and the inverse of ``covariance()``;
* the adaptive random walk reading a SHARED object that the caller updates first is bit for bit
  the random walk with its own private statistics;
* integrators: dimension-robust with a full mass ``M`` follows the leapfrog trajectory with the
  same ``M`` (tiny step), identity mass bit-identical to the pre-C1 formula;
* ``integration_time``: ``num_steps`` follows ``T / step`` (trace column), both/neither rules;
* dual-averaging carry: continued with the same mass (first step = the carried ``exp(log_eps_bar)``,
  ``m`` continues), fresh with a changed one;
* ``mass="covariance"`` resolution (identity below 100 states, precision above, manifest record);
* runs: statistics carried across stages (counts in ``carry_over/``), a later adaptive random walk
  installs its shape at its first period boundary, continuation carries them, ``problem=`` drops them.
* stage-end pooling (bug fix 2026-10-11): ``begin_stage``/``finish_stage`` over several fake ranks
  and stages equals numpy on the concatenation of all states (the former pooling of the full
  objects counted the shared start R times); one row / empty start bit-identical to the former
  formula; ``n_seen`` (author decision 1, 2026-10-11) through update, discount, pooling, the
  shrinkage weight, ``carry_over/`` and continuation (older files: ``n_seen = n``).
Evidence tables: CHANGELOG.md (2026-10-11, C1).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import surrDAMH
from helpers_statistics import FixedInitialSample
from surrDAMH.modules.continuation import load_carry_over
from surrDAMH.modules.posterior_statistics import MASS_MIN_STATES, PosteriorStatistics
from surrDAMH.modules.proposal_builder import build_proposal
from surrDAMH.modules.proposals import (GaussRandomWalk_adaptive, Hamiltonian, Hamiltonian_adaptive,
                                        HamiltonianInfinite, HamiltonianInfinite_adaptive,
                                        num_steps_for_integration_time)
from surrDAMH.proposals import Hamiltonian as HamiltonianSpec
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.parent import Evaluator

pytestmark = pytest.mark.unit

A = np.array([[1.0, 0.5], [0.0, 1.0]])
OBSERVED = A @ np.array([0.7, -0.4])
OBS_SD = np.array([0.3, 0.3])


# ---------------------------------------------------------------------------------------------
# PosteriorStatistics
# ---------------------------------------------------------------------------------------------

def _feed(states, d=None):
    statistics = PosteriorStatistics(states.shape[1] if d is None else d)
    for state in states:
        statistics.update(state)
    return statistics


def test_welford_equals_numpy():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(257, 4)) @ rng.normal(size=(4, 4)) + 3.0
    statistics = _feed(states)
    assert statistics.n == 257
    np.testing.assert_allclose(statistics.mean, states.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(statistics.covariance(shrinkage=False), np.cov(states.T), rtol=1e-12, atol=1e-12)


def test_merge_equals_the_statistics_of_the_concatenation():
    rng = np.random.default_rng(1)
    parts = [rng.normal(size=(n, 3)) + shift for n, shift in ((37, 0.5), (53, -1.0), (1, 4.0))]
    merged = _feed(parts[0]).merge([_feed(parts[1]), _feed(parts[2])])
    everything = np.vstack(parts)
    assert merged.n == len(everything)
    np.testing.assert_allclose(merged.mean, everything.mean(axis=0), atol=1e-12)
    centred = everything - everything.mean(axis=0)
    np.testing.assert_allclose(merged.M2, centred.T @ centred, atol=1e-10)
    # from_rows on the Allgather buffer is the same operation; an empty object contributes nothing
    rows = np.vstack([_feed(p).as_vector() for p in parts] + [PosteriorStatistics(3).as_vector()])
    pooled = PosteriorStatistics.from_rows(rows, 3)
    np.testing.assert_allclose(pooled.as_vector(), merged.as_vector(), atol=1e-12)
    assert PosteriorStatistics.from_rows(PosteriorStatistics(3).as_vector()[None, :], 3).n == 0


def test_vector_round_trip_and_in_place_install():
    statistics = _feed(np.random.default_rng(2).normal(size=(20, 3)))
    vector = statistics.as_vector()
    assert vector.shape == (1 + 3 + 9,)
    again = PosteriorStatistics.from_vector(vector, 3)
    np.testing.assert_array_equal(again.as_vector(), vector)
    holder = PosteriorStatistics(3)
    alias = holder
    holder.install(again)
    assert alias is holder and holder.n == 20  # every reference sees the installed state
    with pytest.raises(ValueError, match="length 13"):
        PosteriorStatistics.from_vector(vector[:-1], 3)


def _run_fake_ranks(chunks):
    """``chunks[stage][rank]``: the states rank ``rank`` adds in stage ``stage``. Every rank holds its
    own object; stage end = Allgather of the increments + ``finish_stage`` (as process_SAMPLER)."""
    d = chunks[0][0].shape[1]
    ranks = [PosteriorStatistics(d) for _ in chunks[0]]
    for stage_chunks in chunks:
        for statistics, states in zip(ranks, stage_chunks):
            statistics.begin_stage()
            for state in states:
                statistics.update(state)
        rows = np.vstack([statistics.stage_increment_vector() for statistics in ranks])
        for statistics in ranks:
            statistics.finish_stage(rows)
    return ranks


@pytest.mark.parametrize("no_ranks", [2, 3])
def test_stage_end_pooling_equals_the_concatenation_over_stages_and_ranks(no_ranks):
    rng = np.random.default_rng(10 + no_ranks)
    d = 3
    chunks = [[rng.normal(size=(int(rng.integers(40, 200)), d)) @ rng.normal(size=(d, d)) + stage
               for _ in range(no_ranks)] for stage in range(3)]
    ranks = _run_fake_ranks(chunks)
    everything = np.vstack([states for stage_chunks in chunks for states in stage_chunks])
    for statistics in ranks:  # identical on every rank
        np.testing.assert_array_equal(statistics.as_vector(), ranks[0].as_vector())
    pooled = ranks[0]
    assert pooled.n == pooled.n_seen == len(everything)
    np.testing.assert_allclose(pooled.mean, everything.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(pooled.covariance(shrinkage=False), np.cov(everything.T), rtol=1e-12, atol=1e-12)
    # the former stage end (pooling the ranks' FULL objects) counted the shared start R times
    start = _feed(np.vstack(chunks[0]))
    full = [start.copy() for _ in range(no_ranks)]
    for statistics, states in zip(full, chunks[1]):
        for state in states:
            statistics.update(state)
    former = PosteriorStatistics.from_rows(np.vstack([f.as_vector() for f in full]), d)
    assert former.n == no_ranks * start.n + sum(len(c) for c in chunks[1])


def test_stage_end_pooling_is_bit_identical_for_one_row_and_for_an_empty_start():
    rng = np.random.default_rng(12)
    # one row (local runner, one-sampler MPI run): the former single-row formula on the full object
    statistics = _feed(rng.normal(size=(150, 2)))
    statistics.begin_stage()
    for state in rng.normal(size=(80, 2)):
        statistics.update(state)
    former = PosteriorStatistics.from_rows(statistics.as_vector()[None, :], 2)
    statistics.finish_stage(statistics.stage_increment_vector()[None, :])
    np.testing.assert_array_equal(statistics.as_vector(), former.as_vector())
    assert statistics.n_seen == 230
    # empty start (a run's first stage) with two ranks: the former pooling of the rows
    a, b = PosteriorStatistics(2), PosteriorStatistics(2)
    for statistics, states in ((a, rng.normal(size=(70, 2))), (b, rng.normal(size=(90, 2)))):
        statistics.begin_stage()
        for state in states:
            statistics.update(state)
    former = PosteriorStatistics.from_rows(np.vstack([a.as_vector(), b.as_vector()]), 2)
    rows = np.vstack([a.stage_increment_vector(), b.stage_increment_vector()])
    a.finish_stage(rows)
    np.testing.assert_array_equal(a.as_vector(), former.as_vector())
    with pytest.raises(RuntimeError, match="begin_stage"):
        a.finish_stage(rows)
    with pytest.raises(RuntimeError):
        a.stage_increment_vector()


def test_n_seen_through_update_discount_pooling_and_shrinkage():
    rng = np.random.default_rng(13)
    d = 2
    ranks = _run_fake_ranks([[rng.normal(size=(500, d)) for _ in range(2)]])
    assert [s.n_seen for s in ranks] == [1000, 1000]
    for statistics in ranks:  # C1b burn-in discount on every rank (rank-identical)
        assert statistics.discount(100) == 1000
        assert statistics.n == 100 and statistics.n_seen == 1000
        # the shrinkage weight uses the undiscounted count: delta = 2d / 1000, not 2d / 100
        assert statistics.shrunk_covariance()[2] == pytest.approx(2 * d / 1000)
    C = ranks[0].covariance(shrinkage=False)
    expected = 0.996 * C + 0.004 * (np.trace(C) / d) * np.eye(d)
    np.testing.assert_allclose(ranks[0].covariance(), expected + 1e-6 * (np.trace(C) / d) * np.eye(d), rtol=1e-12)
    # next stage: increments of 30 and 50 states pool onto the discounted start
    for statistics, states in zip(ranks, (rng.normal(size=(30, d)), rng.normal(size=(50, d)))):
        statistics.begin_stage()
        for state in states:
            statistics.update(state)
    rows = np.vstack([s.stage_increment_vector() for s in ranks])
    for statistics in ranks:
        statistics.finish_stage(rows)
    assert ranks[0].n == 180 and ranks[0].n_seen == 1080
    np.testing.assert_array_equal(ranks[0].as_vector(), ranks[1].as_vector())
    # copies / installs / vectors: n_seen travels with install and copy, a bare vector gives n
    assert ranks[0].copy().n_seen == 1080
    assert PosteriorStatistics.from_vector(ranks[0].as_vector(), d).n_seen == 180
    assert PosteriorStatistics.from_vector(ranks[0].as_vector(), d, n_seen=1080).n_seen == 1080
    assert ranks[0].summary()["n_seen"] == 1080
    # an object whose n was set by hand (n_seen 0) shrinks with n
    manual = PosteriorStatistics(d)
    manual.n, manual.M2 = 10, np.eye(d) * 9.0
    assert manual.shrunk_covariance()[2] == pytest.approx(2 * d / 10)


def _pre_c1_random_walk_estimator(n, scatter, d):
    """``GaussRandomWalk_adaptive._install_covariance_from_statistics`` before C1, without the
    ``2.38^2/d`` factor (copied from proposals.py, 2026-10-10)."""
    C = scatter / (n - 1)
    trace_over_d = float(np.trace(C)) / d
    delta = min(1.0, 2.0 * d / n)
    C = (1.0 - delta) * C + delta * trace_over_d * np.eye(d)
    return C + 1e-6 * trace_over_d * np.eye(d)


@pytest.mark.parametrize("n", [3, 10, 400])
def test_covariance_is_the_random_walks_former_estimator(n):
    d = 4
    statistics = _feed(np.random.default_rng(n).normal(size=(n, d)) * [1.0, 2.0, 0.1, 5.0])
    np.testing.assert_array_equal(statistics.covariance(), _pre_c1_random_walk_estimator(n, statistics.M2, d))
    # and the random walk installs exactly 2.38^2/d times it
    walk = GaussRandomWalk_adaptive(no_parameters=d, statistics=statistics)
    assert walk._install_covariance_from_statistics()
    np.testing.assert_array_equal(walk.base_cov, 2.38 ** 2 / d * statistics.covariance())


def test_precision_is_spd_and_the_inverse():
    statistics = _feed(np.random.default_rng(5).normal(size=(150, 5)) @ np.diag([1.0, 10.0, 0.1, 3.0, 1.0]))
    P = statistics.precision()
    np.testing.assert_array_equal(P, P.T)
    assert np.all(np.linalg.eigvalsh(P) > 0)
    np.testing.assert_allclose(P @ statistics.covariance(), np.eye(5), atol=1e-9)


def test_informative_and_degenerate_statistics():
    statistics = PosteriorStatistics(2)
    assert not statistics.is_informative() and not statistics.is_informative(1)
    with pytest.raises(ValueError, match="at least 2"):
        statistics.covariance()
    for _ in range(MASS_MIN_STATES):
        statistics.update([1.0, 2.0])  # a chain that never moved
    assert statistics.is_informative()
    assert statistics.shrunk_covariance() is None
    with pytest.raises(ValueError, match="degenerate"):
        statistics.precision()


def test_shared_statistics_updated_by_the_caller_are_bit_identical_to_private_ones():
    """The algorithm updates the shared object BEFORE adapt(): same arithmetic, same order."""
    rng = np.random.default_rng(7)
    private = GaussRandomWalk_adaptive(no_parameters=3, sd_or_cov=0.4, seed=3, period=5, warmup=20)
    shared_statistics = PosteriorStatistics(3)
    shared = GaussRandomWalk_adaptive(no_parameters=3, sd_or_cov=0.4, seed=3, period=5, warmup=20,
                                      statistics=shared_statistics)
    for i in range(203):
        proposed, current = rng.normal(size=3), rng.normal(size=3)
        log_alpha = -np.inf if i % 9 == 4 else float(np.log(rng.uniform(0.01, 1.2)))
        private.adapt(proposed, log_alpha, current)
        shared_statistics.update(current)
        shared.adapt(proposed, log_alpha, current)
        for name in ("log_sigma", "base_cov", "L", "sd_or_cov", "mean", "M2"):
            np.testing.assert_array_equal(getattr(shared, name), getattr(private, name), err_msg=name)
    assert shared.adaptation_stats_rows == private.adaptation_stats_rows
    np.testing.assert_array_equal(shared.propose_sample(np.zeros(3)), private.propose_sample(np.zeros(3)))


def test_a_carried_statistics_object_installs_the_shape_at_the_first_period():
    """A later adaptive random-walk stage starts with the pooled statistics (n >= warmup already)."""
    carried = _feed(np.random.default_rng(8).normal(size=(500, 2)) * [3.0, 0.2])
    walk = GaussRandomWalk_adaptive(no_parameters=2, sd_or_cov=1.0, seed=0, period=10, warmup=100,
                                    statistics=carried)
    for i in range(10):
        carried.update(np.zeros(2))
        walk.adapt(np.zeros(2), -1.0, np.zeros(2))
        assert np.isnan(walk._trace_over_d) == (i < 9)
    np.testing.assert_allclose(walk.base_cov, 2.38 ** 2 / 2 * carried.covariance())


# ---------------------------------------------------------------------------------------------
# Integrators with a general mass
# ---------------------------------------------------------------------------------------------

def _gradients(precision):
    """``(grad U_likelihood, grad U_prior)`` of ``U(q) = 0.5 q^T P q`` split as N(0, I) prior."""
    identity = np.eye(precision.shape[0])
    return (lambda q: (precision - identity) @ q, lambda q: identity @ q)


@pytest.mark.parametrize("mass_kind", ["matrix", "vector"])
def test_dimension_robust_follows_the_leapfrog_trajectory_with_the_same_mass(mass_kind):
    rng = np.random.default_rng(11)
    d = 4
    B = rng.normal(size=(d, d))
    precision = B @ B.T / d + np.eye(d)
    if mass_kind == "matrix":
        C = rng.normal(size=(d, d))
        mass = C @ C.T / d + 0.5 * np.eye(d)
    else:
        mass = np.array([0.7, 1.3, 2.0, 0.4])  # sds: M = diag(sd^2)
    T, step = 1.0, 1e-4
    q0, p0 = rng.normal(size=d), rng.normal(size=d)
    trajectories = []
    for proposal_class in (Hamiltonian, HamiltonianInfinite):
        proposal = proposal_class(no_parameters=d, step_size=step, num_steps=int(round(T / step)), sd_or_cov=mass)
        proposal.set_gradient_functions(*_gradients(precision))
        trajectories.append(proposal._leapfrog(q0, p0))
    (q_leap, p_leap), (q_robust, p_robust) = trajectories
    print(f"[C1 integrators, mass={mass_kind}] max|dq| {np.max(np.abs(q_leap - q_robust)):.3e}, "
          f"max|dp| {np.max(np.abs(p_leap - p_robust)):.3e}")
    np.testing.assert_allclose(q_robust, q_leap, rtol=0, atol=1e-6)
    np.testing.assert_allclose(p_robust, p_leap, rtol=0, atol=1e-6)


def test_dimension_robust_identity_mass_is_the_pre_c1_rotation_bit_for_bit():
    d = 3
    precision = np.diag([2.0, 5.0, 1.5])
    proposal = HamiltonianInfinite(no_parameters=d, step_size=0.3, num_steps=7, sd_or_cov=1.0)
    grad_likelihood = _gradients(precision)[0]
    proposal.set_gradient_functions(grad_likelihood, lambda q: q)
    q0, p0 = np.array([0.3, -1.0, 0.5]), np.array([1.1, 0.2, -0.4])
    # reference: the pre-C1 diagonal path with sd = 1 (copied from proposals.py)
    q, p, eps = q0.copy(), p0.copy(), 0.3
    sd = np.ones(d)
    p = p - 0.5 * eps * grad_likelihood(q)
    for i in range(7):
        c, s = np.cos(eps * (1.0 / sd)), np.sin(eps * (1.0 / sd))
        q, p = c * q + (s / sd) * p, c * p - (sd * s) * q
        if i < 6:
            p = p - eps * grad_likelihood(q)
    p = p - 0.5 * eps * grad_likelihood(q)
    q_new, p_end = proposal._leapfrog(q0, p0)
    np.testing.assert_array_equal(q_new, q)
    np.testing.assert_array_equal(p_end, -p)


# ---------------------------------------------------------------------------------------------
# integration_time
# ---------------------------------------------------------------------------------------------

def test_integration_time_rules():
    with pytest.raises(ValueError, match="either num_steps or integration_time"):
        HamiltonianSpec(num_steps=5, integration_time=1.0)
    with pytest.raises(ValueError, match="positive"):
        HamiltonianSpec(integration_time=0.0)
    with pytest.raises(ValueError, match="'covariance'"):
        HamiltonianSpec(mass="precision")
    assert HamiltonianSpec().num_steps == 10 and HamiltonianSpec().integration_time is None
    assert HamiltonianSpec(integration_time=2.0).num_steps is None
    with pytest.raises(ValueError, match="either num_steps or integration_time"):
        Hamiltonian(no_parameters=2, num_steps=5, integration_time=1.0)
    assert num_steps_for_integration_time(3.0, 0.1) == 30
    assert num_steps_for_integration_time(3.0, 7.0) == 1
    assert num_steps_for_integration_time(3.0, 1e-9) == 1000  # safety cap


@pytest.mark.parametrize("proposal_class", [Hamiltonian_adaptive, HamiltonianInfinite_adaptive])
def test_num_steps_follows_the_adapted_step(proposal_class):
    proposal = proposal_class(no_parameters=2, step_size=0.1, num_steps=None, integration_time=2.0, seed=1)
    assert proposal.num_steps == 20
    assert proposal.adaptation_stats_header[-1] == "num_steps"
    rng = np.random.default_rng(0)
    for _ in range(40):
        proposal.adapt(np.zeros(2), float(np.log(rng.uniform(0.5, 1.0))), np.zeros(2))
        assert proposal.num_steps == num_steps_for_integration_time(2.0, proposal.step_size)
    assert len({row[-1] for row in proposal.adaptation_stats_rows}) > 1  # it really followed the step
    assert [row[-1] for row in proposal.adaptation_stats_rows] == \
        [num_steps_for_integration_time(2.0, np.exp(row[2])) for row in proposal.adaptation_stats_rows]
    # without integration_time the trace layout is the pre-C1 one
    assert "num_steps" not in proposal_class(no_parameters=2, step_size=0.1, num_steps=5).adaptation_stats_header


# ---------------------------------------------------------------------------------------------
# Dual-averaging carry and mass resolution (build_proposal)
# ---------------------------------------------------------------------------------------------

def _conf(d=2):
    return SimpleNamespace(no_parameters=d, use_surrogate_gradients=True)


def _prior(d=2):
    return surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(0.0, 1.0) for _ in range(d)])


def _finished_stage_carry(mass=1.0, statistics=None):
    """Carry-over of an adaptive Hamiltonian stage that ran 50 dual-averaging updates."""
    spec = HamiltonianSpec(mass=mass, integration_time=1.0)
    proposal = build_proposal(Stage(algorithm="MH", proposal=spec), _conf(), _prior(), seed=1, statistics=statistics)
    rng = np.random.default_rng(3)
    for _ in range(50):
        proposal.adapt(np.zeros(2), float(np.log(rng.uniform(0.6, 1.0))), np.zeros(2))
    proposal.set_pooled_state(proposal.adapted_state()[None, :])
    return proposal, proposal.carry_over()


def test_dual_averaging_continues_with_the_same_mass(capsys):
    previous, carried = _finished_stage_carry()
    assert set(carried) == {"step_size", "dual_averaging", "hamiltonian_mass"}
    np.testing.assert_array_equal(carried["hamiltonian_mass"], np.eye(2))
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(integration_time=1.0))
    continued = build_proposal(stage, _conf(), _prior(), seed=2, carried=carried)
    assert continued.resolution["dual_averaging"] == "continued"
    # first installed step = the carried exp(log_eps_bar), not 10 * eps0; m continues
    assert continued.step_size == pytest.approx(np.exp(previous.log_eps_bar))
    assert (continued.m, continued.H_bar, continued.mu) == (50, previous.H_bar, previous.mu)
    # the next update is exactly the one the previous stage's recursion would have made
    for log_alpha in (np.log(0.8), np.log(0.4), 0.0):
        continued.adapt(np.zeros(2), log_alpha, np.zeros(2))
        previous.adapt(np.zeros(2), log_alpha, np.zeros(2))
        assert (continued.step_size, continued.log_eps_bar, continued.num_steps) == \
            (previous.step_size, previous.log_eps_bar, previous.num_steps)
    assert continued.m == 53
    # a fresh recursion from the same carried step overshoots to mu = log(10 eps0) (alpha = delta: H_bar = 0)
    fresh = build_proposal(stage, _conf(), _prior(), seed=2, carried={"step_size": carried["step_size"]})
    fresh.adapt(np.zeros(2), np.log(0.8), np.zeros(2))
    assert fresh.step_size == pytest.approx(10 * carried["step_size"])
    assert "mass changed" not in capsys.readouterr().out


def test_dual_averaging_restarts_when_the_mass_changed(capsys):
    _previous, carried = _finished_stage_carry()
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(mass=[1.0, 2.0], integration_time=1.0))
    restarted = build_proposal(stage, _conf(), _prior(), seed=2, carried=carried)
    # C1b (2026-10-11): restart from the carried step rescaled to the new mass, (det M_new/det M_old)^(1/(2d)),
    # with mu = log(2 eps0) instead of log(10 eps0); details in tests/unit/test_c1b.py
    assert restarted.resolution["dual_averaging"] == "restarted (mass changed)"
    eps0 = carried["step_size"] * 4.0 ** 0.25
    assert restarted.m == 0 and restarted.step_size == pytest.approx(eps0)
    assert restarted.mu == pytest.approx(np.log(2 * eps0))
    assert "dual averaging restarted from the carried step" in capsys.readouterr().out
    # a pinned step never continues
    pinned = build_proposal(Stage(algorithm="MH", proposal=HamiltonianSpec(step_size=0.05, adaptive=True)),
                            _conf(), _prior(), seed=2, carried=carried)
    assert pinned.resolution["dual_averaging"] == "fresh" and pinned.m == 0


def test_covariance_mass_resolution(capsys):
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(mass="covariance", integrator="dimension_robust"))
    few = _feed(np.random.default_rng(0).normal(size=(MASS_MIN_STATES - 1, 2)))
    proposal = build_proposal(stage, _conf(), _prior(), seed=1, statistics=few)
    assert proposal.resolution["mass_source"] == "identity (no statistics)"
    np.testing.assert_array_equal(proposal.mass_matrix(), np.eye(2))
    assert "identity mass used" in capsys.readouterr().out
    enough = _feed(np.random.default_rng(0).normal(size=(400, 2)) * [2.0, 0.1])
    proposal = build_proposal(stage, _conf(), _prior(), seed=1, statistics=enough)
    assert proposal.resolution["mass_source"] == "covariance (n=400)"
    np.testing.assert_allclose(proposal.mass_matrix(), enough.precision(), rtol=0, atol=0)
    np.testing.assert_allclose(proposal.M_inv, enough.covariance(), rtol=1e-8)
    # the same statistics give the same mass: a following stage continues the dual averaging
    _p, carried = _finished_stage_carry(mass="covariance", statistics=enough)
    again = build_proposal(Stage(algorithm="MH", proposal=HamiltonianSpec(mass="covariance")), _conf(), _prior(),
                           seed=3, carried=carried, statistics=enough)
    assert again.resolution["dual_averaging"] == "continued"
    # C1b: a large change of the statistics (one far state: the 0.1-sd direction widens) restarts the recursion;
    # a small one continues it (tests/unit/test_c1b.py)
    enough.update([5.0, 5.0])
    changed = build_proposal(Stage(algorithm="MH", proposal=HamiltonianSpec(mass="covariance")), _conf(), _prior(),
                             seed=3, carried=carried, statistics=enough)
    assert changed.resolution["dual_averaging"] == "restarted (mass changed)"


# ---------------------------------------------------------------------------------------------
# Runs: carry across stages and runs
# ---------------------------------------------------------------------------------------------

class _LinearSolver(Solver):
    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return A @ self.parameters


class _LinearEvaluator(Evaluator):
    def __init__(self):
        super().__init__(no_parameters=2, no_observations=2)

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ A.T

    def jacobian(self, datapoints):
        return A, A @ np.asarray(datapoints, dtype=float).reshape(-1)


def _problem(observed=OBSERVED):
    prior = surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(0.0, 1.0) for _ in range(2)])
    return surrDAMH.Problem(prior, surrDAMH.distributions.Normal(mean=observed, sd=OBS_SD), solver=_LinearSolver())


def _run_conf(output_dir):
    return surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedInitialSample([0.3, -0.2]))


def _manifest(output_dir):
    with open(os.path.join(str(output_dir), "sampling_output", "run_manifest.json")) as handle:
        return json.load(handle)


def test_statistics_are_carried_across_stages_and_shape_installs_at_once(tmp_path):
    stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=300),
              Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=200),
              Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=200),
              Stage(algorithm="DAMH", proposal=HamiltonianSpec(mass="covariance", integration_time=1.5),
                    max_evaluations=150)]
    run = _problem().run_sampling_local(_run_conf(tmp_path), stages, surrogate_evaluator=_LinearEvaluator())
    names = [s.name for s in run.stages]
    counts = [load_carry_over(str(tmp_path), name)[1]["posterior_statistics_n"] for name in names]
    assert counts[:3] == [300, 500, 700]  # one state per MH iteration, accumulated, never reset
    assert counts[3] > 700
    # stage 0 installs its first shape at n = warmup (100); stage 2 (adaptive, after 500 states) at once
    trace0 = pd.read_csv(tmp_path / "sampling_output" / "adaptive_stats" / names[0] / "rank0000.csv")
    trace2 = pd.read_csv(tmp_path / "sampling_output" / "adaptive_stats" / names[2] / "rank0000.csv")
    assert trace0["trace_C_over_d"].isna().sum() == 9 and not trace2["trace_C_over_d"].isna().any()
    # the stage-2 shape is estimated from all 500 + 10 states, not from its own 10
    resolution = _manifest(tmp_path)["proposal_resolution"]
    assert [r["name"] for r in resolution] == names
    assert resolution[3]["mass_source"] == "covariance (n=700)" and resolution[3]["integration_time"] == 1.5
    assert (resolution[2]["posterior_statistics_n_start"], resolution[2]["posterior_statistics_n_end"]) == (500, 700)


def test_continuation_carries_the_statistics_and_problem_override_drops_them(tmp_path, capsys):
    run_a = _problem().run_sampling_local(_run_conf(tmp_path / "a"),
                                          [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=250)])
    vector_a = load_carry_over(str(tmp_path / "a"), "alg0000_MH-adaptive")[0]["posterior_statistics"]
    for chains in ("continue", "prior"):
        conf_b = surrDAMH.Configuration(output_dir=str(tmp_path / f"b_{chains}"), use_collector=False,
                                        use_solvers_pool=False)
        run_b = run_a.continue_sampling_local(conf_b, [Stage(algorithm="MH", proposal=RandomWalk(scale=0.4),
                                                             max_evaluations=50)], chains=chains)
        name_b = run_b.stages[0].name
        carried_b = load_carry_over(str(tmp_path / f"b_{chains}"), name_b)[0]["posterior_statistics"]
        assert carried_b[0] == vector_a[0] + 50  # 250 carried + 50 new states
        assert _manifest(tmp_path / f"b_{chains}")["proposal_resolution"][0]["posterior_statistics_n_start"] == 250
    capsys.readouterr()
    conf_c = surrDAMH.Configuration(output_dir=str(tmp_path / "c"), use_collector=False, use_solvers_pool=False)
    run_c = run_a.continue_sampling_local(conf_c, [Stage(algorithm="MH", proposal=RandomWalk(scale=0.4),
                                                         max_evaluations=50)],
                                          problem=_problem(observed=OBSERVED + 0.2))
    assert "carried posterior statistics of the previous run(s) are dropped" in capsys.readouterr().out
    carried_c = load_carry_over(str(tmp_path / "c"), run_c.stages[0].name)[0]["posterior_statistics"]
    assert carried_c[0] == 50


def test_n_seen_is_carried_through_carry_over_and_continuation(tmp_path):
    from surrDAMH.modules.continuation import initial_posterior_statistics
    stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=400, burn_in=True),
              Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=150)]
    run_a = _problem().run_sampling_local(_run_conf(tmp_path / "a"), stages)
    names = [s.name for s in run_a.stages]
    carried = [load_carry_over(str(tmp_path / "a"), name)[0] for name in names]
    assert [int(c["posterior_statistics"][0]) for c in carried] == [100, 250]  # discounted n
    assert [int(c["posterior_statistics_n_seen"]) for c in carried] == [400, 550]  # never discounted
    conf_b = surrDAMH.Configuration(output_dir=str(tmp_path / "b"), use_collector=False, use_solvers_pool=False)
    run_b = run_a.continue_sampling_local(conf_b, [Stage(algorithm="MH", proposal=RandomWalk(scale=0.4),
                                                         max_evaluations=50)])
    carried_b = load_carry_over(str(tmp_path / "b"), run_b.stages[0].name)[0]
    assert int(carried_b["posterior_statistics"][0]) == 300 and int(carried_b["posterior_statistics_n_seen"]) == 600
    # a carry-over written before the key existed: n_seen = n
    old = initial_posterior_statistics({"posterior_statistics": carried[1]["posterior_statistics"]}, 2)
    assert old.n == old.n_seen == 250
    new = initial_posterior_statistics(carried[1], 2)
    assert new.n == 250 and new.n_seen == 550
    # problem= drops the statistics and their n_seen
    conf_c = surrDAMH.Configuration(output_dir=str(tmp_path / "c"), use_collector=False, use_solvers_pool=False)
    run_c = run_a.continue_sampling_local(conf_c, [Stage(algorithm="MH", proposal=RandomWalk(scale=0.4),
                                                         max_evaluations=50)],
                                          problem=_problem(observed=OBSERVED + 0.2))
    carried_c = load_carry_over(str(tmp_path / "c"), run_c.stages[0].name)[0]
    assert int(carried_c["posterior_statistics"][0]) == 50 and int(carried_c["posterior_statistics_n_seen"]) == 50
