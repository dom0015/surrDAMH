#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1b (2026-10-11, author decisions): four follow-ups to C1.

Pinned here:
* J -- ``Hamiltonian(integration_time_jitter=j)``: ``j = 0`` draws nothing (bit-identical
  proposals), ``j > 0`` draws ``u ~ Uniform(1 - j, 1 + j)`` per trajectory from the proposal's own
  generator (distribution of the trajectory lengths), validation, describe/resolution, and a
  jittered DAMH stage samples the closed-form posterior;
* D -- dual-averaging continuation across a small mass change: ``relative_mass_change``,
  ``mass_change_step_factor`` (``(det M_new / det M_old)^(1/(2d))``) on synthetic masses, branch
  selection (continued / restarted) and the restart state (``m = 0``, rescaled step,
  ``mu = log(DA_RESTART_FACTOR * eps0)``), recorded in ``resolution``;
* B -- burn-in discount: ``PosteriorStatistics.discount`` keeps mean and covariance and sets ``n``;
  a ``burn_in`` stage discounts (recorded in ``carry_over/``), other stages do not;
* S -- the S2 exact random walk reads the chain's shared statistics (shape installed at once),
  never updates them (no double count);
* Auto fast-mode constants.
Evidence tables: CHANGELOG.md (2026-10-11, C1b).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

import surrDAMH
from helpers_statistics import (FixedInitialSample, assert_matches_closed_form, chain_statistics,
                                read_weighted_samples)
from surrDAMH.auto import (AUTO_FAST_INTEGRATION_TIME, AUTO_FAST_INTEGRATION_TIME_JITTER, plan_auto)
from surrDAMH.modules.algorithm_interfaces_local import LocalEvaluatorProvider, LocalSolverAdapter
from surrDAMH.modules.algorithms import Algorithm_DAMH, Sample
from surrDAMH.modules.continuation import discount_burn_in_statistics, load_carry_over
from surrDAMH.modules.posterior_statistics import (BURN_IN_STATISTICS_PER_DIMENSION, MASS_MIN_STATES,
                                                   PosteriorStatistics, burn_in_statistics_count)
from surrDAMH.modules.proposal_builder import (DA_CONTINUE_MASS_TOLERANCE, DA_RESTART_FACTOR,
                                               build_exact_step_proposal, build_proposal,
                                               mass_change_step_factor, relative_mass_change)
from surrDAMH.modules.proposals import (GaussRandomWalk, Hamiltonian, Hamiltonian_adaptive,
                                        HamiltonianInfinite, HamiltonianInfinite_adaptive,
                                        INTEGRATION_TIME_MAX_STEPS)
from surrDAMH.proposals import Hamiltonian as HamiltonianSpec
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.parent import Evaluator

pytestmark = pytest.mark.unit

A = np.array([[1.0, 0.5], [0.0, 1.0]])
OBSERVED = A @ np.array([0.7, -0.4])
OBS_SD = np.array([0.3, 0.3])


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


def _prior(d=2):
    return surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(0.0, 1.0) for _ in range(d)])


def _problem():
    return surrDAMH.Problem(_prior(), surrDAMH.distributions.Normal(mean=OBSERVED, sd=OBS_SD),
                            solver=_LinearSolver())


def _posterior():
    cov = np.linalg.inv(np.eye(2) + A.T @ np.diag(1.0 / OBS_SD ** 2) @ A)
    return cov @ (A.T @ np.diag(1.0 / OBS_SD ** 2) @ OBSERVED), cov


def _run_conf(output_dir, initial=(0.3, -0.2), **kwargs):
    return surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedInitialSample(list(initial)), **kwargs)


def _conf():
    from types import SimpleNamespace
    return SimpleNamespace(no_parameters=2, use_surrogate_gradients=True)


def _manifest(output_dir):
    with open(os.path.join(str(output_dir), "sampling_output", "run_manifest.json")) as handle:
        return json.load(handle)


def _feed(states):
    statistics = PosteriorStatistics(states.shape[1])
    for state in states:
        statistics.update(state)
    return statistics


def _wire(proposal):
    """Gradients of the linear-Gaussian log-likelihood / standard-normal log-prior."""
    proposal.set_gradient_functions(
        lambda x: A.T @ ((OBSERVED - A @ x) / OBS_SD ** 2),
        lambda x: -np.asarray(x, dtype=float))
    return proposal


# ---------------------------------------------------------------------------------------------
# J: jittered integration time
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("proposal_class", [Hamiltonian, HamiltonianInfinite, Hamiltonian_adaptive,
                                            HamiltonianInfinite_adaptive])
def test_zero_jitter_is_bit_identical(proposal_class):
    kwargs = dict(no_parameters=2, step_size=0.13, num_steps=None, integration_time=1.1, seed=7,
                  sd_or_cov=[[1.5, 0.2], [0.2, 0.8]])
    plain, zero = _wire(proposal_class(**kwargs)), _wire(proposal_class(integration_time_jitter=0.0, **kwargs))
    x_plain = x_zero = np.array([0.3, -0.2])
    for _ in range(25):
        x_plain, x_zero = plain.propose_sample(x_plain), zero.propose_sample(x_zero)
        assert np.array_equal(x_plain, x_zero) and np.array_equal(plain.p_proposed_end, zero.p_proposed_end)
        if hasattr(plain, "adapt"):
            plain.adapt(x_plain, np.log(0.7), x_plain)
            zero.adapt(x_zero, np.log(0.7), x_zero)
    assert plain._generator.uniform() == zero._generator.uniform()  # same stream position


def test_jitter_draws_the_trajectory_length_from_the_own_generator():
    T, j, step = 1.0, 0.5, 0.01
    proposal = _wire(Hamiltonian(no_parameters=2, step_size=step, num_steps=None, integration_time=T,
                                 integration_time_jitter=j, seed=3))
    reference = np.random.RandomState(3)  # the proposal's own stream: u first, then the momentum
    lengths = []
    x = np.zeros(2)
    for _ in range(400):
        u = reference.uniform(1 - j, 1 + j)
        reference.normal(np.zeros(2), proposal.sd_or_cov)
        proposal.propose_sample(x)
        assert proposal.last_num_steps == max(1, round(T * u / step))
        assert proposal.num_steps == round(T / step)  # the nominal count is restored
        lengths.append(proposal.last_num_steps)
    lengths = np.array(lengths)
    assert lengths.min() >= 50 and lengths.max() <= 150
    assert abs(lengths.mean() - 100) < 5 and abs(np.std(lengths) - 100 * j / np.sqrt(3)) < 3  # Uniform(50, 150)
    # the cap applies to the jittered length as well
    tiny = _wire(Hamiltonian(no_parameters=2, step_size=1e-9, num_steps=None, integration_time=T,
                             integration_time_jitter=0.9, seed=1))
    tiny.step_size = 1e-6  # keep the trajectory cheap: 1e6 * u steps -> capped
    tiny.propose_sample(np.zeros(2))
    assert tiny.last_num_steps == INTEGRATION_TIME_MAX_STEPS


def test_jitter_validation_and_records(capsys):
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="integration_time_jitter must be in"):
            HamiltonianSpec(integration_time=1.0, integration_time_jitter=bad)
        with pytest.raises(ValueError, match="integration_time_jitter must be in"):
            Hamiltonian(no_parameters=2, num_steps=None, integration_time=1.0, integration_time_jitter=bad)
    with pytest.raises(ValueError, match="needs integration_time"):
        HamiltonianSpec(integration_time_jitter=0.3)
    with pytest.raises(ValueError, match="needs integration_time"):
        Hamiltonian(no_parameters=2, integration_time_jitter=0.3)
    assert HamiltonianSpec().integration_time_jitter == 0.0
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(integration_time=1.0, integration_time_jitter=0.4))
    assert "integration_time_jitter=0.4: every trajectory uses T * u, u ~ Uniform(0.6, 1.4)" in stage.describe()
    proposal = build_proposal(stage, _conf(), _prior(), seed=1)
    assert proposal.integration_time_jitter == 0.4 and proposal.resolution["integration_time_jitter"] == 0.4


def test_jittered_damh_stage_samples_the_posterior(tmp_path):
    stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=500),
              Stage(algorithm="DAMH", proposal=HamiltonianSpec(mass="covariance", integration_time=math.pi / 2,
                                                               integration_time_jitter=0.5,
                                                               integrator="dimension_robust"),
                    max_evaluations=4000, subchain_length=1)]
    run = _problem().run_sampling_local(_run_conf(tmp_path), stages, surrogate_evaluator=_LinearEvaluator())
    mean, cov = _posterior()
    statistics = chain_statistics(read_weighted_samples(tmp_path, run.stage_results[1].name, 2))
    assert_matches_closed_form(statistics, mean, cov, "C1b jittered covariance-mass HMC")
    assert _manifest(tmp_path)["proposal_resolution"][1]["integration_time_jitter"] == 0.5


# ---------------------------------------------------------------------------------------------
# D: dual-averaging continuation across a small mass change
# ---------------------------------------------------------------------------------------------

def test_relative_mass_change_and_step_factor_on_synthetic_masses():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(4, 4))
    M = B @ B.T + 4 * np.eye(4)
    assert relative_mass_change(M, M) == 0.0
    assert relative_mass_change(1.1 * M, M) == pytest.approx(0.1)
    assert relative_mass_change(np.diag([1.0, 4.0]), np.eye(2)) == pytest.approx(3.0)
    assert relative_mass_change(np.eye(3), np.eye(2)) == np.inf
    # the step scales with the square root of a uniform mass scale c: M -> c M  =>  eps -> sqrt(c) eps
    for c in (0.5, 2.0, 9.0):
        assert mass_change_step_factor(c * M, M) == pytest.approx(np.sqrt(c))
    # M = Sigma_hat^-1: doubling the covariance estimate halves the mass -> step x 1/sqrt(2)
    Sigma_old = np.linalg.inv(M)
    assert mass_change_step_factor(np.linalg.inv(2 * Sigma_old), M) == pytest.approx(1 / np.sqrt(2))
    # = (det Sigma_old / det Sigma_new)^(1/(2d)) for a general change
    Sigma_new = Sigma_old @ np.diag([1.0, 3.0, 0.5, 2.0])
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T) + np.eye(4)
    expected = (np.linalg.det(Sigma_old) / np.linalg.det(Sigma_new)) ** (1 / 8)
    assert mass_change_step_factor(np.linalg.inv(Sigma_new), M) == pytest.approx(expected)
    assert mass_change_step_factor(-np.eye(2), np.eye(2)) == 1.0  # degenerate -> no rescaling
    # the identity-scaled equivalence the derivation rests on: leapfrog with mass c M and step
    # sqrt(c) eps traces the same positions as mass M and step eps
    grad = lambda x: -np.diag([1.0, 25.0]) @ x
    q0, z = np.array([0.4, -0.3]), np.array([0.7, 0.2])

    def positions(mass, step):
        p = _wire_free(Hamiltonian(no_parameters=2, step_size=step, num_steps=7, sd_or_cov=mass, seed=0), grad)
        L = np.linalg.cholesky(mass)
        q, _ = p._leapfrog(q0, L @ z)  # the same standard-normal draw z for both masses
        return q
    base = np.array([[2.0, 0.3], [0.3, 1.0]])
    np.testing.assert_allclose(positions(4.0 * base, 0.05 * mass_change_step_factor(4.0 * base, base)),
                               positions(base, 0.05), rtol=1e-12, atol=1e-12)


def _wire_free(proposal, grad):
    proposal.set_gradient_functions(grad, lambda x: np.zeros_like(x))
    return proposal


def _finished_stage_carry(mass=1.0, statistics=None):
    """Carry-over of an adaptive Hamiltonian stage that ran 50 dual-averaging updates."""
    spec = HamiltonianSpec(mass=mass, integration_time=1.0)
    proposal = build_proposal(Stage(algorithm="MH", proposal=spec), _conf(), _prior(), seed=1, statistics=statistics)
    rng = np.random.default_rng(3)
    for _ in range(50):
        proposal.adapt(np.zeros(2), float(np.log(rng.uniform(0.6, 1.0))), np.zeros(2))
    proposal.set_pooled_state(proposal.adapted_state()[None, :])
    return proposal, proposal.carry_over()


@pytest.mark.parametrize("scale", [1.0, 1.0 + DA_CONTINUE_MASS_TOLERANCE * 0.99, 0.85])
def test_small_mass_change_continues_the_recursion(scale, capsys):
    previous, carried = _finished_stage_carry()
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(mass=float(np.sqrt(scale)), integration_time=1.0))
    continued = build_proposal(stage, _conf(), _prior(), seed=2, carried=carried)
    assert continued.resolution["dual_averaging"] == "continued"
    assert continued.resolution["mass_change_r"] == pytest.approx(abs(scale - 1.0))
    assert (continued.m, continued.H_bar, continued.mu, continued.log_eps_bar) == \
        (50, previous.H_bar, previous.mu, previous.log_eps_bar)
    assert continued.step_size == pytest.approx(carried["step_size"])  # no rescaling when continuing
    assert "mass changed" not in capsys.readouterr().out


def test_large_mass_change_restarts_from_the_rescaled_step(capsys):
    _previous, carried = _finished_stage_carry()
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(mass=[1.0, 2.0], integration_time=1.0))  # M = diag(1, 4)
    restarted = build_proposal(stage, _conf(), _prior(), seed=2, carried=carried)
    resolution = restarted.resolution
    assert resolution["dual_averaging"] == "restarted (mass changed)"
    assert resolution["mass_change_r"] == pytest.approx(3.0) and 3.0 > DA_CONTINUE_MASS_TOLERANCE
    factor = (4.0 / 1.0) ** (1 / 4)  # (det M_new / det M_old)^(1/(2d)), d = 2
    eps0 = carried["step_size"] * factor
    assert restarted.step_size == pytest.approx(eps0) and resolution["step_size_initial"] == pytest.approx(eps0)
    assert (restarted.m, restarted.H_bar) == (0, 0.0)
    assert restarted.mu == pytest.approx(np.log(DA_RESTART_FACTOR * eps0)) and DA_RESTART_FACTOR == 2.0
    assert restarted.log_eps_bar == pytest.approx(np.log(eps0))
    assert restarted.num_steps == round(1.0 / eps0)
    # the first update with alpha = delta (H_bar stays 0) jumps only to 2 x eps0, not 10 x
    restarted.adapt(np.zeros(2), np.log(0.8), np.zeros(2))
    assert restarted.step_size == pytest.approx(DA_RESTART_FACTOR * eps0)
    assert "dual averaging restarted from the carried step" in capsys.readouterr().out
    # a pinned step never continues nor rescales
    pinned = build_proposal(Stage(algorithm="MH", proposal=HamiltonianSpec(step_size=0.05, adaptive=True)),
                            _conf(), _prior(), seed=2, carried=carried)
    assert pinned.resolution["dual_averaging"] == "fresh" and pinned.resolution["mass_change_r"] is None
    assert pinned.m == 0 and pinned.mu == pytest.approx(np.log(10 * 0.05))


def test_covariance_mass_branch_follows_the_statistics():
    rng = np.random.default_rng(0)
    statistics = _feed(rng.normal(size=(400, 2)) * [2.0, 0.1])
    _p, carried = _finished_stage_carry(mass="covariance", statistics=statistics)
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(mass="covariance", integration_time=1.0))
    for state in rng.normal(size=(40, 2)) * [2.0, 0.1]:  # more states from the same target: small change
        statistics.update(state)
    small = build_proposal(stage, _conf(), _prior(), seed=3, carried=carried, statistics=statistics)
    assert small.resolution["dual_averaging"] == "continued"
    assert small.resolution["mass_change_r"] <= DA_CONTINUE_MASS_TOLERANCE
    for state in rng.normal(size=(400, 2)) * [0.5, 0.5]:  # the shape changes a lot
        statistics.update(state)
    large = build_proposal(stage, _conf(), _prior(), seed=3, carried=carried, statistics=statistics)
    assert large.resolution["dual_averaging"] == "restarted (mass changed)"
    expected = mass_change_step_factor(statistics.precision(), carried["hamiltonian_mass"])
    assert large.step_size == pytest.approx(carried["step_size"] * expected)


# ---------------------------------------------------------------------------------------------
# B: burn-in discount of the shared statistics
# ---------------------------------------------------------------------------------------------

def test_discount_preserves_mean_and_covariance_and_sets_n():
    rng = np.random.default_rng(4)
    statistics = _feed(rng.normal(size=(1000, 3)) @ rng.normal(size=(3, 3)) + 2.0)
    mean, cov = statistics.mean.copy(), statistics.covariance(shrinkage=False).copy()
    assert statistics.discount(100) == 1000
    assert statistics.n == 100
    np.testing.assert_array_equal(statistics.mean, mean)
    np.testing.assert_allclose(statistics.covariance(shrinkage=False), cov, rtol=1e-13)
    # a later state now weighs 1/101 in the mean (as after 100 states), not 1/1001
    x = np.full(3, 50.0)
    statistics.update(x)
    np.testing.assert_allclose(statistics.mean, mean + (x - mean) / 101)
    assert statistics.discount(500) is None and statistics.n == 101  # nothing to discount
    with pytest.raises(ValueError):
        statistics.discount(1)
    assert BURN_IN_STATISTICS_PER_DIMENSION == 10
    assert burn_in_statistics_count(2) == MASS_MIN_STATES == 100 and burn_in_statistics_count(37) == 370


def test_only_burn_in_stages_are_discounted():
    statistics = _feed(np.random.default_rng(0).normal(size=(300, 2)))
    summary: dict = {}
    assert discount_burn_in_statistics(Stage(algorithm="MH", max_evaluations=5), statistics, summary) is None
    assert discount_burn_in_statistics(Stage(algorithm="MH", max_evaluations=5, is_excluded=True),
                                       statistics, summary) is None
    assert statistics.n == 300 and summary == {}
    assert discount_burn_in_statistics(Stage(algorithm="MH", max_evaluations=5, burn_in=True), statistics, summary) == 300
    assert statistics.n == 100 and summary == {"posterior_statistics_discounted_from_n": 300}


def test_burn_in_run_records_the_discount(tmp_path):
    stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=400, burn_in=True),
              Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=150, is_excluded=True),
              Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=150)]
    run = _problem().run_sampling_local(_run_conf(tmp_path), stages)
    names = [s.name for s in run.stages]
    carried, summary = load_carry_over(str(tmp_path), names[0])
    assert summary["posterior_statistics_discounted_from_n"] == 400 and summary["posterior_statistics_n"] == 100
    assert carried["posterior_statistics"][0] == 100
    for name in names[1:]:  # is_excluded and plain stages: not discounted
        carried, summary = load_carry_over(str(tmp_path), name)
        assert "posterior_statistics_discounted_from_n" not in summary
    assert [load_carry_over(str(tmp_path), n)[1]["posterior_statistics_n"] for n in names] == [100, 250, 400]
    resolution = _manifest(tmp_path)["proposal_resolution"]
    assert [r["posterior_statistics_discounted_from_n"] for r in resolution] == [400, None, None]


# ---------------------------------------------------------------------------------------------
# S: the S2 exact step reads the shared statistics
# ---------------------------------------------------------------------------------------------

def test_exact_proposal_installs_the_shared_shape_at_once():
    rng = np.random.default_rng(5)
    statistics = _feed(rng.normal(size=(300, 2)) * [1.5, 0.1])
    carried = {"scale": 0.3 * np.eye(2)}
    exact = build_exact_step_proposal(_conf(), _prior(), seed=9, carried=carried, statistics=statistics)
    assert exact.statistics is statistics and not exact._owns_statistics
    np.testing.assert_allclose(exact.base_cov, 2.38 ** 2 / 2 * statistics.covariance(), rtol=1e-12)
    # the carried scale is kept as the trace of the effective covariance, the shape replaced
    assert np.trace(exact.sd_or_cov) == pytest.approx(np.trace(carried["scale"]))
    # too few states: carried scale as it is, shape later
    few = _feed(rng.normal(size=(50, 2)))
    exact_few = build_exact_step_proposal(_conf(), _prior(), seed=9, carried=carried, statistics=few)
    np.testing.assert_allclose(exact_few.sd_or_cov, carried["scale"])
    # without statistics: the private S2/C1 object
    private = build_exact_step_proposal(_conf(), _prior(), seed=9, carried=carried)
    assert private._owns_statistics and private.statistics.n == 0


def test_exact_steps_update_the_shared_statistics_once(tmp_path):
    conf = _run_conf(tmp_path)
    conf.resolve_problem_sizes(2, 2)
    problem = _problem()
    statistics = PosteriorStatistics(2)
    exact = build_exact_step_proposal(conf, problem.prior, seed=5, statistics=statistics)
    stage = Stage(algorithm="DAMH", max_evaluations=600, exact_step_probability=0.5, name="direct")
    stage_proposal = GaussRandomWalk(2, seed=11)
    stage_proposal.set_covariance(0.5)
    algo = Algorithm_DAMH(stage=stage, proposal=stage_proposal, initial_sample=Sample(np.array([0.3, -0.2])),
                          rank_world=0, conf=conf, prior=problem.prior, likelihood=problem.likelihood,
                          observation_provider=LocalSolverAdapter(_LinearSolver()),
                          evaluator_provider=LocalEvaluatorProvider(_LinearEvaluator()), seed=2,
                          exact_step_seed=4, exact_proposal=exact, posterior_statistics=statistics)
    algo.run()
    stats = pd.read_csv(tmp_path / "sampling_output" / "subchain_stats" / "direct" / "rank0000.csv")
    assert algo.counter_exact_steps == int((stats["kernel"] == 1).sum()) > 0
    # ONE update per outer iteration (exact steps included), none by the exact proposal itself
    assert statistics.n == len(stats)
    assert exact.n == algo.counter_exact_steps  # its own adapt() calls
    # and it installed shapes from the shared statistics once they held warmup states
    assert np.isfinite(exact._trace_over_d)


def test_s2_stage_in_a_run_shares_the_statistics(tmp_path, monkeypatch):
    import surrDAMH.runner_local as runner_local
    built = {}
    original = runner_local.build_exact_step_proposal

    def recording(*args, **kwargs):
        built["statistics"] = kwargs.get("statistics")
        built["proposal"] = original(*args, **kwargs)
        return built["proposal"]

    monkeypatch.setattr(runner_local, "build_exact_step_proposal", recording)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=300),
              Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=300, subchain_length=3,
                    exact_step_probability=0.3)]
    run = _problem().run_sampling_local(_run_conf(tmp_path), stages, surrogate_evaluator=_LinearEvaluator())
    assert isinstance(built["statistics"], PosteriorStatistics)
    assert built["proposal"].statistics is built["statistics"]
    name = run.stages[1].name
    stats = pd.read_csv(tmp_path / "sampling_output" / "subchain_stats" / name / "rank0000.csv")
    n0 = load_carry_over(str(tmp_path), run.stages[0].name)[1]["posterior_statistics_n"]
    assert load_carry_over(str(tmp_path), name)[1]["posterior_statistics_n"] == n0 + len(stats)


# ---------------------------------------------------------------------------------------------
# Auto
# ---------------------------------------------------------------------------------------------

def test_auto_fast_mode_constants(tmp_path):
    from types import SimpleNamespace
    assert AUTO_FAST_INTEGRATION_TIME == pytest.approx(3.0) and AUTO_FAST_INTEGRATION_TIME_JITTER == 0.5
    conf = surrDAMH.Configuration(output_dir=str(tmp_path / "out"), use_collector=False, use_solvers_pool=False)
    conf.use_collector = True
    conf.no_samplers = 1
    plan = plan_auto(SimpleNamespace(no_parameters=2, no_observations=2, solver_instance=None), conf,
                     budget=10_000, time_limit=None, mode="fast", local=False, surrogate_updater=None)
    for stage in plan.stages[1:]:
        assert stage.proposal.integration_time == pytest.approx(3.0)
        assert stage.proposal.integration_time_jitter == 0.5
        assert (stage.proposal.mass, stage.proposal.integrator) == ("covariance", "dimension_robust")
    assert "u ~ Uniform(0.5, 1.5) per trajectory" in plan.describe()
    assert "integration_time_jitter=0.5" in plan.manifest_entry()["proposal"]
