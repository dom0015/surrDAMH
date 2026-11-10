#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deferred adaptation of the adaptive proposals (2026-10-23, paper §6.5, "One step of the
production stage", step (4)): ``set_deferred_adaptation`` / ``apply_pending_adaptation``.

In deferred mode ``adapt()`` only gathers the acceptance probabilities (the random walk's
covariance shape still follows its statistics) and ``apply_pending_adaptation()`` advances the
feedback recursion ONCE on their mean and installs the live parameters, so the recursion's index
counts installations (revised 2026-10-23 after a smoke run: a per-sub-step recursion fed by
frozen parameters ran off to the clip). Checked per class:

1. ``set_deferred_adaptation(False)`` on a fresh proposal (and switching on then off before any
   adapt) changes nothing;
2. deferred, never applied: live parameters AND feedback recursion unchanged, proposal stream
   equal to an un-adapted twin's;
3. one apply = one update on the mean acceptance probability (gain ``j**(-0.7)`` / one dual-
   averaging step), True once, then False; regression test for the run-off;
4. ``BlockProposal`` (Mixture) forwards; fixed proposals return False and are unaffected;
5. ``set_pooled_state`` in deferred mode leaves live == recursion.
"""

from __future__ import annotations

import numpy as np
import pytest

from surrDAMH.modules.proposals import (
    BlockProposal,
    GaussRandomWalk,
    GaussRandomWalk_adaptive,
    Hamiltonian,
    Hamiltonian_adaptive,
    HamiltonianInfinite_adaptive,
    PCN,
    PCN_adaptive,
    _own_acceptance_probabilities,
)

pytestmark = pytest.mark.unit

D = 2
N_ADAPT = 60


def _with_gradients(proposal):
    proposal.set_gradient_functions(lambda q: -np.asarray(q, dtype=float),
                                    lambda q: -np.asarray(q, dtype=float))
    return proposal


FACTORIES = {
    "random_walk": lambda: GaussRandomWalk_adaptive(no_parameters=D, sd_or_cov=0.5, seed=3,
                                                    period=5, warmup=10),
    "pcn": lambda: PCN_adaptive(no_parameters=D, beta=0.4, prior_mean=np.zeros(D),
                                prior_sd_or_cov=1.0, seed=3, period=5),
    "hamiltonian": lambda: _with_gradients(Hamiltonian_adaptive(no_parameters=D, step_size=0.3,
                                                                num_steps=4, seed=3, period=5)),
    "hamiltonian_T": lambda: _with_gradients(Hamiltonian_adaptive(no_parameters=D, step_size=0.3,
                                                                  num_steps=None, integration_time=1.2,
                                                                  seed=3, period=5)),
    "hamiltonian_infinite": lambda: _with_gradients(HamiltonianInfinite_adaptive(
        no_parameters=D, step_size=0.3, num_steps=4, seed=3, period=5)),
}


#: centre of the synthetic acceptance probabilities: near each recursion's target
CENTER = {"random_walk": 0.25, "pcn": 0.25, "hamiltonian": 0.8, "hamiltonian_T": 0.8,
          "hamiltonian_infinite": 0.8}


def _adapt_inputs(n=N_ADAPT, seed=11, center=0.25):
    """Synthetic (log alpha, current sample, sub-chain log alphas) triples, -inf included;
    alphas uniform on ``center +- 0.2`` (near the target, so the step does not collapse)."""
    rng = np.random.default_rng(seed)
    inputs = []
    for i in range(n):
        log_alpha = -np.inf if i % 13 == 3 else float(np.log(rng.uniform(center - 0.2, center + 0.2)))
        current = rng.normal(size=D)
        subchain = None if i % 2 else [float(np.log(rng.uniform(center - 0.2, center + 0.2))) for _ in range(5)] + ([-np.inf] if i % 10 == 0 else [])
        inputs.append((log_alpha, current, subchain))
    return inputs


def _feed(proposal, inputs):
    for log_alpha, current, subchain in inputs:
        proposal.adapt(proposed_sample=current + 0.1, log_acceptance_probability=log_alpha,
                       current_sample=current, subchain_log_acceptance_probabilities=subchain)


def _stream(proposal, count=5):
    x = np.array([0.3, -0.2])
    return np.array([proposal.propose_sample(x) for _ in range(count)])


def _recursion(proposal):
    if isinstance(proposal, GaussRandomWalk_adaptive):
        return [proposal.log_sigma, proposal.base_cov, proposal.sd_or_cov, proposal.n_scale]
    if isinstance(proposal, PCN_adaptive):
        return [proposal.logit_beta, proposal.n_scale]
    return [proposal.m, proposal.H_bar, proposal.log_eps_bar, proposal.mu]


def _assert_same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# -- 1 ----------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(FACTORIES))
def test_flag_off_changes_nothing(name):
    reference, switched_off, toggled = FACTORIES[name](), FACTORIES[name](), FACTORIES[name]()
    switched_off.set_deferred_adaptation(False)
    toggled.set_deferred_adaptation(True)
    toggled.set_deferred_adaptation(False)
    inputs = _adapt_inputs(center=CENTER[name])
    for proposal in (reference, switched_off, toggled):
        _feed(proposal, inputs)
    reference_stream = _stream(reference)
    for proposal in (switched_off, toggled):
        _assert_same(_recursion(proposal), _recursion(reference))
        assert proposal.adaptation_stats_rows == reference.adaptation_stats_rows
        np.testing.assert_array_equal(_stream(proposal), reference_stream)
        assert proposal.apply_pending_adaptation() is False


# -- 2 + 3 ------------------------------------------------------------------------------
def _own_alphas(inputs):
    """The acceptance probabilities ``adapt()`` feeds to the proposal's own recursion."""
    return [alpha for log_alpha, _, subchain in inputs
            for alpha in _own_acceptance_probabilities(log_alpha, subchain)]


def _mean(alphas):
    total = 0.0
    for alpha in alphas:
        total += alpha
    return total / len(alphas)


def _sigmoid(logit):
    twin = FACTORIES["pcn"]()
    twin.logit_beta = logit
    return twin._beta_from_logit()


def _feedback_state(proposal):
    """The acceptance-feedback recursion (the random walk: its effective log-scale, invariant
    under the trace-continuous shape installs)."""
    if isinstance(proposal, GaussRandomWalk_adaptive):
        return [proposal._effective_log_scale(), proposal.n_scale]
    if isinstance(proposal, PCN_adaptive):
        return [proposal.logit_beta, proposal.beta, proposal.n_scale]
    return [proposal.m, proposal.H_bar, proposal.log_eps_bar, proposal.mu, proposal.step_size,
            proposal.num_steps]


@pytest.mark.parametrize("name", list(FACTORIES))
def test_deferred_never_applied_moves_only_accumulators(name):
    deferred, untouched, default = FACTORIES[name](), FACTORIES[name](), FACTORIES[name]()
    deferred.set_deferred_adaptation(True)
    inputs = _adapt_inputs(center=CENTER[name])
    _feed(deferred, inputs)
    _feed(default, inputs)
    alphas = _own_alphas(inputs)
    assert deferred._deferred_alpha_count == len(alphas)
    if isinstance(deferred, GaussRandomWalk_adaptive):
        np.testing.assert_allclose(_feedback_state(deferred)[0], _feedback_state(untouched)[0], rtol=1e-13)
        assert deferred.n_scale == 0
        # the state-based shape moves as in default mode
        np.testing.assert_array_equal(deferred.base_cov, default.base_cov)
        assert not np.array_equal(deferred.base_cov, untouched.base_cov)
    else:
        _assert_same(_feedback_state(deferred), _feedback_state(untouched))
    if isinstance(deferred, (GaussRandomWalk_adaptive, PCN_adaptive)):
        # rows keep their adapt() period and the period's mean alpha
        assert [row[:2] for row in deferred.adaptation_stats_rows] == \
            [row[:2] for row in default.adaptation_stats_rows]
    else:
        assert deferred.adaptation_stats_rows == []  # m counts installations: none yet
    np.testing.assert_array_equal(_stream(deferred), _stream(untouched))


@pytest.mark.parametrize("name", list(FACTORIES))
def test_apply_is_one_update_on_the_mean_alpha(name):
    deferred = FACTORIES[name]()
    deferred.set_deferred_adaptation(True)
    expected = FACTORIES[name]()  # updated by hand below
    for batch, seed in enumerate((11, 12), start=1):
        inputs = _adapt_inputs(seed=seed, center=CENTER[name])
        _feed(deferred, inputs)
        mean_alpha = _mean(_own_alphas(inputs))
        if isinstance(deferred, GaussRandomWalk_adaptive):
            log_sigma = deferred.log_sigma + batch ** (-0.7) * (mean_alpha - deferred.target_rate)
        elif isinstance(deferred, PCN_adaptive):
            logit = deferred.logit_beta + batch ** (-0.7) * (mean_alpha - deferred.target_rate)
        else:
            expected._dual_averaging_update(mean_alpha)
        assert deferred.apply_pending_adaptation() is True
        if isinstance(deferred, GaussRandomWalk_adaptive):
            assert deferred.log_sigma == log_sigma
            assert deferred._live_log_sigma == log_sigma
            np.testing.assert_array_equal(deferred._live_L, deferred.L)
            np.testing.assert_allclose(deferred.live_covariance, deferred.sd_or_cov, rtol=1e-12, atol=1e-15)
        elif isinstance(deferred, PCN_adaptive):
            assert deferred.logit_beta == logit
            assert deferred.beta == _sigmoid(logit)
        else:
            _assert_same(_feedback_state(deferred), _feedback_state(expected))
            assert deferred.m == batch
        assert deferred.apply_pending_adaptation() is False  # nothing gathered since
    # leaving deferred mode applies once with the gathered acceptances, then nothing is pending
    _feed(deferred, _adapt_inputs(seed=13, n=5, center=CENTER[name]))
    before = _feedback_state(deferred)
    deferred.set_deferred_adaptation(False)
    assert _feedback_state(deferred) != before
    assert deferred.apply_pending_adaptation() is False


def test_dual_averaging_deferred_does_not_run_off():
    """Regression (2026-10-23 smoke run): alpha = 0 on every sub-step between installations
    drove the per-sub-step recursion to the clip (log step -51.9, num_steps 1000)."""
    proposal = FACTORIES["hamiltonian_T"]()
    untouched = FACTORIES["hamiltonian_T"]()
    proposal.set_deferred_adaptation(True)
    for _ in range(2000):
        proposal.adapt(proposed_sample=np.zeros(D), log_acceptance_probability=-np.inf,
                       current_sample=np.zeros(D))
    assert proposal.log_eps_bar == untouched.log_eps_bar
    assert proposal.step_size == untouched.step_size
    assert proposal.num_steps == untouched.num_steps
    assert proposal.m == 0
    assert proposal.apply_pending_adaptation() is True
    expected = FACTORIES["hamiltonian_T"]()
    expected._dual_averaging_update(0.0)
    assert proposal.m == 1
    assert proposal.step_size == expected.step_size
    assert np.log(proposal.step_size) > proposal.mu - 5.0


def test_random_walk_live_covariance_is_frozen_but_sd_or_cov_follows_recursion():
    proposal = FACTORIES["random_walk"]()
    initial = proposal.sd_or_cov.copy()
    proposal.set_deferred_adaptation(True)
    _feed(proposal, _adapt_inputs())
    np.testing.assert_allclose(proposal.live_covariance, initial, rtol=1e-12)
    assert not np.allclose(proposal.sd_or_cov, initial)  # the shape was installed


def test_pcn_records_recursion_beta_without_installing():
    proposal = FACTORIES["pcn"]()
    proposal.set_deferred_adaptation(True)
    _feed(proposal, _adapt_inputs())
    assert proposal.beta == 0.4
    assert proposal.adaptation_stats_rows[-1][2] == proposal._beta_from_logit()
    assert proposal.carry_over()["beta"] == proposal._beta_from_logit()


# -- 4 ----------------------------------------------------------------------------------
def _mixture(seed=7):
    components = [GaussRandomWalk_adaptive(no_parameters=D, sd_or_cov=0.5, seed=0, period=5, warmup=10),
                  PCN_adaptive(no_parameters=D, beta=0.4, prior_mean=np.zeros(D), prior_sd_or_cov=1.0, seed=0,
                               period=5)]
    return BlockProposal(no_parameters=D, list_of_proposals=components, list_of_groups=[[0, 1]] * 2,
                         seed=seed, is_mixture=True)


def test_mixture_forwards():
    deferred, untouched, default = _mixture(), _mixture(), _mixture()  # default: same draws
    deferred.set_deferred_adaptation(True)
    assert all(component._deferred_adaptation for component in deferred.list_of_proposals)
    x = np.array([0.3, -0.2])
    for log_alpha, current, subchain in _adapt_inputs():
        for mixture in (deferred, untouched, default):
            mixture.choose_group()
        proposed = deferred.propose_sample(x)
        np.testing.assert_array_equal(proposed, untouched.propose_sample(x))  # live never moved
        default.propose_sample(x)
        for mixture in (deferred, default):
            mixture.adapt(proposed_sample=proposed, log_acceptance_probability=log_alpha,
                          current_sample=current, subchain_log_acceptance_probabilities=subchain)
    walk, pcn = deferred.list_of_proposals
    assert walk.n_scale == 0 and pcn.n_scale == 0  # no feedback step before the installation
    assert walk._deferred_alpha_count > 0 and pcn._deferred_alpha_count > 0
    logit = pcn.logit_beta + (pcn._deferred_alpha_sum / pcn._deferred_alpha_count - pcn.target_rate)
    assert deferred.apply_pending_adaptation() is True
    assert pcn.logit_beta == logit and pcn.beta == _sigmoid(logit)
    assert walk._deferred_index == 1 and walk._live_log_sigma == walk.log_sigma
    assert deferred.apply_pending_adaptation() is False


def test_mixture_apply_true_if_only_one_component_changed():
    mixture = _mixture()
    mixture.set_deferred_adaptation(True)
    mixture.group_index = 1  # only the pCN component adapts
    _feed(mixture, _adapt_inputs(n=10))
    assert mixture.apply_pending_adaptation() is True
    assert mixture.apply_pending_adaptation() is False


@pytest.mark.parametrize("make", [
    lambda: GaussRandomWalk(D, 0.5, seed=4),
    lambda: PCN(D, beta=0.3, prior_mean=np.zeros(D), prior_sd_or_cov=1.0, seed=4),
    lambda: _with_gradients(Hamiltonian(D, step_size=0.2, num_steps=3, seed=4)),
])
def test_fixed_proposals_unaffected(make):
    proposal, twin = make(), make()
    proposal.set_deferred_adaptation(True)
    _feed(proposal, _adapt_inputs(n=5))
    assert proposal.apply_pending_adaptation() is False
    proposal.set_deferred_adaptation(False)
    np.testing.assert_array_equal(_stream(proposal), _stream(twin))


# -- 5 ----------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(FACTORIES))
def test_pooled_state_in_deferred_mode_syncs_live(name):
    deferred, default, other = FACTORIES[name](), FACTORIES[name](), FACTORIES[name]()
    deferred.set_deferred_adaptation(True)
    inputs = _adapt_inputs(center=CENTER[name])
    _feed(deferred, inputs)
    deferred.apply_pending_adaptation()
    extra = _adapt_inputs(seed=14, n=5, center=CENTER[name])
    _feed(deferred, extra)  # gathered, then dropped by the pooling
    _feed(default, inputs)
    _feed(default, extra)
    _feed(other, _adapt_inputs(seed=12, center=CENTER[name]))
    states = np.vstack([deferred.adapted_state(), other.adapted_state()])
    deferred.set_pooled_state(states)
    default.set_pooled_state(states)
    if isinstance(deferred, GaussRandomWalk_adaptive):
        # same private statistics (same current samples) -> same pooled shape and scale
        assert deferred.log_sigma == default.log_sigma
        np.testing.assert_array_equal(deferred.base_cov, default.base_cov)
        assert deferred._live_log_sigma == deferred.log_sigma
        np.testing.assert_array_equal(deferred._live_L, deferred.L)
        np.testing.assert_allclose(deferred.live_covariance, deferred.sd_or_cov, rtol=1e-12, atol=1e-15)
    elif isinstance(deferred, PCN_adaptive):
        assert deferred.logit_beta == default.logit_beta
        assert deferred.beta == default.beta == deferred._beta_from_logit()
    else:
        assert deferred.log_eps_bar == default.log_eps_bar
        assert deferred.step_size == default.step_size == float(np.exp(deferred.log_eps_bar))
        assert deferred.num_steps == default.num_steps
    assert deferred.apply_pending_adaptation() is False
    assert deferred.carry_over().keys() == default.carry_over().keys()
