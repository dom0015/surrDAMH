#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S6 (2026-10-10): in a DAMH stage the adaptive random walk and pCN adapt their step on the
sub-chain's OWN per-step acceptance probabilities (one Robbins-Monro step per sub-chain step)
instead of the outer DAMH acceptance; in an MH stage nothing changes, bit for bit.

Pinned here:
* the recursion itself, on stub call sequences: ``n_scale`` advances by K per call, the scale
  equals the hand-computed recursion over the K alphas, the shape statistics / covariance install
  / ``adaptation_stats`` stay keyed on outer calls; with ``None`` the state equals the pre-S6
  formula (reference implementation copied from the code before S6);
* integration on the 2-D linear-Gaussian problem with an EXACT surrogate (surrogate posterior =
  posterior): the K = 20 adapted step lands next to the MH-adapted step and within 2x of the
  fixed-step optimum (pre-S6: 6.7x for the random walk), K = 1 still samples the closed-form
  posterior, and MH stages are byte-identical to the code before S6.
Evidence tables: CHANGELOG.md (2026-10-10, S6) and scratchpad ``s6/`` of that session.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import os

import numpy as np
import pytest

import surrDAMH
from helpers_statistics import FixedInitialSample, assert_matches_closed_form, chain_statistics, read_weighted_samples
from surrDAMH.modules.continuation import load_carry_over
from surrDAMH.modules.proposals import GaussRandomWalk_adaptive, PCN_adaptive, acceptance_probability
from surrDAMH.proposals import PCN, RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.parent import Evaluator

pytestmark = pytest.mark.unit

A = np.array([[1.0, 0.5], [0.0, 1.0]])
OBSERVED = A @ np.array([0.7, -0.4])
OBS_SD = np.array([0.3, 0.3])
POSTERIOR_COV = np.linalg.inv(np.eye(2) + A.T @ np.diag(OBS_SD ** -2) @ A)
POSTERIOR_MEAN = POSTERIOR_COV @ (A.T @ np.diag(OBS_SD ** -2) @ OBSERVED)


# ---------------------------------------------------------------------------------------------
# Reference implementations: the adapt() bodies of the code BEFORE S6 (proposals.py, 2026-10-09)
# ---------------------------------------------------------------------------------------------

def _pre_s6_random_walk_adapt(self, proposed_sample, log_acceptance_probability, current_sample):
    alpha = acceptance_probability(log_acceptance_probability)
    self.n += 1
    self.log_sigma += self.n ** (-0.7) * (alpha - self.target_rate)
    state = np.asarray(current_sample, dtype=float)
    difference = state - self.mean
    self.mean = self.mean + difference / self.n
    self.M2 += np.outer(difference, state - self.mean)
    self._period_alpha_sum += alpha
    self._period_alpha_count += 1
    if self.n >= self.warmup and self.n % self.period == 0 \
            and self._install_covariance_from_statistics(self.n, self.M2):
        self._refactor()
    else:
        self._update_public_covariance()
    if self.n % self.period == 0:
        self.adaptation_stats_rows.append((self.n, self._period_alpha_sum / max(self._period_alpha_count, 1),
                                           self.log_sigma, self._trace_over_d, self._shrinkage_delta))
        self._period_alpha_sum = 0.0
        self._period_alpha_count = 0


def _pre_s6_pcn_adapt(self, proposed_sample, log_acceptance_probability, current_sample):
    alpha = acceptance_probability(log_acceptance_probability)
    self.n += 1
    self.logit_beta += self.n ** (-0.7) * (alpha - self.target_rate)
    self._set_beta_from_logit()
    self._period_alpha_sum += alpha
    self._period_alpha_count += 1
    if self.n % self.period == 0:
        self.adaptation_stats_rows.append((self.n, self._period_alpha_sum / max(self._period_alpha_count, 1),
                                           self.beta))
        self._period_alpha_sum = 0.0
        self._period_alpha_count = 0


def _random_walk(**kwargs):
    return GaussRandomWalk_adaptive(no_parameters=2, sd_or_cov=0.5, seed=3, **kwargs)


def _pcn(**kwargs):
    return PCN_adaptive(no_parameters=2, beta=0.3, prior_mean=np.zeros(2), prior_sd_or_cov=1.0, seed=3, **kwargs)


def _stub_sequence(no_calls, rng_seed=0):
    """(proposed, log alpha, current) triples with a mix of accepted / rejected / -inf calls."""
    rng = np.random.default_rng(rng_seed)
    out = []
    for i in range(no_calls):
        log_alpha = -np.inf if i % 7 == 3 else float(np.log(rng.uniform(0.01, 1.5)))
        out.append((rng.normal(size=2), log_alpha, rng.normal(size=2)))
    return out


def _state(proposal):
    keys = ("n", "log_sigma", "logit_beta", "beta", "mean", "M2", "base_cov", "L", "sd_or_cov",
            "_period_alpha_sum", "_period_alpha_count", "adaptation_stats_rows", "_trace_over_d")
    return {key: copy.deepcopy(getattr(proposal, key)) for key in keys if hasattr(proposal, key)}


def _assert_states_equal(new, old):
    assert new.keys() == old.keys()
    for key in new:
        # assert_array_equal treats NaN == NaN (the random walk logs NaN before the first install)
        np.testing.assert_array_equal(np.asarray(new[key], dtype=float), np.asarray(old[key], dtype=float),
                                      err_msg=key)


# ---------------------------------------------------------------------------------------------
# MH stage (no sub-chain list): bit for bit the pre-S6 recursion
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("make, reference", [(lambda: _random_walk(period=5, warmup=20), _pre_s6_random_walk_adapt),
                                             (lambda: _pcn(period=5), _pre_s6_pcn_adapt)],
                         ids=["random_walk", "pcn"])
def test_without_subchain_list_the_recursion_is_the_pre_s6_one_bit_for_bit(make, reference):
    new, old = make(), make()
    for proposed, log_alpha, current in _stub_sequence(137):
        new.adapt(proposed, log_alpha, current)
        reference(old, proposed, log_alpha, current)
        _assert_states_equal(_state(new), _state(old))
    assert new.n_scale == new.n == 137
    np.testing.assert_array_equal(new.propose_sample(np.zeros(2)), old.propose_sample(np.zeros(2)))


# ---------------------------------------------------------------------------------------------
# DAMH stage: one scale step per sub-chain step, shape and bookkeeping per outer call
# ---------------------------------------------------------------------------------------------

def _subchain_sequence(no_calls, K, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    out = []
    for i in range(no_calls):
        subchain = [-np.inf if (i + j) % 5 == 0 else float(np.log(rng.uniform(0.01, 1.5))) for j in range(K)]
        # the outer value is deliberately extreme: it must not influence the scale
        out.append((rng.normal(size=2), float(rng.choice([0.0, -np.inf])), rng.normal(size=2), subchain))
    return out


@pytest.mark.parametrize("K", [1, 4])
def test_random_walk_scale_runs_one_step_per_subchain_step(K):
    proposal = _random_walk(period=5, warmup=10 ** 9)  # no covariance install: log_sigma is the pure recursion
    sequence = _subchain_sequence(23, K)
    log_sigma, n_scale, states = 0.0, 0, []
    for call, (proposed, outer, current, subchain) in enumerate(sequence, start=1):
        proposal.adapt(proposed, outer, current, subchain_log_acceptance_probabilities=subchain)
        for log_alpha in subchain:
            n_scale += 1
            log_sigma += n_scale ** (-0.7) * (acceptance_probability(log_alpha) - 0.234)
        states.append(current)
        assert proposal.n_scale == n_scale == K * call
        assert proposal.n == call  # the shape counter advances once per outer call
        assert proposal.log_sigma == pytest.approx(log_sigma, rel=0, abs=1e-12)
        np.testing.assert_allclose(proposal.mean, np.mean(states, axis=0), atol=1e-12)  # one Welford update per call
    centred = np.asarray(states) - np.mean(states, axis=0)
    np.testing.assert_allclose(proposal.M2, centred.T @ centred, atol=1e-10)
    np.testing.assert_allclose(proposal.sd_or_cov, np.exp(2 * log_sigma) * proposal.base_cov, rtol=1e-12)
    # adaptation_stats: one row per `period` OUTER calls, n = outer calls, mean = mean of the period's sub-chain alphas
    rows = proposal.adaptation_stats_rows
    assert [row[0] for row in rows] == [5, 10, 15, 20]
    for row in rows:
        alphas = [acceptance_probability(la) for _, _, _, sub in sequence[row[0] - 5:row[0]] for la in sub]
        assert row[1] == pytest.approx(np.mean(alphas), abs=1e-12)


def test_random_walk_covariance_install_keyed_on_outer_calls():
    proposal = _random_walk(period=5, warmup=10)
    for call, (proposed, outer, current, subchain) in enumerate(_subchain_sequence(14, 6), start=1):
        proposal.adapt(proposed, outer, current, subchain_log_acceptance_probabilities=subchain)
        # first install at outer call 10 (n_scale is already 60 at that point, 6x past the warm-up)
        assert np.isnan(proposal._trace_over_d) == (call < 10), call


def test_outer_acceptance_does_not_drive_the_scale_in_a_damh_stage():
    a, b = _random_walk(), _random_walk()
    p, c, sub = np.zeros(2), np.ones(2), [np.log(0.5), -np.inf, 0.0]
    a.adapt(p, 0.0, c, subchain_log_acceptance_probabilities=sub)
    b.adapt(p, -np.inf, c, subchain_log_acceptance_probabilities=sub)
    assert a.log_sigma == b.log_sigma and a.n_scale == b.n_scale == 3


@pytest.mark.parametrize("K", [1, 3])
def test_pcn_logit_runs_one_step_per_subchain_step(K):
    proposal = _pcn(period=4)
    sequence = _subchain_sequence(17, K)
    logit, n_scale = float(np.log(0.3 / 0.7)), 0
    for call, (proposed, outer, current, subchain) in enumerate(sequence, start=1):
        proposal.adapt(proposed, outer, current, subchain_log_acceptance_probabilities=subchain)
        for log_alpha in subchain:
            n_scale += 1
            logit += n_scale ** (-0.7) * (acceptance_probability(log_alpha) - 0.234)
        assert proposal.n_scale == K * call and proposal.n == call
        assert proposal.logit_beta == pytest.approx(logit, rel=0, abs=1e-12)
        assert proposal.beta == pytest.approx(1.0 / (1.0 + np.exp(-logit)), abs=1e-12)
    assert [row[0] for row in proposal.adaptation_stats_rows] == [4, 8, 12, 16]
    assert proposal.adaptation_stats_rows[0][1] == pytest.approx(
        np.mean([acceptance_probability(la) for _, _, _, sub in sequence[:4] for la in sub]), abs=1e-12)


def test_pooled_state_layout():
    """``n_scale`` is not pooled. Since C1 (2026-10-11) the random walk's ``adapted_state`` is only
    the effective log-scale (its Welford statistics are pooled by the runner); pCN's is unchanged."""
    rw, pcn = _random_walk(), _pcn()
    for proposed, outer, current, subchain in _subchain_sequence(12, 5):
        rw.adapt(proposed, outer, current, subchain_log_acceptance_probabilities=subchain)
        pcn.adapt(proposed, outer, current, subchain_log_acceptance_probabilities=subchain)
    assert rw.adapted_state().shape == (1,) and rw.statistics.n == 12
    assert rw.adapted_state()[0] == pytest.approx(rw.log_sigma + 0.5 * np.log(np.trace(rw.base_cov)), abs=1e-14)
    assert pcn.adapted_state().tolist() == [12.0, pcn.logit_beta]


# ---------------------------------------------------------------------------------------------
# Integration: 2-D linear-Gaussian problem, exact surrogate, run_sampling_local
# ---------------------------------------------------------------------------------------------

class _LinearSolver(Solver):
    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return A @ self.parameters


class _ExactEvaluator(Evaluator):
    def __init__(self):
        super().__init__(no_parameters=2, no_observations=2)

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ A.T


def _problem():
    prior = surrDAMH.distributions.PriorIndependentComponents([surrDAMH.distributions.NormalComponent(0.0, 1.0)] * 2)
    return surrDAMH.Problem(prior, surrDAMH.distributions.Normal(mean=OBSERVED, sd=OBS_SD), solver=_LinearSolver())


def _conf(path, **kwargs):
    return surrDAMH.Configuration(output_dir=str(path), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedInitialSample([0.3, -0.2]), **kwargs)


def _sd(scale):
    return float(np.sqrt(np.trace(np.atleast_2d(scale)) / 2))


#: S6 evidence study (scratchpad s6/, 20 000 exact evaluations, 2 seeds x 10-point grid, ESS per sub-chain step):
#: the best fixed random-walk sd-equivalent step at K = 20 on this problem, and the pre-S6 adapted value (3 seeds)
FIXED_STEP_OPTIMUM_SD_K20 = 0.45
PRE_S6_ADAPTED_SD_K20 = 3.03


def test_k20_adaptive_random_walk_adapts_like_an_mh_chain(tmp_path):
    """With an exact surrogate the sub-chain IS an MH chain on the posterior, so its own-acceptance
    adaptation must land where the MH warm-up's does, and near the fixed-step optimum."""
    warm = Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=1000, burn_in=True)
    damh = Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=2000, subchain_length=20)
    run = _problem().run_sampling_local(_conf(tmp_path), [warm, damh], surrogate_evaluator=_ExactEvaluator())
    sd_mh = _sd(load_carry_over(str(tmp_path), run.stage_results[0].name)[0]["scale"])
    sd_k20 = _sd(load_carry_over(str(tmp_path), run.stage_results[1].name)[0]["scale"])
    assert abs(np.log(sd_k20 / sd_mh)) < np.log(1.25), (sd_k20, sd_mh)
    assert sd_k20 / FIXED_STEP_OPTIMUM_SD_K20 < 2.0, sd_k20  # measured 1.59x (20 000 evaluations, 3 seeds)
    assert PRE_S6_ADAPTED_SD_K20 / FIXED_STEP_OPTIMUM_SD_K20 > 2.0  # the defect S6 removes (6.7x)
    assert PRE_S6_ADAPTED_SD_K20 / sd_mh > 2.0


def test_k20_adaptive_pcn_adapts_like_an_mh_chain(tmp_path):
    warm = Stage(algorithm="MH", proposal=PCN(), max_evaluations=1000, burn_in=True)
    damh = Stage(algorithm="DAMH", proposal=PCN(), max_evaluations=2000, subchain_length=20)
    run = _problem().run_sampling_local(_conf(tmp_path), [warm, damh], surrogate_evaluator=_ExactEvaluator())
    beta_mh = float(load_carry_over(str(tmp_path), run.stage_results[0].name)[0]["beta"])
    beta_k20 = float(load_carry_over(str(tmp_path), run.stage_results[1].name)[0]["beta"])
    assert abs(np.log(beta_k20 / beta_mh)) < np.log(1.25), (beta_k20, beta_mh)
    assert beta_k20 < 0.95  # pre-S6 the outer-acceptance feedback drove beta to 1.0 (independence sampler)


def test_k1_adaptive_damh_samples_the_closed_form_posterior(tmp_path):
    warm = Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=500, burn_in=True)
    damh = Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=6000, subchain_length=1)
    run = _problem().run_sampling_local(_conf(tmp_path), [warm, damh], surrogate_evaluator=_ExactEvaluator())
    result = run.stage_results[1]
    statistics = chain_statistics(read_weighted_samples(tmp_path, result.name, 2))
    assert_matches_closed_form(statistics, POSTERIOR_MEAN, POSTERIOR_COV, "S6 adaptive DAMH K=1")


#: md5 of the MH-stage outputs made with the code BEFORE S6 (scratchpad s6/s6_ref_run.py, cases
#: mh_rw_adaptive / mh_pcn_adaptive): csv files hashed raw, npz files as md5(key + bytes, sorted keys)
PRE_S6_MH_MD5 = {
    "rw": {
        "adaptive_stats/alg0000_MH-adaptive/rank0000.csv": "dfb0ee443bded44327b786976921184b",
        "carry_over/alg0000_MH-adaptive.npz": "72274e4f4474996f9c32d4bceafd6e93",
        "last_sample/alg0000_MH-adaptive/rank0000.npz": "560c354950c612c3b5123ba53c2a10a0",
        "notes/alg0000_MH-adaptive/rank0000.csv": "95888ab8b016ae398be1daff0be08e0d",
        "raw_data/alg0000_MH-adaptive/rank0000.csv": "dc88c202efd109d53290e2ee420e34e4",
        "samples/alg0000_MH-adaptive/rank0000.csv": "feb02c0034c715e3243712339333e1f0",
    },
    "pcn": {
        "adaptive_stats/alg0000_MH-adaptive/rank0000.csv": "239ac99ed7793a5c871076b64e6a9284",
        "carry_over/alg0000_MH-adaptive.npz": "be85dedd4bdf504f7969ec2341e35932",
        "last_sample/alg0000_MH-adaptive/rank0000.npz": "d00d1cd023800052ada1a8b6fd42f828",
        "notes/alg0000_MH-adaptive/rank0000.csv": "9976d1e9791c24809d9d673224e2a1c6",
        "raw_data/alg0000_MH-adaptive/rank0000.csv": "512bc50910a7d77d6c7909118cb9324b",
        "samples/alg0000_MH-adaptive/rank0000.csv": "52362014a617d336d8ddc28538c7876e",
    },
}


#: carry_over keys added by C1 (2026-10-11) and the n_seen fix (2026-10-11), not part of the pre-S6
#: reference: stripped before hashing
C1_CARRY_OVER_KEYS = {"carry_over__posterior_statistics", "summary__posterior_statistics_n",
                      "carry_over__posterior_statistics_n_seen"}


def _md5(path):
    if path.endswith(".npz"):
        with np.load(path) as z:
            return hashlib.md5(b"".join(key.encode() + np.ascontiguousarray(z[key]).tobytes()
                                        for key in sorted(z.files) if key not in C1_CARRY_OVER_KEYS)).hexdigest()
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


@pytest.mark.parametrize("kind", ["rw", "pcn"])
def test_mh_stage_outputs_are_byte_identical_to_the_code_before_s6(tmp_path, kind):
    proposal = RandomWalk() if kind == "rw" else PCN()
    _problem().run_sampling_local(
        _conf(tmp_path, save_snapshots_to_file=True, min_snapshots_initial=50, min_snapshots_to_update=40),
        [Stage(algorithm="MH", proposal=proposal, max_evaluations=1500)])
    root = os.path.join(str(tmp_path), "sampling_output")
    produced = {}
    for pattern in ("*/*/rank*.csv", "last_sample/*/rank*.npz", "carry_over/*.npz"):
        for path in glob.glob(os.path.join(root, pattern)):
            key = os.path.relpath(path, root)
            if key.split("/")[0] in ("samples", "raw_data", "notes", "adaptive_stats", "last_sample", "carry_over"):
                produced[key] = _md5(path)
    assert produced == PRE_S6_MH_MD5[kind]
