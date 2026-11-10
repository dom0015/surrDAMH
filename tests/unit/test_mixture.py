#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 (2026-10-12, note 25 decision D8, note 23 "V1"): the combined proposal.

* ``surrDAMH.proposals.Mixture`` -- full-dimension components drawn with fixed probabilities --
  and ``Block`` with ``probabilities`` and adaptive sub-proposals;
* both run as ``modules.proposals.BlockProposal`` (a mixture = every group is all parameters):
  ``adapt`` routed to the component that proposed, ``adapted_state``/``set_pooled_state``
  concatenated/split per component, ``carry_over`` merged for a mixture and ``{}`` for a block,
  ``adapted_summary`` prefixed ``component<k>.``, the adaptation trace of the first adaptive
  component;
* the builder's V1 rule (two adaptive components of one type refused) and per-component guards;
* byte identity of a run with fixed ``Block`` proposals against fingerprints recorded on the
  code BEFORE C2.
"""

from __future__ import annotations

import contextlib
import copy
import glob
import hashlib
import io
import json
import os
import warnings

import numpy as np
import pytest

import surrDAMH
from surrDAMH.configuration import Configuration
from surrDAMH.distributions.gaussian_mixture import GaussianMixture
from surrDAMH.distributions.independent_components import NormalComponent, PriorIndependentComponents
from surrDAMH.modules.continuation import load_carry_over, save_carry_over
from surrDAMH.modules.manifest import _json_safe
from surrDAMH.modules.posterior_statistics import PosteriorStatistics
from surrDAMH.modules.proposal_builder import build_proposal
from surrDAMH.modules.proposals import (BlockProposal, GaussRandomWalk, GaussRandomWalk_adaptive,
                                        Hamiltonian_adaptive, PCN_adaptive, Proposal, subproposal_seed)
from surrDAMH.proposals import PCN as PCNSpec
from surrDAMH.proposals import Block, Hamiltonian, Mixture, RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage, check_stage_list
from surrDAMH.surrogates.parent import Evaluator

pytestmark = [pytest.mark.unit,
              # the "only component k is traced" warning is asserted explicitly where it matters
              pytest.mark.filterwarnings("ignore:.*is written to adaptive_stats:RuntimeWarning")]


def _conf(tmp_path, d=2, use_surrogate_gradients=True):
    return Configuration(no_parameters=d, no_observations=1, output_dir=str(tmp_path), use_collector=False,
                         use_solvers_pool=False, use_surrogate_gradients=use_surrogate_gradients)


def _prior(d=2):
    return PriorIndependentComponents([NormalComponent(mu=0.0, sigma=1.0) for _ in range(d)])


def _build(tmp_path, spec, d=2, carried=None, statistics=None, seed=1, **conf_kwargs):
    stage = Stage(algorithm="MH", proposal=spec, max_evaluations=10)
    return build_proposal(stage=stage, conf=_conf(tmp_path, d=d, **conf_kwargs), prior=_prior(d), seed=seed,
                          carried=carried, stage_index=0, statistics=statistics)


class Spy(Proposal):
    """A component recording its calls; proposes ``x + tag`` and returns ``(tag, 0)``."""

    def __init__(self, tag: float):
        super().__init__()
        self.tag = float(tag)
        self.calls: list[dict] = []
        self.seed = None

    def reseed(self, seed):
        self.seed = seed

    def propose_sample(self, current_sample):
        return np.asarray(current_sample, dtype=float) + self.tag

    def get_log_acceptance_probability(self, *args):
        return self.tag, 0.0

    def adapt(self, proposed_sample, log_acceptance_probability, current_sample,
              subchain_log_acceptance_probabilities=None):
        self.calls.append(dict(proposed=np.array(proposed_sample), la=log_acceptance_probability,
                               current=np.array(current_sample), sub=subchain_log_acceptance_probabilities))


# ------------------------------------------------------------------------------------- specs
def test_mixture_spec_validation():
    with pytest.raises(ValueError, match="at least two"):
        Mixture([RandomWalk()])
    with pytest.raises(ValueError, match="2 probabilities for 3"):
        Mixture([RandomWalk(), PCNSpec(), RandomWalk(scale=1.0)], probabilities=[0.5, 0.5])
    for bad in ([0.5, 0.0], [1.0, -1.0], [1.0, np.nan], [True, 1.0], ["a", 1.0]):
        with pytest.raises(ValueError, match="positive finite"):
            Mixture([RandomWalk(), PCNSpec()], probabilities=bad)
    with pytest.raises(ValueError, match="no nesting"):
        Mixture([RandomWalk(), Block(groups=[[0, 1]], proposals=[RandomWalk(scale=1.0)])])
    with pytest.raises(ValueError, match="no nesting"):
        Mixture([RandomWalk(), Mixture([RandomWalk(), PCNSpec()])])
    with pytest.raises(ValueError, match="cannot be nested"):
        Block(groups=[[0], [1]], proposals=[RandomWalk(), Mixture([RandomWalk(), PCNSpec()])])
    with pytest.raises(ValueError, match="1 probabilities for 2"):
        Block(groups=[[0], [1]], proposals=[RandomWalk(), PCNSpec()], probabilities=[1.0])
    spec = Mixture([RandomWalk(scale=0.1), RandomWalk()], probabilities=[1, 9])
    assert spec.probabilities == [1, 9]  # stored as given, normalised at build time


def test_adaptive_and_needs_gradients_properties():
    assert Mixture([RandomWalk(scale=0.1), PCNSpec(beta=0.3)]).adaptive is False
    assert Mixture([RandomWalk(scale=0.1), PCNSpec()]).adaptive is True
    assert Mixture([RandomWalk(scale=0.1), PCNSpec()]).needs_gradients is False
    assert Mixture([RandomWalk(scale=0.1), Hamiltonian(step_size=0.1)]).needs_gradients is True
    fixed_block = Block(groups=[[0], [1]], proposals=[RandomWalk(scale=0.1), PCNSpec(beta=0.2)])
    assert fixed_block.adaptive is False and Stage(proposal=fixed_block).adaptive is False
    adaptive_block = Block(groups=[[0], [1]], proposals=[RandomWalk(scale=0.1), PCNSpec()])
    assert adaptive_block.adaptive is True and Stage(proposal=adaptive_block).adaptive is True


def test_stage_accepts_mixture_and_describes_it():
    spec = Mixture([RandomWalk(scale=0.1), Hamiltonian(step_size=0.2)], probabilities=[0.3, 0.7])
    stage = Stage(algorithm="MH", proposal=spec, max_evaluations=10, surrogate_model_updates=True)
    # like a bare Hamiltonian: an MH stage whose proposal needs gradients may refresh the surrogate
    assert stage.proposal_needs_gradients() and stage.surrogate_model_updates is True
    text = stage.describe(0)
    assert "Mixture(2 components, probabilities=[0.3, 0.7])" in text
    assert "component 1: Hamiltonian(step_size=0.2" in text
    with pytest.raises(TypeError, match="wrap it"):
        check_stage_list([spec])
    stage = Stage(algorithm="MH", proposal=Mixture([RandomWalk(scale=0.1), RandomWalk(scale=1.0)]),
                  max_evaluations=10, surrogate_model_updates=True)
    assert stage.surrogate_model_updates is False  # no gradients -> refused, as for a bare RandomWalk
    assert "_MH-adaptive" not in surrDAMH.stages.stage_name(stage, 0)


# ---------------------------------------------------------------------------------- dispatch
def test_mixture_draws_components_with_the_given_probabilities_and_uses_the_chosen_one():
    spies = [Spy(1.0), Spy(-2.0)]
    mixture = BlockProposal(no_parameters=3, list_of_proposals=spies, list_of_groups=[[0, 1, 2]] * 2, seed=5,
                            group_probabilities=[1, 4], is_mixture=True)
    np.testing.assert_allclose(mixture.group_probabilities, [0.2, 0.8])
    draws = 20_000
    chosen = []
    x = np.array([0.5, 0.0, -1.0])
    for _ in range(draws):
        mixture.choose_group()
        chosen.append(mixture.group_index)
        tag = spies[mixture.group_index].tag
        np.testing.assert_array_equal(mixture.propose_sample(x), x + tag)
        assert mixture.get_log_acceptance_probability(0.0, 0.0, 0.0, 0.0) == (tag, 0.0)
    fraction = np.mean(np.asarray(chosen) == 0)
    assert abs(fraction - 0.2) < 4 * np.sqrt(0.2 * 0.8 / draws), fraction


def test_adapt_reaches_only_the_component_that_proposed():
    spies = [Spy(1.0), Spy(2.0)]
    mixture = BlockProposal(no_parameters=2, list_of_proposals=spies, list_of_groups=[[0, 1]] * 2, seed=0,
                            is_mixture=True)
    mixture.group_index = 1
    mixture.adapt(proposed_sample=np.array([1.0, 2.0]), log_acceptance_probability=-np.inf,
                  current_sample=np.array([0.0, 0.5]), subchain_log_acceptance_probabilities=[-0.1, -np.inf])
    assert spies[0].calls == [] and len(spies[1].calls) == 1
    call = spies[1].calls[0]
    assert call["la"] == -np.inf and call["sub"] == [-0.1, -np.inf]
    np.testing.assert_array_equal(call["proposed"], [1.0, 2.0])
    np.testing.assert_array_equal(call["current"], [0.0, 0.5])

    # a block slices the samples to the proposing sub-proposal's group
    spies = [Spy(1.0), Spy(2.0)]
    block = BlockProposal(no_parameters=3, list_of_proposals=spies, list_of_groups=[[0, 2], [1]], seed=0)
    block.group_index = 0
    block.adapt(np.array([1.0, 2.0, 3.0]), -0.3, np.array([4.0, 5.0, 6.0]))
    assert spies[1].calls == []
    np.testing.assert_array_equal(spies[0].calls[0]["proposed"], [1.0, 3.0])
    np.testing.assert_array_equal(spies[0].calls[0]["current"], [4.0, 6.0])
    assert spies[0].calls[0]["sub"] is None


def test_adaptive_block_sub_proposal_adapts_in_group_dimension(tmp_path):
    proposal = _build(tmp_path, Block(groups=[[0, 2], [1]], proposals=[RandomWalk(), RandomWalk(scale=0.3)]), d=3)
    adaptive = proposal.list_of_proposals[0]
    assert isinstance(adaptive, GaussRandomWalk_adaptive) and adaptive.no_parameters == 2
    rng = np.random.default_rng(0)
    for _ in range(200):
        proposal.choose_group()
        x = rng.standard_normal(3)
        proposal.adapt(x, 0.0, x)  # always accepted: the scale grows
    assert adaptive.n > 0 and adaptive.statistics.n == adaptive.n and adaptive.log_sigma > 0
    assert proposal.adapted_state().shape == (1,)
    assert proposal.carry_over() == {}
    assert proposal.resolution == {"adaptive_stats_component": 0}


# ----------------------------------------------------------------------- pooling round trip
def _mixed_components(seed):
    return [GaussRandomWalk_adaptive(no_parameters=2, sd_or_cov=0.5, seed=seed),
            GaussRandomWalk(no_parameters=2, sd_or_cov=0.1, seed=seed),
            PCN_adaptive(no_parameters=2, beta=0.4, prior_mean=np.zeros(2), prior_sd_or_cov=1.0, seed=seed)]


def test_adapted_state_and_set_pooled_state_round_trip():
    rng = np.random.default_rng(3)
    mixtures = []
    for rank in range(3):
        mixture = BlockProposal(no_parameters=2, list_of_proposals=_mixed_components(rank),
                                list_of_groups=[[0, 1]] * 3, seed=10 * rank + 1, is_mixture=True)
        for _ in range(150):
            mixture.choose_group()
            x = rng.standard_normal(2)
            mixture.adapt(x, float(np.log(rng.uniform())), x)
        mixtures.append(mixture)
    assert mixtures[0]._state_lengths == [1, 0, 2]
    states = np.vstack([m.adapted_state() for m in mixtures])
    assert states.shape == (3, 3)
    # reference: the components pooled one by one with their own column slices
    reference = copy.deepcopy(mixtures[0].list_of_proposals)
    reference[0].set_pooled_state(states[:, 0:1])
    reference[2].set_pooled_state(states[:, 1:3])
    mixtures[0].set_pooled_state(states)
    np.testing.assert_array_equal(mixtures[0].list_of_proposals[0].sd_or_cov, reference[0].sd_or_cov)
    assert mixtures[0].list_of_proposals[2].beta == reference[2].beta
    assert mixtures[0].list_of_proposals[2].logit_beta == pytest.approx(states[:, 2].mean())
    with pytest.raises(ValueError, match="length 3"):
        mixtures[0].set_pooled_state(states[:, :2])
    fixed = BlockProposal(no_parameters=2, list_of_proposals=[GaussRandomWalk(2, 0.1), GaussRandomWalk(2, 1.0)],
                          list_of_groups=[[0, 1]] * 2, is_mixture=True)
    assert fixed.adapted_state().shape == (0,)


# ------------------------------------------------------------------------------- carry-over
def test_mixture_carry_over_merges_and_feeds_a_following_random_walk(tmp_path):
    spec = Mixture([RandomWalk(scale=0.05), RandomWalk(), PCNSpec()], probabilities=[0.1, 0.6, 0.3])
    proposal = _build(tmp_path, spec)
    assert [type(c) for c in proposal.list_of_proposals] == [GaussRandomWalk, GaussRandomWalk_adaptive,
                                                               PCN_adaptive]
    rng = np.random.default_rng(1)
    for _ in range(300):
        proposal.choose_group()
        x = rng.standard_normal(2)
        proposal.adapt(x, float(np.log(rng.uniform())), x)
    proposal.set_pooled_state(proposal.adapted_state()[None, :])
    carried = proposal.carry_over()
    assert set(carried) == {"scale", "beta"}
    np.testing.assert_array_equal(carried["scale"], proposal.list_of_proposals[1].sd_or_cov)
    assert carried["beta"] == proposal.list_of_proposals[2].beta
    # the next stage's bare random walk starts from the mixture's adaptive random-walk scale
    following = _build(tmp_path, RandomWalk(), carried=carried)
    np.testing.assert_allclose(following.sd_or_cov, carried["scale"])
    # and a mixture consumes carry-over like bare proposals do (the pinned component keeps its step)
    again = _build(tmp_path, Mixture([RandomWalk(scale=0.05), RandomWalk()]), carried=carried)
    np.testing.assert_allclose(again.list_of_proposals[1].sd_or_cov, carried["scale"])
    np.testing.assert_allclose(again.list_of_proposals[0].sd_or_cov, 0.05)


def test_block_hands_nothing_over_even_when_adaptive(tmp_path):
    proposal = _build(tmp_path, Block(groups=[[0], [1]], proposals=[RandomWalk(), PCNSpec()]))
    proposal.set_pooled_state(np.vstack([proposal.adapted_state()] * 2))
    assert proposal.carry_over() == {}
    assert set(proposal.adapted_summary()) >= {"component0.base_cov", "component0.log_sigma", "component1.beta"}


def test_builder_refuses_two_adaptive_components_of_one_type(tmp_path):
    with pytest.raises(ValueError, match=r"components \[0, 2\] are both adaptive RandomWalk.*scale="):
        _build(tmp_path, Mixture([RandomWalk(), PCNSpec(beta=0.2), RandomWalk(target_rate=0.4)]))
    with pytest.raises(ValueError, match="both adaptive Hamiltonian"):
        _build(tmp_path, Mixture([Hamiltonian(), Hamiltonian(integrator="dimension_robust")]))
    # fixed duplicates and an adaptive + pinned pair are fine
    _build(tmp_path, Mixture([RandomWalk(scale=0.1), RandomWalk(scale=1.0)]))
    _build(tmp_path, Mixture([RandomWalk(scale=0.1), RandomWalk()]))


def test_pinned_component_without_step_warns_like_a_bare_proposal(tmp_path, capsys):
    _build(tmp_path, Mixture([RandomWalk(adaptive=False), PCNSpec()]))
    assert "RandomWalk(adaptive=False) without a scale" in capsys.readouterr().out


# ------------------------------------------------------------------------- adaptation trace
def test_trace_forwards_to_the_first_adaptive_component(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        proposal = _build(tmp_path, Mixture([RandomWalk(scale=0.1), RandomWalk()]))
    adaptive = proposal.list_of_proposals[1]
    assert proposal.adaptation_stats_header == GaussRandomWalk_adaptive.adaptation_stats_header
    assert proposal.resolution == {"adaptive_stats_component": 1}
    proposal.group_index = 1
    for _ in range(20):
        proposal.adapt(np.zeros(2), -0.5, np.zeros(2))
    assert len(proposal.adaptation_stats_rows) == 2 and proposal.adaptation_stats_rows is adaptive.adaptation_stats_rows

    with pytest.warns(RuntimeWarning, match=r"components \[0, 1\] adapt, but only component 0 \(PCN_adaptive\)"):
        proposal = _build(tmp_path, Mixture([PCNSpec(), RandomWalk(), RandomWalk(scale=1.0)]))
    assert proposal.adaptation_stats_header == PCN_adaptive.adaptation_stats_header
    assert proposal.resolution == {"adaptive_stats_component": 0}

    fixed = _build(tmp_path, Mixture([RandomWalk(scale=0.1), RandomWalk(scale=1.0)]))
    assert not hasattr(fixed, "resolution")
    assert fixed.adaptation_stats_header == [] and fixed.adaptation_stats_rows == []


# --------------------------------------------------------------------------- builder guards
def test_component_guards(tmp_path):
    stage = Stage(algorithm="MH", proposal=Mixture([RandomWalk(), PCNSpec(beta=0.2)]), max_evaluations=10)
    with pytest.raises(ValueError, match="Gaussian"):
        build_proposal(stage=stage, conf=_conf(tmp_path), prior=GaussianMixture(means=[np.zeros(2), np.ones(2)]),
                       seed=0)
    with pytest.raises(AssertionError, match="use_surrogate_gradients=True"):
        _build(tmp_path, Mixture([RandomWalk(), Hamiltonian(step_size=0.1)]), use_surrogate_gradients=False)
    proposal = _build(tmp_path, Mixture([RandomWalk(), Hamiltonian()]))
    assert proposal.needs_gradients and isinstance(proposal.list_of_proposals[1], Hamiltonian_adaptive)


def test_hamiltonian_component_resolves_its_mass_from_the_statistics(tmp_path):
    statistics = PosteriorStatistics(2)
    rng = np.random.default_rng(0)
    for _ in range(200):
        statistics.update(rng.standard_normal(2) * [1.0, 3.0])
    proposal = _build(tmp_path, Mixture([RandomWalk(scale=0.2), Hamiltonian(mass="covariance")]),
                      statistics=statistics)
    np.testing.assert_allclose(proposal.list_of_proposals[1].mass_matrix(), statistics.precision())
    assert proposal.resolution["adaptive_stats_component"] == 1
    assert proposal.resolution["component1.mass_source"] == "covariance (n=200)"
    # an adaptive random-walk component reads the chain's shared statistics (like a bare one)
    proposal = _build(tmp_path, Mixture([RandomWalk(), PCNSpec(beta=0.3)]), statistics=statistics)
    assert proposal.list_of_proposals[0].statistics is statistics


# ------------------------------------------------------------------------------------ seeds
def test_component_streams_are_distinct_and_reseeded(tmp_path):
    proposal = _build(tmp_path, Mixture([RandomWalk(scale=1.0), RandomWalk(scale=1.0)]), seed=7)
    first = [c._generator.standard_normal() for c in proposal.list_of_proposals]
    expected = [np.random.RandomState(subproposal_seed(7, k)).standard_normal() for k in range(2)]
    assert first == expected
    own = np.random.RandomState(7).standard_normal()
    assert len({own, *first}) == 3
    proposal.reseed(17)
    again = [c._generator.standard_normal() for c in proposal.list_of_proposals]
    assert again == [np.random.RandomState(subproposal_seed(17, k)).standard_normal() for k in range(2)]
    assert proposal._generator.standard_normal() == np.random.RandomState(17).standard_normal()


# ---------------------------------------------------------------- summary / continuation file
def test_adapted_summary_is_json_safe_and_round_trips_through_the_carry_over_file(tmp_path):
    proposal = _build(tmp_path, Mixture([RandomWalk(), PCNSpec(), Hamiltonian(step_size=0.1)]))
    proposal.set_pooled_state(np.vstack([proposal.adapted_state()] * 2))
    summary = proposal.adapted_summary()
    assert {"component0.base_cov", "component0.log_sigma", "component0.n_pooled", "component0.mean",
            "component1.beta"} <= set(summary)
    json.dumps(_json_safe(summary))
    json.dumps(_json_safe(proposal.resolution))

    class C:
        output_dir = str(tmp_path)
    save_carry_over(C(), "alg0000_MH-adaptive", proposal.carry_over(), summary)
    carried, loaded = load_carry_over(str(tmp_path), "alg0000_MH-adaptive")
    assert set(carried) == {"scale", "beta"} and set(loaded) == set(summary)
    np.testing.assert_array_equal(loaded["component0.base_cov"], summary["component0.base_cov"])
    assert loaded["component1.beta"] == summary["component1.beta"]


# --------------------------------------------------------------- run-level: local runner
_A3 = np.array([[1.0, 0.5, 0.2], [0.0, 1.0, -0.3]])


class _LinearSolver3(Solver):
    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return _A3 @ self.parameters


class _LinearGradientEvaluator3(Evaluator):
    def __init__(self):
        super().__init__(no_parameters=3, no_observations=2)
        self.matrix = _A3 + 0.05

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.matrix.T

    def jacobian(self, datapoints):
        x = np.asarray(datapoints, dtype=float).reshape(-1)
        return self.matrix, self.matrix @ x


def _run_local(output_dir, stages):
    prior = surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(3)])
    problem = surrDAMH.Problem(prior, surrDAMH.distributions.Normal(mean=_A3 @ np.array([0.7, -0.4, 0.2]),
                                                                    sd=np.array([0.3, 0.3])),
                               solver=_LinearSolver3())
    conf = surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="prior", save_snapshots_to_file=True,
                                  use_surrogate_gradients=True)
    with contextlib.redirect_stdout(io.StringIO()):
        problem.run_sampling_local(conf, stages, surrogate_evaluator=_LinearGradientEvaluator3())
    return os.path.join(str(output_dir), "sampling_output")


def _fingerprint(sampling_output):
    digest = {}
    for kind in ("samples", "raw_data", "subchain_stats", "adaptive_stats"):
        for path in sorted(glob.glob(os.path.join(sampling_output, kind, "*", "rank*.csv"))):
            digest[os.path.relpath(path, sampling_output)] = hashlib.md5(open(path, "rb").read()).hexdigest()
    for path in sorted(glob.glob(os.path.join(sampling_output, "carry_over", "*.npz"))):
        with np.load(path) as z:
            digest[os.path.relpath(path, sampling_output)] = sorted(z.files) + [hashlib.md5(
                b"".join(np.ascontiguousarray(z[k]).tobytes() for k in sorted(z.files))).hexdigest()]
    with open(os.path.join(sampling_output, "run_manifest.json")) as f:
        digest["proposal_resolution"] = json.load(f)["proposal_resolution"]
    return digest


#: recorded 2026-10-12 with the code BEFORE C2 (dev container, /dolfinx-env python), by the same
#: stages and fingerprint as below: fixed Block proposals in an MH stage (RandomWalk + PCN), a
#: DAMH stage (two random walks) and a DAMH stage with a Hamiltonian sub-proposal
_FIXED_BLOCK_REFERENCE = {
    "carry_over/alg0000_MH.npz": ["carry_over__posterior_statistics", "carry_over__posterior_statistics_n_seen",
                                  "summary__posterior_statistics_n", "51f9e90ad8cf7ad0b8f0dfa95f98709e"],
    "carry_over/alg0001_DAMH-SMU.npz": ["carry_over__posterior_statistics", "carry_over__posterior_statistics_n_seen",
                                        "summary__posterior_statistics_n", "2f6224db5e099ecdc951bd63470b5c0c"],
    "carry_over/alg0002_DAMH-SMU.npz": ["carry_over__posterior_statistics", "carry_over__posterior_statistics_n_seen",
                                        "summary__posterior_statistics_n", "1e468b72e18a3779b77995c1bceefb11"],
    "proposal_resolution": [
        {"name": "alg0000_MH", "posterior_statistics_discounted_from_n": None,
         "posterior_statistics_n_end": 150, "posterior_statistics_n_start": 0},
        {"name": "alg0001_DAMH-SMU", "posterior_statistics_discounted_from_n": None,
         "posterior_statistics_n_end": 276, "posterior_statistics_n_start": 150},
        {"name": "alg0002_DAMH-SMU", "posterior_statistics_discounted_from_n": None,
         "posterior_statistics_n_end": 359, "posterior_statistics_n_start": 276}],
    "raw_data/alg0000_MH/rank0000.csv": "2bcb44cc3a915012c50d7f039e2b56a4",
    "raw_data/alg0001_DAMH-SMU/rank0000.csv": "e5840d88111791cdfa148ac78e4c5f7e",
    "raw_data/alg0002_DAMH-SMU/rank0000.csv": "919106005bb8b055f6e162297db3327a",
    "samples/alg0000_MH/rank0000.csv": "b3cc3472b3f7b22dfe74fd6750c701f9",
    "samples/alg0001_DAMH-SMU/rank0000.csv": "b4a2f0f7a4f688453c7a57563b9f8366",
    "samples/alg0002_DAMH-SMU/rank0000.csv": "dde749ecf5c50559c2f7b9465bc0a0ba",
    "subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "8dc84ec32a910505b224b7c5263d6095",
    "subchain_stats/alg0002_DAMH-SMU/rank0000.csv": "1bcb1d9b4fdf23b1a18748dcc7ab0f4d",
}


def test_fixed_block_runs_are_byte_identical_to_before_c2(tmp_path):
    """Every output file of a run with fixed Block proposals (MH, DAMH, DAMH + Hamiltonian
    sub-proposal) hashes exactly as before C2; no adaptive_stats and no resolution keys appear.
    The hashes are environment-specific (floating-point libraries); if this fails on another
    machine, re-record them there from the pre-C2 commit before concluding anything."""
    stages = [
        Stage(algorithm="MH", max_evaluations=150,
              proposal=Block(groups=[[0, 2], [1]], proposals=[RandomWalk(scale=0.4), PCNSpec(beta=0.3)])),
        Stage(algorithm="DAMH", subchain_length=3, max_evaluations=120,
              proposal=Block(groups=[[0], [1, 2]], proposals=[RandomWalk(scale=[0.5]), RandomWalk(scale=0.3)])),
        Stage(algorithm="DAMH", subchain_length=2, max_evaluations=80,
              proposal=Block(groups=[[1], [0, 2]], proposals=[Hamiltonian(step_size=0.2, num_steps=3),
                                                              RandomWalk(scale=0.3)])),
    ]
    assert _fingerprint(_run_local(tmp_path, stages)) == _FIXED_BLOCK_REFERENCE


def test_local_run_with_an_adaptive_mixture_hands_its_scale_to_the_next_stage(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stages = [Stage(algorithm="MH", max_evaluations=400,
                        proposal=Mixture([RandomWalk(scale=0.05), RandomWalk()], probabilities=[0.1, 0.9])),
                  Stage(algorithm="DAMH", subchain_length=2, max_evaluations=100,
                        proposal=Mixture([RandomWalk(adaptive=False), Hamiltonian(step_size=0.1)]))]
        sampling_output = _run_local(tmp_path, stages)
    with open(os.path.join(sampling_output, "run_manifest.json")) as f:
        resolution = json.load(f)["proposal_resolution"]
    assert resolution[0]["adaptive_stats_component"] == 1
    assert resolution[1]["component1.mass_source"] == "spec"
    with open(os.path.join(sampling_output, "adaptive_stats", "alg0000_MH-adaptive", "rank0000.csv")) as f:
        assert f.readline().strip().split(",") == GaussRandomWalk_adaptive.adaptation_stats_header + ["rank_world"]
    carried, summary = load_carry_over(os.path.dirname(sampling_output), "alg0000_MH-adaptive")
    assert carried["scale"].shape == (3, 3) and "component1.log_sigma" in summary
    assert not os.path.exists(os.path.join(sampling_output, "adaptive_stats", "alg0001_DAMH-SMU"))
