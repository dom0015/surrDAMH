#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 items 4-10 (library_notes/07_testing_plan.md): ``surrDAMH.modules.proposals``.

Sign convention note (relevant to items 5 and 6): ``Hamiltonian``/``HamiltonianInfinite``
consume ``log_likelihood_gradient`` / ``log_prior_gradient`` callables that are, despite the
name, the gradients of the NEGATIVE log-likelihood / NEGATIVE log-prior (i.e. the two
additive pieces of grad(U(q))) -- see ``AlgorithmBase._compute_surrogate_log_likelihood_gradient``
and ``_compute_log_prior_gradient`` in ``modules/algorithms.py``, which build exactly these
callables as ``-likelihood.grad_logpdf(...)`` / ``-prior.grad_logpdf(...)``. Tests below follow
that convention when injecting gradients directly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.stats

from surrDAMH.configuration import Configuration
from surrDAMH.distributions.gaussian_mixture import GaussianMixture
from surrDAMH.distributions.independent_components import NormalComponent, PriorIndependentComponents
from surrDAMH.distributions.normal import Normal
from surrDAMH.distributions.parent import FromScipy
from surrDAMH.modules.proposal_builder import build_proposal
from surrDAMH.modules.posterior_statistics import PosteriorStatistics
from surrDAMH.modules.proposals import (
    BlockProposal,
    GaussRandomWalk,
    GaussRandomWalk_adaptive,
    Hamiltonian,
    HamiltonianInfinite,
    HamiltonianInfinite_adaptive,
    Hamiltonian_adaptive,
    PCN,
    PCN_adaptive,
    Proposal,
    as_covariance_matrix,
    subproposal_seed,
)
from surrDAMH.proposals import Block, RandomWalk
from surrDAMH.proposals import Hamiltonian as HamiltonianSpec
from surrDAMH.proposals import PCN as PCNSpec
from surrDAMH.stages import Stage


@pytest.mark.unit
def test_pcn_invariance_prior_preserving():
    """Item 4: applying PCN.propose_sample repeatedly (no likelihood/accept step) on its
    own is a Markov kernel invariant w.r.t. the prior N(prior_mean, C0); empirical mean/cov
    of a long chain should converge to prior_mean / C0."""
    no_parameters = 2
    prior_mean = np.zeros(no_parameters)
    pcn = PCN(no_parameters=no_parameters, beta=0.3, prior_mean=prior_mean, prior_sd_or_cov=1.0, seed=7)

    rng = np.random.RandomState(11)
    x = rng.randn(no_parameters)
    n_draws = 20000
    samples = np.empty((n_draws, no_parameters))
    for i in range(n_draws):
        x = pcn.propose_sample(x)
        samples[i] = x

    empirical_mean = samples.mean(axis=0)
    empirical_cov = np.cov(samples, rowvar=False)

    assert np.allclose(empirical_mean, prior_mean, atol=0.15)
    assert np.allclose(np.diag(empirical_cov), np.ones(no_parameters), atol=0.2)
    assert abs(empirical_cov[0, 1]) < 0.15


@pytest.mark.unit
def test_leapfrog_energy_error_scales_as_step_size_squared():
    """Item 5: for a quadratic potential U(q) = 0.5 q^T Q q, halving the leapfrog step size
    (while doubling num_steps to keep the integrated time fixed) should divide |Delta H| by
    approximately 4 (asymptotic O(eps^2) local-then-global error of the leapfrog scheme).
    Loose tolerance since this is a scaling check, not an exact one."""
    Q = np.array([[2.0, 0.3], [0.3, 1.0]])

    def delta_H(step_size: float, num_steps: int) -> float:
        ham = Hamiltonian(no_parameters=2, step_size=step_size, num_steps=num_steps, sd_or_cov=1.0, seed=5)
        ham.log_likelihood_gradient = lambda q: Q @ q  # grad(U) contribution, see module docstring
        ham.log_prior_gradient = lambda q: np.zeros_like(q)
        rng = np.random.RandomState(3)
        q0 = rng.randn(2)
        p0 = rng.randn(2)
        U = lambda q: 0.5 * q @ Q @ q
        H0 = U(q0) + 0.5 * p0 @ ham.M_inv @ p0
        q1, p1 = ham._leapfrog(q0, p0)
        H1 = U(q1) + 0.5 * p1 @ ham.M_inv @ p1
        return abs(H1 - H0)

    dH_coarse = delta_H(step_size=0.1, num_steps=10)
    dH_fine = delta_H(step_size=0.05, num_steps=20)

    assert dH_fine > 0.0  # sanity: not a degenerate zero-error trajectory
    ratio = dH_coarse / dH_fine
    assert 3.0 < ratio < 5.0, f"expected |dH| to shrink ~4x when halving eps, got ratio={ratio}"


@pytest.mark.unit
@pytest.mark.parametrize("proposal_cls", [Hamiltonian, HamiltonianInfinite])
def test_leapfrog_reversibility(proposal_cls):
    """Item 6: leapfrog(leapfrog(q, p)) returns to (q, p) to ~1e-10, for both Hamiltonian and
    HamiltonianInfinite (using the momentum flip already built into ``_leapfrog``'s return)."""
    Q = np.eye(3) * 1.5
    prop = proposal_cls(no_parameters=3, step_size=0.05, num_steps=10, sd_or_cov=1.0, seed=2)
    prop.log_likelihood_gradient = lambda q: Q @ q
    prop.log_prior_gradient = lambda q: np.zeros_like(q)

    rng = np.random.RandomState(1)
    q0 = rng.randn(3)
    p0 = rng.randn(3)

    q1, p1 = prop._leapfrog(q0, p0)
    q2, p2 = prop._leapfrog(q1, p1)

    assert np.allclose(q2, q0, atol=1e-10)
    assert np.allclose(p2, p0, atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize(
    "sd_or_cov",
    [
        1.0,
        np.array([0.3, 2.0, 1.7]),
        np.array([[2.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 1.1]]),  # SPD, not diagonal
    ],
    ids=["scalar_mass", "diagonal_mass", "full_spd_mass"],
)
def test_hamiltonian_infinite_prior_kinetic_flow_preserves_energy_for_any_positive_definite_mass(sd_or_cov):
    """Item 7 (widened 2026-09-18): HamiltonianInfinite._apply_prior_kinetic_flow preserves
    0.5 q.q + 0.5 p^T M^-1 p EXACTLY for any positive-definite mass M = diag(sd_or_cov**2)
    or the full sd_or_cov matrix -- not only for sd_or_cov == 1 as this test previously
    checked. Energy preservation (this test) is a strictly weaker property than
    prior-preservation (only true for internal prior N(0,I), the diagonal/full mass
    cases above still assume that prior and are not separately checked here) -- see the
    class docstring for the distinction."""
    prop = HamiltonianInfinite(no_parameters=3, step_size=0.13, num_steps=5, sd_or_cov=sd_or_cov, seed=1)
    rng = np.random.RandomState(0)
    q = rng.randn(3)
    p = rng.randn(3)

    energy_before = 0.5 * np.dot(q, q) + 0.5 * p @ prop.M_inv @ p
    q_new, p_new = prop._apply_prior_kinetic_flow(q, p)
    energy_after = 0.5 * np.dot(q_new, q_new) + 0.5 * p_new @ prop.M_inv @ p_new

    assert energy_after == pytest.approx(energy_before, abs=1e-10)


@pytest.mark.unit
def test_adaptive_random_walk_reaches_full_rank_covariance():
    """
    Item 8, rewritten for the 2026-09-20 internals: the shrinkage-regularised estimate is
    full rank from the first installed covariance on, even at ``no_parameters=20`` with far
    fewer states than that (``delta = min(1, 2d/n)`` mixes in ``(tr C/d) I``, and the
    relative ``1e-6`` ridge keeps it non-singular). The old implementation needed the
    unbounded sample history to reach full rank at all.
    """
    no_parameters = 20
    proposal = GaussRandomWalk_adaptive(no_parameters=no_parameters, period=10, warmup=20, seed=1)
    rng = np.random.RandomState(0)
    state = np.zeros(no_parameters)
    for _ in range(5 * proposal.period):  # 5 adaptation periods, fewer states than parameters
        proposed = state + rng.randn(no_parameters)
        state = proposed if rng.rand() < 0.3 else state
        proposal.adapt(proposed_sample=proposed, log_acceptance_probability=-abs(rng.randn()),
                       current_sample=state)

    assert proposal.sd_or_cov.ndim == 2
    assert np.linalg.matrix_rank(proposal.sd_or_cov) == no_parameters
    assert np.all(np.isfinite(proposal.sd_or_cov))
    # one stats row per completed period, with the documented column count
    assert len(proposal.adaptation_stats_rows) == 5
    assert all(len(row) == len(proposal.adaptation_stats_header)
               for row in proposal.adaptation_stats_rows)


# ---------------------------------------------------------------------------
# RandomWalk(target_rate=...) is forwarded to the adaptive proposal by build_proposal
# (G1, 2026-09-17; rewired to the new adaptive classes on 2026-09-20 and to the proposal
# spec's own ``target_rate`` field on 2026-09-21).
# ---------------------------------------------------------------------------

def _adaptive_stage(target_rate=None) -> Stage:
    return Stage(algorithm="MH", proposal=RandomWalk(scale=0.5, adaptive=True, target_rate=target_rate),
                 max_evaluations=10)


def _run_adapt_sequence(proposal, no_parameters=3, no_periods=8, seed=0):
    """Feed a fixed pseudo-random adapt() sequence into ``proposal`` and return its covariance."""
    rng = np.random.RandomState(seed)
    state = np.zeros(no_parameters)
    for _ in range(no_periods * proposal.period):
        proposed = state + rng.randn(no_parameters)
        state = proposed if rng.rand() < 0.3 else state
        proposal.adapt(proposed_sample=proposed,
                       log_acceptance_probability=rng.uniform(-3.0, 0.0),
                       current_sample=state)
    return np.array(proposal.sd_or_cov, copy=True)


@pytest.mark.unit
def test_build_proposal_target_rate_none_keeps_the_class_default(tmp_path):
    """``RandomWalk(target_rate=None)`` leaves each adaptive class at its own published default:
    0.234 for the random walk / pCN (Robbins-Monro), 0.8 for the dual-averaged step size."""
    conf = _make_conf(tmp_path)
    prior = Normal(mean=np.zeros(2), sd=1.0)
    proposal = build_proposal(stage=_adaptive_stage(), conf=conf, prior=prior, seed=1, stage_index=0)

    assert isinstance(proposal, GaussRandomWalk_adaptive)
    assert proposal.target_rate == 0.234
    assert proposal.period == 10 and proposal.warmup == 100
    assert not hasattr(proposal, "corr_limit") and not hasattr(proposal, "sample_limit")


@pytest.mark.unit
def test_build_proposal_target_rate_changes_adapted_covariance(tmp_path):
    """``RandomWalk(target_rate=0.1)`` must reach the proposal and change the adapted covariance
    compared to the 0.234 default."""
    conf = Configuration(no_parameters=3, no_observations=1, output_dir=str(tmp_path),
                         use_collector=False, use_solvers_pool=False)
    prior = Normal(mean=np.zeros(3), sd=1.0)

    default_proposal = build_proposal(stage=_adaptive_stage(), conf=conf, prior=prior, seed=1, stage_index=0)
    retargeted_proposal = build_proposal(stage=_adaptive_stage(target_rate=0.1), conf=conf,
                                         prior=prior, seed=1, stage_index=0)
    assert default_proposal.target_rate == 0.234
    assert retargeted_proposal.target_rate == 0.1

    cov_default = _run_adapt_sequence(default_proposal)
    cov_retargeted = _run_adapt_sequence(retargeted_proposal)
    assert not np.allclose(cov_default, cov_retargeted), (
        "target_rate=0.1 produced the same covariance as the 0.234 default"
    )


@pytest.mark.unit
def test_adaptive_random_walk_all_rejected_period_lowers_the_scale_without_warning():
    """
    New-internals replacement of the old degenerate-weights guard test: a whole period of
    ``log_acceptance_probability=-inf`` (acceptance probability 0, what DAMH feeds a
    pre-rejected iteration) must simply drive the Robbins-Monro log-scale DOWN, leave
    ``sd_or_cov`` finite and positive definite, and emit no warning at all -- the old
    weighted-covariance code raised/warned on exactly this input.
    """
    proposal = GaussRandomWalk_adaptive(no_parameters=3, period=5, seed=1)
    log_sigma_before = proposal.log_sigma
    state = np.array([0.3, -0.2, 0.1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(proposal.period):
            proposal.adapt(proposed_sample=state + 1.0, log_acceptance_probability=-np.inf,
                           current_sample=state)
    assert caught == [], [str(w.message) for w in caught]

    assert proposal.log_sigma < log_sigma_before
    assert np.all(np.isfinite(proposal.sd_or_cov))
    assert np.all(np.linalg.eigvalsh(proposal.sd_or_cov) > 0)


@pytest.mark.unit
def test_block_proposal_subproposals_reseeded_per_rank():
    """Item 9 / G5 (fixed 2026-09-17): two BlockProposals built the way two MPI ranks build
    them (same user sub-proposal seed, different per-rank block seed from ``build_proposal``)
    must draw different increments. Before G5 the sub-proposal kept its own seed, so all
    chains proposed identically."""

    def build_block_proposal_as_one_rank_would(seed: int) -> BlockProposal:
        # the user writes one sub-proposal seed into the Stage; only the block seed is per-rank
        sub_proposal = GaussRandomWalk(no_parameters=2, seed=0)
        return BlockProposal(no_parameters=2, list_of_proposals=[sub_proposal], list_of_groups=[[0, 1]], seed=seed)

    rank0_proposal = build_block_proposal_as_one_rank_would(seed=1)    # 10*(no_stages*0+0)+1
    rank1_proposal = build_block_proposal_as_one_rank_would(seed=11)   # 10*(no_stages*1+0)+1
    rank0_proposal.choose_group()
    rank1_proposal.choose_group()

    draw0 = rank0_proposal.propose_sample(np.zeros(2))
    draw1 = rank1_proposal.propose_sample(np.zeros(2))

    assert not np.allclose(draw0, draw1), "expected per-rank reseeding (G5) to make the draws differ"


@pytest.mark.unit
def test_block_proposal_subproposal_seeds_are_reproducible_and_distinct():
    """G5: the sub-proposal seeds are a pure function of (block seed, sub-proposal index),
    injective, and disjoint from the seed0/proposal/algorithm seed family."""
    seeds = {(block_seed, index): subproposal_seed(block_seed, index)
             for block_seed in range(0, 200, 7) for index in range(5)}
    assert len(set(seeds.values())) == len(seeds), "sub-proposal seeds collide"
    assert min(seeds.values()) >= 2**31, "sub-proposal seeds must not overlap the seed0 family"
    assert subproposal_seed(11, 2) == subproposal_seed(11, 2)  # deterministic
    with pytest.raises(ValueError):
        subproposal_seed(1, 1000)


@pytest.mark.unit
def test_block_proposal_reseeds_subproposals_per_stage():
    """G5/A13: the same sub-proposal OBJECT reused by two stages is re-seeded from each
    stage's own block seed, so stage 2 does not continue stage 1's stream and two identical
    stages produce identical increments."""
    shared_sub_proposal = GaussRandomWalk(no_parameters=1, sd_or_cov=0.5, seed=0)

    def draw_with_block_seed(seed: int) -> float:
        block = BlockProposal(no_parameters=1, list_of_proposals=[shared_sub_proposal],
                              list_of_groups=[[0]], seed=seed)
        block.choose_group()
        return float(block.propose_sample(np.zeros(1))[0])

    stage0_draw = draw_with_block_seed(1)
    stage1_draw = draw_with_block_seed(11)
    stage0_draw_again = draw_with_block_seed(1)

    assert stage0_draw != stage1_draw
    assert stage0_draw == stage0_draw_again, "re-seeding must make each stage's stream reproducible"


@pytest.mark.unit
def test_proposal_reseed_replaces_the_generator_of_every_shipped_proposal():
    """G5: ``reseed`` works for every proposal class that owns a generator."""
    instances = [
        GaussRandomWalk(no_parameters=2, sd_or_cov=0.5, seed=0),
        GaussRandomWalk_adaptive(no_parameters=2, sd_or_cov=0.5, seed=0),
        PCN(no_parameters=2, beta=0.3, prior_mean=np.zeros(2), prior_sd_or_cov=1.0, seed=0),
        Hamiltonian(no_parameters=2, sd_or_cov=0.5, seed=0),
        HamiltonianInfinite(no_parameters=2, sd_or_cov=0.5, seed=0),
    ]
    for proposal in instances:
        before = proposal._generator.get_state()[1].copy()
        proposal.reseed(12345)
        after = proposal._generator.get_state()[1]
        assert not np.array_equal(before, after), type(proposal).__name__
        # and the same seed gives the same state again
        other = np.random.RandomState(12345)
        np.testing.assert_array_equal(after, other.get_state()[1])


@pytest.mark.unit
def test_proposal_reseed_warns_for_a_proposal_without_a_generator():
    """G5: a custom Proposal that owns a generator under another name must be told, rather
    than silently sharing a stream between chains."""

    class GeneratorlessProposal(Proposal):
        def propose_sample(self, current_sample):
            return current_sample

    with pytest.warns(RuntimeWarning, match="no '_generator' attribute"):
        GeneratorlessProposal().reseed(7)


# ---------------------------------------------------------------------------
# G2 (decided 2026-09-17): as_covariance_matrix normalises every accepted sd_or_cov
# spelling to a 2-D covariance matrix, so the adaptive Allreduce cannot see two shapes.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_as_covariance_matrix_normalises_scalar_vector_and_matrix():
    np.testing.assert_array_equal(as_covariance_matrix(0.5, 3), np.diag([0.25, 0.25, 0.25]))
    np.testing.assert_array_equal(as_covariance_matrix([0.5, 2.0], 2), np.diag([0.25, 4.0]))

    cov = np.array([[1.0, 0.3], [0.3, 2.0]])
    normalised = as_covariance_matrix(cov, 2)
    np.testing.assert_array_equal(normalised, cov)  # 2-D input passes through unchanged
    assert normalised is not cov  # but is a fresh, contiguous buffer safe to Allreduce from
    assert normalised.flags["C_CONTIGUOUS"]
    assert normalised.dtype == np.float64


@pytest.mark.unit
def test_adaptive_random_walk_state_and_covariance_shapes_never_depend_on_the_adapt_count():
    """
    What finding 2.5 was about, checked on the 2026-09-20 hand-over: a rank that has adapted
    and one that has not must produce the same buffer shapes, so the cross-rank exchange can
    never abort with ``Message truncated``. ``sd_or_cov`` is now 2-D from construction (no
    ``as_covariance_matrix`` normalisation needed in the runners) and ``adapted_state()`` has a
    length fixed by ``no_parameters`` alone.
    """
    no_parameters = 2
    not_adapted = GaussRandomWalk_adaptive(no_parameters=no_parameters, sd_or_cov=0.5, period=5,
                                           warmup=10, seed=1)
    adapted = GaussRandomWalk_adaptive(no_parameters=no_parameters, sd_or_cov=0.5, period=5,
                                       warmup=10, seed=1)
    rng = np.random.RandomState(0)
    state = np.zeros(no_parameters)
    for _ in range(3 * adapted.period):
        proposed = state + rng.randn(no_parameters)
        state = proposed if rng.rand() < 0.3 else state
        adapted.adapt(proposed_sample=proposed, log_acceptance_probability=rng.uniform(-3.0, 0.0),
                      current_sample=state)

    assert np.asarray(not_adapted.sd_or_cov).shape == (no_parameters, no_parameters)
    assert np.asarray(adapted.sd_or_cov).shape == (no_parameters, no_parameters)
    assert not_adapted.adapted_state().shape == adapted.adapted_state().shape
    # C1 (2026-10-11): the effective log-scale only; the statistics travel as
    # PosteriorStatistics.as_vector(), whose length is fixed by no_parameters as well
    assert adapted.adapted_state().size == 1
    assert adapted.statistics.as_vector().size == not_adapted.statistics.as_vector().size \
        == 1 + no_parameters + no_parameters ** 2


@pytest.mark.unit
def test_as_covariance_matrix_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        as_covariance_matrix([1.0, 2.0, 3.0], 2)
    with pytest.raises(ValueError):
        as_covariance_matrix(np.ones((2, 3)), 2)
    with pytest.raises(ValueError):
        as_covariance_matrix(np.ones((2, 2, 2)), 2)


@pytest.mark.unit
def test_build_proposal_accepts_adaptive_block_proposal(tmp_path):
    """
    C2 (2026-10-12) lifts G2: a Block SUB-PROPOSAL left adaptive (its step field is None, e.g.
    ``RandomWalk()``) used to raise because a ``BlockProposal`` never called ``adapt()`` on its
    members; it now forwards ``adapt()`` to the member that proposed (``tests/unit/test_mixture.py``),
    so the block builds, the stage is adaptive, and the block hands nothing over.
    """
    conf = _make_conf(tmp_path)
    prior = Normal(mean=np.zeros(2), sd=1.0)
    spec = Block(groups=[[0], [1]], proposals=[RandomWalk(), RandomWalk(scale=0.5)])
    stage = Stage(algorithm="MH", max_evaluations=5, proposal=spec)
    assert spec.adaptive and stage.adaptive
    proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=0, stage_index=0)
    assert isinstance(proposal, BlockProposal)
    assert isinstance(proposal.list_of_proposals[0], GaussRandomWalk_adaptive)
    assert proposal.carry_over() == {}


@pytest.mark.unit
def test_gauss_random_walk_adaptive_inside_block_proposal_constructs():
    """Item 10: GaussRandomWalk_adaptive inside a BlockProposal constructs without
    AttributeError (fixed by WS0 item C8: needs_gradients/subchain_length class defaults on
    Proposal, so GaussRandomWalk_adaptive -- which bypasses GaussRandomWalk.__init__ -- still
    exposes them)."""
    sub_proposal = GaussRandomWalk_adaptive(no_parameters=2, seed=0)
    block = BlockProposal(no_parameters=2, list_of_proposals=[sub_proposal], list_of_groups=[[0, 1]], seed=0)
    assert block.needs_gradients is False


# ---------------------------------------------------------------------------
# WS7 guards-only slice (library_notes/09_improvement_plan.md): build_proposal guards.
# ---------------------------------------------------------------------------

def _make_conf(tmp_path, use_surrogate_gradients=True):
    return Configuration(
        no_parameters=2,
        no_observations=1,
        output_dir=str(tmp_path),
        use_collector=False,
        use_solvers_pool=False,
        use_surrogate_gradients=use_surrogate_gradients,
    )


@pytest.mark.unit
def test_build_proposal_pcn_accepts_prior_independent_components(tmp_path):
    """Finding 1.10/A19: PriorIndependentComponents' internal prior is standard normal by
    design (pinned by test_distributions.py::TestPriorIndependentComponentsLogpdfDesign),
    so it qualifies as the Gaussian internal prior pCN's acceptance ratio requires."""
    conf = _make_conf(tmp_path)
    prior = PriorIndependentComponents([NormalComponent(mu=0.0, sigma=1.0), NormalComponent(mu=0.0, sigma=1.0)])
    stage = Stage(algorithm="MH", proposal=PCNSpec(beta=0.2), max_evaluations=10)
    proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=0)
    assert isinstance(proposal, PCN)


@pytest.mark.unit
def test_build_proposal_pcn_accepts_normal_prior(tmp_path):
    conf = _make_conf(tmp_path)
    prior = Normal(mean=np.zeros(2), sd=1.0)
    stage = Stage(algorithm="MH", proposal=PCNSpec(beta=0.2), max_evaluations=10)
    proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=0)
    assert isinstance(proposal, PCN)


@pytest.mark.unit
def test_build_proposal_pcn_rejects_gaussian_mixture_prior(tmp_path):
    """GaussianMixture is not the exact Gaussian PCN's dropped prior ratio assumes."""
    conf = _make_conf(tmp_path)
    prior = GaussianMixture(means=[np.zeros(2), np.ones(2)])
    stage = Stage(algorithm="MH", proposal=PCNSpec(beta=0.2), max_evaluations=10)
    with pytest.raises(ValueError, match="Gaussian"):
        build_proposal(stage=stage, conf=conf, prior=prior, seed=0)


@pytest.mark.unit
def test_build_proposal_pcn_rejects_from_scipy_prior_even_when_wrapping_a_normal(tmp_path):
    """FromScipy is rejected unconditionally for pCN (see _prior_is_gaussian docstring):
    a frozen scipy multivariate_normal cannot be reliably identified as Gaussian from the
    outside, and FromScipy does not implement get_covariance() regardless."""
    conf = _make_conf(tmp_path)
    prior = FromScipy(scipy.stats.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2)))
    stage = Stage(algorithm="MH", proposal=PCNSpec(beta=0.2), max_evaluations=10)
    with pytest.raises(ValueError, match="Gaussian"):
        build_proposal(stage=stage, conf=conf, prior=prior, seed=0)


@pytest.mark.unit
def test_build_proposal_hamiltonian_requires_use_surrogate_gradients(tmp_path):
    """Generalisation of the former B3 assert: a bare Hamiltonian proposal still requires
    conf.use_surrogate_gradients=True."""
    conf = _make_conf(tmp_path, use_surrogate_gradients=False)
    prior = PriorIndependentComponents([NormalComponent(), NormalComponent()])
    stage = Stage(algorithm="MH", proposal=HamiltonianSpec(), max_evaluations=10)
    with pytest.raises(AssertionError, match="use_surrogate_gradients=True"):
        build_proposal(stage=stage, conf=conf, prior=prior, seed=0)


@pytest.mark.unit
def test_build_proposal_block_with_hamiltonian_sub_proposal_requires_use_surrogate_gradients(tmp_path):
    """New coverage: a BlockProposal containing a Hamiltonian sub-proposal must trigger the
    same guard as a bare Hamiltonian proposal (BlockProposal.needs_gradients is True iff any
    sub-proposal needs gradients, computed in BlockProposal.__init__)."""
    conf = _make_conf(tmp_path, use_surrogate_gradients=False)
    prior = PriorIndependentComponents([NormalComponent(), NormalComponent()])
    # steps pinned: a Block sub-proposal must not be adaptive (see
    # test_build_proposal_rejects_adaptive_block_proposal above)
    block_list = [HamiltonianSpec(step_size=0.1), RandomWalk(scale=0.5)]
    stage = Stage(
        algorithm="MH",
        proposal=Block(groups=[[0], [1]], proposals=block_list),
        max_evaluations=10,
    )
    with pytest.raises(AssertionError, match="use_surrogate_gradients=True"):
        build_proposal(stage=stage, conf=conf, prior=prior, seed=0)


@pytest.mark.unit
def test_build_proposal_block_with_hamiltonian_sub_proposal_constructs_when_gradients_enabled(tmp_path):
    conf = _make_conf(tmp_path, use_surrogate_gradients=True)
    prior = PriorIndependentComponents([NormalComponent(), NormalComponent()])
    block_list = [HamiltonianSpec(step_size=0.1), RandomWalk(scale=0.5)]
    stage = Stage(
        algorithm="MH",
        proposal=Block(groups=[[0], [1]], proposals=block_list),
        max_evaluations=10,
    )
    proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=0)
    assert isinstance(proposal, BlockProposal)
    assert proposal.needs_gradients is True


@pytest.mark.unit
def test_build_proposal_block_of_gauss_random_walks_constructs_without_gradients(tmp_path):
    """A block of proposals that need no gradients must construct fine even with
    use_surrogate_gradients=False (the guard must not be a blanket ban on block proposals)."""
    conf = _make_conf(tmp_path, use_surrogate_gradients=False)
    prior = PriorIndependentComponents([NormalComponent(), NormalComponent()])
    block_list = [RandomWalk(scale=0.5), RandomWalk(scale=0.5)]
    stage = Stage(
        algorithm="MH",
        proposal=Block(groups=[[0], [1]], proposals=block_list),
        max_evaluations=10,
    )
    proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=0)
    assert isinstance(proposal, BlockProposal)
    assert proposal.needs_gradients is False


# ---------------------------------------------------------------------------
# WS7: BlockProposal evaluates the inactive groups' gradient coordinates at the
# full current sample instead of zero-filling them
# ---------------------------------------------------------------------------

def _quadratic_gradient_recorder(Q):
    """``(grad_U, calls)``: grad of U(q) = 0.5 q^T Q q, recording every argument it is given."""
    calls: list[np.ndarray] = []

    def grad_U(q):
        calls.append(np.array(q, dtype=float).copy())
        return Q @ np.asarray(q, dtype=float)

    return grad_U, calls


def _legacy_zero_filled_argument(self, x, group):
    """The pre-WS7 wrapper: inactive groups zero-filled instead of taken from current_sample."""
    full_argument = np.zeros(self.no_parameters)
    full_argument[group] = x
    return full_argument


def _block_with_hamiltonian_group0(Q, legacy: bool):
    """BlockProposal([Hamiltonian on parameter 0, GaussRandomWalk on parameter 1]) driven by
    grad U(q) = Q q; ``legacy=True`` restores the pre-WS7 zero-filled gradient argument."""
    sub_hamiltonian = Hamiltonian(no_parameters=1, step_size=0.3, num_steps=4, sd_or_cov=1.0, seed=1)
    sub_random_walk = GaussRandomWalk(no_parameters=1, sd_or_cov=0.5, seed=2)
    block = BlockProposal(no_parameters=2, list_of_proposals=[sub_hamiltonian, sub_random_walk],
                          list_of_groups=[[0], [1]], seed=0)
    if legacy:
        block._full_gradient_argument = _legacy_zero_filled_argument.__get__(block, BlockProposal)
    grad_likelihood, calls = _quadratic_gradient_recorder(Q)
    # zero prior gradient: the whole gradient field is the (possibly coupled) likelihood part
    block.set_gradient_functions(grad_likelihood, lambda q: np.zeros(2))
    return block, calls


COUPLED_Q = np.array([[1.0, 0.8], [0.8, 1.5]])  # NOT block diagonal -> the groups are coupled
SEPARABLE_Q = np.array([[1.0, 0.0], [0.0, 1.5]])  # block diagonal -> genuinely separable model
CURRENT_SAMPLE = np.array([2.0, -3.0])


@pytest.mark.unit
def test_block_proposal_gradient_uses_current_value_of_the_inactive_group():
    """
    WS7: while group [0] is being updated, the gradient of the coupled model must be evaluated
    with parameter 1 at its CURRENT value (-3.0), not at 0.
    """
    block, calls = _block_with_hamiltonian_group0(COUPLED_Q, legacy=False)
    block.group_index = 0  # Hamiltonian group is the active one
    block.propose_sample(CURRENT_SAMPLE)

    assert calls, "the Hamiltonian sub-proposal never evaluated the gradient"
    inactive = np.array([call[1] for call in calls])
    assert np.all(inactive == CURRENT_SAMPLE[1]), (
        f"inactive group's gradient coordinate should be {CURRENT_SAMPLE[1]}, got {np.unique(inactive)}")
    active = np.array([call[0] for call in calls])
    assert np.unique(active).size > 1, "the leapfrog should evaluate the gradient along a trajectory"

    legacy_block, legacy_calls = _block_with_hamiltonian_group0(COUPLED_Q, legacy=True)
    legacy_block.group_index = 0
    legacy_block.propose_sample(CURRENT_SAMPLE)
    assert np.all(np.array([call[1] for call in legacy_calls]) == 0.0)  # the behaviour being replaced


@pytest.mark.unit
def test_block_proposal_gradient_change_moves_the_trajectory_for_a_coupled_model():
    """WS7: for a coupled (non-block-diagonal) model the proposed sample really differs from
    what the pre-WS7 zero-filled gradient would have produced (same seeds on both sides)."""
    block, _ = _block_with_hamiltonian_group0(COUPLED_Q, legacy=False)
    legacy_block, _ = _block_with_hamiltonian_group0(COUPLED_Q, legacy=True)
    block.group_index = legacy_block.group_index = 0

    proposed = block.propose_sample(CURRENT_SAMPLE)
    legacy_proposed = legacy_block.propose_sample(CURRENT_SAMPLE)

    assert proposed[1] == legacy_proposed[1] == CURRENT_SAMPLE[1]  # inactive group is not moved
    assert not np.isclose(proposed[0], legacy_proposed[0]), (
        f"the coupled model should give a different trajectory, got {proposed} vs {legacy_proposed}")


@pytest.mark.unit
def test_block_proposal_gradient_change_is_a_no_op_for_a_separable_model():
    """WS7 companion: with a block-diagonal (separable) model the new and the pre-WS7 gradient
    arguments give bit-identical proposals -- the fix only changes what it has to change."""
    for group_index in (0, 1):
        block, _ = _block_with_hamiltonian_group0(SEPARABLE_Q, legacy=False)
        legacy_block, _ = _block_with_hamiltonian_group0(SEPARABLE_Q, legacy=True)
        block.group_index = legacy_block.group_index = group_index

        proposed = block.propose_sample(CURRENT_SAMPLE)
        legacy_proposed = legacy_block.propose_sample(CURRENT_SAMPLE)
        assert np.array_equal(proposed, legacy_proposed), (
            f"group {group_index}: separable model must be unaffected, got {proposed} vs {legacy_proposed}")


@pytest.mark.unit
def test_block_proposal_tracks_the_current_sample_across_calls():
    """``current_sample`` starts at zeros (pre-WS7 behaviour before the chain moves) and follows
    every ``propose_sample`` call, so a later group update sees the updated state."""
    block, _ = _block_with_hamiltonian_group0(COUPLED_Q, legacy=False)
    assert np.array_equal(block.current_sample, np.zeros(2))

    block.group_index = 1  # random-walk group; the tracked state must be updated anyway
    first = block.propose_sample(CURRENT_SAMPLE)
    assert np.array_equal(block.current_sample, CURRENT_SAMPLE)

    block.group_index = 0
    block.propose_sample(first)
    assert np.array_equal(block.current_sample, first)


# ===========================================================================
# 2026-09-20 adaptivity: shrinkage+Robbins-Monro random walk, adaptive pCN,
# dual-averaged Hamiltonian step size (library_notes/16_... §5 items 2, 4, 6)
# ===========================================================================

def _anisotropic_gaussian(no_parameters: int, condition_number: float, seed: int = 20260920):
    """``C = Q diag(lambda) Q^T`` with ``lambda`` log-spaced 0.5 -> 0.5/kappa, ``Q`` random
    orthogonal -- the target family of the prototype study (``RESULTS_PROTO.md`` §0)."""
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(no_parameters, no_parameters))
    eigenvalues = np.logspace(np.log10(0.5), np.log10(0.5 / condition_number), no_parameters)
    C = Q @ np.diag(eigenvalues) @ Q.T
    return 0.5 * (C + C.T)


def _gaussian_split(precision):
    """``(log_likelihood, log_prior)`` of the target ``N(0, C)`` split as prior ``N(0, I)``
    times a likelihood, which is the split every proposal in this library assumes: pCN drops
    the prior ratio and ``HamiltonianInfinite`` integrates the prior part exactly."""
    def log_prior(x):
        return -0.5 * float(x @ x)

    def log_likelihood(x):
        return -0.5 * float(x @ precision @ x) + 0.5 * float(x @ x)

    return log_likelihood, log_prior


def _metropolis_loop(proposal, log_likelihood, log_prior, no_steps, x0, seed=0):
    """Plain MH loop driving ``propose_sample``/``get_log_acceptance_probability``/``adapt``
    exactly as ``Algorithm_MH.run`` does -- in particular the acceptance probability comes from
    the PROPOSAL, so the pCN prior cancellation and the Hamiltonian momentum term are included.

    Returns ``(states, accepted_flags)``; the accept/reject decision uses its own generator so
    the proposal's stream is not disturbed.
    """
    rng = np.random.RandomState(seed)
    x = np.asarray(x0, dtype=float).copy()
    log_l, log_pi = log_likelihood(x), log_prior(x)
    states = np.empty((no_steps, len(x)))
    accepted = np.zeros(no_steps, dtype=bool)
    for i in range(no_steps):
        y = proposal.propose_sample(x)
        log_l_proposed, log_pi_proposed = log_likelihood(y), log_prior(y)
        likelihood_part, prior_part = proposal.get_log_acceptance_probability(
            log_l_proposed, log_l, log_pi_proposed, log_pi)
        log_alpha = likelihood_part + prior_part
        if np.log(rng.uniform()) < log_alpha:
            x, log_l, log_pi = y, log_l_proposed, log_pi_proposed
            accepted[i] = True
        proposal.adapt(proposed_sample=y, log_acceptance_probability=log_alpha, current_sample=x)
        states[i] = x
    return states, accepted


@pytest.mark.unit
def test_adaptive_random_walk_reaches_target_rate_scale_and_shape():
    """
    Item 1: on a d=10, kappa=10 Gaussian, started from the isotropic ``sd=1`` guess, 20 000
    ``adapt()`` steps must coerce the acceptance rate to 0.234, learn the SHAPE of the target
    covariance and land on the ``2.38**2/d`` scale. Tolerances taken from the prototype study
    (`RESULTS_PROTO.md` §2, "AM-shrink+RM": acceptance within 0.006 of target, shape error
    0.077 at d10 kappa10, scale ratio 1.06).
    """
    d, condition_number, no_steps = 10, 10.0, 20_000
    C = _anisotropic_gaussian(d, condition_number)
    log_likelihood, log_prior = _gaussian_split(np.linalg.inv(C))

    proposal = GaussRandomWalk_adaptive(no_parameters=d, sd_or_cov=1.0, seed=3)
    _states, accepted = _metropolis_loop(proposal, log_likelihood, log_prior, no_steps,
                                         np.zeros(d), seed=11)

    rate = float(accepted[-5000:].mean())
    P = np.asarray(proposal.sd_or_cov)
    shape_error = (np.linalg.norm(P / np.trace(P) - C / np.trace(C), "fro")
                   / np.linalg.norm(C / np.trace(C), "fro"))
    scale_ratio = np.trace(P) / np.trace((2.38 ** 2 / d) * C)

    print(f"[adaptive RW] acceptance {rate:.3f}, shape error {shape_error:.3f}, "
          f"scale ratio {scale_ratio:.3f}")
    assert abs(rate - 0.234) < 0.03, f"acceptance {rate:.4f}, target 0.234 +- 0.03"
    assert shape_error < 0.25, f"shape error {shape_error:.4f}, tolerance 0.25"
    assert 0.5 <= scale_ratio <= 2.0, f"scale ratio {scale_ratio:.4f}, tolerance [0.5, 2.0]"


@pytest.mark.unit
def test_adaptive_random_walk_pooling_is_the_statistic_of_the_concatenated_chains():
    """
    Item 2: ``set_pooled_state`` of a single chain's own state is a no-op for the proposal
    covariance, and pooling two chains reproduces exactly the ``(n, mean, M2)`` of their
    concatenated states (Chan-Golub-LeVeque combination).
    """
    d = 3
    rng = np.random.RandomState(0)
    chains = [rng.randn(37, d) + 0.5, rng.randn(53, d) - 1.0]

    def feed(states, seed):
        proposal = GaussRandomWalk_adaptive(no_parameters=d, sd_or_cov=0.7, period=5, warmup=10,
                                            seed=seed)
        for state in states:
            proposal.adapt(proposed_sample=state, log_acceptance_probability=-0.5,
                           current_sample=state)
        return proposal

    # (a) pooling one rank's own state leaves the adapted covariance untouched
    solo = feed(chains[0], seed=1)
    before = np.array(solo.sd_or_cov, copy=True)
    solo.set_pooled_state(solo.adapted_state()[None, :])
    np.testing.assert_allclose(solo.sd_or_cov, before, rtol=0, atol=1e-12)

    # (b) two ranks pooled == the statistics of the concatenated states; since C1 (2026-10-11) the
    # runner pools the PosteriorStatistics (one Allgather of as_vector()) and installs them in
    # place BEFORE set_pooled_state, which exchanges only the effective log-scale
    rank0, rank1 = feed(chains[0], seed=1), feed(chains[1], seed=2)
    pooled = PosteriorStatistics.from_rows(np.vstack([rank0.statistics.as_vector(),
                                                      rank1.statistics.as_vector()]), d)
    states = np.vstack([rank0.adapted_state(), rank1.adapted_state()])
    rank0.statistics.install(pooled)
    rank1.statistics.install(pooled)
    rank0.set_pooled_state(states)

    everything = np.vstack(chains)
    expected_mean = everything.mean(axis=0)
    centered = everything - expected_mean
    expected_M2 = centered.T @ centered
    assert rank0.statistics.n == len(everything)
    np.testing.assert_allclose(rank0.mean, expected_mean, rtol=0, atol=1e-8)
    np.testing.assert_allclose(rank0.M2, expected_M2, rtol=0, atol=1e-8)
    # both ranks reach the same pooled covariance, which is the point of the hand-over
    rank1.set_pooled_state(states)
    np.testing.assert_allclose(rank0.sd_or_cov, rank1.sd_or_cov, rtol=0, atol=1e-12)
    assert rank0.carry_over()["scale"] is rank0.sd_or_cov


@pytest.mark.unit
def test_adaptive_pcn_reaches_target_rate_and_pools_the_logit():
    """
    Item 4: adaptive pCN on a d=10 Gaussian with a standard-normal prior and a kappa=100
    posterior must coerce the acceptance rate to 0.234 with ``beta`` staying strictly inside
    (0, 1); pooling two chains averages the logit (`RESULTS_PROTO.md` §7).
    """
    d, no_steps = 10, 20_000
    posterior_cov = _anisotropic_gaussian(d, 100.0)
    # pCN's acceptance ratio drops the prior term, which _gaussian_split makes correct here:
    # the prior really is the standard normal the proposal assumes
    log_likelihood, log_prior = _gaussian_split(np.linalg.inv(posterior_cov))

    proposal = PCN_adaptive(no_parameters=d, beta=0.5, prior_mean=np.zeros(d),
                            prior_sd_or_cov=1.0, seed=5)
    _states, accepted = _metropolis_loop(proposal, log_likelihood, log_prior, no_steps,
                                         np.zeros(d), seed=13)

    rate = float(accepted[-5000:].mean())
    print(f"[adaptive pCN] acceptance {rate:.3f}, beta {proposal.beta:.4f}")
    assert abs(rate - 0.234) < 0.03, f"acceptance {rate:.4f}, target 0.234 +- 0.03"
    assert 0.02 < proposal.beta < 0.8, proposal.beta
    assert len(proposal.adaptation_stats_rows) == no_steps // proposal.period

    other = PCN_adaptive(no_parameters=d, beta=0.5, prior_mean=np.zeros(d), prior_sd_or_cov=1.0,
                         seed=6)
    other.logit_beta = proposal.logit_beta - 1.5
    other._set_beta_from_logit()
    states = np.vstack([proposal.adapted_state(), other.adapted_state()])
    expected_logit = 0.5 * (proposal.logit_beta + other.logit_beta)
    for chain in (proposal, other):
        chain.set_pooled_state(states)
        assert chain.logit_beta == pytest.approx(expected_logit)
        assert chain.carry_over()["beta"] == pytest.approx(1.0 / (1.0 + np.exp(-expected_logit)))


def _hamiltonian_gradient_functions(precision):
    """The ``(grad(-log L), grad(-log prior))`` pair the Hamiltonian proposals consume, for a
    target ``N(0, C)`` split as prior ``N(0, I)`` times the rest (see the module docstring)."""
    identity = np.eye(precision.shape[0])
    return (lambda q: (precision - identity) @ q, lambda q: identity @ q)


@pytest.mark.unit
@pytest.mark.parametrize("proposal_class", [Hamiltonian_adaptive, HamiltonianInfinite_adaptive])
def test_dual_averaging_step_size_hits_its_target_and_stays_stable(proposal_class):
    """
    Item 5: with EXACT gradients, mass ``I`` and ``num_steps=10`` on a d=10 Gaussian, the
    dual-averaged step size must coerce the acceptance rate to ``delta=0.8``, be insensitive to
    the initial guess (``step_size_0`` 0.01 vs 1.0 agree within 1.5x) and stay below the
    leapfrog stability limit ``2*sqrt(lambda_min(C))`` for mass ``I``
    (`RESULTS_PROTO.md` §5.1/§5.2).
    """
    d, no_steps = 10, 4_000
    C = _anisotropic_gaussian(d, 10.0)
    precision = np.linalg.inv(C)
    log_likelihood, log_prior = _gaussian_split(precision)
    stability_limit = 2.0 * np.sqrt(np.min(np.linalg.eigvalsh(C)))

    step_size_bars = {}
    for step_size_0 in (0.01, 1.0):
        proposal = proposal_class(no_parameters=d, step_size=step_size_0, num_steps=10,
                                  sd_or_cov=1.0, seed=7)
        proposal.set_gradient_functions(*_hamiltonian_gradient_functions(precision))
        _states, accepted = _metropolis_loop(proposal, log_likelihood, log_prior, no_steps,
                                             np.zeros(d), seed=17)

        rate = float(accepted[-1000:].mean())
        step_size_bar = float(np.exp(proposal.log_eps_bar))
        step_size_bars[step_size_0] = step_size_bar
        print(f"[{proposal_class.__name__}] eps0={step_size_0}: acceptance {rate:.3f}, "
              f"eps_bar {step_size_bar:.4f} (stability limit {stability_limit:.4f})")
        assert abs(rate - 0.8) < 0.1, f"acceptance {rate:.4f}, target 0.8 +- 0.1"
        assert step_size_bar < stability_limit, (
            f"eps_bar {step_size_bar:.4f} is at/above the leapfrog stability limit "
            f"{stability_limit:.4f}")
        assert proposal.m == no_steps  # one dual-averaging update per MH iteration
        assert len(proposal.adaptation_stats_rows) == no_steps // proposal.period

    ratio = step_size_bars[0.01] / step_size_bars[1.0]
    assert 1 / 1.5 < ratio < 1.5, f"eps_bar depends on eps0: {step_size_bars} (ratio {ratio:.3f})"


@pytest.mark.unit
def test_dual_averaging_uses_the_subchain_probabilities_and_pools_log_eps_bar():
    """The DAMH path: one dual-averaging update per SUB-CHAIN step (`16` §5 item 4), and the
    cross-rank hand-over freezes the live step size at the averaged ``eps_bar``."""
    proposal = Hamiltonian_adaptive(no_parameters=2, step_size=0.1, num_steps=5, sd_or_cov=1.0,
                                    seed=1)
    proposal.adapt(proposed_sample=np.zeros(2), log_acceptance_probability=-np.inf,
                   current_sample=np.zeros(2),
                   subchain_log_acceptance_probabilities=[-0.1, -0.2, -0.3])
    assert proposal.m == 3, "one update per sub-chain step"

    other = Hamiltonian_adaptive(no_parameters=2, step_size=0.1, num_steps=5, sd_or_cov=1.0, seed=2)
    other.adapt(proposed_sample=np.zeros(2), log_acceptance_probability=-2.0,
                current_sample=np.zeros(2))
    assert other.m == 1, "without a sub-chain list, one update from the outer probability"

    states = np.vstack([proposal.adapted_state(), other.adapted_state()])
    expected = 0.5 * (proposal.log_eps_bar + other.log_eps_bar)
    for chain in (proposal, other):
        chain.set_pooled_state(states)
        assert chain.log_eps_bar == pytest.approx(expected)
        assert chain.step_size == pytest.approx(np.exp(expected))
        assert chain.carry_over()["step_size"] == pytest.approx(np.exp(expected))
    # the mass and the trajectory length are NOT adapted
    np.testing.assert_array_equal(proposal.sd_or_cov, np.ones(2))
    assert proposal.num_steps == 5


@pytest.mark.unit
def test_build_proposal_selects_the_adaptive_class_of_every_spec_type(tmp_path):
    """
    Item 6, first half, rewritten for the 2026-09-21 proposal-spec design: there is no more
    stage-level ``adaptive`` flag or ``proposal_type`` string -- each spec's OWN ``adaptive``
    resolves to True iff its step field (``scale``/``beta``/``step_size``) was left ``None``
    ("adapt unless pinned"). This checks that resolution picks the adaptive runtime class for
    all four proposal families, and the plain one when the step is pinned (``Block`` is not
    covered here: it is never adaptive, see test_build_proposal_rejects_adaptive_block_proposal).
    """
    conf = _make_conf(tmp_path, use_surrogate_gradients=True)
    prior = Normal(mean=np.zeros(2), sd=1.0)
    adaptive_specs = {
        "RandomWalk": (RandomWalk(), GaussRandomWalk_adaptive),
        "PCN": (PCNSpec(), PCN_adaptive),
        "Hamiltonian": (HamiltonianSpec(), Hamiltonian_adaptive),
        "HamiltonianInfinite": (HamiltonianSpec(integrator="dimension_robust"), HamiltonianInfinite_adaptive),
    }
    for label, (spec, proposal_class) in adaptive_specs.items():
        stage = Stage(algorithm="MH", proposal=spec, max_evaluations=10)
        assert stage.adaptive is True, f"{label}: a step-less spec should adapt by default"
        proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=1, stage_index=0)
        assert isinstance(proposal, proposal_class), label
    # a pinned step (explicit scale/beta/step_size) keeps the plain classes
    fixed_specs = {
        "RandomWalk": (RandomWalk(scale=1.0), GaussRandomWalk),
        "PCN": (PCNSpec(beta=0.5), PCN),
        "Hamiltonian": (HamiltonianSpec(step_size=0.1), Hamiltonian),
        "HamiltonianInfinite": (HamiltonianSpec(step_size=0.1, integrator="dimension_robust"), HamiltonianInfinite),
    }
    for label, (spec, proposal_class) in fixed_specs.items():
        stage = Stage(algorithm="MH", proposal=spec, max_evaluations=10)
        assert stage.adaptive is False, f"{label}: an explicit step should pin it"
        proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=1, stage_index=0)
        assert type(proposal) is proposal_class, label


@pytest.mark.unit
def test_removed_stage_fields_raise_type_error():
    """Item 6: ``adaptive_corr_limit``/``adaptive_sample_limit`` are gone from ``Stage``
    entirely (no-compat rule), so a script that still passes them fails loudly."""
    with pytest.raises(TypeError):
        Stage(adaptive_corr_limit=0.3, max_evaluations=10)
    with pytest.raises(TypeError):
        Stage(adaptive_sample_limit=50, max_evaluations=10)


@pytest.mark.unit
def test_build_proposal_resolves_none_beta_and_step_size_from_the_carry_over(tmp_path):
    """Item 6: ``PCN(beta=None)`` / ``Hamiltonian(step_size=None)`` take the carried value, and
    fall back to the classes' own defaults (0.5 / 0.1) when nothing was carried."""
    conf = _make_conf(tmp_path, use_surrogate_gradients=True)
    prior = Normal(mean=np.zeros(2), sd=1.0)

    pcn_stage = Stage(algorithm="MH", proposal=PCNSpec(), max_evaluations=10)
    assert pcn_stage.proposal.beta is None
    assert build_proposal(stage=pcn_stage, conf=conf, prior=prior, seed=1).beta == 0.5
    carried_pcn = build_proposal(stage=pcn_stage, conf=conf, prior=prior, seed=1,
                                 carried={"beta": 0.17})
    assert carried_pcn.beta == 0.17
    # an explicit value always wins over the carry-over
    explicit = Stage(algorithm="MH", proposal=PCNSpec(beta=0.25), max_evaluations=10)
    assert build_proposal(stage=explicit, conf=conf, prior=prior, seed=1,
                          carried={"beta": 0.17}).beta == 0.25

    hamiltonian_stage = Stage(algorithm="MH", proposal=HamiltonianSpec(), max_evaluations=10)
    assert hamiltonian_stage.proposal.step_size is None
    assert build_proposal(stage=hamiltonian_stage, conf=conf, prior=prior, seed=1).step_size == 0.1
    carried_hamiltonian = build_proposal(stage=hamiltonian_stage, conf=conf, prior=prior, seed=1,
                                         carried={"step_size": 0.037})
    assert carried_hamiltonian.step_size == 0.037


# ---------------------------------------------------------------------------
# 2026-09-20: the leapfrog integrators abandon a divergent trajectory
# ---------------------------------------------------------------------------

def _old_hamiltonian_leapfrog(prop, q0, p0_start):
    """The pre-2026-09-20 body of ``Hamiltonian._leapfrog``, copied verbatim (``self`` ->
    ``prop``), so that the finiteness checks added to the live version can be proven to change
    no floating-point result."""
    q = q0.copy()
    p = p0_start.copy()
    p = p - 0.5 * prop.step_size * (prop.log_likelihood_gradient(q) + prop.log_prior_gradient(q))  # half step for momentum
    for i in range(prop.num_steps):
        q = q + prop.step_size * prop.M_inv @ p  # full step for position
        if i < prop.num_steps - 1:
            p = p - prop.step_size * (prop.log_likelihood_gradient(q) + prop.log_prior_gradient(q))  # full step for momentum (except at the end)
    p = p - 0.5 * prop.step_size * (prop.log_likelihood_gradient(q) + prop.log_prior_gradient(q))  # final half step for momentum
    p_end = -p  # negate momentum (for reversibility — does not affect acceptance)
    return q, p_end


def _old_hamiltonian_infinite_leapfrog(prop, q0, p0_start):
    """The pre-2026-09-20 body of ``HamiltonianInfinite._leapfrog``, copied verbatim."""
    q = q0.copy()
    p = p0_start.copy()
    p = p - 0.5 * prop.step_size * prop.log_likelihood_gradient(q)  # half step for momentum
    for i in range(prop.num_steps):
        q, p = prop._apply_prior_kinetic_flow(q, p)
        if i < prop.num_steps - 1:
            p = p - prop.step_size * prop.log_likelihood_gradient(q)  # full step for momentum (except at the end)
    p = p - 0.5 * prop.step_size * prop.log_likelihood_gradient(q)  # final half step for momentum
    p_end = -p  # negate momentum (for reversibility — does not affect acceptance)
    return q, p_end


FULL_SPD_MASS = np.array([[2.0, 0.3, -0.1], [0.3, 1.5, 0.2], [-0.1, 0.2, 1.1]])


@pytest.mark.unit
@pytest.mark.parametrize("proposal_cls, old_leapfrog",
                         [(Hamiltonian, _old_hamiltonian_leapfrog),
                          (HamiltonianInfinite, _old_hamiltonian_infinite_leapfrog)],
                         ids=["Hamiltonian", "HamiltonianInfinite"])
@pytest.mark.parametrize("sd_or_cov", [1.0, FULL_SPD_MASS], ids=["mass_I", "full_mass"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_leapfrog_is_bit_identical_to_the_pre_divergence_check_version(proposal_cls, old_leapfrog,
                                                                      sd_or_cov, seed):
    """
    The divergence guard added on 2026-09-20 only *reads* ``q``/``p``, so on a finite
    trajectory the result must be bit-identical to the loop body it replaced (reproduced
    verbatim in ``_old_*_leapfrog`` above). 5 seeds, d = 3, mass I and a full SPD mass.
    """
    Q = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, -0.2], [0.0, -0.2, 1.1]])
    rng = np.random.RandomState(seed)
    q0 = rng.randn(3)
    p0 = rng.randn(3)

    def make():
        prop = proposal_cls(no_parameters=3, step_size=0.07, num_steps=12, sd_or_cov=sd_or_cov,
                            seed=seed)
        prop.log_likelihood_gradient = lambda q: Q @ q
        prop.log_prior_gradient = lambda q: 0.5 * q
        return prop

    q_new, p_new = make()._leapfrog(q0, p0)
    q_old, p_old = old_leapfrog(make(), q0, p0)

    assert np.array_equal(q_new, q_old)
    assert np.array_equal(p_new, p_old)
    assert np.all(np.isfinite(q_new))


@pytest.mark.unit
@pytest.mark.parametrize("proposal_cls, step_size, sd_or_cov",
                         [(Hamiltonian, 1e4, 1.0), (HamiltonianInfinite, 1e6, 1e-3)],
                         ids=["Hamiltonian", "HamiltonianInfinite"])
def test_leapfrog_stops_integrating_once_the_trajectory_is_not_finite(proposal_cls, step_size,
                                                                      sd_or_cov):
    """
    2026-09-20: with a gradient of ``1e300 * ones`` the trajectory leaves the finite range; the
    integrator must then return the non-finite position immediately (so that
    ``AlgorithmBase._reject_nonfinite_proposal`` rejects the proposal without calling the full
    model) instead of spending the remaining ``num_steps`` surrogate-gradient evaluations on it.

    The step sizes differ per class only to make the divergence happen within a short
    trajectory: ``HamiltonianInfinite``'s free flow is a rotation, which cannot amplify the
    state by itself.
    """
    num_steps = 10
    calls = []

    def gradient(q):
        calls.append(np.asarray(q).copy())
        return 1e300 * np.ones_like(q)

    prop = proposal_cls(no_parameters=3, step_size=step_size, num_steps=num_steps,
                        sd_or_cov=sd_or_cov, seed=0)
    prop.log_likelihood_gradient = gradient
    prop.log_prior_gradient = lambda q: np.zeros_like(q)

    with np.errstate(over="ignore", invalid="ignore"):
        q, _p_end = prop._leapfrog(np.array([0.1, 0.2, 0.3]), np.ones(3))

    assert not np.all(np.isfinite(q)), "the diverging trajectory should return a non-finite q"
    assert len(calls) < num_steps + 1, (
        f"the integrator kept evaluating the gradient after diverging: {len(calls)} calls")


# ---------------------------------------------------------------------------
# 2026-09-21: the adaptive random walk's effective covariance is continuous when its own
# covariance estimate replaces the starting one, and an adaptive RWMH stage therefore no longer
# needs a starting ``RandomWalk(scale=...)`` (build_proposal derives one from the prior).
# ---------------------------------------------------------------------------

def _run_adaptive_random_walk_on_gaussian(proposal, no_parameters, posterior_sd, no_iterations,
                                          seed=1, record_at=()):
    """Plain MH on N(0, posterior_sd^2 I) driving ``proposal.adapt``; returns the effective
    proposal sd ``sqrt(tr sd_or_cov / d)`` at the iterations in ``record_at`` and the mean
    acceptance probability over the last 500 iterations (from the adaptation trace)."""
    rng = np.random.RandomState(seed)
    state = np.zeros(no_parameters)

    def log_posterior(x):
        return -0.5 * float(np.sum((x / posterior_sd) ** 2))

    recorded = {}
    for n in range(1, no_iterations + 1):
        proposed = proposal.propose_sample(state)
        log_alpha = min(0.0, log_posterior(proposed) - log_posterior(state))
        if np.log(rng.rand()) < log_alpha:
            state = proposed
        proposal.adapt(proposed, log_alpha, state)
        if n in record_at:
            recorded[n] = float(np.sqrt(np.trace(proposal.sd_or_cov) / no_parameters))
    tail = proposal.adaptation_stats_rows[-(500 // proposal.period):]
    return recorded, float(np.mean([row[1] for row in tail]))


@pytest.mark.unit
def test_adaptive_random_walk_effective_scale_is_continuous_at_install():
    """From sd 0.001 on a unit-scale target the scale recursion has raised ``log_sigma`` by
    ~ln(1000) by the time the empirical covariance is installed at ``n = warmup``. Before the
    2026-09-21 fix the effective sd jumped 0.8 -> 776 there (the learned scale was applied to a
    base 10^6 times larger) and the chain froze; now the install leaves the effective trace
    unchanged and the start is irrelevant after the warm-up."""
    d = 4
    proposal = GaussRandomWalk_adaptive(no_parameters=d, sd_or_cov=0.001, seed=0)  # warmup=100
    recorded, tail_rate = _run_adaptive_random_walk_on_gaussian(
        proposal, d, posterior_sd=1.0, no_iterations=3000, record_at=(99, 100, 3000))

    # continuity across the install: one scale-recursion step changes log_sigma by at most
    # 100^(-0.7) * 0.766 ~ 0.03, i.e. the effective sd by < 4 %
    assert abs(np.log(recorded[100] / recorded[99])) < 0.05, recorded
    # and the run ends near the 2.38/sqrt(d) optimum with the target acceptance
    assert 0.8 < recorded[3000] < 2.0, recorded
    assert 0.15 < tail_rate < 0.32, tail_rate


@pytest.mark.unit
@pytest.mark.parametrize("initial_sd, posterior_sd", [(100.0, 1.0), (1.0, 1e-3)])
def test_adaptive_random_walk_recovers_from_a_start_far_from_the_posterior_scale(initial_sd, posterior_sd):
    """Too large a start, and the mirror case of a unit start on a tiny posterior (which is what
    a fixed 'reasonable default' start would face): both end at the optimum scale."""
    d = 4
    proposal = GaussRandomWalk_adaptive(no_parameters=d, sd_or_cov=initial_sd, seed=0)
    recorded, tail_rate = _run_adaptive_random_walk_on_gaussian(
        proposal, d, posterior_sd=posterior_sd, no_iterations=5000, record_at=(5000,))
    optimum = 2.38 / np.sqrt(d) * posterior_sd
    assert 0.7 * optimum < recorded[5000] < 1.5 * optimum, (recorded, optimum)
    assert 0.15 < tail_rate < 0.32, tail_rate


@pytest.mark.unit
def test_adaptive_random_walk_install_keeps_the_effective_trace_exactly():
    """Unit-level statement of the invariant: ``tr(sd_or_cov)`` before and after the install
    that ``adapt`` triggers at ``n = warmup`` differ only by that call's own scale step."""
    d = 3
    proposal = GaussRandomWalk_adaptive(no_parameters=d, sd_or_cov=0.01, seed=0, period=5, warmup=20)
    rng = np.random.RandomState(0)
    state = np.zeros(d)
    for _ in range(proposal.warmup - 1):
        state = state + rng.randn(d)
        proposal.adapt(state, 0.0, state)  # alpha = 1
    trace_before = np.trace(proposal.sd_or_cov)
    log_sigma_before = proposal.log_sigma
    proposal.adapt(state, 0.0, state)  # n == warmup: installs the empirical covariance
    assert proposal.n == proposal.warmup and np.isfinite(proposal._trace_over_d)
    own_scale_step = proposal.n ** (-0.7) * (1.0 - proposal.target_rate)
    assert np.isclose(np.trace(proposal.sd_or_cov), trace_before * np.exp(2.0 * own_scale_step))
    # the base_cov really changed (this is not a no-op install)
    assert not np.isclose(proposal.log_sigma, log_sigma_before + own_scale_step)


@pytest.mark.unit
def test_build_proposal_adaptive_rwmh_default_start_from_normal_prior(tmp_path):
    """No ``scale``, nothing carried over, ``RandomWalk(adaptive=True)``: the start is the
    prior covariance scaled by 2.38^2/d (here ``Normal(sd=2)`` in d=2)."""
    conf = _make_conf(tmp_path)
    prior = Normal(mean=np.zeros(2), sd=2.0)
    stage = Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=10)
    assert stage.proposal.scale is None
    proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=1, stage_index=0)
    assert isinstance(proposal, GaussRandomWalk_adaptive)
    np.testing.assert_allclose(proposal.sd_or_cov, (2.38 ** 2 / 2) * 4.0 * np.eye(2))


@pytest.mark.unit
def test_build_proposal_adaptive_rwmh_default_start_from_independent_components_and_matrix(tmp_path):
    conf = _make_conf(tmp_path)
    stage = Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=10)
    prior = PriorIndependentComponents([NormalComponent(0.0, 1.0), NormalComponent(1.0, 3.0)])
    proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=1, stage_index=0)
    np.testing.assert_allclose(proposal.sd_or_cov, (2.38 ** 2 / 2) * np.eye(2))  # internal N(0, I)

    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    proposal = build_proposal(stage=stage, conf=conf, prior=Normal(mean=np.zeros(2), cov=cov),
                              seed=1, stage_index=0)
    np.testing.assert_allclose(proposal.sd_or_cov, (2.38 ** 2 / 2) * cov)


@pytest.mark.unit
def test_build_proposal_adaptive_rwmh_default_start_falls_back_to_unit_sd_with_warning(tmp_path, capsys):
    conf = _make_conf(tmp_path)
    stage = Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=10)
    prior = FromScipy(scipy.stats.multivariate_normal(mean=np.zeros(2), cov=4.0 * np.eye(2)))
    proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=1, stage_index=0)
    np.testing.assert_allclose(proposal.sd_or_cov, np.eye(2))
    out = capsys.readouterr().out
    assert "Warning" in out and "FromScipy" in out and "get_covariance" in out


@pytest.mark.unit
def test_build_proposal_carried_covariance_beats_the_default_start(tmp_path):
    conf = _make_conf(tmp_path)
    stage = Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=10)
    carried_cov = np.diag([0.1, 0.2])
    proposal = build_proposal(stage=stage, conf=conf, prior=Normal(mean=np.zeros(2), sd=1.0), seed=1,
                              carried={"scale": carried_cov}, stage_index=1)
    np.testing.assert_allclose(proposal.sd_or_cov, carried_cov)


@pytest.mark.unit
def test_build_proposal_non_adaptive_rwmh_without_start_or_carry_over_warns_and_uses_default(tmp_path, capsys):
    """2026-09-21 (author decision): warn and use the prior-scaled default instead of raising."""
    conf = _make_conf(tmp_path)
    stage = Stage(algorithm="MH", proposal=RandomWalk(adaptive=False), max_evaluations=10)
    capsys.readouterr()
    proposal = build_proposal(stage=stage, conf=conf, prior=Normal(mean=np.zeros(2), sd=2.0), seed=1, stage_index=0)
    assert ("Warning: stage 0: RandomWalk(adaptive=False) without a scale"
           in capsys.readouterr().out)
    assert type(proposal) is GaussRandomWalk
    # Normal.get_covariance() returns the sd VECTOR, so the default is sqrt(2.38^2/d) * sd
    np.testing.assert_allclose(proposal.sd_or_cov, np.sqrt(2.38 ** 2 / 2) * 2.0 * np.ones(2))


# ----------------------------------------------------------------------------------------
# adapted_summary() (2026-09-21): diagnostic-only snapshot for the carry_over report, never
# consumed by build_proposal (see Proposal.adapted_summary docstring)
# ----------------------------------------------------------------------------------------
@pytest.mark.unit
def test_base_proposal_adapted_summary_is_empty():
    assert GaussRandomWalk(no_parameters=2).adapted_summary() == {}


@pytest.mark.unit
def test_gauss_random_walk_adaptive_adapted_summary_keys_and_shapes():
    d = 3
    proposal = GaussRandomWalk_adaptive(no_parameters=d, sd_or_cov=0.5, period=5, warmup=10, seed=1)
    for state in np.random.RandomState(0).randn(60, d):
        proposal.adapt(proposed_sample=state, log_acceptance_probability=-0.3, current_sample=state)
    proposal.set_pooled_state(proposal.adapted_state()[None, :])

    summary = proposal.adapted_summary()
    assert set(summary) == {"base_cov", "log_sigma", "n_pooled", "mean"}
    assert summary["base_cov"].shape == (d, d)
    np.testing.assert_allclose(summary["base_cov"], proposal.base_cov)
    assert isinstance(summary["log_sigma"], float)
    assert summary["log_sigma"] == pytest.approx(proposal.log_sigma)
    assert isinstance(summary["n_pooled"], int)
    assert summary["n_pooled"] == proposal.n
    assert summary["mean"].shape == (d,)
    np.testing.assert_allclose(summary["mean"], proposal.mean)


@pytest.mark.unit
def test_pcn_adaptive_adapted_summary():
    proposal = PCN_adaptive(no_parameters=2, beta=0.4, prior_mean=np.zeros(2), prior_sd_or_cov=1.0, seed=1)
    proposal.adapt(proposed_sample=np.zeros(2), log_acceptance_probability=-0.5, current_sample=np.zeros(2))
    summary = proposal.adapted_summary()
    assert set(summary) == {"beta"}
    assert summary["beta"] == pytest.approx(proposal.beta)


@pytest.mark.unit
@pytest.mark.parametrize("proposal_class", [Hamiltonian_adaptive, HamiltonianInfinite_adaptive])
def test_dual_averaging_adapted_summary_matches_carry_over(proposal_class):
    proposal = proposal_class(no_parameters=2, step_size=0.1, num_steps=5, sd_or_cov=1.0, seed=1)
    proposal.adapt(proposed_sample=np.zeros(2), log_acceptance_probability=-0.4, current_sample=np.zeros(2))
    summary = proposal.adapted_summary()
    assert set(summary) == {"step_size"}
    assert summary["step_size"] == pytest.approx(proposal.carry_over()["step_size"])
