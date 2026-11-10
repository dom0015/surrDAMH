#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 items 1-3 (library_notes/07_testing_plan.md): ``Stage`` and ``Configuration``
dataclass construction / validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from surrDAMH.configuration import Configuration
from surrDAMH.distributions.independent_components import (
    NormalComponent,
    PriorIndependentComponents,
)
from surrDAMH.distributions.normal import Normal
from surrDAMH.proposals import Block, Hamiltonian, PCN, RandomWalk
from surrDAMH.stages import Stage, stage_name


@pytest.mark.unit
def test_stage_no_longer_has_the_pre_2026_09_21_proposal_knobs():
    """
    Item 1, updated for the 2026-09-21 proposal-spec rewrite: every adaptive/proposal knob that
    used to live directly on ``Stage`` (``adaptive_target_rate``, ``pcn_beta``,
    ``hamiltonian_step_size``, and the even older ``adaptive_corr_limit``/
    ``adaptive_sample_limit``) was removed with no compatibility shim (author's no-compat rule).
    The equivalents now live on ``stage.proposal`` (a ``RandomWalk``/``PCN``/``Hamiltonian``/
    ``Block`` from ``surrDAMH.proposals``); passing any of the removed kwargs to ``Stage(...)``
    still raises ``TypeError`` (plain dataclass, no **kwargs).
    """
    stage = Stage(proposal=PCN(target_rate=0.4))
    assert stage.proposal.target_rate == 0.4
    for removed in ("adaptive_target_rate", "pcn_beta", "hamiltonian_step_size",
                    "adaptive_corr_limit", "adaptive_sample_limit"):
        assert not hasattr(stage, removed)
        with pytest.raises(TypeError):
            Stage(**{removed: 0.4})


@pytest.mark.unit
def test_stage_no_longer_disables_adaptive_for_pcn(capsys):
    """Adaptive pCN is supported since 2026-09-20 (Robbins-Monro on the logit of beta), so a
    ``PCN(adaptive=True)`` proposal stays adaptive."""
    stage = Stage(algorithm="MH", proposal=PCN(adaptive=True), max_evaluations=10)
    assert stage.adaptive is True
    assert "adaptive" not in capsys.readouterr().out


@pytest.mark.unit
def test_stage_algorithm_is_an_unenforced_literal():
    """
    Item 2. ``Stage.algorithm`` is typed as ``Literal["MH", "DAMH"]`` but Python does
    not enforce ``Literal`` at runtime, so construction with an invalid value (e.g. the
    lowercase ``"damh"``) does NOT raise here.
    """
    stage = Stage(algorithm="damh")
    assert stage.algorithm == "damh"


@pytest.mark.unit
def test_invalid_algorithm_raises_in_run_sampler(tmp_path):
    """
    Item 2 continued: the actual validation of ``algorithm`` happens later, in
    ``surrDAMH.stages.stage_name`` as called from ``surrDAMH.process_SAMPLER.run_SAMPLER``
    (WS0 item B6): ``raise ValueError(f"unknown algorithm {stage.algorithm!r} in stage {i} ...")``.

    This calls the real ``run_SAMPLER`` (not a copy of its logic) with
    ``use_collector=False, use_solvers_pool=False`` so it runs entirely on one MPI rank
    (no spawned solver pool / collector needed) and hits the ValueError after the (valid)
    default RWMH proposal is built but before any solver machinery is touched.
    """
    from surrDAMH.process_SAMPLER import run_SAMPLER

    prior = PriorIndependentComponents([NormalComponent(mu=0.0, sigma=1.0), NormalComponent(mu=0.0, sigma=1.0)])
    likelihood = Normal(mean=np.zeros(2), sd=np.ones(2))
    conf = Configuration(
        no_parameters=2,
        no_observations=2,
        output_dir=str(tmp_path),
        use_collector=False,
        use_solvers_pool=False,
    )
    bad_stage = Stage(algorithm="damh", proposal=RandomWalk(scale=1.0), max_evaluations=1)

    with pytest.raises(ValueError, match=r"unknown algorithm 'damh'"):
        run_SAMPLER(
            conf=conf,
            prior=prior,
            likelihood=likelihood,
            list_of_stages=[bad_stage],
            solver_instance=None,
            surrogate_evaluator=None,
        )


@pytest.mark.unit
def test_configuration_world_size_assertion_message(tmp_path):
    """
    Item 3: under plain ``pytest`` (mpi4py auto-inits a size-1 ``COMM_WORLD``),
    ``use_collector=True, use_solvers_pool=True`` needs ``no_samplers = size_world - 2 > 0``,
    i.e. at least 3 ranks. Assert the exact message from configuration.py (WS0 item D11).
    """
    with pytest.raises(AssertionError, match=r"use at least 3 MPI processes with collector and solvers pool"):
        Configuration(
            no_parameters=2,
            no_observations=2,
            output_dir=str(tmp_path),
            use_collector=True,
            use_solvers_pool=True,
        )


# ---------------------------------------------------------------------------
# WS7: surrogate_model_updates on an MH stage with a Hamiltonian-family proposal
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_surrogate_model_updates_defaults_are_unchanged():
    """Historical defaults: on for DAMH, off for MH (unconditionally)."""
    assert Stage(algorithm="DAMH", max_evaluations=10).surrogate_model_updates is True
    assert Stage(algorithm="MH", max_evaluations=10).surrogate_model_updates is False
    assert Stage(algorithm="MH", proposal=Hamiltonian(),
                 max_evaluations=10).surrogate_model_updates is False


@pytest.mark.unit
def test_surrogate_model_updates_opt_in_on_mh_with_gradient_proposals():
    """
    WS7: an MH stage whose proposal needs surrogate gradients may keep
    ``surrogate_model_updates=True`` (refresh the gradient surrogate during the stage).
    """
    for integrator in ("leapfrog", "dimension_robust"):
        stage = Stage(algorithm="MH", proposal=Hamiltonian(integrator=integrator),
                      surrogate_model_updates=True, max_evaluations=10)
        assert stage.surrogate_model_updates is True, integrator
        assert stage.proposal_needs_gradients() is True, integrator

    block_stage = Stage(algorithm="MH",
                        proposal=Block(groups=[[0], [1]],
                                      proposals=[Hamiltonian(step_size=0.1),
                                                 RandomWalk(scale=0.5)]),
                        surrogate_model_updates=True, max_evaluations=10)
    assert block_stage.proposal_needs_gradients() is True
    assert block_stage.surrogate_model_updates is True


@pytest.mark.unit
def test_surrogate_model_updates_refused_on_mh_without_gradient_proposal(capsys):
    """WS7: on any other MH stage ``True`` is still forced back to ``False``, now with a
    warning instead of silently."""
    proposals = (
        RandomWalk(),
        PCN(),
        Block(groups=[[0], [1]], proposals=[RandomWalk(scale=0.5), RandomWalk(scale=0.5)]),
    )
    for proposal in proposals:
        stage = Stage(algorithm="MH", proposal=proposal,
                      surrogate_model_updates=True, max_evaluations=10)
        assert stage.surrogate_model_updates is False, proposal
        assert stage.proposal_needs_gradients() is False, proposal
        assert "surrogate_model_updates" in capsys.readouterr().out


@pytest.mark.unit
def test_surrogate_model_updates_explicit_false_is_kept():
    """WS7: an explicit ``False`` is never turned into anything else."""
    assert Stage(algorithm="DAMH", surrogate_model_updates=False,
                 max_evaluations=10).surrogate_model_updates is False
    assert Stage(algorithm="MH", proposal=Hamiltonian(), surrogate_model_updates=False,
                 max_evaluations=10).surrogate_model_updates is False


@pytest.mark.unit
def test_stage_name_is_unaffected_by_the_mh_opt_in():
    """WS7: the DAMH-SMU suffix is a DAMH-only naming rule; an opted-in MH stage stays ``_MH``."""
    stage = Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1),  # pinned step -> non-adaptive
                  surrogate_model_updates=True, max_evaluations=10)
    assert stage_name(stage, 1) == "alg0001_MH"


@pytest.mark.unit
def test_adaptive_default_is_adapt_unless_the_step_is_pinned():
    """2026-09-21 (author decision): each proposal spec's ``adaptive=None`` resolves to True iff
    its step field was left None (``RandomWalk.__post_init__`` etc.); ``Stage.adaptive`` mirrors
    ``stage.proposal.adaptive``, and a ``Block`` never adapts regardless of its sub-proposals."""
    assert Stage(max_evaluations=10).adaptive is True  # default proposal: RandomWalk(scale=None)
    assert Stage(proposal=RandomWalk(scale=0.5), max_evaluations=10).adaptive is False
    assert Stage(proposal=RandomWalk(scale=0.5, adaptive=True), max_evaluations=10).adaptive is True
    assert Stage(proposal=RandomWalk(adaptive=False), max_evaluations=10).adaptive is False
    assert Stage(proposal=PCN(), max_evaluations=10).adaptive is True
    assert Stage(proposal=PCN(beta=0.2), max_evaluations=10).adaptive is False
    for integrator in ("leapfrog", "dimension_robust"):
        assert Stage(proposal=Hamiltonian(integrator=integrator), max_evaluations=10).adaptive is True
        assert Stage(proposal=Hamiltonian(integrator=integrator, step_size=0.05),
                     max_evaluations=10).adaptive is False

    block = Stage(proposal=Block(groups=[[0]], proposals=[RandomWalk(scale=0.5)]), max_evaluations=10)
    assert block.adaptive is False  # a Block never adapts, regardless of its sub-proposals
    # there is no way to force a Block stage adaptive any more: `adaptive` is a plain class
    # attribute on Block, not a dataclass field, so passing it raises (this replaces the old
    # `proposal_type="block", adaptive=True` combination, which used to raise later in
    # build_proposal instead)
    with pytest.raises(TypeError):
        Block(groups=[[0]], proposals=[RandomWalk(scale=0.5)], adaptive=True)


@pytest.mark.unit
def test_target_rate_with_adaptive_false_prints_a_warning_at_construction(capsys):
    """2026-09-22: target_rate is unused by a fixed proposal; the spec says so when built, not only
    in describe() at start-up."""
    from surrDAMH.proposals import PCN, Hamiltonian, RandomWalk
    capsys.readouterr()
    RandomWalk(scale=0.5, adaptive=False, target_rate=0.5)
    out = capsys.readouterr().out
    assert "RandomWalk(target_rate=0.5, adaptive=False)" in out and "only used when the proposal adapts" in out
    for spec in (RandomWalk(scale=0.5, target_rate=0.3, adaptive=True), RandomWalk(target_rate=0.3),
                 PCN(beta=0.2), Hamiltonian(step_size=0.1)):
        capsys.readouterr()
        assert capsys.readouterr().out == "", spec  # adaptive, or no target_rate: silent
    capsys.readouterr()
    PCN(beta=0.2, adaptive=False, target_rate=0.3); Hamiltonian(step_size=0.1, target_rate=0.9)
    assert capsys.readouterr().out.count("only used when the proposal adapts") == 2


@pytest.mark.unit
def test_stage_list_with_a_bare_proposal_is_refused_with_a_hint():
    """2026-09-22: the mistake from the author's template (a RandomWalk appended as a stage)."""
    from surrDAMH.proposals import RandomWalk
    from surrDAMH.stages import check_stage_list
    with pytest.raises(TypeError, match=r"list_of_stages\[1\] is a RandomWalk.*wrap it: Stage\(proposal="):
        check_stage_list([Stage(max_evaluations=5), RandomWalk(scale=0.5)])
    with pytest.raises(TypeError, match="got a single Stage"):
        check_stage_list(Stage(max_evaluations=5))
    check_stage_list([Stage(max_evaluations=5), Stage(algorithm="DAMH", max_evaluations=5)])  # fine


@pytest.mark.unit
def test_stages_module_does_not_leak_the_proposal_specs():
    """2026-09-22: surrDAMH.stages.RandomWalk used to resolve (an import leak) and invited
    appending a proposal in place of a stage."""
    import surrDAMH.stages as stages
    for name in ("RandomWalk", "PCN", "Hamiltonian", "Block", "ProposalSpec", "STEP_FIELD"):
        assert not hasattr(stages, name), name
    assert hasattr(stages, "Stage") and hasattr(stages, "stage_name")


@pytest.mark.unit
def test_wasted_snapshot_notes_flag_only_a_frozen_tail():
    """2026-09-22 (GRF scheme study): a frozen-surrogate stage that still streams snapshots, with no
    later stage that would pick up a retrained surrogate, gets a start-up note; other layouts do not."""
    from surrDAMH.stages import wasted_snapshot_notes
    mh, smu = Stage(max_evaluations=5), Stage(algorithm="DAMH", max_evaluations=5)
    frozen = Stage(algorithm="DAMH", max_evaluations=5, surrogate_model_updates=False)
    frozen_quiet = Stage(algorithm="DAMH", max_evaluations=5, surrogate_model_updates=False, send_snapshots_to_collector=False)
    assert len(wasted_snapshot_notes([mh, smu, frozen])) == 1 and "stage 2" in wasted_snapshot_notes([mh, smu, frozen])[0]
    assert wasted_snapshot_notes([mh, smu, frozen_quiet]) == []
    assert wasted_snapshot_notes([mh, frozen, smu]) == []          # a later SMU stage still consumes them
    assert wasted_snapshot_notes([mh, smu]) == []                   # SMU stage: its own retrains use them
    assert wasted_snapshot_notes([mh]) == []

