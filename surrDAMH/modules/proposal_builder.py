#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-stage proposal construction, shared by the MPI sampler process and the
standalone local runner.

This module is intentionally MPI-free: it only depends on ``Stage``, the
configuration attributes ``no_parameters`` / ``use_surrogate_gradients`` and the
prior distribution, so that ``surrDAMH.runner_local`` can reuse it without
importing ``surrDAMH.process_SAMPLER`` (which imports ``mpi4py``).

The logic started as a verbatim extraction of the block that used to live inline in
``process_SAMPLER.run_SAMPLER`` (seed usage is still exactly that block's). It is also the one
place that resolves the ``Stage`` fields left ``None`` against the ``carried`` hand-over of the
previous adaptive stages, and that maps ``Stage.adaptive`` to the adaptive proposal class of
each ``proposal_type`` (2026-09-20).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from surrDAMH.distributions.independent_components import PriorIndependentComponents
from surrDAMH.distributions.normal import Normal
from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules import proposals
from surrDAMH.modules.proposals import Proposal
from surrDAMH.stages import Stage

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing mpi4py at runtime
    from surrDAMH.configuration import Configuration


def _prior_is_gaussian(prior: Distribution) -> bool:
    """True iff ``prior`` is known to be exactly N(prior.mean, prior.get_covariance()).

    ``PCN.get_log_acceptance_probability`` drops the prior ratio, which is only correct
    when the internal prior really is that Gaussian. ``Normal`` and
    ``PriorIndependentComponents`` (whose internal prior is standard normal by design,
    see ``tests/unit/test_distributions.py::TestPriorIndependentComponentsLogpdfDesign``)
    qualify. ``FromScipy`` is rejected unconditionally: a frozen
    ``scipy.stats.multivariate_normal`` exposes no public attribute identifying its
    distribution family (only the private ``_dist``), and a frozen univariate
    ``scipy.stats.norm``'s public ``.dist`` does not reliably identify it as Gaussian
    either -- there is no non-fragile way to detect Gaussianity from the outside, and
    even if there were, ``FromScipy`` does not implement ``get_covariance()``. Anything
    else (e.g. ``GaussianMixture``) is rejected too.
    """
    return isinstance(prior, (Normal, PriorIndependentComponents))


def _default_adaptive_random_walk_covariance(prior: Distribution, no_parameters: int,
                                              stage_index: int | None):
    """
    Starting covariance of an adaptive random walk whose stage left ``proposal_sd_or_cov=None``
    and that has nothing to carry over (2026-09-21).

    ``GaussRandomWalk_adaptive`` learns both scale and shape and keeps the effective covariance
    continuous when its covariance estimate replaces the starting one, so the starting point
    only shapes the warm-up. The textbook start is the prior covariance scaled by ``2.38^2 / d``
    (Gelman-Roberts-Gilks), which has the units of the problem; it is used whenever the prior
    implements ``get_covariance()`` (``Normal``, ``PriorIndependentComponents``). Any other
    prior (``FromScipy``, ``GaussianMixture``) falls back to the unit sd ``1.0`` with a printed
    warning, since nothing tells its scale from the outside.
    """
    scale_factor = 2.38 ** 2 / no_parameters
    try:
        prior_sd_or_cov = np.asarray(prior.get_covariance(), dtype=float)
    except NotImplementedError:
        print(f"Warning: stage {stage_index}: proposal_sd_or_cov is None and the prior of type "
              f"'{type(prior).__name__}' has no get_covariance(); the adaptive random walk starts "
              "from sd 1.0 (only the warm-up depends on this starting point)")
        return 1.0
    if prior_sd_or_cov.ndim == 2:
        return scale_factor * prior_sd_or_cov  # covariance matrix
    # scalar or vector of STANDARD DEVIATIONS (the Distribution.get_covariance() contract)
    return np.sqrt(scale_factor) * prior_sd_or_cov


def build_proposal(stage: Stage, conf: "Configuration", prior: Distribution, seed: int,
                   carried: dict | None = None, stage_index: int | None = None) -> Proposal:
    """
    Build the proposal distribution of one sampling stage.

    Args:
        stage: the stage whose proposal is constructed.
        conf: configuration (only ``no_parameters`` and ``use_surrogate_gradients`` are used).
        prior: prior distribution (used by the pCN proposal, and for the starting covariance of
            an adaptive random walk that has no ``proposal_sd_or_cov`` and nothing to carry over).
        seed: seed of the proposal's random generator.
        carried: merged carry-over of every previous adaptive stage of this run
            (``Proposal.carry_over()`` keyed by ``Stage`` field name, 2026-09-20). It supplies
            the value of every stage field that is left ``None``:
            ``proposal_sd_or_cov`` (from an adaptive random walk), ``pcn_beta`` (from an
            adaptive pCN stage, default 0.5) and ``hamiltonian_step_size`` (the dual-averaged
            step size of an adaptive Hamiltonian stage, default 0.1). Replaces the former
            ``prev_cov`` argument. A random-walk stage with ``proposal_sd_or_cov=None`` and
            nothing to carry over starts from the prior-derived default of
            ``_default_adaptive_random_walk_covariance`` when ``adaptive=True`` and raises
            ``ValueError`` otherwise (2026-09-21).
        stage_index: index of the stage in the list of stages (only used in error messages).

    Returns:
        The proposal instance for this stage.
    """
    carried = {} if carried is None else carried
    # only forwarded when the stage sets it; otherwise each adaptive class keeps its own
    # default target (0.234 for the random walk and pCN, 0.8 for the dual-averaged step size)
    adaptive_kwargs: dict = {}
    if stage.adaptive and stage.adaptive_target_rate is not None:
        adaptive_kwargs["target_rate"] = stage.adaptive_target_rate

    if stage.proposal_type == "pCN":
        if not _prior_is_gaussian(prior):
            raise ValueError(
                "pCN proposal requires a Gaussian internal prior: "
                "PCN.get_log_acceptance_probability drops the prior ratio, which is only "
                "correct when the internal prior is exactly N(prior.mean, "
                "prior.get_covariance()). Got prior of type "
                f"'{type(prior).__name__}'. Use surrDAMH.distributions.Normal or "
                "PriorIndependentComponents (internal prior is standard normal by design) "
                "instead; FromScipy and GaussianMixture priors are not accepted for pCN."
            )
        pcn_beta = stage.pcn_beta
        if pcn_beta is None:
            pcn_beta = carried.get("pcn_beta", 0.5)
        pcn_class = proposals.PCN_adaptive if stage.adaptive else proposals.PCN
        my_Prop = pcn_class(
            no_parameters=conf.no_parameters,
            beta=pcn_beta,
            prior_mean=prior.mean,
            prior_sd_or_cov=prior.get_covariance(),
            seed=seed,
            **adaptive_kwargs,
        )
    elif stage.proposal_type in ("Hamiltonian", "HamiltonianInfinite"):
        # The carried random-walk covariance is deliberately NOT used as the mass: `16` §5
        # item 5 measured M = cov as a poor mass (M = cov^-1 is the good one) and carrying the
        # inverse needs the step size re-adapted at fixed integration time -- an open decision.
        hamiltonian_sd_or_cov = stage.proposal_sd_or_cov
        if hamiltonian_sd_or_cov is None:
            hamiltonian_sd_or_cov = 1.0
        step_size = stage.hamiltonian_step_size
        if step_size is None:
            step_size = carried.get("hamiltonian_step_size", 0.1)
        if stage.proposal_type == "Hamiltonian":
            hamiltonian_class = proposals.Hamiltonian_adaptive if stage.adaptive else proposals.Hamiltonian
        else:
            hamiltonian_class = (proposals.HamiltonianInfinite_adaptive if stage.adaptive
                                 else proposals.HamiltonianInfinite)
        my_Prop = hamiltonian_class(
            no_parameters=conf.no_parameters,
            seed=seed,
            num_steps=stage.hamiltonian_num_steps,
            step_size=step_size,
            sd_or_cov=hamiltonian_sd_or_cov,
            **adaptive_kwargs,
        )
    elif stage.proposal_type == "block":
        if stage.adaptive:
            # G2 (2026-09-17): a BlockProposal has no 'sd_or_cov', so the per-stage adaptive
            # covariance reduction used to die with AttributeError on every sampler at the end
            # of the stage (finding 2.5). Fail fast at construction instead.
            raise ValueError(
                f"stage {stage_index}: adaptive covariance reduction is not defined for block "
                "proposals; adapt the sub-proposals instead (pass GaussRandomWalk_adaptive "
                "instances in block_proposal_list and leave Stage.adaptive=False)"
            )
        my_Prop = proposals.BlockProposal(
            no_parameters=conf.no_parameters,
            list_of_groups=stage.block_proposal_groups,
            list_of_proposals=stage.block_proposal_list,
            seed=seed
        )
    else:
        if stage.adaptive:
            my_Prop = proposals.GaussRandomWalk_adaptive(no_parameters=conf.no_parameters, seed=seed,
                                                         **adaptive_kwargs)
        else:
            my_Prop = proposals.GaussRandomWalk(no_parameters=conf.no_parameters, seed=seed)
        sd_or_cov = stage.proposal_sd_or_cov
        if sd_or_cov is None:
            sd_or_cov = carried.get("proposal_sd_or_cov")
        if sd_or_cov is None:
            if not stage.adaptive:
                raise ValueError(
                    f"stage {stage_index}: proposal_sd_or_cov is None, no earlier adaptive stage "
                    "provides a covariance to carry over, and a non-adaptive random walk cannot "
                    "recover from a wrong scale: set proposal_sd_or_cov or use adaptive=True"
                )
            sd_or_cov = _default_adaptive_random_walk_covariance(prior, conf.no_parameters, stage_index)
        my_Prop.set_covariance(sd_or_cov=sd_or_cov)

    # Generalises the former Hamiltonian-only / HamiltonianInfinite-only asserts (B3) to any
    # proposal that needs gradients, including a BlockProposal whose sub-proposals include a
    # Hamiltonian one (BlockProposal.__init__ sets its own needs_gradients to True iff any
    # sub-proposal needs gradients). Same message and exception type as before: one code path.
    assert not getattr(my_Prop, "needs_gradients", False) or conf.use_surrogate_gradients, \
        "Hamiltonian proposals need use_surrogate_gradients=True (it may have been disabled by SamplingFramework, see warnings)"
    return my_Prop


__all__ = ["build_proposal"]
