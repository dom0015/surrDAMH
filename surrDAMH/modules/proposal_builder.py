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
each proposal spec class (``surrDAMH.proposals``, 2026-09-21).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from surrDAMH.distributions.independent_components import PriorIndependentComponents
from surrDAMH.distributions.normal import Normal, StandardizedNormal
from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules import proposals
from surrDAMH.modules.proposals import Proposal
from surrDAMH.proposals import PCN, Block, Hamiltonian, RandomWalk
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
    return isinstance(prior, (Normal, StandardizedNormal, PriorIndependentComponents))


def _default_adaptive_random_walk_covariance(prior: Distribution, no_parameters: int,
                                              stage_index: int | None):
    """
    Starting covariance of an adaptive random walk whose stage left ``RandomWalk.scale=None``
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
        print(f"Warning: stage {stage_index}: RandomWalk.scale is None and the prior of type "
              f"'{type(prior).__name__}' has no get_covariance(); the adaptive random walk starts "
              "from sd 1.0 (only the warm-up depends on this starting point)")
        return 1.0
    if prior_sd_or_cov.ndim == 2:
        return scale_factor * prior_sd_or_cov  # covariance matrix
    # scalar or vector of STANDARD DEVIATIONS (the Distribution.get_covariance() contract)
    return np.sqrt(scale_factor) * prior_sd_or_cov


def _slice_prior(prior: Distribution, group: list[int]):
    """``(mean, sd_or_cov)`` of ``prior`` restricted to the parameters in ``group`` (for a pCN
    sub-proposal of a ``Block``): a vector of sds is indexed, a covariance matrix is sub-blocked."""
    mean = np.asarray(prior.mean, dtype=float)[group]
    cov = np.asarray(prior.get_covariance(), dtype=float)
    if cov.ndim == 2:
        return mean, cov[np.ix_(group, group)]
    return mean, cov[group]


def _build_simple(spec, no_parameters: int, prior: Distribution, seed: int, carried: dict,
                  conf, stage_index, prior_mean=None, prior_sd_or_cov=None,
                  warn_fixed_default: bool = True) -> Proposal:
    """One runtime proposal for a ``RandomWalk``/``PCN``/``Hamiltonian`` spec."""
    adaptive_kwargs: dict = {}
    if spec.adaptive and spec.target_rate is not None:
        adaptive_kwargs["target_rate"] = spec.target_rate

    if isinstance(spec, PCN):
        if not _prior_is_gaussian(prior):
            raise ValueError(
                "PCN proposal requires a Gaussian internal prior: "
                "PCN.get_log_acceptance_probability drops the prior ratio, which is only "
                "correct when the internal prior is exactly N(prior.mean, "
                "prior.get_covariance()). Got prior of type "
                f"'{type(prior).__name__}'. Use surrDAMH.distributions.Normal or "
                "PriorIndependentComponents (internal prior is standard normal by design) "
                "instead; FromScipy and GaussianMixture priors are not accepted for PCN."
            )
        beta = spec.beta
        if beta is None:
            beta = carried.get("beta", 0.5)
        pcn_class = proposals.PCN_adaptive if spec.adaptive else proposals.PCN
        return pcn_class(
            no_parameters=no_parameters,
            beta=beta,
            prior_mean=prior.mean if prior_mean is None else prior_mean,
            prior_sd_or_cov=prior.get_covariance() if prior_sd_or_cov is None else prior_sd_or_cov,
            seed=seed,
            **adaptive_kwargs,
        )
    if isinstance(spec, Hamiltonian):
        # The carried random-walk covariance is deliberately NOT used as the mass: `16` §5
        # item 5 measured M = cov as a poor mass (M = cov^-1 is the good one) and carrying the
        # inverse needs the step size re-adapted at fixed integration time -- an open decision.
        step_size = spec.step_size
        if step_size is None:
            step_size = carried.get("step_size", 0.1)
        if spec.integrator == "leapfrog":
            hamiltonian_class = proposals.Hamiltonian_adaptive if spec.adaptive else proposals.Hamiltonian
        else:
            hamiltonian_class = (proposals.HamiltonianInfinite_adaptive if spec.adaptive
                                 else proposals.HamiltonianInfinite)
        return hamiltonian_class(
            no_parameters=no_parameters,
            seed=seed,
            num_steps=spec.num_steps,
            step_size=step_size,
            sd_or_cov=spec.mass,
            **adaptive_kwargs,
        )
    assert isinstance(spec, RandomWalk), type(spec)
    if spec.adaptive:
        my_Prop = proposals.GaussRandomWalk_adaptive(no_parameters=no_parameters, seed=seed, **adaptive_kwargs)
    else:
        my_Prop = proposals.GaussRandomWalk(no_parameters=no_parameters, seed=seed)
    sd_or_cov = spec.scale
    if sd_or_cov is None:
        sd_or_cov = carried.get("scale")
    if sd_or_cov is None:
        if not spec.adaptive and warn_fixed_default:
            # 2026-09-21 (author decision): warn instead of raising -- a fixed random walk
            # cannot recover from a wrong scale, so the user should know it was guessed
            print(f"Warning: stage {stage_index}: RandomWalk(adaptive=False) without a scale and no "
                  "earlier adaptive stage provides one; using the prior covariance scaled by "
                  "2.38^2/d for the WHOLE stage (give scale=..., or leave adaptive on to tune it)",
                  flush=True)
        if prior_sd_or_cov is None:
            sd_or_cov = _default_adaptive_random_walk_covariance(prior, no_parameters, stage_index)
        else:  # a Block sub-proposal: the sliced prior
            scale_factor = 2.38 ** 2 / no_parameters
            arr = np.asarray(prior_sd_or_cov, dtype=float)
            sd_or_cov = scale_factor * arr if arr.ndim == 2 else np.sqrt(scale_factor) * arr
    my_Prop.set_covariance(sd_or_cov=sd_or_cov)
    return my_Prop


def build_proposal(stage: Stage, conf: "Configuration", prior: Distribution, seed: int,
                   carried: dict | None = None, stage_index: int | None = None) -> Proposal:
    """
    Build the runtime proposal of one sampling stage from its ``stage.proposal`` spec
    (``surrDAMH.proposals``: ``RandomWalk``/``PCN``/``Hamiltonian``/``Block``).

    Args:
        stage: the stage whose proposal is constructed.
        conf: configuration (only ``no_parameters`` and ``use_surrogate_gradients`` are used).
        prior: prior distribution (used by the pCN proposal, and for the starting covariance of
            a random walk that has no ``scale`` and nothing to carry over).
        seed: seed of the proposal's random generator.
        carried: merged carry-over of every previous adaptive stage of this run
            (``Proposal.carry_over()`` keyed by the spec's step field, 2026-09-21: ``"scale"``
            from an adaptive random walk, ``"beta"`` from an adaptive pCN stage (default 0.5),
            ``"step_size"`` from an adaptive Hamiltonian stage (default 0.1)). It supplies the
            value of every step field left ``None``. A random walk with ``scale=None`` and
            nothing to carry over starts from the prior-derived default of
            ``_default_adaptive_random_walk_covariance``; a non-adaptive one additionally prints
            a warning, since that guess is then used for the whole stage.
        stage_index: index of the stage in the list of stages (only used in messages).

    Returns:
        The proposal instance for this stage.

    Raises:
        ValueError: ``PCN`` with a non-Gaussian internal prior; a ``Block`` whose sub-proposal
            is adaptive (a block proposal never adapts, give every sub-proposal its step);
            a gradient-needing proposal without ``conf.use_surrogate_gradients`` in effect.
    """
    carried = {} if carried is None else carried
    spec = stage.proposal

    if isinstance(spec, Block):
        sub_proposals = []
        for k, (group, sub_spec) in enumerate(zip(spec.groups, spec.proposals)):
            if sub_spec.adaptive:
                # G2 (2026-09-17): a BlockProposal never calls adapt() on its members, so an
                # adaptive sub-proposal would silently stay at its start value. Fail fast.
                raise ValueError(
                    f"stage {stage_index}: adaptive covariance reduction is not defined for block "
                    f"proposals (sub-proposal {k}: {sub_spec!r}); give every sub-proposal its step "
                    "(scale/beta/step_size) or set adaptive=False on it"
                )
            group = [int(i) for i in group]
            mean_g, cov_g = (None, None)
            if isinstance(sub_spec, (PCN, RandomWalk)) and _prior_is_gaussian(prior):
                mean_g, cov_g = _slice_prior(prior, group)
            sub_proposals.append(_build_simple(sub_spec, no_parameters=len(group), prior=prior,
                                               seed=proposals.subproposal_seed(seed, k), carried={},
                                               conf=conf, stage_index=stage_index,
                                               prior_mean=mean_g, prior_sd_or_cov=cov_g,
                                               warn_fixed_default=False))
        my_Prop = proposals.BlockProposal(
            no_parameters=conf.no_parameters,
            list_of_groups=[list(g) for g in spec.groups],
            list_of_proposals=sub_proposals,
            seed=seed
        )
    else:
        my_Prop = _build_simple(spec, no_parameters=conf.no_parameters, prior=prior, seed=seed,
                                carried=carried, conf=conf, stage_index=stage_index)

    # Generalises the former Hamiltonian-only / HamiltonianInfinite-only asserts (B3) to any
    # proposal that needs gradients, including a BlockProposal whose sub-proposals include a
    # Hamiltonian one (BlockProposal.__init__ sets its own needs_gradients to True iff any
    # sub-proposal needs gradients). Same message and exception type as before: one code path.
    assert not getattr(my_Prop, "needs_gradients", False) or conf.use_surrogate_gradients, \
        "Hamiltonian proposals need use_surrogate_gradients=True (it may have been disabled by Problem.run_sampling, see warnings)"
    return my_Prop


__all__ = ["build_proposal"]
