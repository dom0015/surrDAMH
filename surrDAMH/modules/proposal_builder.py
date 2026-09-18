#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-stage proposal construction, shared by the MPI sampler process and the
standalone local runner.

This module is intentionally MPI-free: it only depends on ``Stage``, the
configuration attributes ``no_parameters`` / ``use_surrogate_gradients`` and the
prior distribution, so that ``surrDAMH.runner_local`` can reuse it without
importing ``surrDAMH.process_SAMPLER`` (which imports ``mpi4py``).

The logic is a verbatim extraction of the block that used to live inline in
``process_SAMPLER.run_SAMPLER``; behaviour (including seed usage) is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy.typing as npt

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


def build_proposal(stage: Stage, conf: "Configuration", prior: Distribution, seed: int,
                   prev_cov: npt.ArrayLike | None = None, stage_index: int | None = None) -> Proposal:
    """
    Build the proposal distribution of one sampling stage.

    Args:
        stage: the stage whose proposal is constructed.
        conf: configuration (only ``no_parameters`` and ``use_surrogate_gradients`` are used).
        prior: prior distribution (only used by the pCN proposal).
        seed: seed of the proposal's random generator.
        prev_cov: proposal covariance carried over from the previous stage; used
            when ``stage.proposal_sd_or_cov`` is not specified.
        stage_index: index of the stage in the list of stages (only used in error messages).

    Returns:
        The proposal instance for this stage.
    """
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
        my_Prop = proposals.PCN(
            no_parameters=conf.no_parameters,
            beta=stage.pcn_beta,
            prior_mean=prior.mean,
            prior_sd_or_cov=prior.get_covariance(),
            seed=seed
        )
    elif stage.proposal_type == "Hamiltonian":
        hamiltonian_sd_or_cov = stage.proposal_sd_or_cov
        if hamiltonian_sd_or_cov is None:
            hamiltonian_sd_or_cov = 1.0
        my_Prop = proposals.Hamiltonian(
            no_parameters=conf.no_parameters,
            seed=seed,
            num_steps=stage.hamiltonian_num_steps,
            step_size=stage.hamiltonian_step_size,
            sd_or_cov=hamiltonian_sd_or_cov,
        )
    elif stage.proposal_type == "HamiltonianInfinite":
        hamiltonian_sd_or_cov = stage.proposal_sd_or_cov
        if hamiltonian_sd_or_cov is None:
            hamiltonian_sd_or_cov = 1.0
        my_Prop = proposals.HamiltonianInfinite(
            no_parameters=conf.no_parameters,
            seed=seed,
            num_steps=stage.hamiltonian_num_steps,
            step_size=stage.hamiltonian_step_size,
            sd_or_cov=hamiltonian_sd_or_cov,
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
            # G1 (2026-09-17): the three Stage.adaptive_* fields are now honoured. Each is
            # only forwarded when the stage sets it, so a stage that leaves them None gets
            # GaussRandomWalk_adaptive's own defaults -- the values that were hard-coded
            # before G1 (target_rate=0.25, corr_limit=0.3, unbounded sample history) -- and
            # is therefore bit-identical to the pre-G1 behaviour.
            adaptive_kwargs: dict = {}
            if stage.adaptive_target_rate is not None:
                adaptive_kwargs["target_rate"] = stage.adaptive_target_rate
            if stage.adaptive_corr_limit is not None:
                adaptive_kwargs["corr_limit"] = stage.adaptive_corr_limit
            if stage.adaptive_sample_limit is not None:
                adaptive_kwargs["sample_limit"] = stage.adaptive_sample_limit
            my_Prop = proposals.GaussRandomWalk_adaptive(no_parameters=conf.no_parameters, seed=seed,
                                                         **adaptive_kwargs)
        else:
            my_Prop = proposals.GaussRandomWalk(no_parameters=conf.no_parameters, seed=seed)
        if stage.proposal_sd_or_cov is None:
            assert prev_cov is not None, f"proposal sd/cov not specified for stage {stage_index}"
            my_Prop.set_covariance(sd_or_cov=prev_cov)
        else:
            my_Prop.set_covariance(sd_or_cov=stage.proposal_sd_or_cov)

    # Generalises the former Hamiltonian-only / HamiltonianInfinite-only asserts (B3) to any
    # proposal that needs gradients, including a BlockProposal whose sub-proposals include a
    # Hamiltonian one (BlockProposal.__init__ sets its own needs_gradients to True iff any
    # sub-proposal needs gradients). Same message and exception type as before: one code path.
    assert not getattr(my_Prop, "needs_gradients", False) or conf.use_surrogate_gradients, \
        "Hamiltonian proposals need use_surrogate_gradients=True (it may have been disabled by SamplingFramework, see warnings)"
    return my_Prop


__all__ = ["build_proposal"]
