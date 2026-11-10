#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The exact prior-proposal step of a DAMH stage (2026-10-23, paper §6.5;
``Stage.exact_step_proposal="prior"``).

``PriorIndependence`` is the runtime proposal of the S2 exact step when the stage asks for the
prior: an independence sampler whose proposal IS the prior, drawn in the chain's internal space.
It is built by ``proposal_builder.build_exact_step_proposal(kind="prior")`` and used only by
``Algorithm_DAMH._exact_iteration``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from surrDAMH.distributions.parent import Distribution, rvs_with_generator
from surrDAMH.modules.proposals import Proposal


class PriorIndependence(Proposal):
    """
    Independence proposal ``y ~ prior`` (2026-10-23, paper §6.5), drawn in the INTERNAL space with
    ``rvs_with_generator(prior, np.random.default_rng(seed))``; the current state is ignored.

    Acceptance: with target ``pi(y) ∝ L(y) prior(y)`` and proposal density ``prior(y)``, the MH
    ratio is ``L(y) prior(y) prior(x) / (L(x) prior(x) prior(y)) = L(y) / L(x)``: the prior cancels
    for every shipped prior, whose internal ``logpdf`` and ``rvs`` describe the same distribution by
    design (standard normal for ``Normal``/``PriorIndependentComponents``, the identity
    parametrisation for ``GaussianMixture`` and ``FromScipy``). ``get_log_acceptance_probability``
    therefore returns ``(log L(y) - log L(x), 0.0)``, the convention of ``PCN``.

    With a prior bound (S0, ``Problem(prior_bound=...)``) the target is the prior truncated to the
    box; a draw outside it is rejected unevaluated by ``_exact_iteration`` (tag ``-3``), which is the
    MH kernel of the truncated target with this untruncated proposal (``min{1, 1_box(y) L(y)/L(x)}``).

    Nothing adapts (``adapt`` is the base-class no-op), and ``needs_gradients`` is ``False``.
    ``resolution = {"exact_step_proposal": "prior"}`` is merged into the stage's entry of the
    manifest's ``proposal_resolution`` by both runners.
    """

    def __init__(self, prior: Distribution, no_parameters: int, seed: int = 0) -> None:
        super().__init__()
        self.prior = prior
        self.no_parameters = int(no_parameters)
        self._generator = np.random.default_rng(seed)
        self.resolution = {"exact_step_proposal": "prior"}

    def propose_sample(self, current_sample: npt.NDArray) -> npt.NDArray:
        # GaussianMixture.rvs(n=1, generator) returns shape (1, d); a scalar FromScipy prior a 0-d value
        draw = np.asarray(rvs_with_generator(self.prior, self._generator), dtype=float).reshape(-1)
        if draw.size != self.no_parameters:
            raise ValueError(f"PriorIndependence: the prior's rvs() returned {draw.size} values, expected "
                             f"{self.no_parameters}")
        return draw

    def get_log_acceptance_probability(self, log_likelihood_proposed: float, log_likelihood_current: float,
                                       log_prior_proposed: float, log_prior_current: float) -> tuple[float, float]:
        # the prior cancels against the proposal density (see the class docstring)
        return log_likelihood_proposed - log_likelihood_current, 0.0

    def reseed(self, seed: int) -> None:
        """A fresh ``np.random.default_rng(seed)`` (the stream is a ``Generator``, not a ``RandomState``)."""
        self._generator = np.random.default_rng(seed)


__all__ = ["PriorIndependence"]
