#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Installation rules for the surrogate versions a chain receives during a stage
(2026-10-23, paper §6.5; ``pdf_reports/adaptive_nn_ergodicity_paper/main.tex`` def:randomised and def:averaged).

A DAMH stage with ``surrogate_model_updates=True`` (DAMH-SMU) keeps receiving retrained
surrogates from the collector. Without a rule (``Stage.installation=None``, the default) every
received version is installed at the next poll, once per sub-chain -- the historical behaviour.
The two rules below make the installed surrogate satisfy diminishing adaptation::

    from surrDAMH.installation import RandomisedInstallation, AveragedInstallation
    Stage(algorithm="DAMH", installation=RandomisedInstallation(n_lambda=50, a_lambda=0.5), ...)
    Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5),
          installation=AveragedInstallation(j0=5, a_zeta=0.6, weight_box=10.0), ...)

* ``RandomisedInstallation``: before outer iteration ``n + 1`` the chain draws ``U_n`` from its
  own installation stream and installs the NEWEST version it has received, together with the
  current values of its adaptive sub-chain parameters, iff ``U_n < lambda_n`` (and a version is
  pending); skipped versions are dropped. Nothing is assumed about the surrogate or the sub-chain.
* ``AveragedInstallation``: every received version (projected componentwise onto
  ``[-weight_box, weight_box]`` when ``weight_box`` is set) is averaged into the running average
  with gain ``zeta_j`` and the average is installed at the next poll. Needs fixed sub-chain
  parameters and an averageable evaluator (the neural network). ``AveragedInstallation(j0=1,
  a_zeta=1.0)`` installs the running mean of the RECEIVED versions; the legacy flag
  ``polyak_averaging=True`` keeps the network in use at the stage start in that mean (weight
  ``1/k``) and stays as it is.

Semantics in the sampler: ``docs/stages.md``, "Installation rules (2026-10-23)".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class RandomisedInstallation:
    """
    Randomised installation (paper def:randomised, 2026-10-23, paper §6.5): install the newest received
    surrogate version before outer iteration ``n + 1`` with probability
    ``lambda_n = min{1, (n / n_lambda)^(-a_lambda)}`` (``lambda_n = 1`` for ``n <= n_lambda``).

    Args:
        n_lambda: ``>= 1``, finite: the number of outer iterations during which every pending
            version is installed at once.
        a_lambda: in ``(0, 1]``: the decay exponent; ``a_lambda <= 1`` keeps ``sum lambda_n``
            infinite, so the chain keeps installing as long as new versions arrive.
    """

    n_lambda: float
    a_lambda: float

    def __post_init__(self):
        n_lambda = _real(self.n_lambda, "RandomisedInstallation.n_lambda")
        a_lambda = _real(self.a_lambda, "RandomisedInstallation.a_lambda")
        if not (math.isfinite(n_lambda) and n_lambda >= 1.0):
            raise ValueError(f"RandomisedInstallation.n_lambda must be a finite number >= 1, got {self.n_lambda!r}")
        if not (0.0 < a_lambda <= 1.0):
            raise ValueError(f"RandomisedInstallation.a_lambda must be in (0, 1], got {self.a_lambda!r}")
        object.__setattr__(self, "n_lambda", n_lambda)
        object.__setattr__(self, "a_lambda", a_lambda)

    def probability(self, n: int) -> float:
        """``lambda_n = min{1, (n / n_lambda)^(-a_lambda)}``; ``1.0`` for ``n <= 0``."""
        if n <= 0:
            return 1.0
        return min(1.0, (float(n) / self.n_lambda) ** (-self.a_lambda))


@dataclass(frozen=True)
class AveragedInstallation:
    """
    Averaged installation (paper def:averaged, 2026-10-23, paper §6.5): the ``j``-th version received
    in the stage enters the running average with gain ``zeta_j = min{1, (j / j0)^(-a_zeta)}``,
    ``average_j = average_{j-1} + zeta_j * (version_j - average_{j-1})``, and the average is
    installed. ``average_0`` is the network in use at the stage start (projected onto the weight
    box); ``zeta_1 = 1``, so the first received version replaces it.

    Args:
        j0: integer ``>= 1``: the warm-up, ``zeta_j = 1`` (no averaging) for ``j <= j0``.
        a_zeta: in ``(0, 1]``: the decay exponent of the gain; ``j0 = 1, a_zeta = 1`` is the
            running mean of the received versions (``Stage.polyak_averaging=True`` is the legacy
            running mean that also keeps the stage-start network in, with weight ``1/k``).
        weight_box: ``None`` (no projection) or ``omega > 0`` finite: every received version's
            model parameters are clipped to ``[-omega, omega]`` componentwise before averaging
            (the error-model entries of ``Stage.surrogate_error_model`` are not clipped).
    """

    j0: int = 1
    a_zeta: float = 1.0
    weight_box: float | None = None

    def __post_init__(self):
        if isinstance(self.j0, (bool, np.bool_)) or not isinstance(self.j0, (int, np.integer)) or self.j0 < 1:
            raise ValueError(f"AveragedInstallation.j0 must be an integer >= 1, got {self.j0!r}")
        object.__setattr__(self, "j0", int(self.j0))
        a_zeta = _real(self.a_zeta, "AveragedInstallation.a_zeta")
        if not (0.0 < a_zeta <= 1.0):
            raise ValueError(f"AveragedInstallation.a_zeta must be in (0, 1], got {self.a_zeta!r}")
        object.__setattr__(self, "a_zeta", a_zeta)
        if self.weight_box is not None:
            omega = _real(self.weight_box, "AveragedInstallation.weight_box")
            if not (math.isfinite(omega) and omega > 0.0):
                raise ValueError(f"AveragedInstallation.weight_box must be None or a positive finite number, "
                                 f"got {self.weight_box!r}")
            object.__setattr__(self, "weight_box", omega)

    def gain(self, j: int) -> float:
        """``zeta_j = min{1, (j / j0)^(-a_zeta)}`` for ``j >= 1``."""
        if j < 1:
            raise ValueError(f"AveragedInstallation.gain: j must be >= 1, got {j!r}")
        return min(1.0, (float(j) / self.j0) ** (-self.a_zeta))


__all__ = ["RandomisedInstallation", "AveragedInstallation"]
