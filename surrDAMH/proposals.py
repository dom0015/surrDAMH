#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proposal specifications for ``surrDAMH.stages.Stage(proposal=...)`` (2026-09-21).

These are small, picklable descriptions of *which* proposal a stage uses and with which
settings. The sampler turns them into the runtime classes of ``surrDAMH.modules.proposals``
(``modules.proposal_builder.build_proposal``), one fresh object per stage and per chain, with
the chain's own seed. A spec holds no random state and can be reused in several stages.

The step of every proposal (``scale`` / ``beta`` / ``step_size``) is optional. Leave it
``None`` and the stage uses the value tuned by the previous adaptive stage of the same
proposal type, else that proposal's default; ``adaptive`` then defaults to ``True``
("adapt unless the step is pinned"). Give the step and the proposal is fixed unless you also
pass ``adaptive=True``, which then tunes it starting from the given value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import numpy.typing as npt


def _warn_unused_target_rate(spec) -> None:
    """``target_rate`` only steers the adaptation; with ``adaptive=False`` it is silently unused
    by the builder, so say so here, when the user can still see the line that set it (2026-09-22)."""
    if spec.target_rate is not None and not spec.adaptive:
        print(f"Warning: {type(spec).__name__}(target_rate={spec.target_rate}, adaptive=False): target_rate "
              "is only used when the proposal adapts; drop it or set adaptive=True", flush=True)


@dataclass
class RandomWalk:
    """
    Gaussian random walk (Metropolis) proposal, the default of every stage.

    Args:
        scale: step in the prior's internal space: a scalar sd, a vector of per-parameter sds,
            or a covariance matrix. ``None`` (default): the covariance tuned by the previous
            adaptive random-walk stage, else the prior covariance scaled by ``2.38^2/d``.
        adaptive: tune ``scale`` online (covariance estimate plus acceptance-rate scaling).
            ``None`` (default) = ``True`` iff ``scale`` is ``None``. ``adaptive=True`` with a
            ``scale`` starts tuning from that value; ``adaptive=False`` without a ``scale``
            runs the whole stage at the default (a warning is printed).
        target_rate: acceptance rate the tuning aims at; ``None`` = 0.234.
    """
    scale: float | npt.ArrayLike | None = None
    adaptive: bool | None = None
    target_rate: float | None = None

    needs_gradients = False  # class attribute, not a field

    def __post_init__(self) -> None:
        if self.adaptive is None:
            self.adaptive = self.scale is None
        _warn_unused_target_rate(self)


@dataclass
class PCN:
    """
    Preconditioned Crank-Nicolson proposal; needs a Gaussian internal prior
    (``Normal`` or ``PriorIndependentComponents``).

    Args:
        beta: step in ``(0, 1)``; ``None`` (default): the value tuned by the previous adaptive
            pCN stage, else 0.5.
        adaptive: tune ``beta`` online; ``None`` (default) = ``True`` iff ``beta`` is ``None``.
        target_rate: acceptance rate the tuning aims at; ``None`` = 0.234.
    """
    beta: float | None = None
    adaptive: bool | None = None
    target_rate: float | None = None

    needs_gradients = False

    def __post_init__(self) -> None:
        if self.adaptive is None:
            self.adaptive = self.beta is None
        _warn_unused_target_rate(self)


@dataclass
class Hamiltonian:
    """
    Hamiltonian (HMC) proposal driven by the surrogate's gradients; needs a surrogate that
    provides them and ``Configuration.use_surrogate_gradients=True``.

    Args:
        step_size: leapfrog step; ``None`` (default): the value tuned by the previous adaptive
            Hamiltonian stage, else 0.1.
        num_steps: leapfrog steps per proposal. ``None`` (default): 10, unless
            ``integration_time`` is given (giving both raises ``ValueError``).
        integration_time: trajectory length ``T`` in time, instead of ``num_steps``: every
            proposal uses ``num_steps = max(1, round(T / step_size))`` (at most 1000), recomputed
            whenever the dual averaging changes the step, so the trajectory length stays ``T``
            while the step adapts (C1, 2026-10-11).
        integration_time_jitter: ``j`` in ``[0, 1)``, only with ``integration_time`` (C1b,
            2026-10-11): every trajectory uses the length ``T u`` with ``u ~ Uniform(1 - j, 1 + j)``
            drawn from the proposal's own generator (``num_steps = max(1, round(T u / step_size))``,
            at most 1000). ``0.0`` (default) draws nothing. Valid because the length is drawn
            independently of the state (a mixture of reversible kernels is reversible); it removes
            the antithetic behaviour of a fixed ``T`` (e.g. the deficit of ESS for second moments).
        mass: mass matrix as a scalar, a vector of per-parameter values, or a matrix; 1.0 is
            the natural choice for the standard-normal internal prior. ``"covariance"``: the
            inverse of the chain's posterior covariance estimate (the regularised precision of
            the sample statistics carried from all previous stages and runs, pooled over the
            chains), resolved at the start of the stage and fixed within it; the identity (with
            a printed note) while fewer than 100 states have been seen.
        integrator: ``"leapfrog"`` = standard HMC. ``"dimension_robust"`` = the prior part of
            the dynamics is solved exactly (harmonic-oscillator rotation) and only the
            likelihood gradient is integrated numerically -- the splitting scheme of Beskos,
            Pinski, Sanz-Serna & Stuart (2011), recommended for a standard-normal internal
            prior; its step-size acceptance rate does not degrade as the dimension grows,
            unlike plain leapfrog.
        adaptive: tune ``step_size`` online (dual averaging); ``None`` (default) = ``True``
            iff ``step_size`` is ``None``. ``mass`` and ``num_steps`` are never tuned (with
            ``integration_time``, ``num_steps`` follows the step). A stage with ``step_size=None``
            whose mass equals the previous adaptive Hamiltonian stage's continues that stage's
            dual averaging instead of restarting it.
        target_rate: acceptance rate the tuning aims at; ``None`` = 0.8.
    """
    step_size: float | None = None
    num_steps: int | None = None
    mass: float | npt.ArrayLike | Literal["covariance"] = 1.0
    integrator: Literal["leapfrog", "dimension_robust"] = "leapfrog"
    adaptive: bool | None = None
    target_rate: float | None = None
    integration_time: float | None = None
    integration_time_jitter: float = 0.0

    needs_gradients = True

    def __post_init__(self) -> None:
        if self.integrator not in ("leapfrog", "dimension_robust"):
            raise ValueError(f"Hamiltonian.integrator must be 'leapfrog' or 'dimension_robust', got {self.integrator!r}")
        if isinstance(self.mass, str) and self.mass != "covariance":
            raise ValueError(f"Hamiltonian.mass must be a number, a vector, a matrix or 'covariance', got {self.mass!r}")
        if self.integration_time is not None:
            if self.num_steps is not None:
                raise ValueError("Hamiltonian: give either num_steps or integration_time, not both")
            if not float(self.integration_time) > 0.0:
                raise ValueError(f"Hamiltonian.integration_time must be positive, got {self.integration_time!r}")
        elif self.num_steps is None:
            self.num_steps = 10
        jitter = float(self.integration_time_jitter)
        if not 0.0 <= jitter < 1.0:
            raise ValueError(f"Hamiltonian.integration_time_jitter must be in [0, 1), got {self.integration_time_jitter!r}")
        if jitter > 0.0 and self.integration_time is None:
            raise ValueError("Hamiltonian: integration_time_jitter > 0 needs integration_time")
        if self.adaptive is None:
            self.adaptive = self.step_size is None
        _warn_unused_target_rate(self)


@dataclass
class Block:
    """
    Updates one group of parameters at a time, each group with its own proposal.

    Args:
        groups: parameter indices of each group, e.g. ``[[0, 1], [2, 3, 4]]``; the groups
            must cover all parameters without overlap.
        proposals: one ``RandomWalk``/``PCN``/``Hamiltonian`` per group. Their steps must be
            given (a block proposal does not adapt; ``adaptive=True`` inside a block raises).
    """
    groups: list[list[int]] = field(default_factory=list)
    proposals: list[Union[RandomWalk, PCN, Hamiltonian]] = field(default_factory=list)

    adaptive = False  # class attribute: a block never adapts

    def __post_init__(self) -> None:
        if len(self.groups) != len(self.proposals):
            raise ValueError(f"Block: {len(self.groups)} groups but {len(self.proposals)} proposals")
        if not self.groups:
            raise ValueError("Block needs at least one group")
        if any(isinstance(p, Block) for p in self.proposals):
            raise ValueError("Block proposals cannot be nested")

    @property
    def needs_gradients(self) -> bool:
        return any(p.needs_gradients for p in self.proposals)


ProposalSpec = Union[RandomWalk, PCN, Hamiltonian, Block]

#: Name of the step field of each spec type (what "adapt unless pinned" looks at, and the
#: key under which an adaptive stage hands its tuned value to the following stages).
STEP_FIELD: dict[type, str] = {RandomWalk: "scale", PCN: "beta", Hamiltonian: "step_size"}

__all__ = ["RandomWalk", "PCN", "Hamiltonian", "Block", "ProposalSpec", "STEP_FIELD"]
