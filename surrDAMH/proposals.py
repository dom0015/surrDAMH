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

import numpy as np
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
        beta: step in ``(0, 1]``; ``None`` (default): the value tuned by the previous adaptive
            pCN stage, else 0.5. ``beta=1`` proposes an independent draw from the prior, so the
            acceptance probability is the likelihood ratio (a "prior-draw" component, e.g. inside
            a ``Mixture``; only sensible in low dimension). An adaptive pCN keeps ``beta`` in
            ``(0, 1)``.
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
        tempered_fraction: ``f`` in ``[0, 0.5]`` (C3, 2026-10-12): share of each trajectory's
            ``L`` steps tempered at EACH end, ``n_t = min(floor(f L), L // 2)`` (with jitter, ``L``
            is the trajectory's own count). ``0.0`` (default) = off, bit-identical to no tempering.
        tempered_beta: ``beta`` in ``(0, 1]``: inverse temperature reached between heating and
            cooling; ``1.0`` (default) = off. Give both or neither (``f > 0`` iff ``beta < 1``).
        likelihood_weight: ``tau`` in ``[0, 1]`` (flattened likelihood, note 32 §1): every
            momentum kick of the trajectory uses ``tau * grad log L`` instead of ``grad log L``
            (the prior part, the position updates and the flip are unchanged); the map stays
            volume-preserving and reversible and the acceptance test uses the exact energy, so the
            posterior is preserved. ``1.0`` (default) = off, bit-identical; ``tau = 0`` = prior
            dynamics (with ``integrator="dimension_robust"``, ``mass=1`` and
            ``integration_time=pi/2`` an independent prior draw). Like a tempered proposal it does
            not adapt its step (``adaptive=True`` raises): use it as a pinned ``Mixture``
            component with ``adaptive=False``. May be combined with the tempered fields.

    Tempered trajectory (Neal 2011, §5.5.2; note 30): the leapfrog runs on the ordinary
    gradient, but the momentum is multiplied by ``sqrt(alpha)``, ``alpha = beta^(-1/n_t)``,
    before each of the first ``n_t`` position updates and divided by it after each of the last
    ``n_t``; the middle of the trajectory thus moves at kinetic temperature ``1/beta`` (enough
    to climb a barrier of order ``log(1/beta)`` nats), while the map stays volume-preserving
    and reversible, so the acceptance test is unchanged. A trajectory with ``L < 2`` steps gets
    ``n_t = 0`` and is not tempered; such trajectories are counted (``notes`` column
    ``tempered_skipped``, stage-end print). A tempered proposal does not adapt its step
    (``adaptive=True`` raises): use it as a pinned component of a ``Mixture`` next to an
    adaptive Hamiltonian, with ``adaptive=False`` (it then takes the carried ``step_size``) and
    a 2-4 times longer ``integration_time``::

        Mixture([Hamiltonian(integrator="dimension_robust", integration_time=3),
                 Hamiltonian(integrator="dimension_robust", integration_time=9, adaptive=False,
                             tempered_fraction=0.3, tempered_beta=0.1)],
                probabilities=[0.8, 0.2])
    """
    step_size: float | None = None
    num_steps: int | None = None
    mass: float | npt.ArrayLike | Literal["covariance"] = 1.0
    integrator: Literal["leapfrog", "dimension_robust"] = "leapfrog"
    adaptive: bool | None = None
    target_rate: float | None = None
    integration_time: float | None = None
    integration_time_jitter: float = 0.0
    tempered_fraction: float = 0.0
    tempered_beta: float = 1.0
    likelihood_weight: float = 1.0

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
        fraction, beta = float(self.tempered_fraction), float(self.tempered_beta)
        if not 0.0 <= fraction <= 0.5:
            raise ValueError(f"Hamiltonian.tempered_fraction must be in [0, 0.5], got {self.tempered_fraction!r}")
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"Hamiltonian.tempered_beta must be in (0, 1], got {self.tempered_beta!r}")
        if (fraction > 0.0) != (beta < 1.0):
            raise ValueError("Hamiltonian: tempered_fraction > 0 and tempered_beta < 1 go together (got "
                             f"tempered_fraction={self.tempered_fraction!r}, tempered_beta={self.tempered_beta!r}); "
                             "give both to temper the trajectory, or neither")
        if fraction > 0.0 and self.adaptive:
            raise ValueError("Hamiltonian: a tempered component does not adapt its step: pin step_size or set "
                             "adaptive=False; see note 30 §4")
        weight = float(self.likelihood_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Hamiltonian.likelihood_weight must be in [0, 1], got {self.likelihood_weight!r}")
        if weight < 1.0 and self.adaptive:
            raise ValueError("Hamiltonian: a flattened-likelihood component does not adapt its step: pin step_size "
                             "or set adaptive=False; see note 32 §1.2")
        _warn_unused_target_rate(self)


def _check_probabilities(name: str, probabilities, count: int) -> None:
    """``probabilities`` is ``None`` or ``count`` positive finite numbers (normalised at build time)."""
    if probabilities is None:
        return
    values = list(probabilities)
    if len(values) != count:
        raise ValueError(f"{name}: {len(values)} probabilities for {count} proposals")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
                or not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name}: every probability must be a positive finite number, got {probabilities!r}")


@dataclass
class Block:
    """
    Updates one group of parameters at a time, each group with its own proposal.

    Args:
        groups: parameter indices of each group, e.g. ``[[0, 1], [2, 3, 4]]``; the groups
            must cover all parameters without overlap.
        proposals: one ``RandomWalk``/``PCN``/``Hamiltonian`` per group (no ``Block`` or
            ``Mixture``). Each is built in its group's dimension and drawn with its own
            per-chain stream.
        probabilities: probability of drawing each group per (outer) iteration; ``None``
            (default) = uniform. Positive numbers, normalised when the proposal is built.

    Adaptation (C2, 2026-10-12): a sub-proposal MAY adapt (its step left ``None``, or
    ``adaptive=True``); ``adapt()`` reaches only the sub-proposal of the group that proposed. A
    ``Block`` is adaptive iff one of its sub-proposals is. Its tuned values are pooled over the
    chains at the end of the stage but a Block hands NOTHING over to the following stages: its
    sub-proposals live in group dimension, so their values cannot seed a full-dimension
    proposal. An adaptive sub-proposal starts from its default step (prior-derived for a random
    walk, 0.5 for pCN, 0.1 for Hamiltonian), never from an earlier stage's carry-over.
    """
    groups: list[list[int]] = field(default_factory=list)
    proposals: list[Union[RandomWalk, PCN, Hamiltonian]] = field(default_factory=list)
    probabilities: list[float] | None = None

    def __post_init__(self) -> None:
        if len(self.groups) != len(self.proposals):
            raise ValueError(f"Block: {len(self.groups)} groups but {len(self.proposals)} proposals")
        if not self.groups:
            raise ValueError("Block needs at least one group")
        if any(isinstance(p, (Block, Mixture)) for p in self.proposals):
            raise ValueError("Block proposals cannot be nested (no Block or Mixture inside a Block)")
        _check_probabilities("Block", self.probabilities, len(self.proposals))

    @property
    def adaptive(self) -> bool:
        """True iff any sub-proposal adapts (C2; a Block of fixed sub-proposals is not adaptive)."""
        return any(bool(p.adaptive) for p in self.proposals)

    @property
    def needs_gradients(self) -> bool:
        return any(p.needs_gradients for p in self.proposals)


@dataclass
class Mixture:
    """
    Random choice between several full-dimension proposals (C2, 2026-10-12): every (outer)
    iteration draws one component with the given probabilities and that component proposes
    the move of ALL parameters (in a DAMH stage: the whole sub-chain of that iteration). A
    fixed-weight mixture of reversible kernels is reversible, so the target is unchanged.

    Args:
        proposals: two or more ``RandomWalk``/``PCN``/``Hamiltonian`` specs (no ``Block`` or
            ``Mixture``). Each is built exactly as it would be as the stage's only proposal: it
            consumes the carried steps of earlier stages, a ``Hamiltonian(mass="covariance")``
            its mass from the chain's statistics, a ``PCN`` needs a Gaussian prior and a
            ``Hamiltonian`` needs ``use_surrogate_gradients``.
        probabilities: probability of each component; ``None`` (default) = uniform. Positive
            numbers, normalised when the proposal is built.

    Adaptation: a component adapts as on its own (step ``None`` = adaptive), but only on the
    iterations it proposed; ``adapt()`` reaches only that component. The mixture is adaptive iff
    one of its components is. At the end of the stage the adaptive components are pooled over
    the chains and HAND OVER their tuned values exactly like bare proposals of their type (a
    following ``RandomWalk()`` stage starts from the mixture's adaptive random walk's scale).

    V1 restriction: at most one ADAPTIVE component per type (e.g. not two ``RandomWalk()``);
    their carry-over keys would collide, so building such a mixture raises ``ValueError``. Pin
    the step of one of them (it is then fixed) or use different types. A typical use is a fixed
    safety kernel next to an adaptive one::

        Mixture([RandomWalk(scale=0.05), RandomWalk()], probabilities=[0.1, 0.9])

    Adaptation trace: ``adaptive_stats/<stage>/rank%04d.csv`` records the FIRST adaptive
    component only (a warning names it when several adapt); the manifest's
    ``proposal_resolution`` records its index as ``adaptive_stats_component``.
    """
    proposals: list[Union[RandomWalk, PCN, Hamiltonian]] = field(default_factory=list)
    probabilities: list[float] | None = None

    def __post_init__(self) -> None:
        self.proposals = list(self.proposals)
        if len(self.proposals) < 2:
            raise ValueError(f"Mixture needs at least two proposals, got {len(self.proposals)}")
        for p in self.proposals:
            if isinstance(p, (Block, Mixture)):
                raise ValueError("Mixture components cannot be a Block or a Mixture (no nesting)")
            if not isinstance(p, (RandomWalk, PCN, Hamiltonian)):
                raise TypeError(f"Mixture components must be RandomWalk, PCN or Hamiltonian, got {type(p).__name__}")
        _check_probabilities("Mixture", self.probabilities, len(self.proposals))

    @property
    def adaptive(self) -> bool:
        """True iff any component adapts."""
        return any(bool(p.adaptive) for p in self.proposals)

    @property
    def needs_gradients(self) -> bool:
        return any(p.needs_gradients for p in self.proposals)


ProposalSpec = Union[RandomWalk, PCN, Hamiltonian, Block, Mixture]

#: Name of the step field of each spec type (what "adapt unless pinned" looks at, and the
#: key under which an adaptive stage hands its tuned value to the following stages).
STEP_FIELD: dict[type, str] = {RandomWalk: "scale", PCN: "beta", Hamiltonian: "step_size"}

__all__ = ["RandomWalk", "PCN", "Hamiltonian", "Block", "Mixture", "ProposalSpec", "STEP_FIELD"]
