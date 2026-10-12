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
from surrDAMH.modules.posterior_statistics import MASS_MIN_STATES, PosteriorStatistics
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


#: relative tolerance under which two resolved Hamiltonian masses count as "the same mass" (C1);
#: kept for reference -- the continuation decision uses DA_CONTINUE_MASS_TOLERANCE since C1b
MASS_UNCHANGED_RTOL = 1e-12
#: C1b (2026-10-11) placeholder: a following adaptive Hamiltonian stage CONTINUES the carried dual
#: averaging when the relative spectral-norm change of the mass ``||M_new - M_old||_2 / ||M_old||_2``
#: is at most this; otherwise it restarts from the rescaled carried step (``mass_change_step_factor``)
DA_CONTINUE_MASS_TOLERANCE = 0.2
#: C1b placeholder: restart offset after a large mass change, ``mu = log(DA_RESTART_FACTOR * eps0)``
#: (the fresh-recursion default is 10: the rescaled carried step is already close, so a small push)
DA_RESTART_FACTOR = 2.0


def _resolve_hamiltonian_mass(spec, no_parameters: int, statistics: PosteriorStatistics | None,
                              stage_index) -> tuple:
    """``(sd_or_cov for the runtime class, mass source string)`` of a ``Hamiltonian`` spec (C1).

    ``mass="covariance"``: the regularised precision ``statistics.precision()`` once the
    statistics hold ``MASS_MIN_STATES`` states, else the identity (``1.0``, i.e. exactly a
    ``mass=1.0`` stage) with a printed note. Any other mass is passed through unchanged."""
    if not (isinstance(spec.mass, str) and spec.mass == "covariance"):
        return spec.mass, "spec"
    if statistics is not None and statistics.is_informative(MASS_MIN_STATES):
        try:
            return statistics.precision(), f"covariance (n={statistics.n})"
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"Note: stage {stage_index}: Hamiltonian(mass='covariance'): the posterior statistics "
                  f"(n={statistics.n}) give no usable covariance ({exc}); identity mass used", flush=True)
            return 1.0, "identity (degenerate statistics)"
    n = 0 if statistics is None else statistics.n
    print(f"Note: stage {stage_index}: Hamiltonian(mass='covariance'): only {n} chain states seen "
          f"(< {MASS_MIN_STATES}); identity mass used for this stage", flush=True)
    return 1.0, "identity (no statistics)"


def _same_mass(mass_a, mass_b) -> bool:
    a, b = np.asarray(mass_a, dtype=float), np.asarray(mass_b, dtype=float)
    if a.shape != b.shape:
        return False
    return bool(np.max(np.abs(a - b)) <= MASS_UNCHANGED_RTOL * max(np.max(np.abs(b)), 1e-300))


def relative_mass_change(mass_new, mass_old) -> float:
    """``r = ||M_new - M_old||_2 / ||M_old||_2`` (spectral norms) of two full ``(d, d)`` masses (C1b);
    ``inf`` for different shapes or a degenerate ``M_old``."""
    new, old = np.asarray(mass_new, dtype=float), np.asarray(mass_old, dtype=float)
    if new.shape != old.shape or new.ndim != 2:
        return float("inf")
    denominator = float(np.linalg.norm(old, 2))
    if not (np.isfinite(denominator) and denominator > 0.0):
        return float("inf")
    return float(np.linalg.norm(new - old, 2)) / denominator


def mass_change_step_factor(mass_new, mass_old) -> float:
    """
    Factor by which the leapfrog step should change when the mass changes from ``M_old`` to
    ``M_new`` with the target fixed (C1b): ``(det M_new / det M_old)^(1 / (2d))``.

    Derivation, for ``H = U(q) + 1/2 p^T M^-1 p``: with ``M^-1 = L L^T`` and ``q = L z`` the
    leapfrog (and the dimension-robust splitting) with mass ``M`` and step ``eps`` is exactly the
    one with identity mass on ``U(L z)``. Its stability and energy error are governed by
    ``eps * omega_i``, where ``omega_i^2`` are the eigenvalues of ``L^T (Hess U) L``, i.e. of
    ``M^-1 Hess U``. Keeping ``eps * omega`` at the same typical value for a fixed target
    (``Hess U ~ Sigma^-1``) gives ``eps proportional to typical eig(M Sigma)^(1/2)``; with the
    geometric mean as the "typical" eigenvalue, ``det(M Sigma)^(1/(2d))``, the target cancels in
    the ratio: ``eps_new / eps_old = (det M_new / det M_old)^(1/(2d))``. For ``M = Sigma_hat^-1``
    this is ``(det Sigma_hat_old / det Sigma_hat_new)^(1/(2d))``: a LARGER covariance estimate
    (lighter mass, faster velocities ``M^-1 p``) needs a SMALLER step. (The spec's tentative
    ``sqrt(det Sigma_hat_new / det Sigma_hat_old)^(1/d)`` has the inverse exponent; checked:
    ``Sigma_hat_new = 2 Sigma_hat_old`` doubles the velocity variance, the step must shrink by
    ``1/sqrt 2``, which this factor gives.) ``1.0`` if a determinant is not positive/finite.
    """
    new, old = np.asarray(mass_new, dtype=float), np.asarray(mass_old, dtype=float)
    d = new.shape[0]
    sign_new, logdet_new = np.linalg.slogdet(new)
    sign_old, logdet_old = np.linalg.slogdet(old)
    if sign_new <= 0 or sign_old <= 0 or not (np.isfinite(logdet_new) and np.isfinite(logdet_old)):
        return 1.0
    return float(np.exp((logdet_new - logdet_old) / (2.0 * d)))


def _build_simple(spec, no_parameters: int, prior: Distribution, seed: int, carried: dict,
                  conf, stage_index, prior_mean=None, prior_sd_or_cov=None,
                  warn_fixed_default: bool = True,
                  statistics: PosteriorStatistics | None = None) -> Proposal:
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
        # item 5 measured M = cov as a poor mass. M = cov^-1 is the good one: mass="covariance"
        # (C1) takes it from the chain's posterior statistics, resolved here, at stage start.
        step_size = spec.step_size
        if step_size is None:
            step_size = carried.get("step_size", 0.1)
        if spec.integrator == "leapfrog":
            hamiltonian_class = proposals.Hamiltonian_adaptive if spec.adaptive else proposals.Hamiltonian
        else:
            hamiltonian_class = (proposals.HamiltonianInfinite_adaptive if spec.adaptive
                                 else proposals.HamiltonianInfinite)
        mass, mass_source = _resolve_hamiltonian_mass(spec, no_parameters, statistics, stage_index)
        proposal = hamiltonian_class(
            no_parameters=no_parameters,
            seed=seed,
            num_steps=spec.num_steps,
            step_size=step_size,
            sd_or_cov=mass,
            integration_time=getattr(spec, "integration_time", None),
            integration_time_jitter=getattr(spec, "integration_time_jitter", 0.0),
            **adaptive_kwargs,
        )
        # dual-averaging carry (C1, rule of C1b): when this stage adapts from the carried step,
        # continue the previous adaptive Hamiltonian stage's recursion if the mass changed by at
        # most DA_CONTINUE_MASS_TOLERANCE (relative spectral norm); else restart it from the
        # carried step rescaled to the new mass, with the small offset DA_RESTART_FACTOR
        dual_averaging = None
        mass_change = None
        step_size_initial = float(step_size)
        if spec.adaptive:
            dual_averaging = "fresh"
            if spec.step_size is None and "dual_averaging" in carried and "hamiltonian_mass" in carried:
                mass_new, mass_old = proposal.mass_matrix(), np.asarray(carried["hamiltonian_mass"], dtype=float)
                mass_change = relative_mass_change(mass_new, mass_old)
                if mass_change <= DA_CONTINUE_MASS_TOLERANCE:
                    proposal.continue_recursion(carried["dual_averaging"])
                    dual_averaging = "continued"
                else:
                    factor = mass_change_step_factor(mass_new, mass_old)
                    step_size_initial = float(step_size) * factor
                    proposal.restart_recursion(step_size_initial, DA_RESTART_FACTOR)
                    dual_averaging = "restarted (mass changed)"
                    print(f"Note: stage {stage_index}: mass changed (relative change {mass_change:.3g} > "
                          f"{DA_CONTINUE_MASS_TOLERANCE:g}): dual averaging restarted from the carried step "
                          f"{float(step_size):.6g} x {factor:.4g} = {step_size_initial:.6g} "
                          f"(mu = log({DA_RESTART_FACTOR:g} x step))", flush=True)
        proposal.resolution = {"mass_source": mass_source,
                               "integration_time": getattr(spec, "integration_time", None),
                               "integration_time_jitter": float(getattr(spec, "integration_time_jitter", 0.0)),
                               "num_steps_initial": int(proposal.num_steps),
                               "step_size_initial": step_size_initial,
                               "dual_averaging": dual_averaging,
                               "mass_change_r": mass_change}
        return proposal
    assert isinstance(spec, RandomWalk), type(spec)
    if spec.adaptive:
        # C1: the shape is estimated from the chain's shared posterior statistics (None = private)
        my_Prop = proposals.GaussRandomWalk_adaptive(no_parameters=no_parameters, seed=seed,
                                                     statistics=statistics, **adaptive_kwargs)
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
                   carried: dict | None = None, stage_index: int | None = None,
                   statistics: PosteriorStatistics | None = None) -> Proposal:
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
        statistics: the chain's persistent ``PosteriorStatistics`` (C1): read by an adaptive
            random walk (its shape) and by ``Hamiltonian(mass="covariance")`` (its mass, resolved
            now). ``None``: a private object / the identity mass. A Hamiltonian's carried
            ``"dual_averaging"`` state is continued when the resolved mass is within
            ``DA_CONTINUE_MASS_TOLERANCE`` (relative spectral norm) of the carried
            ``"hamiltonian_mass"``, else restarted from the carried step times
            ``mass_change_step_factor`` (C1b); the runtime Hamiltonian records what was resolved
            in its ``resolution`` dict (mass source, integration time and jitter, dual averaging
            branch, relative mass change ``mass_change_r``).

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
                                carried=carried, conf=conf, stage_index=stage_index, statistics=statistics)

    # Generalises the former Hamiltonian-only / HamiltonianInfinite-only asserts (B3) to any
    # proposal that needs gradients, including a BlockProposal whose sub-proposals include a
    # Hamiltonian one (BlockProposal.__init__ sets its own needs_gradients to True iff any
    # sub-proposal needs gradients). Same message and exception type as before: one code path.
    assert not getattr(my_Prop, "needs_gradients", False) or conf.use_surrogate_gradients, \
        "Hamiltonian proposals need use_surrogate_gradients=True (it may have been disabled by Problem.run_sampling, see warnings)"
    return my_Prop


def build_exact_step_proposal(conf: "Configuration", prior: Distribution, seed: int,
                              carried: dict | None = None, stage_index: int | None = None,
                              statistics: PosteriorStatistics | None = None,
                              scale: float | None = None) -> Proposal:
    """
    The exact step's own proposal of a DAMH stage with ``Stage.exact_step_probability > 0``
    (S2, 2026-10-09): the spec ``RandomWalk()`` -- adaptive, starting from the carried
    ``"scale"`` when an earlier adaptive random-walk stage handed one over, else from the
    prior-derived default -- built exactly like a stage proposal, with its own ``seed``
    (``seed0 + EXACT_PROPOSAL_SEED_OFFSET``). Its SCALE adapts on the exact step's own
    acceptances; it is not pooled across ranks, not carried over and its adaptation trace is not
    written.

    Shape (C1b, 2026-10-11): ``statistics`` = the chain's shared ``PosteriorStatistics`` (both
    runners pass it, as for the stage's random walk). The exact random walk then reads its shape
    from them: immediately at construction when they already hold ``warmup`` (100) states (the
    carried / default scale is kept as the trace of the effective covariance, the shape replaced),
    and at every ``period``-th of its own ``adapt()`` calls afterwards. It never updates them: the
    algorithm updates the shared object once per outer iteration, exact steps included, so there
    is no double count. ``statistics=None``: a PRIVATE object updated by its own ``adapt()`` (the
    S2/C1 behaviour). ``carried`` is read, never modified.

    Fixed scale (``Stage.exact_step_scale``, study 29, 2026-10-12): ``scale`` given = a NON-adaptive
    ``GaussRandomWalk`` with covariance ``scale * (2.38^2 / d) * Sigma_hat``, ``Sigma_hat`` =
    ``statistics.covariance()`` now (stage start) when the statistics hold ``MASS_MIN_STATES`` states,
    else the identity with a printed note. Same ``seed``. The proposal carries a ``resolution`` dict
    (``exact_step_scale``, ``exact_step_covariance_source``, ``exact_step_covariance_trace``).
    ``scale=None``: the adaptive random walk above (unchanged).
    """
    if scale is not None:
        d = int(conf.no_parameters)
        factor = float(scale) * 2.38 ** 2 / d
        covariance, source = None, None
        if statistics is not None and statistics.is_informative(MASS_MIN_STATES):
            try:
                covariance, source = statistics.covariance(), f"covariance (n={statistics.n})"
            except ValueError as exc:
                print(f"Note: stage {stage_index}: exact_step_scale: the posterior statistics (n={statistics.n}) "
                      f"give no usable covariance ({exc}); identity used", flush=True)
        else:
            n = 0 if statistics is None else statistics.n
            print(f"Note: stage {stage_index}: exact_step_scale: only {n} chain states seen (< {MASS_MIN_STATES}); "
                  "the exact random walk uses the identity covariance shape", flush=True)
        if covariance is None:
            covariance, source = np.eye(d), "identity"
        proposal = proposals.GaussRandomWalk(no_parameters=d, sd_or_cov=factor * np.asarray(covariance, dtype=float),
                                             seed=seed)
        proposal.resolution = {"exact_step_scale": float(scale), "exact_step_covariance_source": source,
                               "exact_step_covariance_trace": float(np.trace(proposal.sd_or_cov))}
        return proposal
    proposal = _build_simple(RandomWalk(), no_parameters=conf.no_parameters, prior=prior, seed=seed,
                             carried={} if carried is None else carried, conf=conf, stage_index=stage_index,
                             statistics=statistics)
    if statistics is not None and statistics.n >= proposal.warmup \
            and proposal._install_covariance_from_statistics():
        proposal._refactor()
    return proposal


__all__ = ["build_proposal", "build_exact_step_proposal", "relative_mass_change", "mass_change_step_factor",
           "DA_CONTINUE_MASS_TOLERANCE", "DA_RESTART_FACTOR"]
