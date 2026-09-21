#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np
import numpy.typing as npt
from typing import Callable


def as_covariance_matrix(sd_or_cov: npt.ArrayLike, no_parameters: int) -> npt.NDArray:
    """
    Normalise a proposal scale to a full ``(no_parameters, no_parameters)`` covariance matrix.

    ``GaussRandomWalk.set_covariance`` accepts a scalar, a vector of standard deviations or a
    covariance matrix; this function turns any of those into the one 2-D spelling.

    History: it was introduced for the per-stage ``Allreduce`` of the adaptive covariance (G2,
    2026-09-17), which used to mix 1-D and 2-D buffers across ranks and abort the job with
    ``Message truncated`` (finding 2.5). That reduction is gone -- the ranks now exchange the
    adaptation's sufficient statistics through ``GaussRandomWalk_adaptive.adapted_state()``,
    whose length is fixed by ``no_parameters`` alone -- and ``GaussRandomWalk_adaptive`` keeps
    ``sd_or_cov`` 2-D at all times, so neither runner needs this any more. It stays because it
    is the one place that defines what an accepted ``sd_or_cov`` is, and the adaptive random
    walk uses it to normalise its initial covariance.

    Args:
        sd_or_cov: scalar standard deviation, 1-D vector of standard deviations, or a 2-D
            covariance matrix.
        no_parameters: dimension of the parameter space.

    Returns:
        A fresh, contiguous ``(no_parameters, no_parameters)`` float64 covariance matrix.
    """
    array = np.asarray(sd_or_cov, dtype=np.float64)
    if array.ndim == 0:
        return np.eye(no_parameters) * float(array) ** 2
    if array.ndim == 1:
        if array.shape != (no_parameters,):
            raise ValueError(f"expected {no_parameters} standard deviations, got shape {array.shape}")
        return np.diag(array ** 2)
    if array.ndim == 2:
        if array.shape != (no_parameters, no_parameters):
            raise ValueError(f"expected a ({no_parameters}, {no_parameters}) covariance matrix, "
                             f"got shape {array.shape}")
        return np.ascontiguousarray(array.copy())
    raise ValueError(f"sd_or_cov must be scalar, 1-D or 2-D, got {array.ndim} dimensions")


def _is_finite(vector: npt.NDArray) -> bool:
    """``True`` iff every entry of ``vector`` is finite (leapfrog divergence check, 2026-09-20)."""
    return bool(np.all(np.isfinite(vector)))


def acceptance_probability(log_acceptance_probability: float) -> float:
    """
    ``min(1, exp(log_acceptance_probability))``, without overflow and without warnings.

    ``-inf`` (DAMH's pre-rejected iterations) maps to ``0.0``, a non-negative log-ratio to
    ``1.0`` without ever calling ``exp`` on a large number, and ``nan`` (which no code path
    should produce) to ``0.0`` rather than propagating.
    """
    value = float(log_acceptance_probability)
    if np.isnan(value):
        return 0.0
    if value >= 0.0:
        return 1.0
    return float(np.exp(value))


_SUBPROPOSAL_SEED_BASE = 2**31  # disjoint from the seed0 = 10*(no_stages*rank + i) family


def subproposal_seed(block_seed: int, index: int) -> int:
    """
    Seed of sub-proposal ``index`` of a ``BlockProposal`` whose own seed is ``block_seed``.

    G5 (2026-09-17): the sub-proposals used to keep whatever seed the user gave them when
    building the ``Stage``, so every MPI rank ran the SAME sub-proposal stream and the chains
    shared their proposal increments (finding 1.9/A13). They are now re-seeded from the
    block's per-rank, per-stage seed instead.

    The map is injective for up to 1000 sub-proposals: ``base + 1000*block_seed + index``
    with ``index < 1000`` determines ``(block_seed, index)`` uniquely. ``base = 2**31``
    keeps the result out of the ``seed0``/``proposal_seed``/``algorithm_seed`` range, so a
    sub-proposal can never share a stream with another chain's top-level proposal either.
    """
    if not 0 <= index < 1000:
        raise ValueError(f"BlockProposal supports at most 1000 sub-proposals, got index {index}")
    return (_SUBPROPOSAL_SEED_BASE + 1000 * int(block_seed) + int(index)) % 2**32


class Proposal:
    """Parent class for proposal distributions."""

    # class-level defaults so that subclasses skipping super().__init__() still expose them
    needs_gradients: bool = False
    subchain_length: int = 1

    def __init__(self) -> None:
        self.log_likelihood_gradient: Callable
        self.log_prior_gradient: Callable
        self.no_parameters: int
        self.needs_gradients: bool = False
        self.subchain_length: int = 1
        pass

    def propose_sample(self, current_sample: npt.NDArray) -> npt.NDArray:
        """Gets current sample, returns proposed sample."""
        raise NotImplementedError

    def get_log_acceptance_probability(self, log_likelihood_proposed: float, log_likelihood_current: float,
                                       log_prior_proposed: float, log_prior_current: float) -> tuple[float, float]:
        """
        Gets logarithms of likelihoods and priors of proposed and current sample.
        Returns two parts of log acceptance probability: one related to likelihoods and one related to priors.
        Logarithm of acceptance ratio can be obtained as a sum of output values.
        """
        raise NotImplementedError

    def reseed(self, seed: int) -> None:
        """
        Replace this proposal's random generator by a freshly seeded one (G5).

        Used by ``BlockProposal`` to give each sub-proposal a per-rank, per-stage stream
        instead of whatever seed it was constructed with. Every proposal in this module keeps
        its generator in ``self._generator`` as an ``np.random.RandomState``, which this
        default implementation re-creates. A custom ``Proposal`` that owns a generator under
        a different name must override this method; if it owns none, override it with ``pass``
        to silence the warning below.
        """
        if hasattr(self, "_generator"):
            self._generator = np.random.RandomState(seed=seed)
            return
        warnings.warn(
            f"{type(self).__name__}.reseed() found no '_generator' attribute to re-seed; if this "
            "proposal owns a random generator, override reseed() so that block sub-proposals and "
            "MPI chains do not share random streams (G5)",
            RuntimeWarning,
            stacklevel=2,
        )

    def adapt(self, proposed_sample: npt.NDArray, log_acceptance_probability: float,
              current_sample: npt.NDArray,
              subchain_log_acceptance_probabilities: list[float] | None = None) -> None:
        """
        Feed one accept/reject outcome to the proposal; no-op unless the proposal adapts.

        Called by the algorithms once per (outer) iteration, AFTER the accept/reject decision
        has been applied to the chain (2026-09-20), with

        * ``proposed_sample``: the proposed parameters,
        * ``log_acceptance_probability``: the log of their Metropolis acceptance probability
          (``-inf`` = probability 0, used by DAMH for pre-rejected iterations),
        * ``current_sample``: the chain state AFTER the decision (the proposal if it was
          accepted, the previous state otherwise) -- what a Haario-style covariance estimator
          needs,
        * ``subchain_log_acceptance_probabilities``: in a DAMH stage, the per-step log
          acceptance probabilities the surrogate sub-chain computed (one per sub-chain step,
          in order); ``None`` in an MH stage.

        The fixed signature makes a wrong keyword an error for every proposal, not only for
        the adaptive ones.
        """

    def choose_group(self):
        """
        Chooses the group of parameters to update (only for block proposal).
        """
        pass

    def set_gradient_functions(self, log_likelihood_gradient_function: Callable, log_prior_gradient_function: Callable):
        self.log_likelihood_gradient = log_likelihood_gradient_function
        self.log_prior_gradient = log_prior_gradient_function


class GaussRandomWalk(Proposal):  # initiated by SAMPLERs
    """Symmetric Gaussian random-walk proposal (no prior assumption): ``x' = x + N(0, sd_or_cov)``."""

    def __init__(self, no_parameters, sd_or_cov=1.0, seed=0) -> None:
        super().__init__()
        self.no_parameters = no_parameters
        self._generator = np.random.RandomState(seed=seed)
        self.set_covariance(sd_or_cov=sd_or_cov)

    def set_covariance(self, sd_or_cov: npt.ArrayLike) -> None:
        # sd_or_cov is scalar/vector/covariance matrix:
        if np.isscalar(sd_or_cov):
            self.sd_or_cov = np.full((self.no_parameters,), sd_or_cov)
        else:
            self.sd_or_cov = np.array(sd_or_cov)
        if self.sd_or_cov.ndim == 1:  # proposal - normal uncorrelated
            self.propose_sample = self._propose_sample_uncorrelated
        else:  # proposal - normal correlated
            self.propose_sample = self._propose_sample_multivariate

    def _propose_sample_uncorrelated(self, current_sample: npt.NDArray) -> npt.NDArray:
        proposed_sample = self._generator.normal(current_sample, self.sd_or_cov)
        return proposed_sample

    def _propose_sample_multivariate(self, current_sample: npt.NDArray) -> npt.NDArray:
        proposed_sample = self._generator.multivariate_normal(current_sample, self.sd_or_cov)
        return proposed_sample

    def get_log_acceptance_probability(self, log_likelihood_proposed, log_likelihood_current,
                                       log_prior_proposed, log_prior_current) -> tuple[float, float]:
        # simple since the proposal distribution is symmetrical
        likelihood_part = log_likelihood_proposed - log_likelihood_current
        prior_part = log_prior_proposed - log_prior_current
        return likelihood_part, prior_part


class PCN(Proposal):  # preconditioned Crank-Nicolson proposal
    """
    Preconditioned Crank-Nicolson proposal: ``get_log_acceptance_probability`` drops
    the prior ratio unconditionally, which is only correct when the internal prior
    really is ``N(prior_mean, prior_sd_or_cov)`` -- ``build_proposal`` enforces this by
    only constructing a ``PCN`` for a ``Normal`` or ``PriorIndependentComponents``
    prior (whose internal prior is standard normal by design), see
    ``proposal_builder.py:_prior_is_gaussian``. Constructing this class directly with a
    non-Gaussian prior silently produces an incorrect chain; go through
    ``build_proposal``/``Stage(proposal_type="pCN")`` instead of instantiating it by hand.
    """

    def __init__(self, no_parameters: int, beta: float, prior_mean: npt.NDArray,
                 prior_sd_or_cov: npt.ArrayLike, seed: int = 0) -> None:
        """
        Args:
            no_parameters: number of parameters
            beta: pCN step size in (0, 1]
            prior_mean: mean of the Gaussian prior
            prior_sd_or_cov: standard deviations (1D) or covariance matrix (2D) of the prior
            seed: random seed for reproducibility
        """
        super().__init__()
        assert 0 < beta <= 1, f"pCN beta must be in (0, 1], got {beta}"
        self.no_parameters = no_parameters
        self.beta = beta
        self.prior_mean = np.array(prior_mean)
        self._generator = np.random.RandomState(seed=seed)
        self._set_prior_covariance(prior_sd_or_cov)

    def _set_prior_covariance(self, sd_or_cov: npt.ArrayLike) -> None:
        if np.isscalar(sd_or_cov):
            self._prior_sd = np.full((self.no_parameters,), sd_or_cov)
            self._is_diagonal = True
        else:
            arr = np.array(sd_or_cov)
            if arr.ndim == 1:
                self._prior_sd = arr
                self._is_diagonal = True
            else:
                self._prior_cov = arr
                self._is_diagonal = False

    def propose_sample(self, current_sample: npt.NDArray) -> npt.NDArray:
        # xi* = m + sqrt(1 - beta^2) * (xi - m) + beta * eta,  eta ~ N(0, C_0)
        shift = current_sample - self.prior_mean
        if self._is_diagonal:
            noise = self._generator.normal(np.zeros(self.no_parameters), self._prior_sd)
        else:
            noise = self._generator.multivariate_normal(np.zeros(self.no_parameters), self._prior_cov)
        return self.prior_mean + np.sqrt(1 - self.beta**2) * shift + self.beta * noise

    def get_log_acceptance_probability(self, log_likelihood_proposed: float, log_likelihood_current: float,
                                       log_prior_proposed: float, log_prior_current: float) -> tuple[float, float]:
        # pCN: prior cancels by construction, acceptance depends only on likelihood
        return log_likelihood_proposed - log_likelihood_current, 0.0


class GaussRandomWalk_adaptive(GaussRandomWalk):  # initiated by SAMPLERs
    """
    Adaptive Gaussian random walk: shrinkage-regularised Haario covariance for the SHAPE and a
    Robbins-Monro recursion on the log-scale for the SCALE (rewritten 2026-09-20, recommendation
    2 of ``library_notes/16_adaptivity_options_research_2026-09-20.md`` §5; prototype formulas in
    that study's ``proto/RESULTS_PROTO.md`` §1, "AMShrinkRM").

    Per ``adapt()`` call (one per outer iteration of the stage)::

        alpha        = min(1, exp(log_acceptance_probability))
        n           += 1
        log_sigma   += n**(-0.7) * (alpha - target_rate)          [Andrieu & Thoms 2008 Alg. 4]
        (mean, M2)   = Welford update with the POST-decision chain state

    and, every ``period`` iterations once ``n >= warmup``::

        C            = M2 / (n - 1)                               [Haario et al. 2001 estimator]
        delta        = min(1, 2d/n)
        C            = (1 - delta) * C + delta * (tr C / d) * I    [Ledoit-Wolf-style shrinkage]
        base_cov     = (2.38**2/d) * (C + 1e-6 * (tr C / d) * I)   [relative ridge]

    The proposal is ``x' = x + exp(log_sigma) * L z``, ``L L^T = base_cov``, ``z ~ N(0, I)``;
    ``self.sd_or_cov`` is kept equal to ``exp(2*log_sigma) * base_cov`` (always 2-D) so that every
    existing reader of it -- the stage hand-over, ``as_covariance_matrix``, the start-up prints --
    keeps working.

    What was REMOVED with the rewrite (breaking, see CHANGELOG): ``corr_limit`` (the entrywise
    correlation clip, which made the covariance indefinite and had numpy draw ~every proposal
    through an SVD fallback -- 19 750 warnings in a 20 000-evaluation prototype run), the
    ``sample_limit`` bounded history (unstable: a 4-orders-of-magnitude scale blow-up in the
    2026-09-18 study), the ``coef`` acceptance feedback, the stored sample/weight history and the
    degenerate-weights guard they needed. The ridge is RELATIVE to ``tr C / d`` rather than the
    prototype's absolute ``1e-6``, so the regularisation does not depend on the units of the
    parameters.

    DAMH stages: ``adapt()`` is called on **every** outer iteration -- with the exact-posterior
    acceptance probability of the sub-chain endpoint when the sub-chain moved, and with
    ``-inf`` (probability 0) on a pre-rejected iteration. The rate driven to ``target_rate`` is
    therefore the *overall* acceptance of an outer iteration; scoring only the moved iterations
    feeds back the conditional second-stage rate, which tends to 1 and makes the scale diverge
    (`16` §2 item 1, §4.2). For ``subchain_max_length > 1`` the whole multi-step move is scored,
    so the per-step scale ends above its own optimum -- prefer ``subchain_max_length=1`` when
    adapting.

    Cross-rank hand-over: ``adapted_state()`` / ``set_pooled_state()`` / ``carry_over()`` pool the
    SUFFICIENT STATISTICS of all chains (Chan et al. parallel combination of ``(n, mean, M2)``,
    plus the mean log-scale) instead of averaging per-rank covariances; see the runners.
    """

    #: column names of ``adaptive_stats/<stage>/rank%04d.csv`` (see docs/outputs.md)
    adaptation_stats_header = ["n", "mean_acceptance_probability", "log_sigma",
                              "trace_C_over_d", "shrinkage_delta"]

    def __init__(self, no_parameters: int, sd_or_cov: npt.ArrayLike = 1.0, seed: int = 0,
                 target_rate: float = 0.234, period: int = 10, warmup: int = 100) -> None:
        """
        Args:
            no_parameters: dimension of the parameter space.
            sd_or_cov: initial proposal scale (scalar sd, vector of sds, or covariance matrix).
                Only the starting point of the recursion -- both scale and shape are learned.
            seed: seed of this proposal's random generator.
            target_rate: target acceptance rate of the Robbins-Monro scale recursion
                (``Stage.adaptive_target_rate``; 0.234 is the high-dimensional RWM optimum).
            period: re-estimate the covariance every ``period`` ``adapt()`` calls.
            warmup: no covariance is installed before ``n >= warmup`` (the scale recursion runs
                from the first call).
        """
        Proposal.__init__(self)  # not GaussRandomWalk.__init__: the covariance is set below
        self.no_parameters = int(no_parameters)
        self._generator = np.random.RandomState(seed=seed)
        self.target_rate = float(target_rate)
        self.period = int(period)
        self.warmup = int(warmup)
        self.scale_factor = 2.38 ** 2 / self.no_parameters

        self.n = 0
        self.mean = np.zeros(self.no_parameters)
        self.M2 = np.zeros((self.no_parameters, self.no_parameters))
        self.log_sigma = 0.0
        self._trace_over_d = np.nan  # of the last installed covariance estimate
        self._shrinkage_delta = np.nan
        self._period_alpha_sum = 0.0
        self._period_alpha_count = 0
        self.adaptation_stats_rows: list[tuple] = []

        self.set_covariance(sd_or_cov=sd_or_cov)

    # -- proposal -----------------------------------------------------------------------
    def set_covariance(self, sd_or_cov: npt.ArrayLike) -> None:
        """Replace the (unscaled) base covariance; ``log_sigma`` and the statistics are kept.

        Overrides ``GaussRandomWalk.set_covariance``, which would install one of the two
        fixed-covariance ``propose_sample`` implementations as an instance attribute and thereby
        shadow this class's own ``propose_sample``.
        """
        self.base_cov = as_covariance_matrix(sd_or_cov, self.no_parameters)
        self._refactor()

    def _update_public_covariance(self) -> None:
        """``sd_or_cov`` is the covariance the proposal actually draws with, ``exp(2 log_sigma)
        * base_cov``; it depends on the scale, which changes on every ``adapt()`` call."""
        self.sd_or_cov = np.exp(2.0 * self.log_sigma) * self.base_cov

    def _refactor(self) -> None:
        """Recompute the Cholesky factor of ``base_cov`` and the public ``sd_or_cov``.

        Only needed when ``base_cov`` itself changed: ``propose_sample`` multiplies ``L`` by
        ``exp(log_sigma)``, so a scale-only update needs no factorisation.
        """
        symmetric = 0.5 * (self.base_cov + self.base_cov.T)
        try:
            self.L = np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError:
            # never observed with the shrinkage+ridge above; kept so that a pathological
            # estimate degrades to its PSD projection instead of aborting the run
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
            self.L = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 1e-14, None)))
        self._update_public_covariance()

    def propose_sample(self, current_sample: npt.NDArray) -> npt.NDArray:
        z = self._generator.standard_normal(self.no_parameters)
        return current_sample + np.exp(self.log_sigma) * (self.L @ z)

    # -- adaptation ---------------------------------------------------------------------
    def adapt(self, proposed_sample: npt.NDArray, log_acceptance_probability: float,
              current_sample: npt.NDArray,
              subchain_log_acceptance_probabilities: list[float] | None = None) -> None:
        alpha = acceptance_probability(log_acceptance_probability)
        self.n += 1
        self.log_sigma += self.n ** (-0.7) * (alpha - self.target_rate)

        # Welford update of the mean/scatter of the POST-decision chain states
        state = np.asarray(current_sample, dtype=float)
        difference = state - self.mean
        self.mean = self.mean + difference / self.n
        self.M2 += np.outer(difference, state - self.mean)

        self._period_alpha_sum += alpha
        self._period_alpha_count += 1
        if self.n >= self.warmup and self.n % self.period == 0 \
                and self._install_covariance_from_statistics(self.n, self.M2):
            self._refactor()  # base_cov changed: re-factorise (once per period at most)
        else:
            self._update_public_covariance()  # scale-only update, no factorisation needed
        if self.n % self.period == 0:
            self.adaptation_stats_rows.append((
                self.n,
                self._period_alpha_sum / max(self._period_alpha_count, 1),
                self.log_sigma,
                self._trace_over_d,
                self._shrinkage_delta,
            ))
            self._period_alpha_sum = 0.0
            self._period_alpha_count = 0

    def _install_covariance_from_statistics(self, n: int, scatter: npt.NDArray) -> bool:
        """``base_cov`` from ``(n, M2)``: shrinkage towards ``(tr C / d) I`` plus a relative ridge.

        Returns ``True`` iff a new ``base_cov`` was installed (so the caller knows whether the
        Cholesky factor has to be recomputed)."""
        d = self.no_parameters
        if n < 2:
            return False
        C = scatter / (n - 1)
        trace_over_d = float(np.trace(C)) / d
        if not np.all(np.isfinite(C)) or not np.isfinite(trace_over_d) or trace_over_d <= 0.0:
            # a chain that never moved (or a non-finite statistic): keep the previous covariance
            # instead of installing a singular one the proposal could never leave
            return False
        delta = min(1.0, 2.0 * d / n)
        C = (1.0 - delta) * C + delta * trace_over_d * np.eye(d)
        self.base_cov = self.scale_factor * (C + 1e-6 * trace_over_d * np.eye(d))
        self._trace_over_d = trace_over_d
        self._shrinkage_delta = delta
        return True

    # -- cross-rank hand-over -----------------------------------------------------------
    def adapted_state(self) -> npt.NDArray:
        """``[n, mean (d), M2 (d*d), log_sigma]`` -- fixed length, same on every rank."""
        return np.concatenate((
            np.array([float(self.n)]),
            self.mean.astype(np.float64).ravel(),
            self.M2.astype(np.float64).ravel(),
            np.array([float(self.log_sigma)]),
        ))

    def set_pooled_state(self, states: npt.NDArray) -> None:
        """
        Combine the ``adapted_state()`` rows of all chains into this proposal.

        ``(n_r, mean_r, M2_r)`` are combined by the Chan-Golub-LeVeque parallel formula, which
        gives exactly the statistics of the concatenated chains; ``log_sigma`` is averaged.
        Pooling the SAMPLES (rather than averaging per-rank covariances) is what Craiu et al.
        (2009) / Solonen et al. (2012) recommend and what removes the 6-17x rank disagreement
        the 2026-09-18 study measured.

        With a single row there is nothing to pool: the statistics are this chain's own and the
        adapted covariance is left EXACTLY as the stage ended it (a covariance rebuilt at an
        arbitrary ``n`` would differ from the one installed at the last period boundary). That
        keeps ``run_local`` reproducing MPI chain 0 of a one-sampler run.
        """
        states = np.atleast_2d(np.asarray(states, dtype=float))
        d = self.no_parameters
        expected = 1 + d + d * d + 1
        if states.shape[1] != expected:
            raise ValueError(f"expected adapted-state rows of length {expected}, got {states.shape}")

        counts = states[:, 0]
        means = states[:, 1:1 + d]
        scatters = states[:, 1 + d:1 + d + d * d].reshape(-1, d, d)
        total = float(counts.sum())
        if total > 0:
            pooled_mean = (counts[:, None] * means).sum(axis=0) / total
            pooled_scatter = scatters.sum(axis=0)
            for count, mean in zip(counts, means):
                shift = mean - pooled_mean
                pooled_scatter = pooled_scatter + count * np.outer(shift, shift)
            self.n = int(round(total))
            self.mean = pooled_mean
            self.M2 = pooled_scatter
        self.log_sigma = float(states[:, -1].mean())

        if states.shape[0] > 1 and self.n >= self.warmup:
            self._install_covariance_from_statistics(self.n, self.M2)
        self._refactor()

    def carry_over(self) -> dict:
        """Stage fields this adapted proposal hands to the following stages."""
        return {"proposal_sd_or_cov": self.sd_or_cov}


class PCN_adaptive(PCN):
    """
    ``PCN`` whose step size ``beta`` is adapted by a Robbins-Monro recursion on its logit
    (new 2026-09-20, recommendation 6 of `16` §5; prototype `RESULTS_PROTO.md` §1 "P6", §7)::

        alpha        = min(1, exp(log_acceptance_probability))
        n           += 1
        logit_beta  += n**(-0.7) * (alpha - target_rate)
        beta         = sigmoid(logit_beta)

    Adapting the logit keeps ``beta`` in ``(0, 1)`` by construction, so no clipping is needed.
    Target 0.234 (not 0.5) is what the prototype measured: it recovers 0.77-0.99 of the ESS of
    the best fixed ``beta`` on every problem and beats target 0.5 by 1.03-2.14x, while the
    optimal ``beta`` spans 0.1-0.8 across problems, i.e. a fixed default cannot serve them all.

    In a DAMH stage the feedback is the same overall outer acceptance probability
    ``GaussRandomWalk_adaptive`` uses (pre-rejected iterations scored as 0).
    """

    adaptation_stats_header = ["n", "mean_acceptance_probability", "beta"]

    def __init__(self, no_parameters: int, beta: float, prior_mean: npt.NDArray,
                 prior_sd_or_cov: npt.ArrayLike, seed: int = 0,
                 target_rate: float = 0.234, period: int = 10) -> None:
        if not 0.0 < beta < 1.0:
            raise ValueError(f"adaptive pCN needs an initial beta strictly inside (0, 1), got {beta}")
        super().__init__(no_parameters=no_parameters, beta=beta, prior_mean=prior_mean,
                         prior_sd_or_cov=prior_sd_or_cov, seed=seed)
        self.target_rate = float(target_rate)
        self.period = int(period)
        self.n = 0
        self.logit_beta = float(np.log(beta / (1.0 - beta)))
        self._period_alpha_sum = 0.0
        self._period_alpha_count = 0
        self.adaptation_stats_rows: list[tuple] = []

    def adapt(self, proposed_sample: npt.NDArray, log_acceptance_probability: float,
              current_sample: npt.NDArray,
              subchain_log_acceptance_probabilities: list[float] | None = None) -> None:
        alpha = acceptance_probability(log_acceptance_probability)
        self.n += 1
        self.logit_beta += self.n ** (-0.7) * (alpha - self.target_rate)
        self._set_beta_from_logit()
        self._period_alpha_sum += alpha
        self._period_alpha_count += 1
        if self.n % self.period == 0:
            self.adaptation_stats_rows.append((
                self.n, self._period_alpha_sum / max(self._period_alpha_count, 1), self.beta))
            self._period_alpha_sum = 0.0
            self._period_alpha_count = 0

    def _set_beta_from_logit(self) -> None:
        # numerically stable sigmoid: never 0 or 1 for a finite logit
        if self.logit_beta >= 0.0:
            self.beta = 1.0 / (1.0 + float(np.exp(-self.logit_beta)))
        else:
            exponential = float(np.exp(self.logit_beta))
            self.beta = exponential / (1.0 + exponential)

    def adapted_state(self) -> npt.NDArray:
        return np.array([float(self.n), float(self.logit_beta)])

    def set_pooled_state(self, states: npt.NDArray) -> None:
        """Average the logit over the chains (``n`` stays this chain's own count)."""
        states = np.atleast_2d(np.asarray(states, dtype=float))
        if states.shape[1] != 2:
            raise ValueError(f"expected adapted-state rows of length 2, got {states.shape}")
        self.logit_beta = float(states[:, 1].mean())
        self._set_beta_from_logit()

    def carry_over(self) -> dict:
        return {"pcn_beta": self.beta}


class Hamiltonian(Proposal):
    """
    Standard Hamiltonian (leapfrog) proposal with mass matrix ``sd_or_cov``.
    ``needs_gradients=True``: requires ``set_gradient_functions`` to be called before
    ``propose_sample`` (``build_proposal``/the algorithm classes wire this to the
    surrogate's ``vjp``/``jacobian``, never the exact model -- see
    ``docs/stages.md``), and requires ``Configuration.use_surrogate_gradients=True``
    in effect (``build_proposal`` asserts this).
    """

    def __init__(self, no_parameters, step_size=0.1, num_steps=10, sd_or_cov=1.0, seed=0) -> None:
        """
        Hamiltonian proposal with leapfrog integrator for Hamiltonian dynamics.

        Parameters:
            no_parameters: number of parameters
            log_likelihood_gradient: callable — gradient of log-likelihood
            log_prior_gradient: callable — gradient of log-prior
            sd_or_cov: scalar, vector of standard deviations, or covariance matrix
            step_size: float — leapfrog step size
            num_steps: int — number of leapfrog steps
            seed: random seed for reproducibility
        """
        super().__init__()
        self.no_parameters = no_parameters
        self.step_size = step_size
        self.num_steps = num_steps
        self._generator = np.random.RandomState(seed=seed)
        self.set_covariance(sd_or_cov=sd_or_cov)
        self.p_current_start: npt.NDArray # current momentum, will be set in propose_sample
        self.p_proposed_end: npt.NDArray # proposed momentum, will be set in propose_sample
        self.M: npt.NDArray  # mass matrix, will be set in set_covariance
        self.M_inv: npt.NDArray  # inverse mass matrix, will be set in set_covariance
        self.needs_gradients = True

    def set_covariance(self, sd_or_cov: npt.ArrayLike) -> None:
        # sd_or_cov is scalar/vector/covariance matrix:
        if np.isscalar(sd_or_cov):
            self.sd_or_cov = np.full((self.no_parameters,), sd_or_cov)
        else:
            self.sd_or_cov = np.array(sd_or_cov)
        if self.sd_or_cov.ndim == 1:  # proposal - normal uncorrelated
            self.generate_normal_sample = self._generate_normal_sample_uncorrelated
            variances = self.sd_or_cov**2
            self.M = np.diag(variances)
            self.M_inv = np.diag(1/variances)
            self._mass_is_diagonal = True
            self._mass_sd = self.sd_or_cov.copy()
        else:  # proposal - normal correlated
            self.generate_normal_sample = self._generate_normal_sample_multivariate
            self.M = self.sd_or_cov
            self.M_inv = np.linalg.inv(self.sd_or_cov)
            self._mass_is_diagonal = False
            eigenvalues, eigenvectors = np.linalg.eigh(self.M)
            if np.any(eigenvalues <= 0.0):
                raise ValueError("Hamiltonian mass matrix must be positive definite")
            self._mass_eigenvalues = eigenvalues
            self._mass_eigenvectors = eigenvectors

    def _generate_normal_sample_uncorrelated(self, mean: npt.NDArray) -> npt.NDArray:
        return self._generator.normal(mean, self.sd_or_cov)

    def _generate_normal_sample_multivariate(self, mean: npt.NDArray) -> npt.NDArray:
        return self._generator.multivariate_normal(mean, self.sd_or_cov)

    def propose_sample(self, current_sample: npt.NDArray) -> npt.NDArray:
        self.p_current_start = self.generate_normal_sample(np.zeros((self.no_parameters,)))
        q, p_end = self._leapfrog(current_sample, self.p_current_start)
        self.p_proposed_end = p_end
        return q

    def get_log_acceptance_probability(self, log_likelihood_proposed, log_likelihood_current,
                                       log_prior_proposed, log_prior_current) -> tuple[float, float]:
        # Total Hamiltonian H(q, p) = U(q) + 0.5 * p^T M^{-1} p
        #    return U(q) + 0.5 * p @ M_inv @ p
        likelihood_part = log_likelihood_proposed - log_likelihood_current
        prior_part = log_prior_proposed - log_prior_current
        p_part = 0.5 * (self.p_current_start @ self.M_inv @ self.p_current_start - self.p_proposed_end @ self.M_inv @ self.p_proposed_end)
        # p_part included into the prior part output
        return likelihood_part, prior_part + p_part

    def _leapfrog(self, q0: npt.NDArray, p0_start: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Leapfrog integrator for Hamiltonian dynamics.
        
        Parameters:
            q0: ndarray, shape (no_parameters,) — initial position
            p0_start: ndarray, shape (no_parameters,) — initial momentum
        
        Returns:
            q: ndarray — final position
            p_end: ndarray — final momentum (after momentum flip)

        Divergent trajectories (2026-09-20): as soon as ``q`` or ``p`` stops being finite the
        integration is abandoned and the non-finite ``q`` is returned, so the remaining
        surrogate-gradient evaluations are saved and the algorithm rejects the proposal without
        calling the full model (``AlgorithmBase._reject_nonfinite_proposal``). The checks read
        the state but never modify it, so a finite trajectory is bit-identical to the version
        without them.
        """
        q = q0.copy()
        p = p0_start.copy()
        p = p - 0.5 * self.step_size * (self.log_likelihood_gradient(q) + self.log_prior_gradient(q))  # half step for momentum
        if not _is_finite(p):
            return q, -p
        for i in range(self.num_steps):
            q = q + self.step_size * self.M_inv @ p  # full step for position
            if not _is_finite(q):
                return q, -p
            if i < self.num_steps - 1:
                p = p - self.step_size * (self.log_likelihood_gradient(q) + self.log_prior_gradient(q))  # full step for momentum (except at the end)
                if not _is_finite(p):
                    return q, -p
        p = p - 0.5 * self.step_size * (self.log_likelihood_gradient(q) + self.log_prior_gradient(q))  # final half step for momentum
        p_end = -p  # negate momentum (for reversibility — does not affect acceptance)
        return q, p_end


class HamiltonianInfinite(Hamiltonian):
    """
    Hamiltonian proposal whose free flow is the exact solution of the harmonic oscillator
    H(q, p) = 0.5 q^T q + 0.5 p^T M^{-1} p (a rotation in phase space, ``M = diag(sd^2)``
    or the full matrix passed as ``sd_or_cov``), so that only the likelihood gradient is
    integrated numerically (split integrator).

    ``sd_or_cov`` is permanently the MASS matrix, not the prior covariance (decision 8,
    `library_notes/09_improvement_plan.md` §3 — kept, not reinterpreted: the rotation's
    angle depends only on the product mass*prior_covariance, so it cannot be told apart
    from a prior-covariance parametrisation without also changing the mixing amplitudes,
    which the harmonic-oscillator flow fixes). The rotation preserves
    ``0.5*q@q + 0.5*p@M_inv@p`` EXACTLY for any positive-definite mass ``M`` (verified
    2026-09-18, not only for ``sd_or_cov == 1`` as an earlier version of this docstring
    and its test claimed) -- but it is prior-preserving (i.e. it makes the proposal
    exactly reversible w.r.t. an N(0,I) target with no extra prior-ratio correction) only
    when the INTERNAL prior actually is N(0, I). That holds for every prior class this
    library ships with an internal-space design (`PriorIndependentComponents`); a
    hand-built `Normal` prior with `sd != 1` / `cov != I` would still be a VALID
    Metropolis-Hastings proposal (the map stays volume-preserving and reversible --
    composition of symplectic maps and a momentum flip -- and the real prior ratio still
    enters through `get_log_acceptance_probability`), just an inefficient one for that
    prior. `sd_or_cov=1` (the default) is the canonical dimension-robust choice for the
    N(0,I) internal prior (mass = inverse prior covariance, i.e. mass = I here).
    """

    def _apply_prior_kinetic_flow(self, q: npt.NDArray, p: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        if self._mass_is_diagonal:
            frequencies = 1.0 / self._mass_sd
            angles = self.step_size * frequencies
            cos_angles = np.cos(angles)
            sin_angles = np.sin(angles)
            q_new = cos_angles * q + (sin_angles / self._mass_sd) * p
            p_new = cos_angles * p - (self._mass_sd * sin_angles) * q
            return q_new, p_new

        sqrt_eigenvalues = np.sqrt(self._mass_eigenvalues)
        angles = self.step_size / sqrt_eigenvalues
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        q_eigen = self._mass_eigenvectors.T @ q
        p_eigen = self._mass_eigenvectors.T @ p
        q_eigen_new = cos_angles * q_eigen + (sin_angles / sqrt_eigenvalues) * p_eigen
        p_eigen_new = cos_angles * p_eigen - (sqrt_eigenvalues * sin_angles) * q_eigen
        q_new = self._mass_eigenvectors @ q_eigen_new
        p_new = self._mass_eigenvectors @ p_eigen_new
        return q_new, p_new

    def _leapfrog(self, q0: npt.NDArray, p0_start: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Gaussian prior preserving integrator for Hamiltonian dynamics.
        
        Parameters:
            q0: ndarray, shape (no_parameters,) — initial position
            p0_start: ndarray, shape (no_parameters,) — initial velocity
        
        Returns:
            q: ndarray — final position
            p_end: ndarray — final velocity (after velocity flip)

        Divergent trajectories are abandoned as in ``Hamiltonian._leapfrog`` (2026-09-20); the
        checks do not touch the state, so a finite trajectory is bit-identical to the version
        without them.
        """
        q = q0.copy()
        p = p0_start.copy()
        p = p - 0.5 * self.step_size * self.log_likelihood_gradient(q)  # half step for momentum
        if not _is_finite(p):
            return q, -p
        for i in range(self.num_steps):
            q, p = self._apply_prior_kinetic_flow(q, p)
            if not (_is_finite(q) and _is_finite(p)):
                return q, -p
            if i < self.num_steps - 1:
                p = p - self.step_size * self.log_likelihood_gradient(q)  # full step for momentum (except at the end)
                if not _is_finite(p):
                    return q, -p
        p = p - 0.5 * self.step_size * self.log_likelihood_gradient(q)  # final half step for momentum
        p_end = -p  # negate momentum (for reversibility — does not affect acceptance)
        return q, p_end


#: safety bound on ``|log eps - mu|`` in the dual-averaging recursion. ``H_bar`` is bounded by 1
#: in absolute value, so ``log eps`` can transiently reach ``mu +- 20*sqrt(m)`` before the
#: feedback catches up; clipping keeps ``exp`` from overflowing to ``inf`` (which would poison
#: the leapfrog with NaNs) without touching any converged run -- ``exp(50) ~ 5e21``.
_DUAL_AVERAGING_LOG_STEP_CLIP = 50.0


class _DualAveragingStepSize:
    """
    Dual-averaging adaptation of the leapfrog ``step_size`` (Hoffman & Gelman 2014, *The
    No-U-Turn Sampler*, JMLR 15, Algorithm 5 / §3.2.1), mixed into the Hamiltonian proposals
    (new 2026-09-20, recommendation 4 of `16` §5; prototype `RESULTS_PROTO.md` §5).

    Per update::

        alpha        = min(1, exp(la))
        m           += 1
        H_bar        = (1 - 1/(m+t0)) * H_bar + (delta - alpha)/(m+t0)
        log_eps      = mu - sqrt(m)/gamma * H_bar          # the step size actually used next
        eta          = m**(-kappa)
        log_eps_bar  = eta*log_eps + (1-eta)*log_eps_bar   # the averaged, frozen step size

    with ``mu = log(10*step_size_0)``, ``gamma = 0.05``, ``t0 = 10``, ``kappa = 0.75`` and
    ``delta = target_rate`` defaulting to **0.8** (not Beskos et al.'s 0.65: at 0.65 the
    prototype overshot by ~2x into the ``eps*L ~ 2*pi`` resonance when the mass was
    well-conditioned, `RESULTS_PROTO.md` §5.2).

    The first few updates deliberately push the step size UP (``mu = log(10*step_size_0)``), which
    can send an early leapfrog trajectory off to non-finite positions; that is harmless, because a
    divergent trajectory is abandoned by the integrator, rejected without any model evaluation and
    without being sent to the surrogate collector (``AlgorithmBase._reject_nonfinite_proposal``),
    and scores ``alpha = 0`` here -- exactly the "divergent transition" signal the recursion needs.

    In a DAMH stage one update is done per **sub-chain** step, on the sub-chain's own acceptance
    probability against the surrogate posterior: the leapfrog integrates the SURROGATE gradient
    field, so that is the Hamiltonian whose energy error the step size controls, and the outer
    DAMH acceptance is flat over a 20x step range and therefore uninformative (`16` §5 item 4,
    `15` §2.7). In an MH stage the single outer acceptance probability is used.

    The mass (``sd_or_cov``) and ``num_steps`` are NOT adapted. Only ``step_size`` changes, and
    nothing derived from it is cached: ``Hamiltonian._leapfrog`` reads ``self.step_size`` per
    step and ``HamiltonianInfinite._apply_prior_kinetic_flow`` recomputes its rotation angles
    (``step_size / sqrt(mass eigenvalue)``) on every call, so the integrator can never run with
    a stale angle. ``set_step_size`` is the single place that changes it.
    """

    adaptation_stats_header = ["m", "mean_acceptance_probability", "log_step_size",
                              "log_step_size_bar"]

    def __init__(self, *args, target_rate: float = 0.8, gamma: float = 0.05, t0: float = 10.0,
                 kappa: float = 0.75, period: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]
        step_size_0 = float(self.step_size)  # type: ignore[attr-defined]
        if not step_size_0 > 0.0:
            raise ValueError(f"dual averaging needs a positive initial step_size, got {step_size_0}")
        self.target_rate = float(target_rate)
        self.gamma = float(gamma)
        self.t0 = float(t0)
        self.kappa = float(kappa)
        self.period = int(period)
        self.mu = float(np.log(10.0 * step_size_0))
        self.H_bar = 0.0
        self.log_eps_bar = float(np.log(step_size_0))
        self.m = 0
        self._period_alpha_sum = 0.0
        self._period_alpha_count = 0
        self.adaptation_stats_rows: list[tuple] = []

    def set_step_size(self, step_size: float) -> None:
        """Install a new leapfrog step size (nothing derived from it is cached, see the docstring)."""
        self.step_size = float(step_size)  # type: ignore[attr-defined]

    def adapt(self, proposed_sample: npt.NDArray, log_acceptance_probability: float,
              current_sample: npt.NDArray,
              subchain_log_acceptance_probabilities: list[float] | None = None) -> None:
        if subchain_log_acceptance_probabilities is None:
            log_probabilities = [log_acceptance_probability]
        else:
            log_probabilities = list(subchain_log_acceptance_probabilities)
        for log_probability in log_probabilities:
            self._dual_averaging_update(acceptance_probability(log_probability))

    def _dual_averaging_update(self, alpha: float) -> None:
        self.m += 1
        m = self.m
        self.H_bar = (1.0 - 1.0 / (m + self.t0)) * self.H_bar + (self.target_rate - alpha) / (m + self.t0)
        log_eps = self.mu - np.sqrt(m) / self.gamma * self.H_bar
        log_eps = float(np.clip(log_eps, self.mu - _DUAL_AVERAGING_LOG_STEP_CLIP,
                                self.mu + _DUAL_AVERAGING_LOG_STEP_CLIP))
        eta = m ** (-self.kappa)
        self.log_eps_bar = eta * log_eps + (1.0 - eta) * self.log_eps_bar
        self.set_step_size(np.exp(log_eps))

        self._period_alpha_sum += alpha
        self._period_alpha_count += 1
        if m % self.period == 0:
            self.adaptation_stats_rows.append((
                m, self._period_alpha_sum / max(self._period_alpha_count, 1),
                log_eps, self.log_eps_bar))
            self._period_alpha_sum = 0.0
            self._period_alpha_count = 0

    # -- cross-rank hand-over -----------------------------------------------------------
    def adapted_state(self) -> npt.NDArray:
        return np.array([float(self.m), float(self.H_bar), float(self.log_eps_bar), float(self.mu)])

    def set_pooled_state(self, states: npt.NDArray) -> None:
        """Average ``log_eps_bar`` over the chains and freeze the live step size at it.

        ``m``/``H_bar``/``mu`` stay this chain's own: they only drive the recursion, which does
        not continue past the end of the stage (the next stage builds a fresh proposal).
        """
        states = np.atleast_2d(np.asarray(states, dtype=float))
        if states.shape[1] != 4:
            raise ValueError(f"expected adapted-state rows of length 4, got {states.shape}")
        self.log_eps_bar = float(states[:, 2].mean())
        self.set_step_size(np.exp(self.log_eps_bar))

    def carry_over(self) -> dict:
        return {"hamiltonian_step_size": float(np.exp(self.log_eps_bar))}


class Hamiltonian_adaptive(_DualAveragingStepSize, Hamiltonian):
    """``Hamiltonian`` with a dual-averaged leapfrog step size; see ``_DualAveragingStepSize``."""


class HamiltonianInfinite_adaptive(_DualAveragingStepSize, HamiltonianInfinite):
    """``HamiltonianInfinite`` with a dual-averaged leapfrog step size (the split integrator's
    rotation angles are recomputed from ``step_size`` on every call, so they are never stale)."""


class BlockProposal(Proposal):
    """
    Applies a different sub-``Proposal`` to a different, non-overlapping group of
    parameters on each call (``choose_group`` picks the group; only that group is
    updated per proposal). ``needs_gradients`` is True iff any sub-proposal needs
    gradients, in which case ``set_gradient_functions`` wraps the full gradient
    functions so that each Hamiltonian-family sub-proposal only sees its own group's
    slice of the full gradient.

    Gradient evaluation point (WS7, 2026-09-18): the wrapped gradient functions build the
    full argument by putting the sub-proposal's trial coordinates into the active group and
    the CURRENT SAMPLE's coordinates (tracked in ``self.current_sample``, updated by
    ``propose_sample``) into the inactive groups. Before this change the inactive groups were
    zero-filled, which evaluates the gradient at a point the chain is not at unless the model
    is separable (e.g. linear in each group). This was never a correctness bug -- any gradient
    field yields a valid reversible HMC proposal, and the acceptance ratio uses the exact model
    -- but it degraded the leapfrog trajectory for nonlinear models. For a genuinely separable
    model the two versions agree exactly (the inactive coordinates do not enter the active
    group's gradient components at all), so only nonlinear/coupled models see a different
    sample stream.

    Sub-proposal seeding (G5, 2026-09-17): the sub-proposals in ``list_of_proposals`` are
    re-seeded in ``__init__`` from this block's own ``seed`` via ``subproposal_seed``, so the
    seeds they were constructed with in the ``Stage`` are IGNORED. Because ``build_proposal``
    passes a per-rank, per-stage seed, every MPI chain now runs an independent sub-proposal
    stream (before G5 all chains shared one, finding 1.9) and a sub-proposal object reused
    across stages is re-seeded at the start of each stage instead of continuing the previous
    stage's stream (finding A13).
    """

    def __init__(self, no_parameters: int, list_of_proposals: list[Proposal], list_of_groups: list[list[int]], seed: int = 0, group_probabilities: list[float] | None = None) -> None:
        """ Block proposal that applies different proposal distributions to different groups of parameters.
        Args:
            no_parameters: total number of parameters
            list_of_proposals: list of Proposal instances, one for each group
            list_of_groups: list of lists of parameter indices, one for each group. The groups should cover all parameters without overlap.
            group_probabilities: optional list of probabilities for selecting each group. If None, groups are selected uniformly.
            seed: random seed for reproducibility; it also determines the sub-proposals'
                seeds (see the class docstring and ``subproposal_seed``), overriding whatever
                seed each sub-proposal was constructed with.
        """
        super().__init__()
        self.list_of_proposals = list_of_proposals
        self.list_of_groups = list_of_groups
        self.no_parameters = no_parameters
        self._generator = np.random.RandomState(seed=seed)
        self._reseed_subproposals(seed)
        if group_probabilities is None:
            self.group_probabilities = np.full((len(list_of_groups),), 1/len(list_of_groups))
        else:
            self.group_probabilities = np.array(group_probabilities) / np.sum(group_probabilities)
        self.no_groups = len(list_of_groups)
        self.group_index = 0  # will be set in propose_sample
        self.needs_gradients = any(proposal.needs_gradients for proposal in list_of_proposals)
        # Full current state of the chain, kept up to date by propose_sample and used by the
        # wrapped gradient functions to fill in the coordinates of the groups that are NOT
        # being updated (WS7). Zeros until the first propose_sample call, which reproduces the
        # pre-WS7 behaviour for a gradient evaluated before the chain ever moved.
        self.current_sample = np.zeros(no_parameters)

    def _reseed_subproposals(self, seed: int) -> None:
        """Give every sub-proposal its own deterministic stream derived from ``seed`` (G5)."""
        for index, proposal in enumerate(self.list_of_proposals):
            proposal.reseed(subproposal_seed(seed, index))

    def reseed(self, seed: int) -> None:
        """Re-seed this block's group-choice generator AND every sub-proposal (G5)."""
        super().reseed(seed)
        self._reseed_subproposals(seed)

    def choose_group(self) -> None:
        self.group_index = self._generator.choice(self.no_groups, p=self.group_probabilities)
        # TODO: or in a deterministic way (cycling through groups)
        
    def propose_sample(self, current_sample: npt.NDArray) -> npt.NDArray:
        group = self.list_of_groups[self.group_index]
        proposal = self.list_of_proposals[self.group_index]
        # Remember the full current state BEFORE the sub-proposal runs: a Hamiltonian
        # sub-proposal evaluates the wrapped gradient functions inside its leapfrog
        # integrator, and those need the inactive groups' current coordinates (WS7).
        self.current_sample = np.asarray(current_sample, dtype=float).copy()
        # Propose new values for the selected group:
        proposed_sample = current_sample.copy()
        proposed_sample[group] = proposal.propose_sample(current_sample[group])
        return proposed_sample
    
    def get_log_acceptance_probability(self, log_likelihood_proposed: float, log_likelihood_current: float,
                                       log_prior_proposed: float, log_prior_current: float) -> tuple[float, float]:
        # Get acceptance probability from the proposal that was used for the selected group:
        proposal = self.list_of_proposals[self.group_index]
        return proposal.get_log_acceptance_probability(log_likelihood_proposed, log_likelihood_current,
                                                      log_prior_proposed, log_prior_current)

    def _full_gradient_argument(self, x: npt.NDArray, group) -> npt.NDArray:
        """
        Full parameter vector at which a group's gradient contribution is evaluated.

        The active ``group`` gets the sub-proposal's trial coordinates ``x``, every other
        group gets the chain's CURRENT value (``self.current_sample``, WS7) instead of the
        zeros used before. A group's sub-proposal only ever sees ``[group]`` of the result,
        so for a separable model the inactive coordinates cancel out and this is bit-identical
        to the old zero-filled version; for a coupled model it is the gradient at the point
        the chain actually occupies.
        """
        full_argument = self.current_sample.copy()
        full_argument[group] = x
        return full_argument

    def set_gradient_functions(self, log_likelihood_gradient_function: Callable, log_prior_gradient_function: Callable):
        # The sub-proposals are reused unmodified: each one is handed gradient functions that
        # take/return only its own group's coordinates, built from the full ones here.
        for proposal, group in zip(self.list_of_proposals, self.list_of_groups):
            if proposal.needs_gradients:
                def group_log_likelihood_gradient_function(x, group=group):
                    return log_likelihood_gradient_function(self._full_gradient_argument(x, group))[group]
                def group_log_prior_gradient_function(x, group=group):
                    return log_prior_gradient_function(self._full_gradient_argument(x, group))[group]
                proposal.set_gradient_functions(group_log_likelihood_gradient_function, group_log_prior_gradient_function)