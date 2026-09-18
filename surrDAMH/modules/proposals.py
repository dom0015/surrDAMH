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
    covariance matrix, and ``GaussRandomWalk_adaptive`` REPLACES a 1-D ``sd_or_cov`` by a 2-D
    covariance the first time it adapts. The per-stage reduction of the adaptive covariance
    (``process_SAMPLER``'s ``Allreduce``) therefore used to mix 1-D and 2-D buffers across
    ranks whenever the ranks' ``adapt()`` counts straddled the adaptation period, which
    aborted the whole job with ``Message truncated`` (finding 2.5). Both runners now call this
    function on every rank before the reduction, so the shapes always agree and the next stage
    always receives a 2-D matrix (G2, 2026-09-17).

    Consequence for the edge case where NO rank ever adapted: the next stage is handed
    ``diag(sd**2)`` instead of the 1-D ``sd`` vector, i.e. it draws with
    ``multivariate_normal`` instead of independent ``normal`` calls -- the same distribution
    but a different RNG stream, so such runs are not bit-identical to before G2.

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

    def adapt(self, **kwargs):
        """
        Adapts the proposal distribution.
        """
        pass

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
    ``GaussRandomWalk`` whose covariance is periodically re-estimated from the
    accepted-vs-rejected history (Haario-style adaptation) to chase ``target_rate``.

    ``target_rate``, ``corr_limit`` and ``sample_limit`` are set from
    ``Stage.adaptive_target_rate`` / ``adaptive_corr_limit`` / ``adaptive_sample_limit``
    by ``build_proposal`` whenever those fields are not ``None`` (G1, 2026-09-17); a stage
    that leaves them ``None`` gets the constructor defaults below, which are the values
    hard-coded before G1. With ``sample_limit=None`` (the default) the sample history
    accumulates without bound across the whole stage; with an integer only the last
    ``sample_limit`` proposals are kept, which bounds memory and makes the adaptation
    forget the early, badly-scaled part of the chain.

    Note for DAMH stages: ``adapt()`` is only called once per OUTER step (from the exact
    acceptance test on the sub-chain endpoint), so ``target_rate`` is a target for the
    outer/second-stage acceptance rate, not the sub-chain rate.
    """

    def __init__(self, no_parameters: int, sd_or_cov: npt.ArrayLike = 1.0, seed: int = 0,
                 target_rate: float = 0.25, corr_limit: float = 0.3,
                 period: int = 10, sample_limit: int | None = None) -> None:
        """
        Args:
            target_rate (float): target acceptance rate
            corr_limit (float): maximal alowed correlation of proposal distribution
            period (int): number of proposed samples to adapt
            sample_limit (int | None): if given, only the last ``sample_limit`` proposed
                samples/acceptance weights are kept and used to estimate the covariance
                and the current acceptance rate; ``None`` keeps the whole history.
        """
        Proposal.__init__(self)  # not GaussRandomWalk.__init__: covariance is set below
        self.no_parameters = no_parameters
        self._generator = np.random.RandomState(seed=seed)
        self.set_covariance(sd_or_cov=sd_or_cov)

        self.target_rate = target_rate
        self.corr_limit = corr_limit
        self.period = period
        self.sample_limit = sample_limit
        self.counter = 0  # counter of proposed samples

        self.samples = np.empty((0, self.no_parameters))
        self.aweights = np.empty((0,), dtype=float)
        self.init_flag = True
        self.coef = 1

        if self.sd_or_cov.ndim == 1:
            self.initial_sd = self.sd_or_cov
        else:
            self.initial_sd = np.sqrt(np.diag(self.sd_or_cov))

    def adapt(self, proposed_sample: npt.NDArray, log_acceptance_probability: float):
        self.samples = np.vstack((self.samples, proposed_sample))
        # TODO: temporary thing?
        acceptance_probability = min(1.0, np.exp(log_acceptance_probability))
        self.aweights = np.append(self.aweights, acceptance_probability)
        if self.sample_limit is not None and len(self.aweights) > self.sample_limit:
            # bounded history (G1, Stage.adaptive_sample_limit): keep only the most recent
            # sample_limit proposals; with sample_limit=None nothing is trimmed, which is
            # bit-identical to the pre-G1 behaviour.
            self.samples = self.samples[-self.sample_limit:]
            self.aweights = self.aweights[-self.sample_limit:]

        self.counter += 1
        if self.counter % self.period == 0:
            # Guard (WS7 "NaN-safe weights", finding A15c/d): if every acceptance weight in
            # this period is zero (e.g. all proposals in the period were rejected) or
            # non-finite, np.cov(..., aweights=...) divides by their sum and raises
            # ZeroDivisionError; even when it does not raise, a degenerate weight set can
            # yield a covariance with a zero/non-finite diagonal. In either case skip this
            # adaptation step and keep the previous proposal covariance instead of crashing
            # or corrupting sd_or_cov; the arithmetic below is otherwise unchanged, so every
            # non-degenerate adaptation is bit-identical to before this guard was added.
            weights_sum = np.sum(self.aweights)
            degenerate = not np.isfinite(weights_sum) or weights_sum == 0 or np.any(~np.isfinite(self.aweights))
            sample_cov = None
            diag = None
            if not degenerate:
                sample_cov = np.cov(self.samples, aweights=self.aweights, rowvar=False)
                diag = np.diag(sample_cov)
                degenerate = bool(np.any(~np.isfinite(diag)) or np.any(diag == 0))
            if degenerate:
                warnings.warn(
                    "adaptive proposal: degenerate acceptance weights/covariance in this "
                    "period, keeping the previous proposal covariance",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                current_rate = np.mean(self.aweights)
                sd = np.sqrt(diag)
                sample_corr = sample_cov/sd.reshape((self.no_parameters, 1))
                sample_corr = sample_corr/sd.reshape(1, self.no_parameters)
                sample_corr[sample_corr < -self.corr_limit] = -self.corr_limit
                sample_corr[sample_corr > self.corr_limit] = self.corr_limit
                np.fill_diagonal(sample_corr, 1.0)
                sample_cov = sample_corr*sd.reshape((self.no_parameters, 1))
                sample_cov = sample_cov*sd.reshape((1, self.no_parameters))
                if self.init_flag:
                    self.init_flag = False
                    self.coef = np.mean(self.initial_sd/sd)
                ratio = current_rate/self.target_rate
                if ratio > 1.2:  # acceptance rate is too high:
                    self.coef = self.coef*min(ratio**(2/self.no_parameters), 2.0)
                    self.set_covariance(self.coef*sample_cov)
                elif (1/ratio) > 1.2:  # acceptance rate is too low:
                    self.coef = self.coef*max(ratio**(2/self.no_parameters), 0.5)
                    self.set_covariance(self.coef*sample_cov)


class Hamiltonian(Proposal):
    """
    Standard Hamiltonian (leapfrog) proposal with mass matrix ``sd_or_cov``.
    ``needs_gradients=True``: requires ``set_gradient_functions`` to be called before
    ``propose_sample`` (``build_proposal``/the algorithm classes wire this to the
    surrogate's ``vjp``/``jacobian``, never the exact model -- see
    ``docs/concepts.md``), and requires ``Configuration.use_surrogate_gradients=True``
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
        """
        q = q0.copy()
        p = p0_start.copy()
        p = p - 0.5 * self.step_size * (self.log_likelihood_gradient(q) + self.log_prior_gradient(q))  # half step for momentum
        for i in range(self.num_steps): 
            q = q + self.step_size * self.M_inv @ p  # full step for position
            if i < self.num_steps - 1:
                p = p - self.step_size * (self.log_likelihood_gradient(q) + self.log_prior_gradient(q))  # full step for momentum (except at the end)
        p = p - 0.5 * self.step_size * (self.log_likelihood_gradient(q) + self.log_prior_gradient(q))  # final half step for momentum
        p_end = -p  # negate momentum (for reversibility — does not affect acceptance)
        return q, p_end


class HamiltonianInfinite(Hamiltonian):
    """
    Hamiltonian proposal whose free flow is the exact solution of the harmonic oscillator
    H(q, p) = 0.5 q^T q + 0.5 p^T M^{-1} p (a rotation in phase space), so that only the
    likelihood gradient is integrated numerically (split integrator).

    The rotation is prior-preserving only when the INTERNAL prior is N(0, I) (zero mean,
    identity covariance) and the mass matrix is M = diag(sd^2) built from ``sd_or_cov``
    (or the full matrix passed as ``sd_or_cov``). For any other prior the map is still
    volume-preserving and reversible (composition of symplectic maps and a momentum flip),
    hence a valid Metropolis-Hastings proposal whose prior term enters through
    ``get_log_acceptance_probability``; it is just no longer prior-preserving.
    The relation between the mass matrix and the prior covariance is documented as-is
    (library_notes/09 decision 8).
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
        """
        q = q0.copy()
        p = p0_start.copy()
        p = p - 0.5 * self.step_size * self.log_likelihood_gradient(q)  # half step for momentum
        for i in range(self.num_steps):
            q, p = self._apply_prior_kinetic_flow(q, p)
            if i < self.num_steps - 1:
                p = p - self.step_size * self.log_likelihood_gradient(q)  # full step for momentum (except at the end)
        p = p - 0.5 * self.step_size * self.log_likelihood_gradient(q)  # final half step for momentum
        p_end = -p  # negate momentum (for reversibility — does not affect acceptance)
        return q, p_end


class BlockProposal(Proposal):
    """
    Applies a different sub-``Proposal`` to a different, non-overlapping group of
    parameters on each call (``choose_group`` picks the group; only that group is
    updated per proposal). ``needs_gradients`` is True iff any sub-proposal needs
    gradients, in which case ``set_gradient_functions`` wraps the full gradient
    functions to only expose that group's slice to each Hamiltonian-family
    sub-proposal (other groups' gradient components are zeroed, not the sample's
    actual values -- a known inefficiency for nonlinear models, not a correctness
    bug: any gradient field still yields a valid reversible HMC proposal).

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

    def set_gradient_functions(self, log_likelihood_gradient_function: Callable, log_prior_gradient_function: Callable):
        for proposal, group in zip(self.list_of_proposals, self.list_of_groups):
            if proposal.needs_gradients:
                # TODO: this is a bit hacky, we create gradient functions for the group that call the full gradient functions with zeroes for other parameters. It would be cleaner if the proposal could directly call the full gradient functions with the full parameter vector and just use the relevant parts, but this way we can reuse existing proposals without modification.
                def group_log_likelihood_gradient_function(x, group=group):
                    full_argument = np.zeros(self.no_parameters)
                    full_argument[group] = x
                    return log_likelihood_gradient_function(full_argument)[group]
                def group_log_prior_gradient_function(x, group=group):
                    full_argument = np.zeros(self.no_parameters)
                    full_argument[group] = x
                    return log_prior_gradient_function(full_argument)[group]
                proposal.set_gradient_functions(group_log_likelihood_gradient_function, group_log_prior_gradient_function)