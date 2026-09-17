#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np
import numpy.typing as npt
from typing import Callable


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
    Sample history accumulates without bound across the whole stage (``self.samples``
    is never trimmed or reset, finding G1) and ``corr_limit`` is fixed at construction
    (``Stage.adaptive_corr_limit``/``adaptive_sample_limit`` are NOT wired to this class,
    see ``stages.py``). Note for DAMH stages: ``adapt()`` is only called once per OUTER
    step (from the exact acceptance test on the sub-chain endpoint), so ``target_rate``
    is a target for the outer/second-stage acceptance rate, not the sub-chain rate.
    """

    def __init__(self, no_parameters: int, sd_or_cov: npt.ArrayLike = 1.0, seed: int = 0,
                 target_rate: float = 0.25, corr_limit: float = 0.3,
                 period: int = 10) -> None:
        """
        Args:
            target_rate (float): target acceptance rate
            corr_limit (float): maximal alowed correlation of proposal distribution
            period (int): number of proposed samples to adapt
        """
        Proposal.__init__(self)  # not GaussRandomWalk.__init__: covariance is set below
        self.no_parameters = no_parameters
        self._generator = np.random.RandomState(seed=seed)
        self.set_covariance(sd_or_cov=sd_or_cov)

        self.target_rate = target_rate
        self.corr_limit = corr_limit
        self.period = period
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

    Known limitation: the sub-proposals' own RNGs are NOT re-seeded per MPI rank
    (finding 1.9/G5), so identical ``seed``s across chains can produce correlated
    group choices/increments across ranks -- do not rely on independence between
    chains built this way until G5 is addressed.
    """

    def __init__(self, no_parameters: int, list_of_proposals: list[Proposal], list_of_groups: list[list[int]], seed: int = 0, group_probabilities: list[float] | None = None) -> None:
        """ Block proposal that applies different proposal distributions to different groups of parameters.
        Args:
            no_parameters: total number of parameters
            list_of_proposals: list of Proposal instances, one for each group
            list_of_groups: list of lists of parameter indices, one for each group. The groups should cover all parameters without overlap.
            group_probabilities: optional list of probabilities for selecting each group. If None, groups are selected uniformly.
            seed: random seed for reproducibility
        """
        super().__init__()
        self.list_of_proposals = list_of_proposals
        self.list_of_groups = list_of_groups
        self.no_parameters = no_parameters
        self._generator = np.random.RandomState(seed=seed)
        if group_probabilities is None:
            self.group_probabilities = np.full((len(list_of_groups),), 1/len(list_of_groups))
        else:
            self.group_probabilities = np.array(group_probabilities) / np.sum(group_probabilities)
        self.no_groups = len(list_of_groups)
        self.group_index = 0  # will be set in propose_sample
        self.needs_gradients = any(proposal.needs_gradients for proposal in list_of_proposals)

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