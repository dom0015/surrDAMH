#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt


class Proposal:
    """Parent class for proposal distributions."""

    def __init__(self) -> None:
        pass

    def propose_sample(self, current_sample: npt.NDArray) -> npt.NDArray:
        """Gets current sample, returns proposed sample."""
        raise NotImplementedError

    def get_log_acceptance_probability(self, log_likelihood_proposed: float, log_likelihood_current: float,
                                       log_prior_proposed: float, log_prior_current: float) -> float:
        """
        Gets logarithms of likelihoods and priors of proposed and current sample.
        Returns logarithm of acceptance probability.
        """
        raise NotImplementedError

    def adapt(self, **kwargs):
        """
        Adapts the proposal distribution.
        """
        pass


class GaussRandomWalk(Proposal):  # initiated by SAMPLERs
    def __init__(self, no_parameters, sd_or_cov=1.0, seed=0) -> None:
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
                                       log_prior_proposed, log_prior_current) -> float:
        # simple since the proposal distribution is symmetrical
        log_posterior_proposed = log_likelihood_proposed + log_prior_proposed
        log_posterior_current = log_likelihood_current + log_prior_current
        return log_posterior_proposed - log_posterior_current


class PCN(Proposal):  # preconditioned Crank-Nicolson proposal
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
                                       log_prior_proposed: float, log_prior_current: float) -> float:
        # pCN: prior cancels by construction, acceptance depends only on likelihood
        return log_likelihood_proposed - log_likelihood_current


class GaussRandomWalk_adaptive(GaussRandomWalk):  # initiated by SAMPLERs
    def __init__(self, no_parameters: int, sd_or_cov: npt.ArrayLike = 1.0, seed: int = 0,
                 target_rate: float = 0.25, corr_limit: float = 0.3,
                 period: int = 10) -> None:
        """
        Args:
            target_rate (float): target acceptance rate
            corr_limit (float): maximal alowed correlation of proposal distribution
            period (int): number of proposed samples to adapt
        """
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

    def adapt(self, proposed_sample: npt.NDArray, acceptance_probability: float):
        self.samples = np.vstack((self.samples, proposed_sample))
        self.aweights = np.append(self.aweights, acceptance_probability)

        self.counter += 1
        if self.counter % self.period == 0:
            current_rate = np.mean(self.aweights)
            sample_cov = np.cov(self.samples, aweights=self.aweights, rowvar=False)
            sd = np.sqrt(np.diag(sample_cov))
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


class Hamiltonian(Proposal):  # Hamiltonian proposal
    def __init__(self, no_parameters, grad_U, step_size=0.1, num_steps=10, sd_or_cov=1.0, seed=0) -> None:
        """
        Hamiltonian proposal with leapfrog integrator for Hamiltonian dynamics.

        Parameters:
            no_parameters: number of parameters
            grad_U: callable — gradient of potential energy
            sd_or_cov: scalar, vector of standard deviations, or covariance matrix
            step_size: float — leapfrog step size
            num_steps: int — number of leapfrog steps
            seed: random seed for reproducibility
        """
        self.no_parameters = no_parameters
        self.grad_U = grad_U
        self.step_size = step_size
        self.num_steps = num_steps
        self._generator = np.random.RandomState(seed=seed)
        self.set_covariance(sd_or_cov=sd_or_cov)
        self.p_current: npt.NDArray # current momentum, will be set in propose_sample
        self.p_proposed: npt.NDArray # proposed momentum, will be set in propose_sample
        self.M_inv: npt.NDArray  # inverse mass matrix, will be set in set_covariance

    def set_covariance(self, sd_or_cov: npt.ArrayLike) -> None:
        # sd_or_cov is scalar/vector/covariance matrix:
        if np.isscalar(sd_or_cov):
            self.sd_or_cov = np.full((self.no_parameters,), sd_or_cov)
        else:
            self.sd_or_cov = np.array(sd_or_cov)
        if self.sd_or_cov.ndim == 1:  # proposal - normal uncorrelated
            self.generate_normal_sample = self._generate_normal_sample_uncorrelated
            variances = self.sd_or_cov**2
            self.M_inv = np.diag(1/variances)
        else:  # proposal - normal correlated
            self.generate_normal_sample = self._generate_normal_sample_multivariate
            self.M_inv = np.linalg.inv(self.sd_or_cov)

    def _generate_normal_sample_uncorrelated(self, mean: npt.NDArray) -> npt.NDArray:
        return self._generator.normal(mean, self.sd_or_cov)

    def _generate_normal_sample_multivariate(self, mean: npt.NDArray) -> npt.NDArray:
        return self._generator.multivariate_normal(mean, self.sd_or_cov)

    def propose_sample(self, current_sample: npt.NDArray) -> npt.NDArray:
        p0 = self.generate_normal_sample(np.zeros((self.no_parameters,)))
        q, p = self._leapfrog(current_sample, p0)
        self.p_current = p0
        self.p_proposed = p
        return q

    def get_log_acceptance_probability(self, log_likelihood_proposed, log_likelihood_current,
                                       log_prior_proposed, log_prior_current) -> float:
        # Total Hamiltonian H(q, p) = U(q) + 0.5 * p^T M^{-1} p
        #    return U(q) + 0.5 * p @ M_inv @ p
        H_current  = -log_likelihood_current - log_prior_current + 0.5 * self.p_current @ self.M_inv @ self.p_current
        H_proposed = -log_likelihood_proposed - log_prior_proposed + 0.5 * self.p_proposed @ self.M_inv @ self.p_proposed
        return H_current - H_proposed  # log acceptance probability = - (H_proposed - H_current)

    def _leapfrog(self, q0: npt.NDArray, p0: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Leapfrog integrator for Hamiltonian dynamics.
        
        Parameters:
            q0: ndarray, shape (no_parameters,) — initial position
            p0: ndarray, shape (no_parameters,) — initial momentum
        
        Returns:
            q: ndarray — final position
            p: ndarray — final momentum (after momentum flip)
        """
        q = q0.copy()
        p = p0.copy()
        p = p - 0.5 * self.step_size * self.grad_U(q)  # half step for momentum
        for i in range(self.num_steps): 
            q = q + self.step_size * self.M_inv @ p  # full step for position
            if i < self.num_steps - 1:
                p = p - self.step_size * self.grad_U(q)  # full step for momentum (except at the end)
        p = p - 0.5 * self.step_size * self.grad_U(q)  # final half step for momentum
        p = -p  # negate momentum (for reversibility — does not affect acceptance)
        return q, p