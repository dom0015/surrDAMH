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

    def get_log_acceptance_probability(self, log_posterior_proposed: float, log_posterior_current: float) -> float:
        """
        Gets logarithms of posteriors of current sample and proposed sample.
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
            self.sd = np.full((self.no_parameters,), sd_or_cov)
        else:
            self.sd = np.array(sd_or_cov)
        if self.sd.ndim == 1:  # proposal - normal uncorrelated
            self.propose_sample = self._propose_sample_uncorrelated
        else:  # proposal - normal correlated
            self.propose_sample = self._propose_sample_multivariate

    def _propose_sample_uncorrelated(self, current_sample: npt.NDArray) -> npt.NDArray:
        proposed_sample = self._generator.normal(current_sample, self.sd)
        return proposed_sample

    def _propose_sample_multivariate(self, current_sample: npt.NDArray) -> npt.NDArray:
        proposed_sample = self._generator.multivariate_normal(current_sample, self.sd)
        return proposed_sample

    def get_log_acceptance_probability(self, log_posterior_proposed, log_posterior_current) -> float:
        # simple since the proposal distribution is symmetrical
        return log_posterior_proposed - log_posterior_current


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
        # idx_accepted = np.empty((0,),dtype=bool)
        self.init_flag = True
        self.coef = 1

        if self.sd.ndim == 1:
            self.initial_sd = self.sd
        else:
            self.initial_sd = np.sqrt(np.diag(self.sd))

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
                # print("COVARIANCE CHANGED (rate too high):", ratio, self.Proposal.proposal_std)
            elif (1/ratio) > 1.2:  # acceptance rate is too low:
                self.coef = self.coef*max(ratio**(2/self.no_parameters), 0.5)
                self.set_covariance(self.coef*sample_cov)
