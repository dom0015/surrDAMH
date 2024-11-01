#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt
from typing import List
from scipy.stats import multivariate_normal

from surrDAMH.distributions.parent import Distribution


class GaussianMixture(Distribution):
    """
    Gaussian mixture distribution given by list of means, list of covariance matrices and list of weights.
    """

    def __init__(self, means: List[npt.NDArray], covs: List[npt.NDArray] | None = None, weights: List[float] | None = None):
        """
        Args:
            means (List of np.ndarray): list of mean vectors for each component
            covs (List of np.ndarray): list of covariance matrices (optional) for each component
            weights (List of float): list of weights for each component
        """
        self.n_components = len(self.means)  # number of components in the mixture
        self.d = means[0].shape[0]  # dimension
        if covs is None:
            covs = [np.eye(self.d)] * self.n_components
        if weights is None:
            weights = [1.0] * self.n_components  # default weights

        self.means = np.array(means)
        self.covs = np.array(covs)
        self.weights = np.array(weights)

        # Normalize the weights to ensure they sum to 1
        self.weights /= np.sum(self.weights)

    def logpdf(self, sample: npt.NDArray) -> float:
        """Calculate the log of the probability density function in given sample."""
        log_prob = 0.0
        for i in range(self.n_components):
            # Gaussian PDF for each component
            mvn = multivariate_normal(mean=self.means[i], cov=self.covs[i])
            log_prob += self.weights[i] * mvn.pdf(sample)

        return np.log(log_prob)

    def rvs(self, n: int = 1) -> npt.NDArray:
        """Generate random samples from the Gaussian mixture."""
        samples = []
        # Choose which component to sample from based on the weights
        components = np.random.choice(self.n_components, size=n, p=self.weights)

        for comp in components:
            mvn = multivariate_normal(mean=self.means[comp], cov=self.covs[comp])
            samples.append(mvn.rvs())

        return np.array(samples)
