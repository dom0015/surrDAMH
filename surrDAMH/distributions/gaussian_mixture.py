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
    No ``transform`` override (identity: internal space == physical space).

    Notes:
        Known, currently-unfixed edge cases (pinned by ``tests/unit/test_distributions.py``,
        see ``library_notes/10_manual_review_notes.md`` §3, finding 1.11): far from every
        component, ``logpdf`` returns ``-inf`` (``log(0)``, no warning) and
        ``grad_logpdf`` returns an all-zero vector instead of pointing back towards the
        mixture; ``rvs(n)`` returns shape ``(n, d)`` even for the default ``n=1``, unlike
        the ``(d,)`` returned by ``Normal``/``PriorIndependentComponents.rvs()``.
    """

    def __init__(self, means: List[npt.NDArray], covs: List[npt.NDArray] | None = None, weights: List[float] | None = None):
        """
        Args:
            means (List of np.ndarray): list of mean vectors for each component
            covs (List of np.ndarray): list of covariance matrices (optional) for each component
            weights (List of float): list of weights for each component
        """
        self.n_components = len(means)  # number of components in the mixture
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

    def grad_logpdf(self, sample: npt.NDArray) -> npt.NDArray:
        """Gradient of the Gaussian mixture log-density."""
        sample = np.asarray(sample)
        component_pdfs = np.empty(self.n_components, dtype=float)
        component_grads = np.empty((self.n_components, self.d), dtype=float)
        for i in range(self.n_components):
            mvn = multivariate_normal(mean=self.means[i], cov=self.covs[i])
            pdf_i = self.weights[i] * mvn.pdf(sample)
            component_pdfs[i] = pdf_i
            component_grads[i, :] = pdf_i * np.linalg.solve(self.covs[i], self.means[i] - sample)

        mixture_pdf = np.sum(component_pdfs)
        if mixture_pdf <= 0.0:
            return np.zeros(self.d)
        return np.sum(component_grads, axis=0) / mixture_pdf

    def rvs(self, n: int = 1, generator: np.random.Generator | None = None) -> npt.NDArray:
        """Generate random samples from the Gaussian mixture.

        ``generator`` (G4): draw both the component choice and the component sample from this
        ``np.random.Generator`` instead of the global, unseeded NumPy RNG. ``None`` keeps the
        historical global-RNG behaviour bit-for-bit.
        """
        samples = []
        # Choose which component to sample from based on the weights
        if generator is not None:
            components = generator.choice(self.n_components, size=n, p=self.weights)
        else:
            components = np.random.choice(self.n_components, size=n, p=self.weights)

        for comp in components:
            mvn = multivariate_normal(mean=self.means[comp], cov=self.covs[comp])
            samples.append(mvn.rvs() if generator is None else mvn.rvs(random_state=generator))

        return np.array(samples)
