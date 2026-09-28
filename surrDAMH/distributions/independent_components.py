#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats

import surrDAMH.distributions.transformations as transformations
from surrDAMH.distributions.parent import Distribution


class UnivariateComponent:
    """
    Base class of the components of ``PriorIndependentComponents``: the prior of ONE
    parameter. ``transform`` maps a standard-normal internal value to this component's
    physical value (an inverse-CDF-style map, see ``surrDAMH.distributions.transformations``).
    Subclass it for a marginal not shipped here (``NormalComponent``, ``UniformComponent``,
    ``LognormalComponent``, ``BetaComponent``).
    """

    def __init__(self):
        pass

    def transform(self, parameter):
        """Maps one standard-normal internal value to this component's physical value."""
        raise NotImplementedError

    def pdf(self, x):
        """Evaluate the marginal prior PDF at value(s) *x* in transformed (physical) space."""
        raise NotImplementedError


class PriorIndependentComponents(Distribution):
    """
    Prior with independent parameters, one component per parameter::

        prior = PriorIndependentComponents([
            NormalComponent(mu=0.0, sigma=1.0),
            LognormalComponent(mu=-2.0, sigma=0.5),
            UniformComponent(a=0.0, b=1.0),
        ])

    The chain samples in the standard normal N(0, I) internal space; ``transform`` maps each
    coordinate to its component's distribution for the solver, the surrogate and the saved
    samples. This is what pCN and the dimension-robust Hamiltonian proposal expect.

    Args:
        list_of_components: one component per parameter, in parameter order.
    """
    # Maintainer note: logpdf/grad_logpdf are exactly the standard-normal log-density/gradient
    # with no Jacobian term for transform -- a design decision, not an approximation: MCMC
    # acceptance ratios are computed entirely in the internal space (pinned by
    # tests/unit/test_distributions.py::TestPriorIndependentComponentsLogpdfDesign).

    def __init__(self, list_of_components: list[UnivariateComponent]):
        self.list_of_components = list_of_components
        self.no_parameters = len(list_of_components)
        self.mean = np.zeros((self.no_parameters,))
        self.sd_approximation = np.ones((self.no_parameters,))

    def transform(self, sample):
        # sample ... numpy array of shape (no_parameters,)
        trans_sample = sample.copy()
        for i in range(self.no_parameters):
            trans_sample[i] = self.list_of_components[i].transform(sample[i])
        return trans_sample

    def logpdf(self, sample):
        """
        Returns logarithm of the value of N(zeros,ones) pdf in given sample
        (up to an additive constant).
        """
        return -0.5*np.dot(sample, sample)

    def grad_logpdf(self, sample):
        """Gradient of the internal standard-normal prior log-density."""
        return -np.asarray(sample)

    def get_covariance(self):
        """Returns vector of standard deviations (ones for N(0,I))."""
        return np.ones(self.no_parameters)

    def rvs(self, generator: np.random.Generator | None = None):
        """
        Returns a random sample from N(zeros,ones).

        ``generator`` (G4): draw from this ``np.random.Generator`` instead of the global,
        unseeded NumPy RNG. ``None`` keeps the historical global-RNG behaviour bit-for-bit.
        """
        if generator is not None:
            return generator.standard_normal(self.no_parameters)
        return np.random.randn(self.no_parameters)


class UniformComponent(UnivariateComponent):
    """Uniform prior of one parameter on ``[a, b]``."""

    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a = a
        self.b = b

    def transform(self, parameter):
        return transformations.normal_to_uniform(parameter, a=self.a, b=self.b, mu=0, sigma=1)

    def pdf(self, x):
        return stats.uniform.pdf(x, loc=self.a, scale=self.b - self.a)


class LognormalComponent(UnivariateComponent):
    """Lognormal prior of one parameter: ``log(x) ~ N(mu, sigma)``."""

    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def transform(self, parameter):
        return transformations.normal_to_lognormal(parameter, mu=self.mu, sigma=self.sigma)

    def pdf(self, x):
        return stats.lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu))


class BetaComponent(UnivariateComponent):
    """Beta(alpha, beta) prior of one parameter on ``[0, 1]``."""

    def __init__(self, alpha=2.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta

    def transform(self, parameter):
        return transformations.normal_to_beta(parameter, mu=0, sigma=1, alpha=self.alpha, beta=self.beta)

    def pdf(self, x):
        return stats.beta.pdf(x, self.alpha, self.beta)


class NormalComponent(UnivariateComponent):
    """Normal prior of one parameter: ``N(mu, sigma)`` (``sigma`` is the standard deviation)."""

    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def transform(self, parameter):
        return parameter*self.sigma + self.mu

    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.mu, scale=self.sigma)

