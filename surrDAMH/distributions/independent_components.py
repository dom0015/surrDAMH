#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats

import surrDAMH.distributions.transformations as transformations
from surrDAMH.distributions.parent import Distribution


class UnivariateComponent:
    def __init__(self):
        pass

    def transform(self, parameter):
        pass

    def pdf(self, x):
        """Evaluate the marginal prior PDF at value(s) *x* in transformed space."""
        raise NotImplementedError


class PriorIndependentComponents(Distribution):
    """
    Internally, the sampling framework uses the Gaussian prior distribution N(zeros,ones).
    Other distributions are transformed to Gaussian, component by component.
    Available univariate components: Uniform, Normal, Lognormal, Beta
    """

    def __init__(self, list_of_components: list[UnivariateComponent]):
        """
        Args:
            list_of_components: list of UnivariateComponent instances
        """
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

    def rvs(self):
        """
        Returns a random sample from N(zeros,ones).
        """
        return np.random.randn(self.no_parameters)


class UniformComponent(UnivariateComponent):
    """
    U(a,b)
    """

    def __init__(self, a: float = 0.0, b: float = 1.0):
        self.a = a
        self.b = b

    def transform(self, parameter):
        return transformations.normal_to_uniform(parameter, a=self.a, b=self.b, mu=0, sigma=1)

    def pdf(self, x):
        return stats.uniform.pdf(x, loc=self.a, scale=self.b - self.a)


class LognormalComponent(UnivariateComponent):
    """
    Lognormal(mu,sigma)
    """

    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def transform(self, parameter):
        return transformations.normal_to_lognormal(parameter, mu=self.mu, sigma=self.sigma)

    def pdf(self, x):
        return stats.lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu))


class BetaComponent(UnivariateComponent):
    """
    Beta(alpha,beta)
    """

    def __init__(self, alpha=2.0, beta=2.0):
        self.alpha = alpha
        self.beta = beta

    def transform(self, parameter):
        return transformations.normal_to_beta(parameter, mu=0, sigma=1, alpha=self.alpha, beta=self.beta)

    def pdf(self, x):
        return stats.beta.pdf(x, self.alpha, self.beta)


class NormalComponent(UnivariateComponent):
    """
    Normal(mu,sigma)
    """

    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def transform(self, parameter):
        return parameter*self.sigma + self.mu

    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.mu, scale=self.sigma)


# Backward-compatible aliases for the component-style interface.
Uniform = UniformComponent
Lognormal = LognormalComponent
Beta = BetaComponent
Normal = NormalComponent
