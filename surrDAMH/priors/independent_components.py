#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from surrDAMH.priors.parent import Prior
import surrDAMH.modules.transformations as transformations
import surrDAMH.priors.normal


class UnivariateComponent:
    def __init__(self):
        pass

    def transform(self, parameter):
        pass


class PriorIndependentComponents(Prior):
    """
    Internally, the sampling framework uses the Gaussian prior distribution N(zeros,ones).
    Other distributions are transformed to Gaussian, component by component.
    Available univariate components: Uniform, Normal, Lognormal, Beta
    """

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

    def calculate_log_prior(self, sample):
        """
        Returns logarithm of the value of N(zeros,ones) pdf in given sample
        (up to an additive constant).
        """
        return -0.5*np.dot(sample, sample)


class Uniform(UnivariateComponent):
    """
    U(a,b)
    """

    def __init__(self, a: float = 0, b: float = 1):
        self.a = a
        self.b = b

    def transform(self, parameter):
        return transformations.normal_to_uniform(parameter, a=self.a, b=self.b, mu=0, sigma=1)


class Lognormal(UnivariateComponent):
    """
    Lognormal(mu,sigma)
    """

    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def transform(self, parameter):
        return transformations.normal_to_lognormal(parameter, mu=self.mu, sigma=self.sigma)


class Beta(UnivariateComponent):
    """
    Lognormal(mu,sigma)
    """

    def __init__(self, alpha=2, beta=2):
        self.alpha = alpha
        self.beta = beta

    def transform(self, parameter):
        return transformations.normal_to_beta(parameter, mu=0, sigma=1, alpha=self.alpha, beta=self.beta)


class Normal(UnivariateComponent):
    """
    Normal(mu,sigma)
    """

    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def transform(self, parameter):
        return parameter*self.sigma + self.mu
