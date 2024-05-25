#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:10:11 2021

@author: domesova
"""

import numpy as np
import numpy.typing as npt
import scipy.stats as stats


def normal_to_lognormal(parameters: npt.NDArray, mu: float, sigma: float):
    # N(0,1) to LogN(mu,sigma)
    return np.exp(parameters*sigma+mu)


def lognormal_to_normal(parameters: npt.NDArray, mu: float, sigma: float):
    # LogN(mu,sigma) to N(0,1)
    return (np.log(parameters)-mu)/sigma


def normal_to_uniform(parameters: npt.NDArray, a: float = 0, b: float = 1, mu: float = 0, sigma: float = 1):
    # N(mu,sigma) to Uni((a,b))
    tmp = stats.norm.cdf(parameters, mu, sigma)
    return a + tmp*(b-a)


def uniform_to_normal(parameters: npt.NDArray, a: float = 0, b: float = 1, mu: float = 0, sigma: float = 1):
    # Uni((mu,sigma)) to N(0,1)
    tmp = (parameters - a)/(b-a)
    return stats.norm.ppf(tmp, mu, sigma)


def beta_to_uniform(parameters: npt.NDArray, a: float = 0, b: float = 1, alpha: float = 2, beta: float = 2):
    # Beta(alpha,beta) to Uni((a,b))
    tmp = stats.beta.cdf(parameters, alpha, beta)
    return a + tmp*(b-a)


def uniform_to_beta(parameters: npt.NDArray, a: float = 0, b: float = 1, alpha: float = 2, beta: float = 2):
    # Uni((a,b)) to Beta(alpha,beta)
    tmp = (parameters - a)/(b-a)
    return stats.beta.ppf(tmp, alpha, beta)


def normal_to_beta(parameters: npt.NDArray, mu: float = 0, sigma: float = 1, alpha: float = 2, beta: float = 2):
    # N(mu,sigma) to Beta(alpha,beta)
    tmp = normal_to_uniform(parameters, mu=mu, sigma=sigma)
    return uniform_to_beta(tmp, alpha=alpha, beta=beta)


def beta_to_normal(parameters: npt.NDArray, mu: float = 0, sigma: float = 1, alpha: float = 2, beta: float = 2):
    # Beta(alpha,beta) to N(mu,sigma)
    tmp = beta_to_uniform(parameters, alpha=alpha, beta=beta)
    return uniform_to_normal(tmp, mu=mu, sigma=sigma)
