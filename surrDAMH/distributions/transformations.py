#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:10:11 2021

@author: domesova
"""

import numpy as np
import scipy.stats as stats


def normal_to_lognormal(parameters, mu, sigma):
    # N(0,1) to LogN(mu,sigma)
    return np.exp(parameters*sigma+mu)


def lognormal_to_normal(parameters, mu, sigma):
    # LogN(mu,sigma) to N(0,1)
    return (np.log(parameters)-mu)/sigma


def normal_to_uniform(parameters, a=0, b=1, mu=0, sigma=1):
    # N(mu,sigma) to Uni((a,b))
    tmp = stats.norm.cdf(parameters, mu, sigma)
    return a + tmp*(b-a)


def uniform_to_normal(parameters, a=0, b=1, mu=0, sigma=1):
    # Uni((mu,sigma)) to N(0,1)
    tmp = (parameters - a)/(b-a)
    return stats.norm.ppf(tmp, mu, sigma)


def beta_to_uniform(parameters, a=0, b=1, alpha=2, beta=2):
    # Beta(alpha,beta) to Uni((a,b))
    tmp = stats.beta.cdf(parameters, alpha, beta)
    return a + tmp*(b-a)


def uniform_to_beta(parameters, a=0, b=1, alpha=2, beta=2):
    # Uni((a,b)) to Beta(alpha,beta)
    tmp = (parameters - a)/(b-a)
    return stats.beta.ppf(tmp, alpha, beta)


def normal_to_beta(parameters, mu=0, sigma=1, alpha=2, beta=2):
    # N(mu,sigma) to Beta(alpha,beta)
    tmp = normal_to_uniform(parameters, mu=mu, sigma=sigma)
    return uniform_to_beta(tmp, alpha=alpha, beta=beta)


def beta_to_normal(parameters, mu=0, sigma=1, alpha=2, beta=2):
    # Beta(alpha,beta) to N(mu,sigma)
    tmp = beta_to_uniform(parameters, alpha=alpha, beta=beta)
    return uniform_to_normal(tmp, mu=mu, sigma=sigma)
