#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:10:11 2021

@author: domesova
"""

import numpy as np
import scipy.stats as stats

def transform(data, settings):
    # data ... numpy array of shape (no_parameters,)
    # settings ... list of lists [type, optionsl inputs]
    trans_data = data.copy()
    for i,parameter in enumerate(data):
        if settings[i] is None:
            pass
        else:
            trans_type = settings[i][0]
            if trans_type=="normal_to_lognormal":
                options = settings[i][1]
                trans_data[i] = normal_to_lognormal(data[i],**options)
            elif trans_type=="normal_to_uniform":
                options = settings[i][1]
                trans_data[i] = normal_to_uniform(data[i],**options)
            elif trans_type=="normal_to_beta":
                options = settings[i][1]
                trans_data[i] = normal_to_beta(data[i],**options)
    return trans_data

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

def beta_to_uniform(parameters, a=0, b=1, alfa=2, beta=2):
    # Beta(alfa,beta) to Uni((a,b))
    tmp = stats.beta.cdf(parameters, alfa, beta)
    return a + tmp*(b-a)

def uniform_to_beta(parameters, a=0, b=1, alfa=2, beta=2):
    # Uni((a,b)) to Beta(alfa,beta)
    tmp = (parameters - a)/(b-a)
    return stats.beta.ppf(tmp, alfa, beta)

def normal_to_beta(parameters, mu=0, sigma=1, alfa=2, beta=2):
    # N(mu,sigma) to Beta(alfa,beta)
    tmp = normal_to_uniform(parameters, mu=mu, sigma=sigma)
    return uniform_to_beta(tmp, alfa=alfa, beta=beta)

def beta_to_normal(parameters, mu=0, sigma=1, alfa=2, beta=2):
    # Beta(alfa,beta) to N(mu,sigma)
    tmp = beta_to_uniform(parameters, alfa=alfa, beta=beta)
    return uniform_to_normal(tmp, mu=mu, sigma=sigma)

## TEST NORMAL-BETA
# mu=0
# sigma=1
# alfa = 5
# beta = 5
# parameters = np.random.randn(100,500)
# parameters = parameters*sigma+mu
# Y=normal_to_beta(parameters, mu=mu, sigma=sigma, alfa=alfa, beta=beta)
# X=beta_to_normal(Y, mu=mu, sigma=sigma, alfa=alfa, beta=beta)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(parameters.reshape((-1)), bins=100)
# plt.figure()
# plt.hist(Y.reshape((-1)), bins=100)
# plt.figure()
# plt.hist(X.reshape((-1)), bins=100)
# plt.show()

## TEST NORMAL-LOGNORMAL
# mu=-35
# sigma=5
# parameters = np.random.randn(100,500)
# parameters = parameters*sigma+mu
# Y=normal_to_lognormal(parameters)
# X=lognormal_to_normal(Y)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(parameters.reshape((-1)), bins=100)
# plt.figure()
# plt.hist(Y.reshape((-1)), bins=200, range=(1e-18,1e-16))
# plt.figure()
# plt.hist(X.reshape((-1)), bins=100)
# plt.show()