#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from surrDAMH.likelihoods.parent import Likelihood


class LikelihoodNormal(Likelihood):
    def __init__(self, no_observations=None, observations=None, sd=1.0, cov=None):
        self.observations = observations
        if no_observations is None:
            no_observations = len(observations)
        self.no_observations = no_observations
        self.noise_mean = np.zeros((no_observations,))

        if cov is not None:  # noise distribution covariance matrix is given
            self.cov = np.array(cov)
            self.calculate_log_likelihoos = self.calculate_log_likelihood_multivariate
        else:  # no covarinace matrix, use sd instead
            if np.isscalar(sd):
                self.sd = np.full((no_observations,), sd)
            else:
                self.sd = np.array(sd)
            # sd is now a numpy array of shape (no_observations,)
            self.calculate_log_likelihood = self.calculate_log_likelihood_uncorrelated

    def calculate_log_likelihood_uncorrelated(self, G_sample):
        v = self.observations - G_sample
        invCv = v/(self.sd**2)
        return -0.5*np.sum(v*invCv)

    def calculate_log_likelihood_multivariate(self, G_sample):
        v = self.observations - G_sample.ravel()  # TODO: ravel? and for normal prior?
        invCv = np.linalg.solve(self.cov, v)
        return -0.5*np.dot(v, invCv)

