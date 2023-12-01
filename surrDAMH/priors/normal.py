#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from surrDAMH.priors.parent import Prior


class PriorNormal(Prior):
    def __init__(self, no_parameters, mean=0.0, sd=1.0, cov=None):
        self.no_parameters = no_parameters
        if np.isscalar(mean):
            self.mean = np.full((no_parameters,), mean)
        else:
            self.mean = np.array(mean)
        # prior mean is now a numpy array of shape (no_parameters,)

        if cov is not None:  # prior covariance matrix is given
            self.cov = np.array(cov)
            self.calculate_log_prior = self.calculate_log_prior_multivariate
            self.sd_approximation = np.sqrt(np.diag(self.cov))
        else:  # no covarinace matrix, use sd instead
            if np.isscalar(sd):
                self.sd = np.full((no_parameters,), sd)
            else:
                self.sd = np.array(sd)
            # prior sd is now a numpy array of shape (no_parameters,)
            self.calculate_log_prior = self.calculate_log_prior_uncorrelated
            self.sd_approximation = self.sd.copy()

    def calculate_log_prior_uncorrelated(self, sample):
        v = sample - self.mean
        invCv = v/(self.sd**2)
        return -0.5*np.dot(v, invCv)

    def calculate_log_prior_multivariate(self, sample):
        v = sample - self.mean
        invCv = np.linalg.solve(self.cov, v)
        return -0.5*np.dot(v, invCv)
