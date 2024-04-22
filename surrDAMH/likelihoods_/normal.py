#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from surrDAMH.likelihoods.parent import Likelihood


class LikelihoodNormal(Likelihood):
    def __init__(self, n=None, mean=0.0, sd=1.0, cov=None):
        self.mean = mean
        if n is None:
            n = len(mean)
        self.n = n

        if cov is not None:  # covariance matrix is given
            self.cov = np.array(cov)
            self.logpdf = self.calculate_logpdf_multivariate
        else:  # no covarinace matrix, use sd instead
            if np.isscalar(sd):
                self.sd = np.full((self.n,), sd)
            else:
                self.sd = np.array(sd)
            # sd is a numpy array of shape (n,)
            self.logpdf = self.calculate_logpdf_uncorrelated

    def calculate_logpdf_uncorrelated(self, sample):
        v = self.mean - sample
        invCv = v/(self.sd**2)
        return -0.5*np.dot(v, invCv)

    def calculate_logpdf_multivariate(self, sample):
        v = self.mean - sample.ravel()
        invCv = np.linalg.solve(self.cov, v)
        return -0.5*np.dot(v, invCv)
