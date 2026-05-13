#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from surrDAMH.distributions.parent import Distribution


class Normal(Distribution):
    """
    Normal distribution:
    - univariate N(mean, sd)
    - multivariate with independent components N(mean, sd)
    - multivariate correlated N(mean, cov)
    """

    def __init__(self, mean: npt.ArrayLike | float = 0.0, cov: npt.NDArray | None = None,
                 sd: float | npt.NDArray = 1.0, d: int | None = None):
        """
        Args:
            mean (float | np.ndarray): mean vector (can be scalar, in that case, d has to be given)
            cov (np.ndarray): covariance matrix (optional)
            sd (float | np.ndarray): standard deviation (scalar or vector) (optional, only if cov is None)
            d (int): number of variables (optional)
        """
        if np.isscalar(mean):
            if d is None:
                d = 1
            self.mean = np.full((d,), mean)
        else:
            self.mean = np.array(mean)
        self.n = len(self.mean)

        if cov is not None:  # covariance matrix is given
            self.cov = np.array(cov)
            self.logpdf = self.calculate_logpdf_multivariate
            self.grad_logpdf = self.calculate_grad_logpdf_multivariate
            self.rvs = self.calculate_rvs_multivariate
        else:  # no covarinace matrix, use sd instead
            if np.isscalar(sd):
                self.sd = np.full((self.n,), sd)
            else:
                self.sd = np.array(sd)
            # sd is a numpy array of shape (n,)
            self.logpdf = self.calculate_logpdf_uncorrelated
            self.grad_logpdf = self.calculate_grad_logpdf_uncorrelated
            self.rvs = self.calculate_rvs_uncorrelated

    def calculate_logpdf_uncorrelated(self, sample):
        """Calculates logpdf of N(mean,sd) up to an additive constant."""
        v = self.mean - sample
        invCv = v/(self.sd**2)
        return -0.5*np.dot(v, invCv)

    def calculate_grad_logpdf_uncorrelated(self, sample):
        """Gradient of logpdf of N(mean,sd) up to an additive constant."""
        return (self.mean - sample)/(self.sd**2)

    def calculate_logpdf_multivariate(self, sample):
        """Calculates logpdf of N(mean,cov) up to an additive constant."""
        v = self.mean - sample.ravel()
        invCv = np.linalg.solve(self.cov, v)
        return -0.5*np.dot(v, invCv)

    def calculate_grad_logpdf_multivariate(self, sample):
        """Gradient of logpdf of N(mean,cov) up to an additive constant."""
        v = self.mean - sample.ravel()
        return np.linalg.solve(self.cov, v)

    def get_covariance(self) -> npt.NDArray:
        """Returns covariance matrix or vector of standard deviations."""
        if hasattr(self, 'cov'):
            return self.cov
        else:
            return self.sd

    def calculate_rvs_uncorrelated(self):
        """Returns a random sample from N(mean,sd)."""
        return np.random.randn(self.n) * self.sd + self.mean

    def calculate_rvs_multivariate(self):
        """Returns a random sample from N(mean,cov)."""
        return np.random.multivariate_normal(self.mean, self.cov)
