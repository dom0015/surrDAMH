#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy.typing as npt


class Distribution:
    """
    Parent class for prior distributions and likelihoods.
    The prior distribution is a function composition: transform(internal_prior()).
    """

    def __init__(self) -> None:
        self.mean = 0.0
        pass

    def transform(self, sample: npt.NDArray) -> npt.NDArray:
        """
        The sample is transformed before the solver is applied to it,
        and (optionally) before writing to a file.
        If not overridden, it remains the identity.
        """
        return sample

    def logpdf(self, sample: npt.NDArray) -> float:
        """
        Returns logarithm of pdf in given sample UP TO AN ADDITIVE CONSTANT.
        (If transformation is used, log-pdf of INTERNAL prior distribution is returned.)
        """
        return 0.0

    def grad_logpdf(self, sample: npt.NDArray) -> npt.NDArray:
        """
        Returns the gradient of ``logpdf(sample)`` with respect to ``sample``.
        """
        raise NotImplementedError("grad_logpdf() not implemented for " + type(self).__name__)

    def get_covariance(self) -> npt.NDArray:
        """
        Returns the covariance matrix (2D array) or vector of variances (1D array)
        of the distribution. Required by pCN proposal.
        """
        raise NotImplementedError("get_covariance() not implemented for " + type(self).__name__)

    def rvs(self) -> npt.NDArray:
        """
        Returns a random sample from the distribution.
        """
        raise NotImplementedError


class FromScipy(Distribution):
    def __init__(self, scipy_rv) -> None:
        """
        Args:
            scipy_rv: class instance with methods "logpdf" and "rvs"
        """
        self.scipy_rv = scipy_rv
        self.logpdf = scipy_rv.logpdf
        self.rvs = scipy_rv.rvs
        