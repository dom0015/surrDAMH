#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from surrDAMH.distributions.parent import Distribution


class Normal(Distribution):
    """
    Gaussian distribution N(mean, sd) or N(mean, cov), usable as prior or likelihood.

    - prior: ``Normal(mean=0, sd=1, dim=no_parameters)``. The sampler standardizes it: the
      chain runs in ``z ~ N(0, I)`` and ``transform(z) = mean + L z`` gives the physical
      parameters (``StandardizedNormal``, applied automatically), so every prior is sampled
      in the same standard-normal internal space.
    - likelihood: ``Normal(mean=observed_data, sd=noise_sd)``, i.e. the data are the mean
      and ``sd`` is the noise level (scalar or one value per observation).
    - correlated: ``Normal(mean=m, cov=C)``.
    """

    def __init__(self, mean: npt.ArrayLike | float = 0.0, cov: npt.NDArray | None = None,
                 sd: float | npt.NDArray = 1.0, dim: int | None = None):
        """
        Args:
            mean: mean vector; a scalar is broadcast to ``dim`` components (``dim`` needed then).
                For a likelihood this is the vector of observed data.
            cov: covariance matrix; if given, ``sd`` is ignored.
            sd: standard deviation, scalar or one value per component (used when ``cov`` is None).
            dim: number of components when ``mean`` is a scalar.
        """
        if np.isscalar(mean):
            if dim is None:
                dim = 1
            self.mean = np.full((dim,), mean)
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

    def calculate_rvs_uncorrelated(self, generator: np.random.Generator | None = None):
        """Returns a random sample from N(mean,sd).

        ``generator`` (G4): draw from this ``np.random.Generator`` instead of the global,
        unseeded NumPy RNG. ``None`` keeps the historical global-RNG behaviour bit-for-bit.
        """
        if generator is not None:
            return generator.standard_normal(self.n) * self.sd + self.mean
        return np.random.randn(self.n) * self.sd + self.mean

    def calculate_rvs_multivariate(self, generator: np.random.Generator | None = None):
        """Returns a random sample from N(mean,cov) (see ``calculate_rvs_uncorrelated``)."""
        if generator is not None:
            return generator.multivariate_normal(self.mean, self.cov)
        return np.random.multivariate_normal(self.mean, self.cov)


class StandardizedNormal(Distribution):
    """
    Internal-space view of a ``Normal`` prior: the chain samples ``z ~ N(0, I)`` and
    ``transform(z) = mean + L z`` (``L L^T = cov``, or ``L = diag(sd)``) gives the physical
    parameters. Built automatically by ``SamplingFramework``, ``run_local`` and ``TestData``
    for every ``Normal`` prior (2026-09-22, author decision), so all priors share the standard
    normal internal space that ``PriorIndependentComponents`` already used: proposal scales,
    pCN and the dimension-robust Hamiltonian proposal then mean the same thing for every prior.

    ``mean`` and ``get_covariance()`` describe the INTERNAL distribution (zeros, ones); the
    physical Gaussian is ``physical``.
    """

    def __init__(self, normal: Normal) -> None:
        self.physical = normal
        self.n = int(normal.n)
        self.no_parameters = self.n
        self.mean = np.zeros(self.n)
        cov = np.asarray(normal.get_covariance(), dtype=float)
        if cov.ndim == 2:
            self._L: npt.NDArray | None = np.linalg.cholesky(cov)
            self._sd = None
        else:
            self._L = None
            self._sd = cov

    def transform(self, sample: npt.NDArray) -> npt.NDArray:
        z = np.asarray(sample, dtype=float)
        if self._L is not None:
            return self.physical.mean + self._L @ z
        return self.physical.mean + self._sd * z

    def logpdf(self, sample: npt.NDArray) -> float:
        """Standard-normal log-density of the INTERNAL sample, up to an additive constant."""
        z = np.asarray(sample, dtype=float)
        return -0.5 * float(np.dot(z, z))

    def grad_logpdf(self, sample: npt.NDArray) -> npt.NDArray:
        return -np.asarray(sample, dtype=float)

    def get_covariance(self) -> npt.NDArray:
        """Vector of internal standard deviations (ones)."""
        return np.ones(self.n)

    def rvs(self, generator: np.random.Generator | None = None) -> npt.NDArray:
        if generator is not None:
            return generator.standard_normal(self.n)
        return np.random.randn(self.n)


def standardize_prior(prior: Distribution) -> Distribution:
    """
    The internal-space prior the sampler works with: a ``Normal`` becomes a
    ``StandardizedNormal``; every other distribution (already internal-space by design, e.g.
    ``PriorIndependentComponents``, or an already standardized one) is returned unchanged.
    Called by ``SamplingFramework.__init__``, ``run_local`` and ``TestData``.
    """
    if isinstance(prior, Normal):
        return StandardizedNormal(prior)
    return prior
