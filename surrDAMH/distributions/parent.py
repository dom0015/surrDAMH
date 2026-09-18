#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt


def rvs_with_generator(distribution, generator: "np.random.Generator | None"):
    """
    Draw one sample from ``distribution``, using ``generator`` if the object supports it.

    Every ``Distribution`` subclass in this package accepts ``rvs(generator=...)`` since G4
    (2026-09-17). User-supplied or test duck-typed "distributions" may still define
    ``rvs(self)`` with no arguments (e.g. the ``FixedSample`` doubles in ``tests/``); those
    are called without the generator.
    """
    try:
        return distribution.rvs(generator=generator)
    except TypeError:
        # legacy duck-typed object whose rvs() takes no arguments (deterministic in practice)
        return distribution.rvs()


class Distribution:
    """
    Parent class for prior distributions and likelihoods (also used for observation
    noise on the likelihood side). See ``docs/concepts.md`` for the internal-vs-physical
    space design this class formalises.

    A prior is the function composition ``transform ∘ internal_prior``: ``rvs()``/
    ``logpdf()``/``grad_logpdf()`` operate on the INTERNAL sample (what the MCMC chain
    actually stores and proposes on), and ``transform()`` maps it to the PHYSICAL
    parameters the solver/surrogate receives (and what gets written to
    ``samples/*.csv`` when ``Configuration.transform_before_saving=True``, the default).
    For a likelihood, ``transform`` is not used; ``logpdf`` is evaluated directly on
    observations.

    Convention used by every subclass here: ``logpdf`` returns the log-density UP TO AN
    ADDITIVE CONSTANT (the constant cancels in every Metropolis-Hastings acceptance
    ratio, so it is never computed). Do not compare `logpdf` values across different
    ``Distribution`` instances/classes expecting them to be on a common normalized scale.
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
        Returns the covariance matrix (2D array) or the vector of STANDARD DEVIATIONS
        (1D array, not variances) of the distribution, as returned by ``Normal`` and
        expected by the pCN proposal. Required by pCN proposal.
        """
        raise NotImplementedError("get_covariance() not implemented for " + type(self).__name__)

    def rvs(self, generator: np.random.Generator | None = None) -> npt.NDArray:
        """
        Returns a random sample from the distribution.

        Args:
            generator: if given, all randomness is drawn from this ``np.random.Generator``
                instead of the global NumPy RNG, which is what makes
                ``initial_sample_type="prior"`` reproducible (G4, 2026-09-17). ``None``
                keeps the historical behaviour (global, unseeded RNG), so external callers
                that do not pass it are unaffected.
        """
        raise NotImplementedError


class FromScipy(Distribution):
    """
    Wraps a frozen scipy.stats distribution (e.g. ``scipy.stats.norm(...)`` or
    ``scipy.stats.multivariate_normal(...)``) as a ``Distribution``: ``logpdf`` is simply
    the scipy object's own method (scipy's ``logpdf`` is normalized, unlike the "up to a
    constant" convention of the other ``Distribution`` subclasses here — harmless for MCMC
    since only differences matter, but do not rely on the constant). ``rvs`` forwards to the
    scipy object, passing an optional ``generator`` through as scipy's ``random_state``
    (G4); any other argument (``size=...``) is forwarded unchanged.

    Notes:
        No ``transform`` override (identity, i.e. internal space == physical space);
        no ``get_covariance()`` (`NotImplementedError` from the base class) or
        ``grad_logpdf()``, so this class cannot be used with pCN (``build_proposal``
        rejects it, see ``proposal_builder.py:_prior_is_gaussian``) or with any
        gradient-based (Hamiltonian) proposal.
    """

    def __init__(self, scipy_rv) -> None:
        """
        Args:
            scipy_rv: class instance with methods "logpdf" and "rvs"
        """
        self.scipy_rv = scipy_rv
        self.logpdf = scipy_rv.logpdf

    def rvs(self, generator: np.random.Generator | None = None, **kwargs) -> npt.NDArray:
        """Forward to the wrapped scipy object; ``generator`` becomes scipy's ``random_state``.

        Before G4 this was the bound ``scipy_rv.rvs`` itself; a call with no arguments is
        unchanged (scipy still uses the global RNG), so existing callers are unaffected.
        """
        if generator is not None:
            return self.scipy_rv.rvs(random_state=generator, **kwargs)
        return self.scipy_rv.rvs(**kwargs)
