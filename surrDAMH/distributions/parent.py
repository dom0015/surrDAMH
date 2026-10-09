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


def distribution_dimension(dist) -> "int | None":
    """
    Number of components of ``dist`` if it can be read off the object, else ``None``.

    Used by ``surrDAMH.Problem`` to find ``no_parameters`` (prior) and ``no_observations``
    (likelihood). Never draws samples. ``Normal``/``StandardizedNormal``: ``n`` (``None`` for a
    dimension-free ``Normal``, i.e. a scalar ``mean`` without ``dim``);
    ``PriorIndependentComponents``: ``no_parameters``; ``GaussianMixture``: the length of the
    mean vectors; ``FromScipy``: the scipy object's ``dim`` attribute if it has one; any other
    object: its ``no_parameters`` attribute, then its ``dim`` attribute.
    """
    if dist is None or getattr(dist, "dimension_free", False):
        return None
    # local imports: those modules import this one
    from surrDAMH.distributions.gaussian_mixture import GaussianMixture
    from surrDAMH.distributions.independent_components import PriorIndependentComponents
    from surrDAMH.distributions.normal import Normal, StandardizedNormal
    if isinstance(dist, (Normal, StandardizedNormal)):
        return int(dist.n)
    if isinstance(dist, PriorIndependentComponents):
        return int(dist.no_parameters)
    if isinstance(dist, GaussianMixture):
        means = np.asarray(dist.means)
        return int(means.shape[1]) if means.ndim == 2 else None
    if isinstance(dist, FromScipy):
        dim = getattr(dist.scipy_rv, "dim", None)
        return _as_dimension(dim)
    for name in ("no_parameters", "dim"):
        value = _as_dimension(getattr(dist, name, None))
        if value is not None:
            return value
    return None


def internal_centre_and_scale(prior, no_parameters: int) -> "tuple[npt.NDArray, npt.NDArray]":
    """
    Per-coordinate centre ``m`` and scale ``s`` of the INTERNAL prior, used by
    ``surrDAMH.Problem(prior_bound=R)`` for the box ``|u_i - m_i| <= R * s_i`` (S0, 2026-10-09).

    ``StandardizedNormal`` and ``PriorIndependentComponents``: ``m = 0``, ``s = 1`` (standard
    normal internal space). ``GaussianMixture``: the mixture's overall mean and standard
    deviation per coordinate (from ``means``, ``covs``, ``weights``). ``FromScipy``: the wrapped
    object's ``mean`` and ``sqrt(diag(cov))`` when it has both as arrays (a frozen
    ``scipy.stats.multivariate_normal``).

    Raises:
        ValueError: any other prior, or a ``FromScipy`` without array ``mean``/``cov``; the
            message names ``prior_bound=None`` (no bound) as the way out.
    """
    from surrDAMH.distributions.gaussian_mixture import GaussianMixture
    from surrDAMH.distributions.independent_components import PriorIndependentComponents
    from surrDAMH.distributions.normal import StandardizedNormal
    d = int(no_parameters)
    if isinstance(prior, (StandardizedNormal, PriorIndependentComponents)):
        return np.zeros(d), np.ones(d)
    if isinstance(prior, GaussianMixture):
        weights = np.asarray(prior.weights, dtype=float)
        means = np.asarray(prior.means, dtype=float)
        variances = np.array([np.diag(np.asarray(c, dtype=float)) for c in prior.covs])
        centre = weights @ means
        second_moment = weights @ (variances + means ** 2)
        return centre, np.sqrt(np.maximum(second_moment - centre ** 2, 0.0))
    if isinstance(prior, FromScipy):
        mean = getattr(prior.scipy_rv, "mean", None)
        cov = getattr(prior.scipy_rv, "cov", None)
        if not callable(mean) and not callable(cov) and mean is not None and cov is not None:
            mean = np.broadcast_to(np.asarray(mean, dtype=float).ravel(), (d,)).copy()
            cov = np.asarray(cov, dtype=float)
            variances = np.diag(cov) if cov.ndim == 2 else np.broadcast_to(cov.ravel(), (d,))
            return mean, np.sqrt(np.asarray(variances, dtype=float))
        raise ValueError(f"prior_bound needs the centre and scale of the prior, but the wrapped scipy object "
                         f"{type(prior.scipy_rv).__name__} has no array 'mean' and 'cov' (only a frozen "
                         "multivariate_normal has them); pass prior_bound=None to sample the unbounded prior")
    raise ValueError(f"prior_bound needs the per-coordinate centre and scale of the prior, which are not known for "
                     f"{type(prior).__name__} (known: Normal, PriorIndependentComponents, GaussianMixture, "
                     "FromScipy(multivariate_normal)); pass prior_bound=None to sample the unbounded prior")


def _as_dimension(value) -> "int | None":
    """``value`` as a positive int, or ``None`` if it is not an integer (bools excluded)."""
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    if isinstance(value, (int, np.integer)) and int(value) > 0:
        return int(value)
    return None


class Distribution:
    """
    Parent class of priors and likelihoods. Ready-made classes live in
    ``surrDAMH.distributions``: ``Normal``, ``PriorIndependentComponents`` (per-parameter
    Normal/Uniform/Lognormal/Beta components), ``GaussianMixture``, ``FromScipy``.
    See ``docs/concepts.md`` for the internal-vs-physical space design.

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
