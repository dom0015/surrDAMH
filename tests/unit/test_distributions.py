#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ``surrDAMH.distributions`` (Phase 1 items 16-21 of
``library_notes/07_testing_plan.md``).

Written against the classes as they actually exist in ``surrDAMH/distributions/``:
``Normal`` (normal.py), ``PriorIndependentComponents`` + its univariate components
(independent_components.py), ``GaussianMixture`` (gaussian_mixture.py), and the
free functions in ``transformations.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as stats
from scipy.stats import multivariate_normal

import surrDAMH.distributions.transformations as transformations
from surrDAMH.distributions.gaussian_mixture import GaussianMixture
from surrDAMH.distributions.independent_components import (
    BetaComponent,
    LognormalComponent,
    NormalComponent,
    PriorIndependentComponents,
    UniformComponent,
)
from surrDAMH.distributions.normal import Normal

pytestmark = pytest.mark.unit


def _fd_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of scalar function ``f`` at ``x``."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        grad[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


# --------------------------------------------------------------------------------------
# Item 16: Normal.logpdf difference between two points vs scipy, for both constructor
# branches (sd scalar/vector, and full cov).
# --------------------------------------------------------------------------------------
class TestNormalLogpdfMatchesScipy:
    def test_sd_scalar(self):
        rng = np.random.default_rng(0)
        d = 4
        mean = rng.normal(size=d)
        sd = 2.5
        dist = Normal(mean=mean, sd=sd)
        mvn = multivariate_normal(mean=mean, cov=(sd ** 2) * np.eye(d))
        x1, x2 = rng.normal(size=d), rng.normal(size=d)
        diff_ours = dist.logpdf(x1) - dist.logpdf(x2)
        diff_scipy = mvn.logpdf(x1) - mvn.logpdf(x2)
        assert diff_ours == pytest.approx(diff_scipy, abs=1e-10)

    def test_sd_vector(self):
        rng = np.random.default_rng(1)
        d = 5
        mean = rng.normal(size=d)
        sd = rng.uniform(0.5, 3.0, size=d)
        dist = Normal(mean=mean, sd=sd)
        mvn = multivariate_normal(mean=mean, cov=np.diag(sd ** 2))
        x1, x2 = rng.normal(size=d), rng.normal(size=d)
        diff_ours = dist.logpdf(x1) - dist.logpdf(x2)
        diff_scipy = mvn.logpdf(x1) - mvn.logpdf(x2)
        assert diff_ours == pytest.approx(diff_scipy, abs=1e-10)

    def test_full_cov(self):
        rng = np.random.default_rng(2)
        d = 3
        mean = rng.normal(size=d)
        A = rng.normal(size=(d, d))
        cov = A @ A.T + d * np.eye(d)  # SPD
        dist = Normal(mean=mean, cov=cov)
        mvn = multivariate_normal(mean=mean, cov=cov)
        x1, x2 = rng.normal(size=d), rng.normal(size=d)
        diff_ours = dist.logpdf(x1) - dist.logpdf(x2)
        diff_scipy = mvn.logpdf(x1) - mvn.logpdf(x2)
        assert diff_ours == pytest.approx(diff_scipy, abs=1e-10)


# --------------------------------------------------------------------------------------
# Item 17: grad_logpdf vs central finite differences.
# --------------------------------------------------------------------------------------
class TestGradLogpdfVsFiniteDifferences:
    def test_normal_sd(self):
        rng = np.random.default_rng(3)
        d = 4
        mean = rng.normal(size=d)
        sd = rng.uniform(0.5, 2.0, size=d)
        dist = Normal(mean=mean, sd=sd)
        x = rng.normal(size=d)
        grad_fd = _fd_grad(dist.logpdf, x)
        assert dist.grad_logpdf(x) == pytest.approx(grad_fd, abs=1e-5)

    def test_normal_cov(self):
        rng = np.random.default_rng(4)
        d = 3
        mean = rng.normal(size=d)
        A = rng.normal(size=(d, d))
        cov = A @ A.T + d * np.eye(d)
        dist = Normal(mean=mean, cov=cov)
        x = rng.normal(size=d)
        grad_fd = _fd_grad(dist.logpdf, x)
        assert dist.grad_logpdf(x) == pytest.approx(grad_fd, abs=1e-5)

    def test_prior_independent_components(self):
        prior = PriorIndependentComponents(
            [UniformComponent(0.0, 1.0), NormalComponent(0.0, 2.0), BetaComponent(2.0, 3.0)]
        )
        x = np.array([0.3, -1.2, 0.5])
        grad_fd = _fd_grad(prior.logpdf, x)
        assert prior.grad_logpdf(x) == pytest.approx(grad_fd, abs=1e-5)

    def test_gaussian_mixture_near_a_component(self):
        means = [np.array([0.0, 0.0]), np.array([5.0, 5.0])]
        covs = [np.eye(2), np.eye(2)]
        mix = GaussianMixture(means, covs)
        x = np.array([0.1, -0.2])
        grad_fd = _fd_grad(mix.logpdf, x)
        assert mix.grad_logpdf(x) == pytest.approx(grad_fd, abs=1e-5)

    def test_gaussian_mixture_far_away_point_is_a_known_unfixed_bug(self):
        """
        finding 1.11 (library_notes/06_findings_consolidated.md): for a point far from
        every mixture component, ``mixture_pdf`` underflows to exactly 0.0 in
        ``GaussianMixture.logpdf``/``grad_logpdf``, giving ``logpdf == -inf`` and
        ``grad_logpdf == 0`` instead of the finite, correctly-signed gradient a
        (numerically stabilised) implementation would give. This is WS6/future-work
        territory, not one of WS0's groups A-E; verified still present by reading
        ``surrDAMH/distributions/gaussian_mixture.py`` and running it directly on
        2026-09-15 (grep for "mixture_pdf <= 0.0" in that file). Test is pinned
        against TODAY's behaviour, not the fix.
        """
        mix = GaussianMixture([np.zeros(2)], [np.eye(2)])
        x = np.array([1000.0, 1000.0])
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            logpdf_value = mix.logpdf(x)
        assert logpdf_value == -np.inf
        assert np.all(mix.grad_logpdf(x) == 0.0)


# --------------------------------------------------------------------------------------
# Item 18: rvs() shape.
# --------------------------------------------------------------------------------------
class TestRvsShape:
    def test_normal_sd(self):
        d = 4
        dist = Normal(mean=np.zeros(d), sd=1.0)
        assert dist.rvs().shape == (d,)

    def test_normal_cov(self):
        d = 3
        dist = Normal(mean=np.zeros(d), cov=np.eye(d))
        assert dist.rvs().shape == (d,)

    def test_prior_independent_components(self):
        prior = PriorIndependentComponents([UniformComponent(), NormalComponent(), BetaComponent()])
        assert prior.rvs().shape == (3,)

    def test_gaussian_mixture_is_a_known_unfixed_bug(self):
        """
        Plan item 18: ``rvs()`` should return shape ``(d,)`` like every other
        ``Distribution`` (this is what ``Proposal.propose_sample`` and
        ``TestData.generate`` expect when calling ``prior.rvs()``). ``GaussianMixture.rvs()``
        instead always builds a Python list of per-draw samples and wraps it with
        ``np.array(...)``, so the default ``n=1`` call returns shape ``(1, d)``. Not
        touched by WS0 groups A-E; verified still present in the working tree on
        2026-09-15. Test pinned against TODAY's behaviour, not the fix.
        """
        d = 2
        mix = GaussianMixture([np.zeros(d)], [np.eye(d)])
        sample = mix.rvs()
        assert sample.shape == (1, d)  # documented bug: a correct rvs() would give (d,)


# --------------------------------------------------------------------------------------
# Item 19: transform round-trip x -> component.transform -> inverse ~= x on [-5, 5].
# --------------------------------------------------------------------------------------
class TestTransformRoundTrip:
    @pytest.mark.parametrize("x", np.linspace(-5, 5, 11))
    def test_uniform(self, x):
        comp = UniformComponent(a=-2.0, b=3.0)
        y = comp.transform(x)
        x_back = transformations.uniform_to_normal(y, a=-2.0, b=3.0, mu=0, sigma=1)
        assert x_back == pytest.approx(x, abs=1e-6)

    @pytest.mark.parametrize("x", np.linspace(-5, 5, 11))
    def test_lognormal(self, x):
        comp = LognormalComponent(mu=0.5, sigma=1.2)
        y = comp.transform(x)
        x_back = transformations.lognormal_to_normal(y, mu=0.5, sigma=1.2)
        assert x_back == pytest.approx(x, abs=1e-8)

    @pytest.mark.parametrize("x", np.linspace(-5, 5, 11))
    def test_beta(self, x):
        comp = BetaComponent(alpha=2.0, beta=3.0)
        y = comp.transform(x)
        x_back = transformations.beta_to_normal(y, mu=0, sigma=1, alpha=2.0, beta=3.0)
        # loosened tolerance at the extremes: beta.cdf/.ppf lose precision near 0/1
        assert x_back == pytest.approx(x, abs=1e-3)


# --------------------------------------------------------------------------------------
# Item 20: marginal KS test of transform(rvs()) against each component's distribution.
# --------------------------------------------------------------------------------------
class TestMarginalKolmogorovSmirnov:
    N = 20_000

    def test_uniform_component(self):
        comp = UniformComponent(a=-1.0, b=4.0)
        rng = np.random.default_rng(42)
        samples = comp.transform(rng.standard_normal(self.N))
        stat, pvalue = stats.kstest(samples, lambda v: stats.uniform.cdf(v, loc=-1.0, scale=5.0))
        assert pvalue > 0.01, f"KS test failed: stat={stat}, p={pvalue}"

    def test_lognormal_component(self):
        comp = LognormalComponent(mu=0.3, sigma=0.8)
        rng = np.random.default_rng(43)
        samples = comp.transform(rng.standard_normal(self.N))
        stat, pvalue = stats.kstest(samples, lambda v: stats.lognorm.cdf(v, s=0.8, scale=np.exp(0.3)))
        assert pvalue > 0.01, f"KS test failed: stat={stat}, p={pvalue}"

    def test_beta_component(self):
        comp = BetaComponent(alpha=2.0, beta=5.0)
        rng = np.random.default_rng(44)
        samples = comp.transform(rng.standard_normal(self.N))
        stat, pvalue = stats.kstest(samples, lambda v: stats.beta.cdf(v, 2.0, 5.0))
        assert pvalue > 0.01, f"KS test failed: stat={stat}, p={pvalue}"


# --------------------------------------------------------------------------------------
# Item 21: pin the design choice that PriorIndependentComponents.logpdf is the plain
# standard-normal log-density with NO Jacobian correction for `transform`.
# --------------------------------------------------------------------------------------
class TestPriorIndependentComponentsLogpdfDesign:
    def test_equals_standard_normal_log_density_without_jacobian(self):
        """
        Design choice documented in ``PriorIndependentComponents``'s class docstring:
        "Internally, the sampling framework uses the Gaussian prior distribution
        N(zeros, ones)." ``logpdf(sample)`` is defined as exactly
        ``-0.5 * sample @ sample`` (see ``independent_components.py``), i.e. the
        INTERNAL standard-normal log-density up to an additive constant, with no
        change-of-variables/Jacobian term added for the (possibly nonlinear)
        `transform` to physical space. This is intentional, not a bug: the transform
        is only ever applied to move samples into physical space for the solver/output,
        while MCMC acceptance ratios operate entirely in the internal N(0, I) space.
        """
        prior = PriorIndependentComponents([UniformComponent(), LognormalComponent(), BetaComponent()])
        rng = np.random.default_rng(5)
        x1, x2 = rng.normal(size=3), rng.normal(size=3)

        diff_ours = prior.logpdf(x1) - prior.logpdf(x2)
        mvn = multivariate_normal(mean=np.zeros(3), cov=np.eye(3))
        diff_scipy = mvn.logpdf(x1) - mvn.logpdf(x2)
        assert diff_ours == pytest.approx(diff_scipy, abs=1e-10)


@pytest.mark.unit
class TestStandardizedNormal:
    """2026-09-22 (author decision): a Normal prior is sampled in the standard-normal internal
    space; ``transform`` maps back to N(mean, cov)."""

    def test_uncorrelated_transform_and_internal_density(self):
        from surrDAMH.distributions import Normal, StandardizedNormal, standardize_prior
        prior = Normal(mean=np.array([8.0, -1.0, 2.0]), sd=np.array([5.0, 0.5, 2.0]))
        z = np.array([1.0, -2.0, 0.5])
        std = standardize_prior(prior)
        assert isinstance(std, StandardizedNormal)
        np.testing.assert_allclose(std.transform(z), prior.mean + prior.sd * z)
        assert std.logpdf(z) == pytest.approx(-0.5 * float(z @ z))
        np.testing.assert_allclose(std.grad_logpdf(z), -z)
        np.testing.assert_allclose(std.mean, np.zeros(3))
        np.testing.assert_allclose(std.get_covariance(), np.ones(3))
        # the internal density of z equals the physical density of transform(z) up to the (constant) Jacobian
        z2 = np.array([0.3, 0.1, -1.2])
        assert (std.logpdf(z) - std.logpdf(z2)) == pytest.approx(
            prior.logpdf(std.transform(z)) - prior.logpdf(std.transform(z2)))

    def test_correlated_transform_reproduces_the_covariance(self):
        from surrDAMH.distributions import Normal, standardize_prior
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        std = standardize_prior(Normal(mean=np.array([1.0, 2.0]), cov=cov))
        rng = np.random.default_rng(0)
        physical = np.vstack([std.transform(std.rvs(generator=rng)) for _ in range(20000)])
        np.testing.assert_allclose(physical.mean(axis=0), [1.0, 2.0], atol=0.06)
        np.testing.assert_allclose(np.cov(physical.T), cov, atol=0.12)

    def test_standardize_prior_leaves_other_priors_and_itself_alone(self):
        from surrDAMH.distributions import (Normal, NormalComponent, PriorIndependentComponents,
                                            standardize_prior)
        pic = PriorIndependentComponents([NormalComponent(), NormalComponent()])
        assert standardize_prior(pic) is pic
        std = standardize_prior(Normal(mean=0, sd=1, dim=2))
        assert standardize_prior(std) is std

    def test_pcn_accepts_a_standardized_normal_prior(self, tmp_path):
        from surrDAMH.configuration import Configuration
        from surrDAMH.distributions import Normal, standardize_prior
        from surrDAMH.modules.proposal_builder import build_proposal
        from surrDAMH.proposals import PCN
        from surrDAMH.stages import Stage
        conf = Configuration(no_parameters=2, no_observations=1, output_dir=str(tmp_path),
                             use_collector=False, use_solvers_pool=False)
        std = standardize_prior(Normal(mean=np.array([8.0, -1.0]), sd=np.array([5.0, 0.5])))
        proposal = build_proposal(Stage(proposal=PCN(beta=0.3), max_evaluations=5), conf, std, seed=1, stage_index=0)
        np.testing.assert_allclose(proposal.prior_mean, np.zeros(2))  # pCN now sees the standard normal

