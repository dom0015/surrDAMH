#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared fixtures for the whole test suite (``tests/unit`` and ``tests/validation``).

The central fixture is ``linear_gaussian_problem``: a linear-Gaussian toy problem
(``G(x) = A @ x``, Gaussian likelihood, independent Gaussian prior) with a closed-form
posterior, so both the unit tests and the ``tests/validation`` checks can compare
sampled/approximated quantities against an exact answer. It lives in this root
``conftest.py`` so that pytest makes it visible to every test subdirectory.

``toy_examples/solver_examples/solver_spec_examples.py`` was considered as the solver
source but its solvers are loaded via ``SolverSpec.solver_module_path`` (a relative path
resolved with ``importlib`` from the ``toy_examples`` working directory), which is not
importable directly under plain ``pytest`` from the repo root. A minimal in-repo
``Solver`` subclass is used instead, per the WS1 task brief.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from surrDAMH.distributions.independent_components import (
    NormalComponent,
    PriorIndependentComponents,
)
from surrDAMH.distributions.normal import Normal
from surrDAMH.solvers import Solver


class LinearSolver(Solver):
    """Minimal deterministic solver: ``G(x) = A @ x``."""

    def __init__(self, A: npt.NDArray) -> None:
        self.A = np.asarray(A, dtype=float)

    def set_parameters(self, parameters: npt.NDArray) -> None:
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self) -> npt.NDArray:
        return self.A @ self.parameters


def closed_form_posterior(A: npt.NDArray, obs_sd: npt.NDArray, prior_mean: npt.NDArray,
                          prior_sd: npt.NDArray, observed_data: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Standard conjugate Gaussian-linear-Gaussian posterior:
    prior N(prior_mean, diag(prior_sd**2)), likelihood N(A x, diag(obs_sd**2)), data ``observed_data``.

    Returns (posterior_mean, posterior_cov).
    """
    prior_cov_inv = np.diag(1.0 / prior_sd**2)
    obs_cov_inv = np.diag(1.0 / obs_sd**2)
    posterior_cov = np.linalg.inv(prior_cov_inv + A.T @ obs_cov_inv @ A)
    posterior_mean = posterior_cov @ (prior_cov_inv @ prior_mean + A.T @ obs_cov_inv @ observed_data)
    return posterior_mean, posterior_cov


@dataclass
class LinearGaussianProblem:
    no_parameters: int
    no_observations: int
    A: npt.NDArray
    obs_sd: npt.NDArray
    prior_mean: npt.NDArray
    prior_sd: npt.NDArray
    observed_data: npt.NDArray
    solver: Solver
    prior: PriorIndependentComponents
    likelihood: Normal
    posterior_mean: npt.NDArray
    posterior_cov: npt.NDArray


@pytest.fixture
def linear_gaussian_problem() -> LinearGaussianProblem:
    """
    ``no_parameters=2`` linear-Gaussian toy problem with a closed-form posterior.

    Internal prior space equals physical space (``NormalComponent(mu=0, sigma=1)``
    is the identity transform), so ``prior.transform`` is a no-op and the solver can be
    evaluated directly on internal-space parameters.
    """
    no_parameters = 2
    no_observations = 2
    A = np.array([[1.0, 0.5], [0.0, 1.0]])
    obs_sd = np.array([0.3, 0.3])
    prior_mean = np.zeros(no_parameters)
    prior_sd = np.ones(no_parameters)
    true_x = np.array([0.7, -0.4])
    observed_data = A @ true_x  # fixed synthetic "observed" data (deterministic fixture)

    solver = LinearSolver(A)
    prior = PriorIndependentComponents([NormalComponent(mu=0.0, sigma=1.0) for _ in range(no_parameters)])
    likelihood = Normal(mean=observed_data, sd=obs_sd)

    posterior_mean, posterior_cov = closed_form_posterior(A, obs_sd, prior_mean, prior_sd, observed_data)

    return LinearGaussianProblem(
        no_parameters=no_parameters,
        no_observations=no_observations,
        A=A,
        obs_sd=obs_sd,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        observed_data=observed_data,
        solver=solver,
        prior=prior,
        likelihood=likelihood,
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
    )
