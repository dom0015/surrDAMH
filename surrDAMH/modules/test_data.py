"""Test data for the surrDAMH module: generation, reuse, and posterior weighting."""

import os

import numpy as np

from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution
from surrDAMH.solver_specification import SolverSpec
from surrDAMH.solvers import Solver, get_solver_from_spec


def _test_data_path(experiment_folder: str) -> str:
    return os.path.join(experiment_folder, "sampling_output", "surrogate_test_data.npz")


def resolve_solver(solver: Solver | SolverSpec) -> Solver:
    if isinstance(solver, Solver):
        return solver
    if isinstance(solver, SolverSpec):
        return get_solver_from_spec(solver)
    raise ValueError("solver must be either a Solver or a SolverSpec instance")


class TestData:
    def __init__(self, parameters: np.ndarray, surrogate_parameters: np.ndarray, observations: np.ndarray,
                 log_posterior: np.ndarray | None = None, weights: np.ndarray | None = None):
        self.parameters = parameters
        self.surrogate_parameters = surrogate_parameters
        self.observations = observations
        self.log_posterior = log_posterior
        self.weights = weights

    @classmethod
    def generate(cls, prior: Distribution, likelihood: Distribution, solver: Solver | SolverSpec,
                 conf: Configuration, size: int = 16, seed: int = 25347) -> "TestData":
        solver = resolve_solver(solver)
        rng_state = np.random.get_state()
        np.random.seed(seed)
        parameters = np.vstack([prior.rvs() for _ in range(size)])
        np.random.set_state(rng_state)

        if conf.transform_before_surrogate:
            surrogate_parameters = np.vstack([prior.transform(p.copy()) for p in parameters])
        else:
            surrogate_parameters = parameters.copy()

        observations = np.zeros((size, conf.no_observations))
        for i, p in enumerate(parameters):
            solver.set_parameters(prior.transform(p.copy()))
            observations[i, :] = np.asarray(solver.get_observations()).reshape(-1)

        test_data = cls(parameters, surrogate_parameters, observations)
        test_data.compute_log_posterior_and_weights(prior, likelihood)
        return test_data

    @classmethod
    def reuse(cls, experiment_folder: str) -> "TestData":
        path = _test_data_path(experiment_folder)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No surrogate test data found in {experiment_folder!r}")
        with np.load(path) as loaded:
            return cls(loaded["parameters"], loaded["surrogate_parameters"], loaded["observations"])

    def save(self, experiment_folder: str) -> None:
        path = _test_data_path(experiment_folder)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, parameters=self.parameters, surrogate_parameters=self.surrogate_parameters,
                 observations=self.observations)  # log_posterior/weights depend on the consumer, not persisted

    def get_size(self) -> int:
        return self.parameters.shape[0]

    def join(self, other: "TestData") -> "TestData":
        return TestData(
            np.vstack([self.parameters, other.parameters]),
            np.vstack([self.surrogate_parameters, other.surrogate_parameters]),
            np.vstack([self.observations, other.observations]),
        )  # caller must recompute posterior/weights after join

    def reduce_size(self, new_size: int) -> "TestData":
        if new_size >= self.get_size():
            return self
        return TestData(self.parameters[:new_size], self.surrogate_parameters[:new_size],
                        self.observations[:new_size])  # caller must recompute posterior/weights after this

    def compute_log_posterior_and_weights(self, prior: Distribution, likelihood: Distribution) -> None:
        log_posterior = np.zeros((self.parameters.shape[0], 1), dtype=float)
        for i, (parameters, obs) in enumerate(zip(self.parameters, self.observations)):
            log_posterior[i, 0] = prior.logpdf(parameters) + likelihood.logpdf(obs)
        shifted = log_posterior - np.max(log_posterior)
        weights = np.exp(shifted)
        weight_sum = np.sum(weights)
        self.log_posterior = log_posterior
        self.weights = (weights / weight_sum if np.isfinite(weight_sum) and weight_sum > 0.0
                        else np.full_like(log_posterior, 1.0 / log_posterior.shape[0]))

    def as_surrogate_test_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.log_posterior is None or self.weights is None:
            raise ValueError("Posterior and weights have not been computed for this TestData")
        return (self.surrogate_parameters, self.observations, self.log_posterior, self.weights)