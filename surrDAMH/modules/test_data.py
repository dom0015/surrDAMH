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
    """
    A fixed held-out set used to monitor surrogate quality during a run
    (``surrogate_quality_test.csv``, written by the collector) rather than to train it.

    ``parameters`` are internal-space (what the prior/proposal use), while
    ``surrogate_parameters`` are whatever space the surrogate itself was built on
    (physical space if ``Configuration.transform_before_surrogate=True``, internal
    otherwise, see ``docs/concepts.md``); ``observations`` are the exact-model outputs
    at ``parameters``. ``log_posterior``/``weights`` (self-normalized importance weights
    from the same log-posterior) are optional and only needed for weighted quality
    metrics; compute them with ``compute_log_posterior_and_weights`` before calling
    ``as_surrogate_test_data()``.
    """

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
        """
        Draws ``size`` fresh samples from ``prior`` (using a temporary, locally-seeded
        NumPy RNG state that is restored afterwards -- does not disturb the sampler's
        own RNG usage) and evaluates the exact model on each. Also computes
        ``log_posterior``/``weights`` immediately (unlike ``reuse()``, which does not).

        Args:
            prior: prior distribution to draw test points from.
            likelihood: likelihood used for the posterior weights.
            solver: forward model, or a ``SolverSpec`` to construct one locally.
            conf: configuration (``no_observations``, ``transform_before_surrogate``).
            size: number of test points.
            seed: seed for the temporary RNG state used to draw the test points.

        Returns:
            A new ``TestData`` with ``log_posterior``/``weights`` already populated.
        """
        solver = resolve_solver(solver)
        # WS4, 2026-09-18: an owned generator instead of save/restore-the-global-seed. Both are
        # deterministic for a given seed and don't disturb unrelated code; this one also matches
        # the generator=-based seeding the rest of the library uses since G4 (modules/seeds.py).
        # Changes the exact test points a given seed produces (a different bit generator).
        rng = np.random.default_rng(seed)
        parameters = np.vstack([prior.rvs(generator=rng) for _ in range(size)])

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
        """
        Loads test points previously written by ``save()`` for ``experiment_folder``
        (``sampling_output/surrogate_test_data.npz``). ``log_posterior``/``weights`` are
        NOT persisted and come back ``None``; call ``compute_log_posterior_and_weights``
        before using the result with a collector.

        Raises:
            FileNotFoundError: if no test-data file exists in ``experiment_folder``.
        """
        path = _test_data_path(experiment_folder)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No surrogate test data found in {experiment_folder!r}")
        with np.load(path) as loaded:
            return cls(loaded["parameters"], loaded["surrogate_parameters"], loaded["observations"])

    def save(self, experiment_folder: str) -> None:
        """Writes ``parameters``/``surrogate_parameters``/``observations`` to ``sampling_output/surrogate_test_data.npz`` under ``experiment_folder``, for later ``reuse()``."""
        path = _test_data_path(experiment_folder)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, parameters=self.parameters, surrogate_parameters=self.surrogate_parameters,
                 observations=self.observations)  # log_posterior/weights depend on the consumer, not persisted

    def get_size(self) -> int:
        """Number of test points."""
        return self.parameters.shape[0]

    def join(self, other: "TestData") -> "TestData":
        """Concatenates two test sets row-wise. Caller must recompute posterior/weights afterwards (not carried over)."""
        return TestData(
            np.vstack([self.parameters, other.parameters]),
            np.vstack([self.surrogate_parameters, other.surrogate_parameters]),
            np.vstack([self.observations, other.observations]),
        )  # caller must recompute posterior/weights after join

    def reduce_size(self, new_size: int) -> "TestData":
        """Truncates to the first ``new_size`` points (no-op if already smaller). Caller must recompute posterior/weights afterwards."""
        if new_size >= self.get_size():
            return self
        return TestData(self.parameters[:new_size], self.surrogate_parameters[:new_size],
                        self.observations[:new_size])  # caller must recompute posterior/weights after this

    def compute_log_posterior_and_weights(self, prior: Distribution, likelihood: Distribution) -> None:
        """
        Fills in ``log_posterior`` (``prior.logpdf(parameters) + likelihood.logpdf(observations)``,
        per point) and self-normalized importance ``weights`` (``exp(log_posterior -
        max) / sum``, or uniform weights if the sum is not finite/positive). Mutates
        ``self`` in place; required before ``as_surrogate_test_data()``.
        """
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
        """
        Returns the ``(surrogate_parameters, observations, log_posterior, weights)``
        tuple expected by ``process_COLLECTOR.run_COLLECTOR``'s ``surrogate_test_data``
        argument.

        Raises:
            ValueError: if ``compute_log_posterior_and_weights`` has not been called yet.
        """
        if self.log_posterior is None or self.weights is None:
            raise ValueError("Posterior and weights have not been computed for this TestData")
        return (self.surrogate_parameters, self.observations, self.log_posterior, self.weights)