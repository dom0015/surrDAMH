#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for helper modules (Phase 1 items 31-33 of
``library_notes/07_testing_plan.md``): ``Gaussian_process.assemble_covariance_matrix``,
``lhs_normal.lhs_normal``, and ``modules.test_data.TestData``.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from scipy.stats import norm

from surrDAMH.configuration import Configuration
from surrDAMH.core import Problem
from surrDAMH.distributions.normal import Normal
from surrDAMH.modules.Gaussian_process import assemble_covariance_matrix
from surrDAMH.modules.lhs_normal import lhs_normal
from surrDAMH.modules.test_data import TestData as SurrogateTestData
from surrDAMH.solvers import Solver

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------
# Item 31: assemble_covariance_matrix with non-constant std -> symmetric and PSD.
# WS0 group C5 applied (see the "block = diag(std) * corr * diag(std)" comment in
# Gaussian_process.py); verified fixed on 2026-09-15.
# --------------------------------------------------------------------------------------
class TestAssembleCovarianceMatrix:
    def test_non_constant_std_is_symmetric_and_psd(self):
        block_spec = {
            "time_grid": np.arange(6),
            "corr_length": 2.0,
            "std": np.array([1.0, 2.0, 3.0, 0.5, 4.0, 1.5]),  # non-constant
        }
        cov = assemble_covariance_matrix([block_spec])
        assert np.allclose(cov, cov.T)
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() > -1e-8  # PSD (up to numerical tolerance)

    def test_multiple_blocks_non_constant_std_still_symmetric_and_psd(self):
        block_a = {"time_grid": np.arange(4), "corr_length": 1.0, "std": np.array([1.0, 5.0, 2.0, 3.0])}
        block_b = {"time_grid": np.arange(3), "corr_length": 0.5, "std": np.array([2.0, 2.0, 6.0]),
                   "cov_type": "squared_exponential"}
        cov = assemble_covariance_matrix([block_a, block_b])
        assert cov.shape == (7, 7)
        assert np.allclose(cov, cov.T)
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() > -1e-8


# --------------------------------------------------------------------------------------
# Item 32: lhs_normal -- stratification property (always holds), and the maximin-quality
# selection, FIXED by G3 (finding 3.8, decided 2026-09-17): ``maxmin`` is now updated
# inside the loop, so the best of the 5 candidates is returned instead of the last one.
# --------------------------------------------------------------------------------------
def _lhs_candidates_and_qualities(n: int, d: int, seed: int):
    """The 5 candidate LHS designs ``lhs_normal`` draws internally, and their maximin qualities."""
    rs = np.random.RandomState(seed)
    candidates = []
    qualities = []
    for _ in range(5):
        R = rs.uniform(size=[n, d])
        P = np.zeros([n, d])
        for col in range(d):
            P[:, col] = rs.permutation(n)
        candidate = (P + R) / n
        distances = np.zeros([n, n])
        for j in range(d):
            temp = np.tile(candidate[:, j].reshape(1, n), (n, 1)) - np.tile(candidate[:, j].reshape(n, 1), (1, n))
            distances = distances + temp * temp
        candidates.append(candidate)
        qualities.append(float(np.min(distances + np.eye(n) * 1000)))
    return candidates, qualities


class TestLHSNormal:
    def test_lhs_normal_no_longer_imports_numpy_matlib(self):
        """
        G3 (2026-09-17): ``lhs_normal.py`` used ``numpy.matlib.repmat`` (which only emitted a
        PendingDeprecationWarning). The two calls were replaced by the exactly equivalent
        ``np.tile`` on 2-D input -- ``repmat``'s only extra behaviour is reshaping 0-D/1-D
        input, which the call sites never used -- so the module no longer needs the import.
        The equivalence itself is checked below on the shapes actually used.
        """
        import warnings

        import surrDAMH.modules.lhs_normal as lhs_module

        assert not hasattr(lhs_module, "matlib")
        with warnings.catch_warnings():  # the deprecation warning is the point of the fix
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            import numpy.matlib
        n = 7
        v = np.random.RandomState(3).uniform(size=n)
        repmat_result = (numpy.matlib.repmat(np.reshape(v, (1, n)), n, 1)
                         - numpy.matlib.repmat(np.reshape(v, (n, 1)), 1, n))
        tile_result = np.tile(np.reshape(v, (1, n)), (n, 1)) - np.tile(np.reshape(v, (n, 1)), (1, n))
        np.testing.assert_array_equal(repmat_result, tile_result)

    def test_one_point_per_stratum_per_dimension(self):
        """
        This stratification property holds regardless of the maximin-selection bug below:
        each of the 5 internal LHS candidates already places exactly one point per stratum
        per dimension (via `np.random.RandomState.permutation`), so whichever candidate
        ends up selected, the returned design still satisfies it.
        """
        n, d = 10, 3
        loc = np.zeros(d)
        design = lhs_normal(loc=loc, scale=1.0, n=n, seed=1)
        assert design.shape == (n, d)
        design_uniform = norm.cdf(design, loc=0, scale=1)  # map back to [0, 1)
        for j in range(d):
            strata = np.floor(design_uniform[:, j] * n).astype(int)
            strata = np.clip(strata, 0, n - 1)
            assert sorted(strata.tolist()) == list(range(n)), (
                f"dimension {j}: expected exactly one sample per stratum, got strata {sorted(strata.tolist())}"
            )

    def test_maximin_best_of_five_candidates_is_returned(self):
        """
        G3 (finding 3.8), decided and applied 2026-09-17: ``lhs_normal`` is documented to
        return the maximin-best of 5 candidate LHS designs, ranked by the minimum squared
        pairwise distance (``quality``). ``maxmin`` used to stay at 0 for the whole loop, so
        every candidate passed ``quality > maxmin`` and the LAST one was returned. It is now
        updated inside the ``if``, so the argmax candidate is returned.

        This test recomputes all 5 candidates (reproducing the internal ``RandomState``
        draws), computes their qualities, and asserts the returned design is the argmax.
        Seed 2 is checked separately because there the argmax is candidate 2, not the last
        one, so the test would fail again if the bug came back.
        """
        n, d = 8, 3
        for seed in range(12):
            design = lhs_normal(loc=np.zeros(d), scale=1.0, n=n, seed=seed)
            design_uniform = norm.cdf(design, loc=0, scale=1)
            candidates, qualities = _lhs_candidates_and_qualities(n, d, seed)
            best = int(np.argmax(qualities))
            assert np.allclose(design_uniform, candidates[best]), (
                f"seed {seed}: lhs_normal did not return the maximin-best candidate "
                f"(expected index {best}, qualities {qualities})"
            )

        # seed 2: the best candidate is index 2, so the old "return the last one" behaviour
        # is distinguishable from the fixed maximin selection.
        candidates, qualities = _lhs_candidates_and_qualities(n, d, seed=2)
        assert int(np.argmax(qualities)) == 2
        design_uniform = norm.cdf(lhs_normal(loc=np.zeros(d), scale=1.0, n=n, seed=2), loc=0, scale=1)
        assert not np.allclose(design_uniform, candidates[-1]), "G3 regression: last candidate returned"


# --------------------------------------------------------------------------------------
# Item 33: TestData -- generate() leaves the global RNG state unchanged; the
# save -> reuse -> compute_log_posterior_and_weights -> as_surrogate_test_data sequence
# (which is what WS0 item A3's fix in core.py, Problem.run_sampling, calls) works standalone.
# --------------------------------------------------------------------------------------
class _LinearSolver(Solver):
    """Minimal deterministic solver: G(x) = 2x."""

    def __init__(self, solver_id: int = 0, output_dir: str | None = None) -> None:
        pass

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters)

    def get_observations(self):
        return self.parameters * 2.0


class TestTestDataGenerateAndReuse:
    def test_generate_leaves_global_rng_state_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf = Configuration(no_parameters=2, no_observations=2, output_dir=tmp,
                                 use_collector=False, use_solvers_pool=False)
            prior = Normal(mean=np.zeros(2), sd=1.0)
            likelihood = Normal(mean=np.zeros(2), sd=1.0)
            solver = _LinearSolver()

            np.random.seed(123)
            state_before = np.random.get_state()
            problem = Problem(prior, likelihood, solver=solver)
            SurrogateTestData.generate(problem, size=8, transform_before_surrogate=conf.transform_before_surrogate,
                                       seed=999)
            state_after = np.random.get_state()

        assert state_before[0] == state_after[0]
        np.testing.assert_array_equal(state_before[1], state_after[1])
        assert state_before[2:] == state_after[2:]

    def test_save_reuse_compute_and_as_surrogate_test_data_sequence(self):
        """Exercises the same call sequence as core.py's collector role (WS0 item A3's fix site):
        TestData.generate -> .save -> .reuse -> .compute_log_posterior_and_weights ->
        .as_surrogate_test_data()."""
        with tempfile.TemporaryDirectory() as tmp:
            conf = Configuration(no_parameters=2, no_observations=2, output_dir=tmp,
                                 use_collector=False, use_solvers_pool=False)
            prior = Normal(mean=np.zeros(2), sd=1.0)
            likelihood = Normal(mean=np.zeros(2), sd=1.0)
            solver = _LinearSolver()

            problem = Problem(prior, likelihood, solver=solver)
            generated = SurrogateTestData.generate(problem, size=8,
                                                   transform_before_surrogate=conf.transform_before_surrogate,
                                                   seed=999)
            generated.save(tmp)

            reused = SurrogateTestData.reuse(tmp)
            reused.compute_log_posterior_and_weights(prior, likelihood)
            surrogate_parameters, observations, log_posterior, weights = reused.as_surrogate_test_data()

        assert surrogate_parameters.shape == (8, 2)
        assert observations.shape == (8, 2)
        assert log_posterior.shape == (8, 1)
        assert weights.shape == (8, 1)
        assert np.all(np.isfinite(log_posterior))
        assert np.all(weights >= 0.0)
        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        # reused parameters/observations must match what generate() computed (round-tripped through .npz)
        np.testing.assert_allclose(surrogate_parameters, generated.surrogate_parameters)
        np.testing.assert_allclose(observations, generated.observations)
