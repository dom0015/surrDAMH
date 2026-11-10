#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ``surrDAMH.surrogates`` (Phase 1 items 22-30 of
``library_notes/07_testing_plan.md``, extended by WS6).

Updaters covered (all importable without heavy/optional external deps such as
FEniCSx): ``KDTreeUpdater`` (nearest_kdtree.py), ``RBFInterpolationUpdater``
(rbf_scipy.py), ``PolynomialSklearnUpdater`` (polynomial_sklearn.py) and
``NeuralNetworkUpdater`` (torch_perceptron_minibatches.py) -- twice, once in
its default minibatch configuration and once in the full-batch L-BFGS preset that
replaced the deleted ``NeuralNetworkUpdaterBasic`` (WS6 decision 4,
``library_notes/12_evaluator_contract_spec.md`` §3).

WS6 contract checks in here:

* ``__call__`` returns ``(n, no_observations)`` for every evaluator and every ``n``,
  including ``n = 1`` (no flattening anywhere any more);
* ``no_parameters``/``no_observations`` are set on every evaluator;
* the ``weighting`` option (``"uniform"`` default vs ``"multiplicity"``) and its
  interaction with ``supports_sample_weights``;
* the NN's ``output_normalization`` (``"likelihood"`` default, ``"manual"``,
  ``"identity"``), its likelihood-derived statistics, the identity fallback warning and
  the checkpoint provenance;
* ``initial_training`` no longer persisting synthetic rows (finding 3.4).
"""

from __future__ import annotations

import os
import pickle
import tempfile
import warnings

import numpy as np
import pytest

from surrDAMH.distributions.normal import Normal
from surrDAMH.surrogates.nearest_kdtree import KDTreeUpdater
from surrDAMH.surrogates.parent import (Evaluator, Updater,
                                        apply_output_normalization_from_likelihood,
                                        likelihood_output_statistics)
from surrDAMH.surrogates.polynomial_sklearn import PolynomialSklearnUpdater
from surrDAMH.surrogates.polynomial_sklearn import num_polynomial_terms
from surrDAMH.surrogates.rbf_scipy import (RBFInterpolationUpdater,
                                           deduplicate_snapshots)
from surrDAMH.surrogates.reuse import _UPDATER_REGISTRY
from surrDAMH.surrogates.torch_perceptron_minibatches import NeuralNetworkUpdater

pytestmark = pytest.mark.unit

NO_PARAMETERS = 3
NO_OBSERVATIONS = 2


def _toy_function(x: np.ndarray) -> np.ndarray:
    """Simple deterministic function of the (small-dimensional) parameters."""
    return np.array([np.sum(x ** 2), np.sum(x)])


def _make_snapshots(n: int = 20, seed: int = 0, noise: float = 1e-3):
    rng = np.random.default_rng(seed)
    par = rng.uniform(-2.0, 2.0, size=(n, NO_PARAMETERS))
    obs = np.array([_toy_function(row) for row in par])
    if noise:
        obs = obs + noise * rng.standard_normal(obs.shape)
    multiplicity = np.ones((n, 1))
    return par.astype(np.float64), obs.astype(np.float64), multiplicity


def _make_kdtree(**kwargs):
    return KDTreeUpdater(NO_PARAMETERS, NO_OBSERVATIONS, no_nearest_neighbors=5, **kwargs)


def _make_rbf(**kwargs):
    return RBFInterpolationUpdater(NO_PARAMETERS, NO_OBSERVATIONS, **kwargs)


def _make_polynomial(**kwargs):
    return PolynomialSklearnUpdater(NO_PARAMETERS, NO_OBSERVATIONS, **kwargs)


def _make_nn_minibatches(**kwargs):
    kwargs.setdefault("output_normalization", "identity")
    return NeuralNetworkUpdater(
        NO_PARAMETERS, NO_OBSERVATIONS, hidden_layer_sizes=(8,), solver="adamw",
        iterations_batch=50, seed=0, verbose=False, **kwargs,
    )


def _make_nn_full_batch(**kwargs):
    """Full-batch L-BFGS preset that replaced ``NeuralNetworkUpdaterBasic`` (spec 12 §3)."""
    kwargs.setdefault("output_normalization", "identity")
    return NeuralNetworkUpdater(
        NO_PARAMETERS, NO_OBSERVATIONS, hidden_layer_sizes=(8,), solver="lbfgs",
        batch_size=None, replay_ratio=0.0, train_on_added_data=False,
        iterations_batch=20, seed=0, verbose=False, **kwargs,
    )


UPDATER_FACTORIES = {
    "kdtree": _make_kdtree,
    "rbf": _make_rbf,
    "polynomial": _make_polynomial,
    "nn_full_batch": _make_nn_full_batch,
    "nn_minibatches": _make_nn_minibatches,
}

NN_UPDATERS = {"nn_full_batch", "nn_minibatches"}
GRADIENT_UPDATERS = NN_UPDATERS
CHECKPOINT_UPDATERS = {"nn_full_batch": _make_nn_full_batch, "nn_minibatches": _make_nn_minibatches}
# updaters whose fit honours per-row weights (Updater.supports_sample_weights)
WEIGHTED_FIT_UPDATERS = {"polynomial", "nn_full_batch", "nn_minibatches"}


# --------------------------------------------------------------------------------------
# WS6 decision 4: ``NeuralNetworkUpdaterBasic`` and its module are gone.
# --------------------------------------------------------------------------------------
class TestNeuralNetworkUpdaterBasicDeleted:
    def test_module_import_fails(self):
        with pytest.raises(ImportError):
            import surrDAMH.surrogates.torch_perceptron  # noqa: F401

    def test_package_attribute_is_gone(self):
        import surrDAMH.surrogates as surrogates

        with pytest.raises(AttributeError):
            surrogates.NeuralNetworkUpdaterBasic  # noqa: B018
        assert "NeuralNetworkUpdaterBasic" not in surrogates.__all__

    def test_every_remaining_updater_is_registered_for_reuse(self):
        registered = set(_UPDATER_REGISTRY)
        assert {"KDTreeUpdater", "RBFInterpolationUpdater", "PolynomialSklearnUpdater",
                "NeuralNetworkUpdater"} <= registered
        assert "NeuralNetworkUpdaterBasic" not in registered


# --------------------------------------------------------------------------------------
# Item 22: add_data / train / get_evaluator, output finite and correctly shaped,
# evaluator picklable and bit-exact after a pickle round-trip.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(UPDATER_FACTORIES))
class TestUpdaterContract:
    def test_add_data_train_evaluate_shape_and_finiteness(self, name):
        updater = UPDATER_FACTORIES[name]()
        par, obs, multiplicity = _make_snapshots(n=20, seed=1)
        updater.add_data(par, obs, multiplicity)
        updater.train()
        evaluator = updater.get_evaluator()

        query = par[:5]
        result = evaluator(query)
        assert np.all(np.isfinite(result))
        # WS6 contract: (n, no_observations) for EVERY updater, no flattened variant left
        assert result.shape == (query.shape[0], NO_OBSERVATIONS)

    @pytest.mark.parametrize("no_datapoints", [1, 2, 7])
    def test_call_returns_two_dimensional_output_for_every_batch_size(self, name, no_datapoints):
        """Spec 12 §2: 2-D output also for ``n = 1`` (this is the actual WS6 change)."""
        updater = UPDATER_FACTORIES[name]()
        par, obs, multiplicity = _make_snapshots(n=20, seed=1)
        updater.add_data(par, obs, multiplicity)
        updater.train()
        evaluator = updater.get_evaluator()

        result = evaluator(par[:no_datapoints])
        assert result.shape == (no_datapoints, NO_OBSERVATIONS)

    def test_evaluator_exposes_dimensions(self, name):
        updater = UPDATER_FACTORIES[name]()
        par, obs, multiplicity = _make_snapshots(n=20, seed=1)
        updater.add_data(par, obs, multiplicity)
        updater.train()
        evaluator = updater.get_evaluator()

        assert evaluator.no_parameters == NO_PARAMETERS
        assert evaluator.no_observations == NO_OBSERVATIONS

    def test_evaluator_picklable_bit_exact(self, name):
        updater = UPDATER_FACTORIES[name]()
        par, obs, multiplicity = _make_snapshots(n=20, seed=2)
        updater.add_data(par, obs, multiplicity)
        updater.train()
        evaluator = updater.get_evaluator()

        query = par[:4]
        direct = evaluator(query)
        reloaded = pickle.loads(pickle.dumps(evaluator))
        roundtrip = reloaded(query)
        np.testing.assert_array_equal(direct, roundtrip)

    def test_default_weighting_is_uniform(self, name):
        assert UPDATER_FACTORIES[name]().weighting == "uniform"

    def test_rejects_unknown_weighting(self, name):
        with pytest.raises(ValueError, match="weighting"):
            UPDATER_FACTORIES[name](weighting="posterior")


# --------------------------------------------------------------------------------------
# Item 23: kd-tree zero-distance guard (WS0 group C4, verified applied).
# --------------------------------------------------------------------------------------
class TestKDTreeExactHit:
    def test_query_training_point_is_finite_and_exact(self):
        updater = _make_kdtree()
        par, obs, multiplicity = _make_snapshots(n=20, seed=3)
        updater.add_data(par, obs, multiplicity)
        evaluator = updater.get_evaluator()

        result = evaluator(par[0].reshape(1, -1))
        assert np.all(np.isfinite(result))
        # C4 fix: exact_hit branch in nearest_kdtree.KDTreeEvaluator.__call__ returns the
        # stored observation exactly instead of a distance-weighted average that would
        # divide by (near-)zero distance. Was NaN before the fix (finding 3.2).
        np.testing.assert_array_equal(result[0], obs[0])


# --------------------------------------------------------------------------------------
# Item 24 / finding 3.3: RBF with duplicated snapshots -- which DAMH produces routinely,
# since the current chain state is re-sent whenever the chain stays where it is.
# Duplicates are now collapsed before the fit (``deduplicate_snapshots``); the old
# shifted-copy fallback (``par`` stacked with ``par+1, par+2, ...``, same observations,
# ``kernel="linear"``), which imposed the false constraint ``f(x) = f(x+k*1)``, is gone.
# --------------------------------------------------------------------------------------
class TestRBFDuplicatedSnapshot:
    def test_duplicate_snapshot_reproduces_observation(self):
        updater = _make_rbf()
        par, obs, _ = _make_snapshots(n=15, seed=4, noise=0.0)
        par_dup = np.vstack([par, par[0:1]])
        obs_dup = np.vstack([obs, obs[0:1]])
        updater.add_data(par_dup, obs_dup, np.ones((par_dup.shape[0], 1)))
        evaluator = updater.get_evaluator()

        result = evaluator(par[0].reshape(1, -1))
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result[0], obs[0], atol=1e-6)

    def test_many_duplicates_interpolate_all_training_points_without_a_fallback(self):
        """
        Several exactly repeated snapshots (one point sent 4 times, another 3 times, as a
        stuck DAMH chain produces): the fit must still *interpolate* every training point,
        and must do so on the first attempt -- no ``ValueError``/``LinAlgError`` caught, no
        smoothing retry (which would warn). Before the fix this raised inside
        ``RBFInterpolator`` ("Singular matrix") and fell back to the shifted-copy model.
        """
        updater = _make_rbf()
        par, obs, _ = _make_snapshots(n=15, seed=4, noise=0.0)
        par_dup = np.vstack([par, par[0:1], par[0:1], par[0:1], par[7:8], par[7:8]])
        obs_dup = np.vstack([obs, obs[0:1], obs[0:1], obs[0:1], obs[7:8], obs[7:8]])
        updater.add_data(par_dup, obs_dup, np.ones((par_dup.shape[0], 1)))

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the smoothing fallback warns; it must not trigger
            evaluator = updater.get_evaluator()

        predicted = evaluator(par)
        assert np.all(np.isfinite(predicted))
        np.testing.assert_allclose(predicted, obs, atol=1e-8)

    def test_duplicates_are_collapsed_with_averaged_observations(self):
        """``deduplicate_snapshots``: one row per location, observations averaged."""
        par = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]])
        obs = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        unique_par, mean_obs = deduplicate_snapshots(par, obs)

        # first-appearance order preserved, group observations averaged ((1+3)/2 = 2)
        np.testing.assert_array_equal(unique_par, par[[0, 1, 3]])
        np.testing.assert_array_equal(mean_obs, np.array([[2.0, 20.0], [2.0, 20.0], [4.0, 40.0]]))

    def test_positive_tolerance_groups_near_duplicates_around_zero(self):
        """``dedup_tolerance>0`` groups by a rounding grid; -0.0 must land in the 0.0 cell."""
        par = np.array([[-1e-9, 0.0, 0.0], [1e-9, 0.0, 0.0], [1.0, 1.0, 1.0]])
        obs = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
        unique_par, mean_obs = deduplicate_snapshots(par, obs, tolerance=1e-6)

        assert unique_par.shape == (2, 3)
        np.testing.assert_allclose(unique_par[0], [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_array_equal(mean_obs, np.array([[2.0, 0.0], [5.0, 0.0]]))

    def test_duplicate_free_data_is_passed_through_untouched(self):
        """Fast path: no duplicates -> the very same arrays, so the fit is bit-identical."""
        par, obs, _ = _make_snapshots(n=10, seed=5)
        unique_par, mean_obs = deduplicate_snapshots(par, obs)
        assert unique_par is par and mean_obs is obs


# --------------------------------------------------------------------------------------
# Finding 3.3: the neighbor cap. Up to ``max_neighbors`` snapshots the fit stays the global
# one (bit-identical to the uncapped code); above it scipy switches to a local fit, which
# must still give finite, accurate values instead of a dense O(N^3) solve.
# --------------------------------------------------------------------------------------
class TestRBFNeighborCap:
    def test_global_fit_below_the_cap(self):
        updater = _make_rbf(max_neighbors=50)
        assert updater._effective_neighbors(20) is None
        assert updater._effective_neighbors(50) is None
        assert updater._effective_neighbors(51) == 50
        assert _make_rbf(max_neighbors=None)._effective_neighbors(10_000) is None
        assert _make_rbf(neighbors=7)._effective_neighbors(3) == 7  # explicit value wins

    def test_below_the_cap_is_bit_identical_to_the_uncapped_fit(self):
        par, obs, multiplicity = _make_snapshots(n=30, seed=12)
        capped, uncapped = _make_rbf(max_neighbors=50), _make_rbf(max_neighbors=None)
        for updater in (capped, uncapped):
            updater.add_data(par, obs, multiplicity)
        query = par[:5] + 0.01
        np.testing.assert_array_equal(capped.get_evaluator()(query),
                                      uncapped.get_evaluator()(query))

    def test_above_the_cap_still_produces_a_reasonable_local_fit(self):
        rng = np.random.default_rng(13)
        par = rng.uniform(-2.0, 2.0, size=(400, NO_PARAMETERS))
        obs = np.array([_toy_function(row) for row in par])
        updater = _make_rbf(max_neighbors=50)
        updater.add_data(par, obs, np.ones((400, 1)))
        evaluator = updater.get_evaluator()

        query = rng.uniform(-1.5, 1.5, size=(25, NO_PARAMETERS))
        predicted = evaluator(query)
        assert predicted.shape == (25, NO_OBSERVATIONS)
        assert np.all(np.isfinite(predicted))
        target = np.array([_toy_function(row) for row in query])
        rmse = np.sqrt(np.mean((predicted - target) ** 2))
        assert rmse < 0.1  # local thin-plate fit over 50 of 400 centres, measured ~0.02


# --------------------------------------------------------------------------------------
# Item 25 / finding 3.10: polynomial surrogate still (all but) exactly fits a quadratic.
#
# The pipeline is ``StandardScaler -> PolynomialFeatures -> Ridge`` since 2026-09-18; the
# unregularized ``LinearRegression`` it replaced reproduced the quadratic to ~1e-14, while
# ``Ridge`` is deliberately biased, so the tolerance below is the ridge bias, not slop:
# with the default ``alpha=1e-6`` the measured RMSE is 1.5e-7 (and it scales linearly with
# ``alpha``: 1.5e-9 at 1e-8, 1.5e-5 at 1e-4), i.e. ~1e-8 relative to the O(10) observations.
# --------------------------------------------------------------------------------------
class TestPolynomialExactQuadratic:
    def test_low_rmse_on_exact_quadratic(self):
        def quadratic(x):
            return np.array([2 * x[0] ** 2 + 3 * x[1] * x[2] - x[1] ** 2 + 5])

        rng = np.random.default_rng(6)
        par = rng.uniform(-2.0, 2.0, size=(200, NO_PARAMETERS))
        obs = np.array([quadratic(row) for row in par])

        updater = PolynomialSklearnUpdater(NO_PARAMETERS, no_observations=1, max_degree=5)
        updater.add_data(par, obs, np.ones((200, 1)))
        evaluator = updater.get_evaluator()

        predicted = evaluator(par)
        # Ridge ravels a single-output target internally and its ``predict`` returns (n,);
        # the evaluator reshapes, so the (n, no_observations) contract survives the switch.
        assert predicted.shape == (200, 1)
        rmse = np.sqrt(np.mean((predicted.reshape(-1) - obs.reshape(-1)) ** 2))
        assert rmse < 1e-5

    def test_ridge_bias_is_proportional_to_alpha(self):
        """The residual above is the ridge penalty, not a conditioning failure."""
        def quadratic(x):
            return np.array([2 * x[0] ** 2 + 3 * x[1] * x[2] - x[1] ** 2 + 5])

        rng = np.random.default_rng(6)
        par = rng.uniform(-2.0, 2.0, size=(200, NO_PARAMETERS))
        obs = np.array([quadratic(row) for row in par])

        def rmse(alpha):
            updater = PolynomialSklearnUpdater(NO_PARAMETERS, no_observations=1, max_degree=5,
                                               alpha=alpha)
            updater.add_data(par, obs, np.ones((200, 1)))
            predicted = updater.get_evaluator()(par)
            return np.sqrt(np.mean((predicted.reshape(-1) - obs.reshape(-1)) ** 2))

        assert rmse(1e-4) / rmse(1e-6) == pytest.approx(100.0, rel=0.05)


# --------------------------------------------------------------------------------------
# Finding 3.10: the minimum-snapshot rule. The degree may never be one whose term count
# exceeds the snapshot count (it used to start at degree 1 unconditionally, i.e. it could
# fit a hyperplane through a single point).
# --------------------------------------------------------------------------------------
class TestPolynomialMinimumSnapshots:
    @pytest.mark.parametrize("no_snapshots", [1, 2, 3, 4, 5, 6, 10, 20, 35, 36, 60, 200])
    def test_degree_never_outruns_the_snapshot_count(self, no_snapshots):
        rng = np.random.default_rng(14)
        par = rng.uniform(-2.0, 2.0, size=(no_snapshots, NO_PARAMETERS))
        obs = np.array([_toy_function(row) for row in par])
        updater = _make_polynomial(max_degree=5)
        updater.add_data(par, obs, np.ones((no_snapshots, 1)))
        evaluator = updater.get_evaluator()

        assert num_polynomial_terms(NO_PARAMETERS, updater.degree) <= no_snapshots
        assert np.all(np.isfinite(evaluator(par)))

    def test_a_single_snapshot_gives_a_constant_fit(self):
        """Degree 0, i.e. the observation itself -- not a degree-1 hyperplane through one point."""
        updater = _make_polynomial()
        updater.add_data(np.array([[1.0, 2.0, 3.0]]), np.array([[7.0, -1.0]]), np.ones((1, 1)))
        evaluator = updater.get_evaluator()

        assert updater.degree == 0
        np.testing.assert_allclose(evaluator(np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])),
                                   [[7.0, -1.0], [7.0, -1.0]], atol=1e-8)

    def test_degree_grows_monotonically_with_the_snapshot_count(self):
        rng = np.random.default_rng(15)
        par = rng.uniform(-2.0, 2.0, size=(60, NO_PARAMETERS))
        obs = np.array([_toy_function(row) for row in par])
        updater = _make_polynomial(max_degree=5)
        degrees = []
        for start in range(0, 60, 5):
            updater.add_data(par[start:start + 5], obs[start:start + 5], np.ones((5, 1)))
            updater.get_evaluator()
            degrees.append(updater.degree)
        assert degrees == sorted(degrees)
        assert degrees[0] == 1 and degrees[-1] >= 3  # 5 snapshots -> degree 1; 60 -> degree 3+


# --------------------------------------------------------------------------------------
# WS6: the classical surrogates must be bit-identical under the new default weighting.
#
# The reference numbers below were produced by the PRE-WS6 code (2026-09-17, commit state
# of the working tree before the WS6 surrogate changes) with the same dataset, which
# contains multiplicity 0 and multiplicity 2 rows -- the pre-WS6 classical updaters
# discarded/ignored the weights entirely, so ``weighting="uniform"`` (the new default) has
# to reproduce them exactly.
#
# ``"polynomial"`` is the ONE deliberate exception: its reference row was re-baselined on
# 2026-09-18 when the fit method itself changed from unscaled ``PolynomialFeatures`` +
# ``LinearRegression`` to ``StandardScaler -> PolynomialFeatures -> Ridge(alpha=1e-6)``
# (finding 3.10). That is a different estimator, not a different weighting, so "unchanged"
# cannot hold and a looser tolerance would only have hidden it. The old row differed from
# the new one by at most 1.5e-6 in absolute value (7.7e-6 relative) -- the ridge bias -- and
# it was:
#     [0.13977384..., 0.19929634..., 3.00073406..., 1.00305664..., 2.50018683..., -0.99656704...]
# ``"kdtree"`` and ``"rbf"`` are untouched: RBF de-duplication returns the arrays unchanged
# when nothing is duplicated (this dataset), and 24 snapshots stay below the neighbor cap,
# so the RBF fit here is still the very same global one.
# --------------------------------------------------------------------------------------
def _weighting_dataset():
    """24 snapshots with mixed multiplicities: 0 (rejected), 2 and 1."""
    rng = np.random.default_rng(0)
    par = rng.uniform(-2.0, 2.0, size=(24, NO_PARAMETERS))
    obs = np.array([_toy_function(row) for row in par]) + 1e-3 * rng.standard_normal((24, NO_OBSERVATIONS))
    multiplicity = np.ones((24, 1))
    multiplicity[::3] = 0.0
    multiplicity[1::3] = 2.0
    query = np.array([[0.1, -0.2, 0.3], [1.0, 1.0, -1.0], [-1.5, 0.5, 0.0]])
    return par, obs, multiplicity, query


PRE_WS6_CLASSICAL_REFERENCE = {
    "kdtree": [1.0579933678243587, 0.040872098991733906, 2.9824102759506044,
               0.23152494662729903, 3.5161670839558745, -0.0605891602392387],
    "rbf": [0.015234176257292703, 0.20018177413637156, 3.600186359010871,
            1.0013261959681892, 2.4200474923527193, -0.9992622036201446],
    # re-baselined 2026-09-18, see the note above (StandardScaler -> Ridge)
    "polynomial": [0.13977494831017118, 0.19929578577863938, 3.0007323696582477,
                   1.0030565447937136, 2.500185786041447, -0.9965658100838426],
}


@pytest.mark.parametrize("name", sorted(PRE_WS6_CLASSICAL_REFERENCE))
def test_classical_surrogates_unchanged_under_default_weighting(name):
    par, obs, multiplicity, query = _weighting_dataset()
    updater = UPDATER_FACTORIES[name]()
    updater.add_data(par, obs, multiplicity)
    updater.train()
    result = updater.get_evaluator()(query)

    assert result.shape == (3, NO_OBSERVATIONS)
    np.testing.assert_array_equal(result.reshape(-1), np.array(PRE_WS6_CLASSICAL_REFERENCE[name]))


# --------------------------------------------------------------------------------------
# WS6 decision 3: the ``weighting`` option.
# --------------------------------------------------------------------------------------
def _dataset_with_zero_multiplicity_rows(altered: bool):
    """
    20 real snapshots plus 4 snapshots carrying multiplicity 0.

    The zero-multiplicity rows sit at the same parameters in both variants but carry
    wildly different observations when ``altered=True``, so any fit that uses them must
    change and any fit that drops them must not.
    """
    par, obs, _ = _make_snapshots(n=20, seed=21, noise=0.0)
    extra_par = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [-0.5, 0.2, 0.1], [0.2, -0.3, 0.4]])
    extra_obs = np.array([_toy_function(row) for row in extra_par])
    if altered:
        extra_obs = extra_obs + 100.0
    parameters = np.vstack([par, extra_par])
    observations = np.vstack([obs, extra_obs])
    multiplicity = np.vstack([np.ones((20, 1)), np.zeros((4, 1))])
    return parameters, observations, multiplicity


QUERY_NEAR_EXTRA_ROWS = np.array([[0.05, 0.05, 0.05], [0.4, 0.4, 0.4], [-0.4, 0.2, 0.1]])


def _fit_and_evaluate(name, query, *, altered=False, **updater_kwargs):
    parameters, observations, multiplicity = _dataset_with_zero_multiplicity_rows(altered)
    updater = UPDATER_FACTORIES[name](**updater_kwargs)
    updater.add_data(parameters, observations, multiplicity)
    updater.train()
    return np.asarray(updater.get_evaluator()(query))


@pytest.mark.parametrize("name", sorted(UPDATER_FACTORIES))
class TestWeightingPolicy:
    def test_multiplicity_fit_ignores_zero_multiplicity_rows(self, name):
        plain = _fit_and_evaluate(name, QUERY_NEAR_EXTRA_ROWS, altered=False, weighting="multiplicity")
        altered = _fit_and_evaluate(name, QUERY_NEAR_EXTRA_ROWS, altered=True, weighting="multiplicity")
        np.testing.assert_array_equal(plain, altered)

    def test_uniform_fit_uses_zero_multiplicity_rows(self, name):
        plain = _fit_and_evaluate(name, QUERY_NEAR_EXTRA_ROWS, altered=False, weighting="uniform")
        altered = _fit_and_evaluate(name, QUERY_NEAR_EXTRA_ROWS, altered=True, weighting="uniform")
        assert not np.allclose(plain, altered), (
            "weighting='uniform' must train on rejected proposals (multiplicity 0) too, so "
            "changing their observations has to change the fit"
        )

    def test_zero_multiplicity_rows_are_not_stored_under_multiplicity_weighting(self, name):
        parameters, observations, multiplicity = _dataset_with_zero_multiplicity_rows(False)
        updater = UPDATER_FACTORIES[name](weighting="multiplicity")
        updater.add_data(parameters, observations, multiplicity)
        assert updater.no_snapshots == 20

        updater_uniform = UPDATER_FACTORIES[name](weighting="uniform")
        updater_uniform.add_data(parameters, observations, multiplicity)
        assert updater_uniform.no_snapshots == 24

    def test_supports_sample_weights_matches_behaviour(self, name):
        """
        ``supports_sample_weights`` must say the truth: with ``weighting="multiplicity"``,
        doubling the multiplicity of half the snapshots changes the fit iff it is ``True``.
        """
        updater_class = type(UPDATER_FACTORIES[name]())
        supports = updater_class.supports_sample_weights
        assert isinstance(supports, bool)
        assert supports == (name in WEIGHTED_FIT_UPDATERS)

        # noisy targets: a noiseless quadratic is fitted exactly by the polynomial
        # surrogate whatever the weights are, which would hide the weighting entirely
        par, obs, _ = _make_snapshots(n=20, seed=22, noise=0.3)
        ones = np.ones((20, 1))
        doubled = ones.copy()
        doubled[:10] = 2.0

        def fit(multiplicity):
            updater = UPDATER_FACTORIES[name](weighting="multiplicity")
            updater.add_data(par, obs, multiplicity)
            updater.train()
            return np.asarray(updater.get_evaluator()(QUERY_NEAR_EXTRA_ROWS))

        equal = np.allclose(fit(ones), fit(doubled))
        assert equal is not supports


class TestNNWeightingModesDiffer:
    def test_uniform_and_multiplicity_produce_different_networks(self):
        """
        The NN default flipped from (unlabelled) multiplicity weighting to ``"uniform"``
        in WS6 -- the two must be distinguishable on a dataset with rejected proposals.
        """
        # altered=True gives the rejected proposals grossly different observations, so a
        # network that trains on them (uniform) cannot match one that drops them
        uniform = _fit_and_evaluate("nn_minibatches", QUERY_NEAR_EXTRA_ROWS, altered=True,
                                    weighting="uniform")
        multiplicity = _fit_and_evaluate("nn_minibatches", QUERY_NEAR_EXTRA_ROWS, altered=True,
                                         weighting="multiplicity")
        assert not np.allclose(uniform, multiplicity, atol=1e-3)


# --------------------------------------------------------------------------------------
# Item 26: NN normalize_outputs / denormalize_outputs round-trip.
# --------------------------------------------------------------------------------------
class TestNNOutputNormalization:
    def test_normalize_denormalize_roundtrip(self):
        rng = np.random.default_rng(7)
        par = rng.uniform(-2.0, 2.0, size=(30, NO_PARAMETERS))
        obs = np.array([_toy_function(row) for row in par])
        updater = NeuralNetworkUpdater(
            NO_PARAMETERS, NO_OBSERVATIONS, hidden_layer_sizes=(8,), iterations_batch=1, seed=0,
            output_normalization="manual",
            output_mean=obs.mean(axis=0), output_scale=obs.std(axis=0) + 1e-3,
        )
        y = obs[0]
        roundtrip = updater.denormalize_outputs(updater.normalize_outputs(y))
        np.testing.assert_allclose(roundtrip, y, atol=1e-4)

    def test_denormalized_evaluator_matches_raw_targets_on_y_eq_x_squared(self):
        rng = np.random.default_rng(8)
        par = rng.uniform(-2.0, 2.0, size=(200, 1))
        obs = par ** 2  # no_observations = 1
        updater = NeuralNetworkUpdater(
            no_parameters=1, no_observations=1, hidden_layer_sizes=(16,), solver="adamw",
            iterations_batch=300, learning_rate=5e-3, seed=0, output_normalization="manual",
            output_mean=obs.mean(axis=0), output_scale=obs.std(axis=0) + 1e-3,
        )
        updater.add_data(par, obs, np.ones((200, 1)))
        updater.train()
        evaluator = updater.get_evaluator()

        test_x = np.linspace(-2.0, 2.0, 20).reshape(-1, 1)
        predicted = evaluator(test_x)
        assert predicted.shape == (20, 1)
        target = (test_x ** 2).reshape(-1)
        rmse = np.sqrt(np.mean((predicted.reshape(-1) - target) ** 2))
        assert rmse < 0.2  # loose: tiny net, few steps, just checks normalization wiring


# --------------------------------------------------------------------------------------
# WS6: output_normalization="likelihood" (the new default) and its fallback.
# --------------------------------------------------------------------------------------
class TestLikelihoodOutputStatistics:
    def test_scalar_sd_is_broadcast(self):
        likelihood = Normal(mean=np.array([3.0, -1.0]), sd=0.5)
        mean, scale = likelihood_output_statistics(likelihood, NO_OBSERVATIONS)
        np.testing.assert_allclose(mean, [3.0, -1.0])
        np.testing.assert_allclose(scale, [0.5, 0.5])

    def test_vector_sd_is_used_per_observation(self):
        likelihood = Normal(mean=np.array([3.0, -1.0]), sd=np.array([0.1, 0.4]))
        mean, scale = likelihood_output_statistics(likelihood, NO_OBSERVATIONS)
        np.testing.assert_allclose(mean, [3.0, -1.0])
        np.testing.assert_allclose(scale, [0.1, 0.4])

    def test_covariance_uses_sqrt_of_the_diagonal(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        likelihood = Normal(mean=np.array([3.0, -1.0]), cov=cov)
        mean, scale = likelihood_output_statistics(likelihood, NO_OBSERVATIONS)
        np.testing.assert_allclose(mean, [3.0, -1.0])
        np.testing.assert_allclose(scale, [0.2, 0.3])

    @pytest.mark.parametrize("likelihood", [
        Normal(mean=np.array([3.0, -1.0, 0.0]), sd=1.0),         # wrong no_observations
        Normal(mean=np.array([3.0, -1.0]), sd=np.array([0.5, 0.0])),  # zero scale
    ])
    def test_unusable_statistics_return_none(self, likelihood):
        assert likelihood_output_statistics(likelihood, NO_OBSERVATIONS) is None


class _LikelihoodWithoutScale:
    """Stand-in likelihood exposing a mean but no ``sd``/``cov``."""

    def __init__(self, mean):
        self.mean = np.asarray(mean, dtype=float)

    def logpdf(self, sample):
        return 0.0


class TestSetOutputNormalizationHook:
    def test_likelihood_statistics_are_adopted(self):
        updater = NeuralNetworkUpdater(NO_PARAMETERS, NO_OBSERVATIONS,
                                                  hidden_layer_sizes=(4,), iterations_batch=1, seed=0)
        assert updater.output_normalization == "likelihood"  # the new default
        assert updater.output_normalization_provenance == "identity"  # not resolved yet

        likelihood = Normal(mean=np.array([3.0, -1.0]), sd=np.array([0.1, 0.4]))
        apply_output_normalization_from_likelihood(updater, likelihood)

        np.testing.assert_allclose(updater.output_mean, [3.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(updater.output_scale, [0.1, 0.4], atol=1e-6)
        assert updater.output_normalization_provenance == "likelihood"

    def test_already_stored_snapshots_are_renormalized(self):
        """``self.obs`` holds normalized targets, so adopting new statistics must rescale them."""
        par, obs, multiplicity = _make_snapshots(n=10, seed=23, noise=0.0)
        updater = NeuralNetworkUpdater(NO_PARAMETERS, NO_OBSERVATIONS,
                                                  hidden_layer_sizes=(4,), iterations_batch=1, seed=0)
        updater.add_data(par, obs, multiplicity)
        apply_output_normalization_from_likelihood(
            updater, Normal(mean=np.array([3.0, -1.0]), sd=np.array([0.1, 0.4])))

        _, stored_observations, _ = updater.get_training_data_arrays()
        np.testing.assert_allclose(stored_observations, obs, rtol=1e-4, atol=1e-3)

    def test_manual_and_identity_modes_ignore_the_hook(self):
        for mode, kwargs in (("identity", {}),
                             ("manual", {"output_mean": np.array([1.0, 2.0]),
                                         "output_scale": np.array([3.0, 4.0])})):
            updater = NeuralNetworkUpdater(
                NO_PARAMETERS, NO_OBSERVATIONS, hidden_layer_sizes=(4,), iterations_batch=1,
                seed=0, output_normalization=mode, **kwargs)
            before = (updater.output_mean.copy(), updater.output_scale.copy())
            apply_output_normalization_from_likelihood(
                updater, Normal(mean=np.array([3.0, -1.0]), sd=np.array([0.1, 0.4])))
            np.testing.assert_array_equal(updater.output_mean, before[0])
            np.testing.assert_array_equal(updater.output_scale, before[1])
            assert updater.output_normalization_provenance == mode

    def test_identity_mode_rejects_explicit_statistics(self):
        with pytest.raises(ValueError, match="identity"):
            NeuralNetworkUpdater(NO_PARAMETERS, NO_OBSERVATIONS,
                                            output_normalization="identity",
                                            output_mean=np.zeros(NO_OBSERVATIONS))

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="output_normalization"):
            NeuralNetworkUpdater(NO_PARAMETERS, NO_OBSERVATIONS,
                                            output_normalization="auto")

    def test_fallback_warns_and_keeps_identity(self):
        updater = NeuralNetworkUpdater(NO_PARAMETERS, NO_OBSERVATIONS,
                                                  hidden_layer_sizes=(4,), iterations_batch=1, seed=0)
        with pytest.warns(RuntimeWarning, match="_LikelihoodWithoutScale"):
            apply_output_normalization_from_likelihood(
                updater, _LikelihoodWithoutScale([3.0, -1.0]))
        np.testing.assert_array_equal(updater.output_mean, np.zeros(NO_OBSERVATIONS))
        np.testing.assert_array_equal(updater.output_scale, np.ones(NO_OBSERVATIONS))
        assert updater.output_normalization_provenance == "identity"

    def test_classical_updaters_are_unaffected(self):
        updater = _make_polynomial()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning must be raised for these
            apply_output_normalization_from_likelihood(
                updater, _LikelihoodWithoutScale([3.0, -1.0]))
        assert Updater.output_normalization == "identity"


class TestIdentityVsLikelihoodNormalizationDiffer:
    def test_default_flip_changes_the_trained_network(self):
        """
        The NN default normalization flipped identity -> likelihood in WS6; on badly scaled
        targets the two must give visibly different fits (numbers reported in
        ``library_notes/10_manual_review_notes.md`` §2.16).
        """
        rng = np.random.default_rng(24)
        par = rng.uniform(-2.0, 2.0, size=(60, NO_PARAMETERS))
        obs = np.array([_toy_function(row) for row in par]) * np.array([100.0, 1.0]) + 500.0
        likelihood = Normal(mean=obs.mean(axis=0), sd=np.array([50.0, 1.0]))
        query = par[:5]

        def train(mode):
            updater = NeuralNetworkUpdater(
                NO_PARAMETERS, NO_OBSERVATIONS, hidden_layer_sizes=(16,), solver="adamw",
                iterations_batch=200, learning_rate=1e-2, seed=0, output_normalization=mode)
            apply_output_normalization_from_likelihood(updater, likelihood)
            updater.add_data(par, obs, np.ones((60, 1)))
            updater.train()
            return np.asarray(updater.get_evaluator()(query))

        identity_prediction = train("identity")
        likelihood_prediction = train("likelihood")
        assert not np.allclose(identity_prediction, likelihood_prediction, rtol=1e-2)


# --------------------------------------------------------------------------------------
# Item 27: NN gradients (jacobian / vjp) vs central finite differences.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(GRADIENT_UPDATERS))
class TestNNGradients:
    def test_jacobian_vs_finite_differences_and_vjp_consistency(self, name):
        rng = np.random.default_rng(9)
        par = rng.uniform(-2.0, 2.0, size=(30, NO_PARAMETERS))
        obs = np.array([_toy_function(row) for row in par])
        updater = UPDATER_FACTORIES[name](
            output_normalization="manual",
            output_mean=obs.mean(axis=0), output_scale=obs.std(axis=0) + 1e-3,
        )
        updater.add_data(par, obs, np.ones((30, 1)))
        updater.train()
        evaluator = updater.get_evaluator()

        x = par[0]
        jac, y = evaluator.jacobian(x)
        assert jac.shape == (NO_OBSERVATIONS, NO_PARAMETERS)
        assert y.shape == (NO_OBSERVATIONS,)

        eps = 1e-4

        def f(point):
            return evaluator(point.reshape(1, -1))[0]

        jac_fd = np.zeros((NO_OBSERVATIONS, NO_PARAMETERS))
        for i in range(NO_PARAMETERS):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            jac_fd[:, i] = (f(xp) - f(xm)) / (2 * eps)
        np.testing.assert_allclose(jac, jac_fd, atol=1e-3, rtol=1e-2)

        v = rng.standard_normal(NO_OBSERVATIONS)
        grad, y2 = evaluator.vjp(x, v)
        np.testing.assert_allclose(grad, jac.T @ v, atol=1e-6)
        np.testing.assert_allclose(y2, y)

        with pytest.raises(ValueError):
            evaluator.jacobian(par[:2])  # batched input must raise, not silently misbehave


# --------------------------------------------------------------------------------------
# Item 28: checkpoint round-trip.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CHECKPOINT_UPDATERS))
class TestCheckpointRoundTrip:
    def test_outputs_identical_after_save_load(self, name):
        par, obs, multiplicity = _make_snapshots(n=20, seed=10)
        updater1 = CHECKPOINT_UPDATERS[name]()
        updater1.add_data(par, obs, multiplicity)
        updater1.train()
        evaluator1 = updater1.get_evaluator()
        query = par[:5]
        out1 = evaluator1(query)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = os.path.join(tmp, "checkpoint.pt")
            data_path = os.path.join(tmp, "data.npz")
            updater1.save_state(checkpoint_path, data_path)

            updater2 = CHECKPOINT_UPDATERS[name]()
            loaded = updater2.load_state(checkpoint_path, data_path)
            evaluator2 = updater2.get_evaluator()
            out2 = evaluator2(query)

        np.testing.assert_array_equal(out1, out2)
        assert loaded is not None
        loaded_par = loaded[0]
        assert loaded_par.shape[0] == par.shape[0]


class TestCheckpointCarriesNormalizationProvenance:
    def test_likelihood_statistics_and_provenance_survive_a_round_trip(self):
        par, obs, multiplicity = _make_snapshots(n=12, seed=25, noise=0.0)
        likelihood = Normal(mean=np.array([3.0, -1.0]), sd=np.array([0.1, 0.4]))
        updater1 = NeuralNetworkUpdater(NO_PARAMETERS, NO_OBSERVATIONS,
                                                   hidden_layer_sizes=(8,), iterations_batch=10,
                                                   seed=0)
        apply_output_normalization_from_likelihood(updater1, likelihood)
        updater1.add_data(par, obs, multiplicity)
        updater1.train()
        query = par[:3]
        out1 = updater1.get_evaluator()(query)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = os.path.join(tmp, "checkpoint.pt")
            data_path = os.path.join(tmp, "data.npz")
            updater1.save_state(checkpoint_path, data_path)

            hparams = updater1._checkpoint_hparams()
            assert hparams["output_normalization"] == "likelihood"
            np.testing.assert_allclose(hparams["output_mean"], [3.0, -1.0], atol=1e-6)
            np.testing.assert_allclose(hparams["output_scale"], [0.1, 0.4], atol=1e-6)
            assert hparams["weighting"] == "uniform"

            # exactly how surrogates.reuse.SurrogateReused rebuilds an updater:
            updater2 = NeuralNetworkUpdater(**hparams)
            updater2.load_state(checkpoint_path, data_path)
            out2 = updater2.get_evaluator()(query)

        np.testing.assert_array_equal(out1, out2)
        assert updater2.output_normalization_provenance == "likelihood"

    def test_identity_normalization_is_stored_as_identity(self):
        updater = NeuralNetworkUpdater(NO_PARAMETERS, NO_OBSERVATIONS,
                                                  hidden_layer_sizes=(4,), iterations_batch=1,
                                                  seed=0)  # "likelihood", never resolved
        hparams = updater._checkpoint_hparams()
        assert hparams["output_normalization"] == "identity"
        assert hparams["output_mean"] is None and hparams["output_scale"] is None
        # round-trips through the constructor, which rejects statistics for "identity"
        NeuralNetworkUpdater(**hparams)


class TestCheckpointRoundTripMinibatchesOnly:
    def test_get_initial_snapshots_reflects_loaded_data_for_minibatches(self):
        """Minibatches-only: ``get_initial_snapshots()`` returns the reloaded snapshots."""
        par, obs, multiplicity = _make_snapshots(n=12, seed=11)
        updater1 = _make_nn_minibatches()
        updater1.add_data(par, obs, multiplicity)
        updater1.train()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = os.path.join(tmp, "checkpoint.pt")
            data_path = os.path.join(tmp, "data.npz")
            updater1.save_state(checkpoint_path, data_path)

            updater2 = _make_nn_minibatches()
            updater2.load_state(checkpoint_path, data_path)
            snapshots = updater2.get_initial_snapshots()

        assert snapshots is not None
        loaded_par = snapshots[0]
        assert loaded_par.shape[0] == 12


# --------------------------------------------------------------------------------------
# Item 29 / finding 3.4: initial_training must NOT persist its synthetic rows (fixed in WS6).
# --------------------------------------------------------------------------------------
class TestInitialTrainingSyntheticRows:
    def test_minibatches_does_not_persist_synthetic_rows(self):
        """
        ``initial_training`` pre-trains the network towards a constant output on synthetic
        ``randn`` parameters. Before WS6 it overwrote ``self.par``/``self.obs``/
        ``no_snapshots`` with those rows, so ``get_training_data_arrays()`` and every
        checkpoint reported fabricated data as real training data (finding 3.4). This test
        used to pin the buggy behaviour (``par.shape[0] == 50``); it now pins the fix.
        """
        updater = NeuralNetworkUpdater(
            no_parameters=2, no_observations=1, hidden_layer_sizes=(4,), iterations_batch=5,
            seed=0, output_normalization="identity",
        )
        updater.initial_training(constant_observations=np.array([3.0]), n=50)
        par, obs, multiplicity = updater.get_training_data_arrays()

        assert updater.no_snapshots == 0
        assert par.shape[0] == 0
        assert obs.shape[0] == 0
        assert multiplicity.shape[0] == 0
        assert updater.get_initial_snapshots() is None

    def test_real_snapshots_survive_initial_training(self):
        par, obs, multiplicity = _make_snapshots(n=7, seed=26, noise=0.0)
        updater = NeuralNetworkUpdater(
            NO_PARAMETERS, NO_OBSERVATIONS, hidden_layer_sizes=(4,), iterations_batch=5,
            seed=0, output_normalization="identity",
        )
        updater.add_data(par, obs, multiplicity)
        updater.initial_training(constant_observations=obs[0], n=30)

        stored_par, stored_obs, _ = updater.get_training_data_arrays()
        assert updater.no_snapshots == 7
        np.testing.assert_allclose(stored_par, par, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(stored_obs, obs, rtol=1e-4, atol=1e-3)


# --------------------------------------------------------------------------------------
# Finding A20 (WS3 item 3): ``SurrogateAsSolver`` must satisfy the ``Solver`` contract,
# i.e. ``get_observations()`` returns shape ``(no_observations,)`` - like a real solver.
# Since WS6 every evaluator returns ``(1, q)`` for a one-point batch, so the ``reshape(-1)``
# in ``SurrogateAsSolver.get_observations`` is provably exact rather than right by accident;
# the flattened variant is kept here as a hostile third-party evaluator.
# --------------------------------------------------------------------------------------
class _BatchedEvaluator(Evaluator):
    """Contract-conforming evaluator: ``(n, no_parameters) -> (n, no_observations)``."""

    def __init__(self, A: np.ndarray) -> None:
        self.A = np.asarray(A, dtype=float)
        super().__init__(self.A.shape[1], self.A.shape[0])

    def __call__(self, datapoints: np.ndarray) -> np.ndarray:
        return np.atleast_2d(datapoints) @ self.A.T


class _FlattenedEvaluator(_BatchedEvaluator):
    """Third-party evaluator that flattens its output, as the torch ones did before WS6."""

    def __call__(self, datapoints: np.ndarray) -> np.ndarray:
        return super().__call__(datapoints).flatten()


@pytest.mark.parametrize("evaluator_class", [_BatchedEvaluator, _FlattenedEvaluator])
@pytest.mark.parametrize("no_observations", [1, 2, 3])
def test_surrogate_as_solver_returns_solver_shaped_observations(evaluator_class, no_observations):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((no_observations, NO_PARAMETERS))
    solver = evaluator_class(A).as_solver()

    parameters = rng.standard_normal(NO_PARAMETERS)  # 1-D, as the algorithms pass it
    solver.set_parameters(parameters)
    observations = solver.get_observations()

    assert observations.shape == (no_observations,)
    np.testing.assert_allclose(observations, A @ parameters)


# --------------------------------------------------------------------------------------
# 2026-09-20: a non-finite snapshot must not be trained on, and a training pass that
# produces non-finite weights must be rolled back (the failure that killed the adaptive
# Hamiltonian runs: one divergent proposal -> NaN weights -> NaN surrogate forever).
# --------------------------------------------------------------------------------------
class TestNeuralNetworkNonFiniteRobustness:
    def test_add_data_skips_non_finite_snapshots(self):
        import torch

        updater = _make_nn_minibatches()
        par, obs, multiplicity = _make_snapshots(n=5, seed=1)
        par[2, 0] = np.inf          # non-finite parameters
        obs[4, 0] = np.nan          # non-finite observations
        with pytest.warns(RuntimeWarning, match="non-finite"):
            updater.add_data(par, obs, multiplicity)

        assert updater.no_snapshots == 3
        assert updater.no_nonfinite_snapshots_skipped == 2
        assert updater.par.shape == (3, NO_PARAMETERS)
        assert updater.obs.shape == (3, NO_OBSERVATIONS)
        assert bool(torch.isfinite(updater.par).all()) and bool(torch.isfinite(updater.obs).all())

        # a second batch of clean data is still accepted and the counter does not grow
        updater.add_data(*_make_snapshots(n=4, seed=2))
        assert updater.no_snapshots == 7
        assert updater.no_nonfinite_snapshots_skipped == 2

    def test_train_rolls_back_non_finite_weights(self, monkeypatch):
        import torch

        updater = _make_nn_minibatches()
        par, obs, multiplicity = _make_snapshots(n=12, seed=3)
        updater.add_data(par, obs, multiplicity)
        updater.train()
        weights_before = {key: value.detach().clone()
                          for key, value in updater.model.state_dict().items()}
        loss_before = updater.last_loss

        def poison_the_weights(indices, max_steps):
            with torch.no_grad():
                next(updater.model.parameters())[0][0] = float("nan")
            return 0.0

        monkeypatch.setattr(updater, "_train_minibatches", poison_the_weights)
        with pytest.warns(RuntimeWarning, match="non-finite weights"):
            updater.train()

        restored = updater.model.state_dict()
        for key, value in weights_before.items():
            assert torch.equal(restored[key], value), f"{key} was not restored"
        assert all(bool(torch.isfinite(value).all()) for value in restored.values())
        assert updater.last_loss == loss_before  # the discarded pass does not report a loss
        # the evaluator built after the rollback is usable again
        predictions = updater.get_evaluator()(par[:3])
        assert np.all(np.isfinite(predictions))


@pytest.mark.unit
def test_every_updater_constructor_parameter_is_annotated():
    """2026-09-22: Pylance showed `Unknown` for unannotated constructor parameters; every
    shipped updater must annotate all of them so completion and hover stay informative."""
    import inspect
    from surrDAMH.surrogates import (KDTreeUpdater, NeuralNetworkUpdater, PolynomialSklearnUpdater,
                                     RBFInterpolationUpdater)
    for cls in (NeuralNetworkUpdater, PolynomialSklearnUpdater, RBFInterpolationUpdater, KDTreeUpdater):
        sig = inspect.signature(cls.__init__)
        unannotated = [n for n, prm in sig.parameters.items() if n != "self" and prm.annotation is inspect.Parameter.empty]
        assert unannotated == [], f"{cls.__name__}: {unannotated}"
        first_two = list(sig.parameters)[1:3]
        assert first_two == ["no_parameters", "no_observations"], cls.__name__


@pytest.mark.unit
def test_needs_retraining_default_and_network_override():
    """2026-09-23: interpolating/least-squares updaters refit only on new data; the network keeps
    improving on unchanged data (GRF scheme study, note 18 §3.1)."""
    from surrDAMH.surrogates import (KDTreeUpdater, NeuralNetworkUpdater, PolynomialSklearnUpdater,
                                     RBFInterpolationUpdater)
    for upd in (PolynomialSklearnUpdater(2, 1), RBFInterpolationUpdater(2, 1), KDTreeUpdater(2, 1, no_nearest_neighbors=2)):
        assert upd.needs_retraining(0) is False and upd.needs_retraining(1) is True, type(upd).__name__
    nn = NeuralNetworkUpdater(2, 1, hidden_layer_sizes=(4,), seed=0, output_normalization="identity")
    assert nn.needs_retraining(0) is False          # nothing to train on yet
    nn.add_data(np.zeros((3, 2)), np.zeros((3, 1)), np.ones((3, 1)))
    assert nn.needs_retraining(0) is True and nn.needs_retraining(5) is True


@pytest.mark.unit
def test_local_manager_with_min_snapshots_to_update_zero_never_refits_identical_data():
    """With the new default 0, ``run_local``'s manager retrains an interpolating updater once per
    new batch and never twice on the same data; a network-like updater is retrained on request."""
    from surrDAMH.modules.algorithm_interfaces_local import LocalSurrogateManager
    from surrDAMH.surrogates import PolynomialSklearnUpdater

    class Counting(PolynomialSklearnUpdater):
        trains = 0
        def train(self):
            type(self).trains += 1
            super().train()

    mgr = LocalSurrogateManager(Counting(2, 1, max_degree=1), min_snapshots_initial=2, min_snapshots_to_update=0)
    for k in range(4):
        mgr.send_to_collector([np.array([k * 0.1, 1.0 - k * 0.1]), np.array([k * 0.3]), 1])
    trains_after_data = Counting.trains
    assert trains_after_data == 3                      # initial fit at 2 snapshots + one per new snapshot
    mgr._maybe_update_evaluator(); mgr._maybe_update_evaluator()
    assert Counting.trains == trains_after_data        # identical data: no refit

    class Learner(Counting):
        def needs_retraining(self, new_snapshots):
            return True
    Counting.trains = 0
    mgr2 = LocalSurrogateManager(Learner(2, 1, max_degree=1), min_snapshots_initial=2, min_snapshots_to_update=0)
    for k in range(3):
        mgr2.send_to_collector([np.array([k * 0.1, 1.0 - k * 0.1]), np.array([k * 0.3]), 1])
    before = Learner.trains                            # the counter lives on the subclass (type(self))
    assert before == 2                                 # initial fit at 2 snapshots + one for the third
    mgr2._maybe_update_evaluator()
    assert Learner.trains == before + 1                # a learner is retrained on unchanged data

