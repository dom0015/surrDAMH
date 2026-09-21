#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import time
import warnings

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RBFInterpolator

from surrDAMH.surrogates.parent import Evaluator, Updater, WeightingPolicy
from surrDAMH.surrogates.reuse import register_updater

#: Smoothing values tried, in order, if the interpolation matrix is still singular after
#: de-duplication (near-duplicate snapshots closer than floating-point exactness). Each one
#: turns the interpolant into a (very lightly) smoothing approximant, which is what a
#: rank-deficient interpolation problem actually admits; see ``_fit_interpolator``.
SMOOTHING_LADDER: tuple[float, ...] = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)


def deduplicate_snapshots(par: npt.NDArray, obs: npt.NDArray,
                          tolerance: float = 0.0) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Collapses repeated parameter rows into one row carrying the mean of their observations.

    DAMH re-sends the *current* chain state as a snapshot whenever the chain stays where it
    is (with an updated multiplicity), so bit-identical duplicate rows are the normal case,
    not an edge case -- and an RBF interpolation matrix with two identical centres is exactly
    singular (finding 3.3).

    Args:
        par: ``(n, no_parameters)`` snapshot parameters.
        obs: ``(n, no_observations)`` snapshot observations.
        tolerance: ``0.0`` (default) groups only **bit-identical** rows, which is exactly what
            DAMH produces; a positive value groups rows that coincide after rounding each
            coordinate to that absolute grid (``round(x / tolerance)``) and uses the group's
            mean parameters as the representative. The default is exact equality on purpose:
            grid rounding is not transitive, so a positive tolerance can still leave two rows
            ``tolerance``-close in different groups -- it is offered as a blunt instrument for
            solvers that return jittered parameters, not as the normal path.

    Returns:
        ``(par, obs)`` in first-appearance order. The **input arrays are returned unchanged
        (same objects) when there is nothing to collapse**, so the fit of a duplicate-free
        dataset is bit-identical to the pre-de-duplication code.

    Observations of a group are averaged rather than "first one wins": for DAMH's exact
    resends they are identical, so averaging is a no-op there, and for a general caller whose
    duplicate locations carry slightly different observations (e.g. a noisy solver) the mean
    is the least-squares-optimal single value to interpolate.
    """
    if par.shape[0] < 2:
        return par, obs

    if tolerance > 0.0:
        # "+ 0.0" normalizes -0.0 to 0.0: np.unique(axis=0) compares rows bytewise, and a
        # coordinate just below zero rounds to -0.0, which would not match the same grid cell
        keys = np.round(par / tolerance) + 0.0
    else:
        keys = par
    _, first_index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    no_groups = first_index.shape[0]
    if no_groups == par.shape[0]:  # nothing duplicated: keep the arrays (and the order) as they are
        return par, obs

    inverse = inverse.reshape(-1)
    # relabel the groups by first appearance, so the surviving rows keep their original order
    order = np.argsort(first_index)
    relabel = np.empty(no_groups, dtype=np.intp)
    relabel[order] = np.arange(no_groups)
    labels = relabel[inverse]

    counts = np.bincount(labels, minlength=no_groups).reshape(-1, 1)
    obs_mean = np.zeros((no_groups, obs.shape[1]), dtype=float)
    np.add.at(obs_mean, labels, obs)
    obs_mean /= counts
    if tolerance > 0.0:  # group mean: the rows are not identical, only tolerance-close
        par_unique = np.zeros((no_groups, par.shape[1]), dtype=float)
        np.add.at(par_unique, labels, par)
        par_unique /= counts
    else:  # identical rows: use them exactly, no rounding
        par_unique = par[first_index[order]]
    return par_unique, obs_mean


class RBFInterpolationEvaluator(Evaluator):
    def __init__(self, no_parameters, no_observations, rbf_interpolator) -> None:
        super().__init__(no_parameters, no_observations)
        self.rbf_interpolator = rbf_interpolator

    def __call__(self, datapoints: npt.NDArray):
        """Evaluates the surrogate model: ``(n, no_parameters) -> (n, no_observations)``."""
        datapoints = datapoints.reshape(-1, self.no_parameters)
        results_interpolated = self.rbf_interpolator(datapoints)
        return results_interpolated


@register_updater
class RBFInterpolationUpdater(Updater):  # initiated by COLLECTOR
    """
    Radial basis function (RBF) interpolation.
    Using scipy.interpolate.RBFInterpolator.

    Weighting: ``supports_sample_weights = False`` -- this is an interpolant, there is no
    per-row weight to give ``scipy.interpolate.RBFInterpolator``. With
    ``weighting="multiplicity"`` the zero-multiplicity snapshots (rejected proposals) are
    dropped and every remaining snapshot is interpolated once; the default
    ``weighting="uniform"`` interpolates every snapshot.
    """

    supports_sample_weights = False

    def __init__(self, no_parameters: int, no_observations: int,
                 neighbors: int | None = None,
                 max_neighbors: int | None = 50,
                 dedup_tolerance: float = 0.0,
                 smoothing: float = 0.0,
                 kernel: str = "thin_plate_spline",
                 epsilon: float | None = None,
                 degree: int | None = None,
                 verbose: bool = False,
                 weighting: WeightingPolicy = "uniform"):
        """
        Args, forwarded to ``scipy.interpolate.RBFInterpolator`` at fit time (see its
        docs for exact semantics): ``neighbors`` (local RBF using only the ``neighbors``
        nearest points if given, global otherwise), ``smoothing``, ``kernel``,
        ``epsilon``, ``degree``.

        Args:
            no_parameters: dimension of the parameter space.
            no_observations: dimension of the observation space.
            neighbors: explicit ``scipy`` ``neighbors`` value. ``None`` (default) lets
                ``max_neighbors`` decide; any other value is used as given, whatever the
                snapshot count, and disables ``max_neighbors``.
            max_neighbors: automatic cap on the neighbor count (finding 3.3). While the
                de-duplicated snapshot count is ``<= max_neighbors`` the fit is the **global**
                one (``neighbors=None``), i.e. bit-identical to the pre-cap behaviour; above
                it, ``scipy`` switches to a local fit over the ``max_neighbors`` nearest
                centres, which replaces one dense ``O(N^3)``/``O(N^2)`` solve by many small
                ones. ``None`` disables the cap and keeps the global fit at every size.
                The default 50 is a practical compromise: a thin-plate-spline patch over 50
                centres is far more than the polynomial part needs in the parameter
                dimensions this library is used in, while 50x50 solves stay negligible.
            dedup_tolerance: passed to ``deduplicate_snapshots`` (0.0 = exact duplicates only).
            smoothing: ``scipy`` smoothing; 0.0 (default) means exact interpolation.
            verbose: print fit timing/shape on every ``get_evaluator()`` call.
            weighting: snapshot-weighting policy, see ``Updater``.

        Notes:
            ``get_evaluator()`` refits from scratch on the full accumulated dataset every
            call. Duplicate snapshot locations -- which DAMH produces routinely -- are
            collapsed before the fit (``deduplicate_snapshots``) instead of being handed to
            ``RBFInterpolator``, where they made the interpolation matrix exactly singular.
            The old fallback for that case (refit on ``par`` stacked with ``par+1, par+2, ...``
            carrying the *same* observations, ``kernel="linear"``) imposed the false
            constraint ``f(x) = f(x + k*1)`` on the surrogate and is gone; what is left is a
            small-smoothing retry ladder (``SMOOTHING_LADDER``) for the residual
            near-duplicate case, which only relaxes exact interpolation.
        """
        super().__init__(no_parameters, no_observations, weighting=weighting)
        self.neighbors = neighbors
        self.max_neighbors = max_neighbors
        self.dedup_tolerance = float(dedup_tolerance)
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.verbose = verbose
        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray,
                 multiplicity: npt.NDArray | None = None):
        """Stores new snapshots; ``weighting="multiplicity"`` drops the zero-multiplicity rows."""
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)
        mask = self._rows_to_use(multiplicity, parameters.shape[0])
        self.par = np.vstack((self.par, parameters[mask]))
        self.obs = np.vstack((self.obs, observations[mask]))
        self.no_snapshots = int(self.par.shape[0])

    def _effective_neighbors(self, no_snapshots: int) -> int | None:
        """``scipy``'s ``neighbors`` argument for this fit: see ``max_neighbors``."""
        if self.neighbors is not None:
            return self.neighbors
        if self.max_neighbors is None or no_snapshots <= self.max_neighbors:
            return None  # global fit, exactly as before the cap was introduced
        return int(self.max_neighbors)

    def _fit_interpolator(self, par: npt.NDArray, obs: npt.NDArray, neighbors: int | None):
        """
        Fits ``RBFInterpolator``, retrying with a little smoothing if the matrix is singular.

        After de-duplication a singular matrix needs two *distinct* but numerically
        indistinguishable centres, i.e. a rank-deficient interpolation problem that has no
        interpolant at all. The principled answer is then to stop insisting on exact
        interpolation, so each retry only raises ``smoothing``; unlike the removed
        shifted-copy fallback it adds no constraint the data do not contain, and it degrades
        continuously (at ``1e-10`` the fit is an interpolant to ~10 significant digits).
        """
        try:
            return RBFInterpolator(par, obs, neighbors=neighbors, smoothing=self.smoothing,
                                   kernel=self.kernel, epsilon=self.epsilon, degree=self.degree)
        except (np.linalg.LinAlgError, ValueError) as first_error:
            base_smoothing = float(np.max(np.asarray(self.smoothing, dtype=float)))
            for smoothing in SMOOTHING_LADDER:
                if smoothing <= base_smoothing:
                    continue
                try:
                    interpolator = RBFInterpolator(
                        par, obs, neighbors=neighbors, smoothing=smoothing,
                        kernel=self.kernel, epsilon=self.epsilon, degree=self.degree)
                except (np.linalg.LinAlgError, ValueError):
                    continue
                warnings.warn(
                    f"RBFInterpolator failed on {par.shape[0]} de-duplicated snapshots "
                    f"({type(first_error).__name__}: {first_error}); refitted with "
                    f"smoothing={smoothing} (approximation instead of exact interpolation).",
                    RuntimeWarning, stacklevel=3)
                return interpolator
            raise

    def get_evaluator(self):
        tt = time.time()
        par, obs = deduplicate_snapshots(self.par, self.obs, self.dedup_tolerance)
        neighbors = self._effective_neighbors(par.shape[0])
        rbf_interpolator = self._fit_interpolator(par, obs, neighbors)
        if self.verbose:
            print("RBF model constructed, time, par, unique par, neighbors:",
                  time.time()-tt, self.par.shape, par.shape[0], neighbors, flush=True)
        return RBFInterpolationEvaluator(self.no_parameters, self.no_observations, rbf_interpolator)
