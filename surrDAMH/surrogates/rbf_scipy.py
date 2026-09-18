#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import time

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RBFInterpolator

from surrDAMH.surrogates.parent import Evaluator, Updater, WeightingPolicy
from surrDAMH.surrogates.reuse import register_updater


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
            verbose: print fit timing/shape on every ``get_evaluator()`` call.
            weighting: snapshot-weighting policy, see ``Updater``.

        Notes:
            ``get_evaluator()`` refits from scratch on the full accumulated dataset
            every call (O(N^3)); duplicated/degenerate snapshot locations raise inside
            ``RBFInterpolator`` and are caught by falling back to a shifted-copy,
            ``kernel="linear"`` construction (see the ``except ValueError`` branch) --
            this changes the fitted surrogate's behaviour, not just its performance.
        """
        super().__init__(no_parameters, no_observations, weighting=weighting)
        self.neighbors = neighbors
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

    def get_evaluator(self):
        tt = time.time()
        try:
            rbf_interpolator = RBFInterpolator(self.par, self.obs, neighbors=self.neighbors, smoothing=self.smoothing,
                                               kernel=self.kernel, epsilon=self.epsilon, degree=self.degree)
        except ValueError as e:
            print("Exception - RBFInterpolation ValueError:", e, flush=True)
            par = self.par.copy()
            obs = self.obs.copy()
            for i in range(self.no_parameters):
                par = np.vstack((par, self.par+i+1))
                obs = np.vstack((obs, self.obs))
            rbf_interpolator = RBFInterpolator(par, obs, kernel="linear", smoothing=1)
        if self.verbose:
            print("RBF model constructed, time, par:", time.time()-tt, self.par.shape, flush=True)
        return RBFInterpolationEvaluator(self.no_parameters, self.no_observations, rbf_interpolator)
