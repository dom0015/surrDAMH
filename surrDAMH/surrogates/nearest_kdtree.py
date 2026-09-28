#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from surrDAMH.surrogates.parent import Evaluator, Updater, WeightingPolicy
from surrDAMH.surrogates.reuse import register_updater


class KDTreeEvaluator(Evaluator):
    def __init__(self, no_parameters, no_observations, kdtree, obs, no_nearest_neighbors) -> None:
        super().__init__(no_parameters, no_observations)
        self.kdtree = kdtree
        self.obs = obs
        self.no_nearest_neighbors = no_nearest_neighbors

    def __call__(self, datapoints: npt.NDArray):
        """Evaluates the surrogate model: ``(n, no_parameters) -> (n, no_observations)``."""
        datapoints = datapoints.reshape(-1, self.no_parameters)
        no_datapoints = datapoints.shape[0]
        distances, indices = self.kdtree.query(datapoints, k=self.no_nearest_neighbors)

        if self.no_nearest_neighbors == 1:
            interpolated_values = self.obs[indices, :]
        else:
            # Use inverse distances as weights for the weighted average
            weights = 1 / np.maximum(distances, 1e-300)  # avoid division by zero
            weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights to sum to 1
            weights = weights.reshape((no_datapoints, self.no_nearest_neighbors, 1))
            interpolated_values = np.sum(self.obs[indices] * weights, axis=1)
            # exact hit of a training point: return its observations instead of the weighted average
            exact_hit = distances[:, 0] == 0
            if np.any(exact_hit):
                interpolated_values[exact_hit] = self.obs[indices[exact_hit, 0]]

        return interpolated_values


# Maintainer note (moved out of the class docstring on 2026-09-22): supports_sample_weights =
# False -- the stored observations are returned/averaged by distance, there is no per-row
# weight; "multiplicity" only drops zero-multiplicity snapshots, "uniform" keeps every snapshot.
@register_updater
class KDTreeUpdater(Updater):  # initiated by COLLECTOR
    """
    Nearest-neighbour surrogate (``scipy.spatial.cKDTree``): the prediction is the
    inverse-distance-weighted average of the closest stored snapshots. Simplest and cheapest,
    piecewise-constant-ish, no gradients::

        updater = surrDAMH.surrogates.KDTreeUpdater(
            no_parameters=conf.no_parameters, no_observations=conf.no_observations,
            no_nearest_neighbors=5)

    Args:
        no_parameters: number of parameters.
        no_observations: number of observations.

        no_nearest_neighbors: snapshots averaged per query; ``1`` = plain nearest neighbour.
            Clamped to the number of stored snapshots while there are fewer.

        weighting: ``"uniform"`` keeps every snapshot; ``"multiplicity"`` drops rejected
            proposals first.
    """

    supports_sample_weights = False

    def __init__(self, no_parameters: int, no_observations: int,
                 # --- model ---
                 no_nearest_neighbors: int,
                 # --- data ---
                 weighting: WeightingPolicy = "uniform") -> None:
        """See the class docstring for every argument."""
        super().__init__(no_parameters, no_observations, weighting=weighting)
        self.no_nearest_neighbors = no_nearest_neighbors

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
        # Build a KDTree from the original points
        kdtree = cKDTree(self.par)
        no_nearest_neighbors = min(self.no_nearest_neighbors, self.par.shape[0])
        return KDTreeEvaluator(self.no_parameters, self.no_observations, kdtree, self.obs,
                               no_nearest_neighbors)
