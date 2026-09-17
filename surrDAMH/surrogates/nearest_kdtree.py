#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from surrDAMH.surrogates.parent import Evaluator, Updater


class KDTreeEvaluator(Evaluator):
    def __init__(self, no_parameters, kdtree, obs, no_nearest_neighbors) -> None:
        self.no_parameters = no_parameters
        self.kdtree = kdtree
        self.obs = obs
        self.no_nearest_neighbors = no_nearest_neighbors

    def __call__(self, datapoints: npt.NDArray):
        # evaluates the surrogate model in datapoints
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


class KDTreeUpdater(Updater):  # initiated by COLLECTOR
    """
    Nearest-neighbor interpolator.
    Using scipy.spatial.cKDTree.
    """

    def __init__(self, no_parameters: int, no_observations: int,
                 no_nearest_neighbors: int):
        """
        Args:
            no_parameters: dimension of the parameter space.
            no_observations: dimension of the observation space.
            no_nearest_neighbors: number of neighbors averaged per query (inverse-distance
                weights); clamped to the number of stored snapshots in ``get_evaluator()``
                if fewer are available. ``1`` = nearest-neighbor lookup, no averaging.
        """
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.no_nearest_neighbors = no_nearest_neighbors

        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray, weights: npt.NDArray | None = None):
        # add new data. ``weights`` is accepted but ignored: every snapshot is treated
        # as equally informative regardless of its (multiplicity) weight.
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)
        self.par = np.vstack((self.par, parameters))
        self.obs = np.vstack((self.obs, observations))

    def get_evaluator(self):
        # Build a KDTree from the original points
        kdtree = cKDTree(self.par)
        no_nearest_neighbors = min(self.no_nearest_neighbors, self.par.shape[0])
        return KDTreeEvaluator(self.no_parameters, kdtree, self.obs, no_nearest_neighbors)
