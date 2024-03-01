#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from surrDAMH.surrogates.parent import Updater, Evaluator
from scipy.spatial import cKDTree


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
            weights = 1 / distances
            weights /= np.sum(weights, axis=1, keepdims=True)  # Normalize weights to sum to 1
            # Perform the weighted average of values from the two nearest neighbors
            # print("???", weights.shape, self.obs[indices].shape, indices.shape)
            # print(weights)
            # print(weights.reshape((no_datapoints, self.no_nearest_neighbors, 1)))
            # print(indices)
            # print(self.obs[indices])
            weights = weights.reshape((no_datapoints, self.no_nearest_neighbors, 1))
            interpolated_values = np.sum(self.obs[indices] * weights, axis=1)

        return interpolated_values


class KDTreeUpdater(Updater):  # initiated by COLLECTOR
    """
    Nearest-neighbor interpolator.
    Using scipy.spatial.cKDTree.
    """

    def __init__(self, no_parameters: int, no_observations: int,
                 no_nearest_neighbors: int):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.no_nearest_neighbors = no_nearest_neighbors

        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))

    def add_data(self, parameters: int, observations: int, weights: npt.NDArray = None):
        # add new data
        # WEIGHTS ARE NOT USED
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)
        self.par = np.vstack((self.par, parameters))
        self.obs = np.vstack((self.obs, observations))

    def get_evaluator(self):
        # Build a KDTree from the original points
        kdtree = cKDTree(self.par)
        no_nearest_neighbors = min(self.no_nearest_neighbors, self.par.shape[0])
        return KDTreeEvaluator(self.no_parameters, kdtree, self.obs, no_nearest_neighbors)
