#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt


class Evaluator:
    def __init__(self) -> None:
        pass

    def __call__(self, datapoints: npt.NDArray) -> npt.NDArray:
        """
        Evaluates the surrogate model in datapoints.

        datapoints shape: (number of datapoints, no_parameters)

        output NDArray shape: (number of datapoints, no_observations)
        """
        pass


class Updater:
    """
    Parent class for surrogate model updaters.
    """

    def __init__(self, no_parameters: int, no_observations: int) -> None:
        pass

    def delayed_init(self, data):
        """
        Additional settings of the surrogate model.
        """
        pass

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray, weights: npt.NDArray) -> None:
        """
        Adds more snapshots to the surrogate model.
        parameters shape: (number of snapshots, no_parameters)
        observations shape: (number of snapshots, no_observations)
        weights shape: (number of snapshots, 1)
        """
        pass

    def get_evaluator(self) -> Evaluator:
        """
        Returns Evaluator instance.
        """
        pass


def closest_point_distance(par, point):
    """
    Calculates distances of the given point to all points in self.par
    and returns the smallest distance.
    """
    distances = np.linalg.norm(par - point, axis=1)
    closest_index = np.argmin(distances)
    smallest_distance = distances[closest_index]
    return smallest_distance


def closest_point_distance_kdtree(par, kdtree, point):
    """
    Returns the distance to the closest point in kdtree.
    """
    closest_index = kdtree.query(point)[1]
    smallest_distance = np.linalg.norm(par[closest_index] - point)
    return smallest_distance


def sigmoid_wrapper(y: npt.NDArray, y_approx: npt.NDArray, dist: float, scale: float):
    # y = vector of observations
    # y_approx = approximation of y obtained using a surrogate model
    # dist = distance to the closest point used for surrogate model construction
    # scale = determines how far from the closest point the surrogate model is considered reliable
    return y_approx + (y - y_approx) / (1 + np.exp(-10 * (dist - 2*scale)/scale))
