#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from surrDAMH.surrogates.parent import Updater, Evaluator
from scipy.interpolate import NearestNDInterpolator


class NearestInterpolationEvaluator(Evaluator):
    def __init__(self, no_parameters, interpolators) -> None:
        self.no_parameters = no_parameters
        self.interpolators = interpolators

    def __call__(self, datapoints: npt.NDArray):
        # evaluates the surrogate model in datapoints
        datapoints = datapoints.reshape(-1, self.no_parameters)
        results_interpolated = np.array([interp(datapoints) for interp in self.interpolators]).T  # TODO check dimensions
        return results_interpolated


class NearestInterpolationUpdater(Updater):  # initiated by COLLECTOR
    """
    Nearest-neighbor interpolator.
    Using scipy.interpolate.NearestNDInterpolator.
    """

    def __init__(self, no_parameters: int, no_observations: int,
                 rescale: bool = False,
                 tree_options: dict | None = None):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.rescale = rescale
        self.tree_options = tree_options
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
        interpolators = [NearestNDInterpolator(self.par, self.obs[:, i], rescale=self.rescale, tree_options=self.tree_options)
                         for i in range(self.no_observations)]

        # try:
        #     rbf_interpolator = RBFInterpolator(self.par, self.obs, neighbors=self.neighbors, smoothing=self.smoothing,
        #                                        kernel=self.kernel, epsilon=self.epsilon, degree=self.degree)
        # except ValueError as e:
        #     print("Exception - RBFInterpolation ValueError:", e)
        #     par = self.par.copy()
        #     obs = self.obs.copy()
        #     for i in range(self.no_parameters):
        #         par = np.vstack((par, self.par+i+1))
        #         obs = np.vstack((obs, self.obs))
        #     rbf_interpolator = RBFInterpolator(par, obs, kernel="linear", smoothing=1)
        return NearestInterpolationEvaluator(self.no_parameters, interpolators)
