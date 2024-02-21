#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from surrDAMH.surrogates.parent import Updater, Evaluator
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


class GaussianSklearnEvaluator(Evaluator):
    def __init__(self, no_parameters, no_observations, model) -> None:
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.model = model

    def __call__(self, datapoints: npt.NDArray):
        # evaluates the surrogate model in datapoints
        datapoints = datapoints.reshape(-1, self.no_parameters)
        res = self.model.predict(datapoints)
        res = res.reshape(-1, self.no_observations)
        return res


class GaussianSklearnUpdater(Updater):  # initiated by COLLECTOR
    """
    Gaussian Process regression.
    Using sklearn.gaussian_process.GaussianProcessRegressor.
    """

    def __init__(self, no_parameters: int, no_observations: int):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))
        self.model = None

    def add_data(self, parameters: int, observations: int, weights: npt.NDArray = None):
        # WEIGHTS ARE NOT USED
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)

        self.par = np.vstack((self.par, parameters))
        self.obs = np.vstack((self.obs, observations))

    def get_evaluator(self):
        # kernel = DotProduct() + WhiteKernel()
        self.model = GaussianProcessRegressor()
        self.model.fit(self.par, self.obs)
        return GaussianSklearnEvaluator(self.no_parameters, self.no_observations, self.model)
