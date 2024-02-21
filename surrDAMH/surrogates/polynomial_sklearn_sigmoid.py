#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from surrDAMH.surrogates.parent import Updater, Evaluator, sigmoid_wrapper, closest_point_distance_kdtree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from scipy.spatial import cKDTree


class PolynomialSklearnEvaluator(Evaluator):
    def __init__(self, no_parameters, model, par, kdtree) -> None:
        self.no_parameters = no_parameters
        self.model = model
        self.par = par
        self.kdtree = kdtree

    def __call__(self, datapoints: npt.NDArray):
        # evaluates the surrogate model in datapoints
        datapoints = datapoints.reshape(-1, self.no_parameters)
        res = self.model.predict(datapoints)
        # for point in datapoints
        for i in range(datapoints.shape[0]):
            point = datapoints[i, :]
            dist = closest_point_distance_kdtree(self.par, self.kdtree,  point)
            res[i, :] = sigmoid_wrapper(y=-3.0, y_approx=res[i, :], dist=dist, scale=0.1)
        return res


class PolynomialSklearnSigmoidUpdater(Updater):  # initiated by COLLECTOR
    def __init__(self, no_parameters: int, no_observations: int, max_degree: int = 5):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_degree = max_degree
        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))
        self.wei = np.empty((0, 1))
        self.current_degree = -1
        self.no_snapshots = 0
        self.no_included_snapshots = 0
        self.model = None

    def add_data(self, parameters: int, observations: int, weights: npt.NDArray = None):
        # add new data to matrices of non-processed data
        # TODO: polynomial surrogate weights
        weights = None  # WEIGHTS ARE NOT USED
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)

        no_new_snapshots = parameters.shape[0]
        self.no_snapshots += no_new_snapshots

        if weights is None:
            weights = np.ones((no_new_snapshots, 1))
        self.par = np.vstack((self.par, parameters))
        self.obs = np.vstack((self.obs, observations))
        self.wei = np.vstack((self.wei, weights))

    def get_evaluator(self):
        # TODO: polynomial degree formula
        degree = int(np.floor(np.log(self.no_snapshots)/np.log(self.no_parameters)))
        degree = min(degree, self.max_degree)
        # update the model if data (and degree) changed:
        if self.no_snapshots > self.no_included_snapshots:
            if degree > self.current_degree:
                self.current_degree = degree
                self.model = make_pipeline(PolynomialFeatures(self.current_degree), LinearRegression())
                print("Polynomial surrogate model degree =", self.current_degree, ", no_snapshots =", self.no_snapshots, flush=True)
            self.model.fit(self.par, self.obs)
            self.no_included_snapshots = self.no_snapshots
        kdtree = cKDTree(self.par)
        return PolynomialSklearnEvaluator(self.no_parameters, self.model, self.par.copy(), kdtree)
