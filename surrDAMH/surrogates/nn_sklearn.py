#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from surrDAMH.surrogates.parent import Updater, Evaluator
from sklearn.neural_network import MLPRegressor


class NNSklearnEvaluator(Evaluator):
    def __init__(self, no_parameters, no_observations, model) -> None:
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.model = model

    def __call__(self, datapoints: npt.NDArray):
        # evaluates the surrogate model in datapoints
        datapoints = datapoints.reshape(-1, self.no_parameters)
        temp = self.model.predict(datapoints)
        return temp.reshape(-1, self.no_observations)


class NNSklearnUpdater(Updater):  # initiated by COLLECTOR
    def __init__(self, no_parameters: int, no_observations: int, hidden_layer_sizes: tuple = (100,),
                 activation: str = 'relu', solver: str = 'adam', max_iter: int = 50000):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))
        self.wei = np.empty((0, 1))
        self.no_snapshots = 0
        self.no_included_snapshots = 0
        self.model = None

    def add_data(self, parameters: int, observations: int, weights: npt.NDArray = None):
        # add new data to matrices of non-processed data
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
        # update the model if data (and degree) changed:
        if self.no_snapshots > self.no_included_snapshots:
            self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
                                      solver=self.solver, max_iter=self.max_iter, verbose=False, learning_rate="adaptive", n_iter_no_change=100)
            print("MPL regressor created, no_snapshots =", self.no_snapshots, flush=True)
            self.model.fit(self.par, self.obs)
            self.no_included_snapshots = self.no_snapshots
        return NNSklearnEvaluator(self.no_parameters, self.no_observations, self.model)
