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


class NNSklearnOngoingUpdater(Updater):  # initiated by COLLECTOR
    def __init__(self, no_parameters: int, no_observations: int, hidden_layer_sizes: tuple = (100,),
                 activation: str = 'relu', solver: str = 'adam', learning_rate_init: float = 1e-3, iterations_batch: int = 100):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.iterations_batch = iterations_batch
        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))
        self.wei = np.empty((0, 1))
        self.no_snapshots = 0
        self.no_included_snapshots = 0
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver,
                                  verbose=False, learning_rate="constant", learning_rate_init=self.learning_rate_init)

    def add_data(self, parameters: int, observations: int, weights: npt.NDArray = None):
        # add new data to matrices of non-processed data
        # WEIGHTS ARE NOT USED
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)

        no_new_snapshots = parameters.shape[0]
        self.no_snapshots += no_new_snapshots

        self.par = np.vstack((self.par, parameters))
        self.obs = np.vstack((self.obs, observations))

        if self.no_observations == 1:
            train_obs = self.obs.ravel()
        else:
            train_obs = self.obs

        # iterations of the learning process
        for i in range(self.iterations_batch):
            self.model.partial_fit(self.par, train_obs)
        print("MPL regressor created, no_snapshots =", self.no_snapshots, self.model.loss_,  flush=True)

    def get_evaluator(self):
        return NNSklearnEvaluator(self.no_parameters, self.no_observations, self.model)
