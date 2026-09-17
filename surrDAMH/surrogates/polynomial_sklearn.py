#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""


import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from surrDAMH.surrogates.parent import Evaluator, Updater


class PolynomialSklearnEvaluator(Evaluator):
    def __init__(self, no_parameters, model) -> None:
        self.no_parameters = no_parameters
        self.model = model

    def __call__(self, datapoints: npt.NDArray):
        # evaluates the surrogate model in datapoints
        datapoints = datapoints.reshape(-1, self.no_parameters)
        return self.model.predict(datapoints)


class PolynomialSklearnUpdater(Updater):  # initiated by COLLECTOR
    """Polynomial regression surrogate (``sklearn`` ``PolynomialFeatures`` + ``LinearRegression``).

    The degree starts at 1 and is grown automatically as more snapshots accumulate
    (``get_evaluator()`` refits from scratch on the full accumulated data whenever the
    snapshot count increases enough to afford the next degree, up to ``max_degree``);
    there is no minimum-snapshot guard before the first fit.
    """

    def __init__(self, no_parameters: int, no_observations: int, max_degree: int = 5):
        """
        Args:
            no_parameters: dimension of the parameter space.
            no_observations: dimension of the observation space.
            max_degree: highest polynomial degree the model is allowed to grow to.
        """
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_degree = max_degree
        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))
        self.wei = np.empty((0, 1))
        self.degree = 1
        self.num_terms = self.no_parameters + 1
        self.terms_multiplicator = (self.no_parameters + self.degree + 1)/(self.degree + 1)
        self.num_snapshots = 0
        self.num_snapshots_current = 0
        self.degree_current = 0
        self.model = None

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray, weights: npt.NDArray | None = None):
        # add new data. The caller's weights are discarded (hard-set to None below, then
        # defaulted to all-ones internally): every snapshot is treated as equally
        # informative regardless of its (multiplicity) weight.
        weights = None
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)

        no_new_snapshots = parameters.shape[0]
        self.num_snapshots += no_new_snapshots

        if weights is None:
            weights = np.ones((no_new_snapshots, 1))
        self.par = np.vstack((self.par, parameters))
        self.obs = np.vstack((self.obs, observations))
        self.wei = np.vstack((self.wei, weights))

    def get_evaluator(self):
        # number of terms: (no_parameters + degree choose degree)
        # degree += 1  =>  num_terms *= (no_parameters + degree)/degree
        while self.num_snapshots > self.num_terms*self.terms_multiplicator and self.degree < self.max_degree:
            self.num_terms *= self.terms_multiplicator
            self.degree += 1
            self.terms_multiplicator = (self.no_parameters + self.degree + 1)/(self.degree + 1)
        if self.num_snapshots > self.num_snapshots_current:  # train the model if num_snapshots increased
            if self.degree > self.degree_current:  # create new model if degree changed
                self.model = make_pipeline(PolynomialFeatures(self.degree), LinearRegression())
                print("Polynomial surrogate model degree increased to ", self.degree, "- no_snapshots =", self.num_snapshots, flush=True)
                self.degree_current = self.degree
            self.model.fit(self.par, self.obs)
            self.num_snapshots_current = self.num_snapshots
        return PolynomialSklearnEvaluator(self.no_parameters, self.model)
