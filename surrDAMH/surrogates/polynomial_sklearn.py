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

from surrDAMH.surrogates.parent import Evaluator, Updater, WeightingPolicy
from surrDAMH.surrogates.reuse import register_updater


class PolynomialSklearnEvaluator(Evaluator):
    def __init__(self, no_parameters, no_observations, model) -> None:
        super().__init__(no_parameters, no_observations)
        self.model = model

    def __call__(self, datapoints: npt.NDArray):
        """Evaluates the surrogate model: ``(n, no_parameters) -> (n, no_observations)``."""
        datapoints = datapoints.reshape(-1, self.no_parameters)
        return self.model.predict(datapoints)


@register_updater
class PolynomialSklearnUpdater(Updater):  # initiated by COLLECTOR
    """Polynomial regression surrogate (``sklearn`` ``PolynomialFeatures`` + ``LinearRegression``).

    The degree starts at 1 and is grown automatically as more snapshots accumulate
    (``get_evaluator()`` refits from scratch on the full accumulated data whenever the
    snapshot count increases enough to afford the next degree, up to ``max_degree``);
    there is no minimum-snapshot guard before the first fit.

    Weighting: ``supports_sample_weights = True`` -- with ``weighting="multiplicity"`` the
    multiplicities are passed to ``LinearRegression.fit(..., sample_weight=...)`` and
    zero-multiplicity snapshots are dropped before they are stored. With the default
    ``weighting="uniform"`` every snapshot counts once and the fit is unweighted.
    """

    supports_sample_weights = True

    def __init__(self, no_parameters: int, no_observations: int, max_degree: int = 5,
                 weighting: WeightingPolicy = "uniform"):
        """
        Args:
            no_parameters: dimension of the parameter space.
            no_observations: dimension of the observation space.
            max_degree: highest polynomial degree the model is allowed to grow to.
            weighting: snapshot-weighting policy, see ``Updater``.
        """
        super().__init__(no_parameters, no_observations, weighting=weighting)
        self.max_degree = max_degree
        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))
        self.mul = np.empty((0, 1))
        self.degree = 1
        self.num_terms = self.no_parameters + 1
        self.terms_multiplicator = (self.no_parameters + self.degree + 1)/(self.degree + 1)
        self.num_snapshots = 0
        self.num_snapshots_current = 0
        self.degree_current = 0
        self.model = None

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray,
                 multiplicity: npt.NDArray | None = None):
        """Stores new snapshots; ``weighting="multiplicity"`` drops the zero-multiplicity rows."""
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)
        if multiplicity is None:
            multiplicity_array = np.ones((parameters.shape[0], 1))
        else:
            multiplicity_array = np.asarray(multiplicity, dtype=float).reshape(-1, 1)

        mask = self._rows_to_use(multiplicity_array, parameters.shape[0])
        parameters = parameters[mask]
        observations = observations[mask]
        multiplicity_array = multiplicity_array[mask]

        self.num_snapshots += parameters.shape[0]
        self.par = np.vstack((self.par, parameters))
        self.obs = np.vstack((self.obs, observations))
        self.mul = np.vstack((self.mul, multiplicity_array))
        self.no_snapshots = self.num_snapshots

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
            sample_weight = self._training_weights(self.mul)
            if sample_weight is None:  # weighting="uniform": unweighted least squares
                self.model.fit(self.par, self.obs)
            else:
                self.model.fit(self.par, self.obs, linearregression__sample_weight=sample_weight)
            self.num_snapshots_current = self.num_snapshots
        return PolynomialSklearnEvaluator(self.no_parameters, self.no_observations, self.model)
