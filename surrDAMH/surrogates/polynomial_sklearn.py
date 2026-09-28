#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import math

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from surrDAMH.surrogates.parent import Evaluator, Updater, WeightingPolicy
from surrDAMH.surrogates.reuse import register_updater

#: Default ridge penalty (finding 3.10). Small enough that the fit stays essentially the
#: least-squares one on well-conditioned data (see ``PolynomialSklearnUpdater.__init__``).
DEFAULT_RIDGE_ALPHA = 1e-6


def num_polynomial_terms(no_parameters: int, degree: int) -> int:
    """Number of terms of a full polynomial of ``degree`` in ``no_parameters`` variables."""
    return math.comb(no_parameters + degree, degree)


class PolynomialSklearnEvaluator(Evaluator):
    def __init__(self, no_parameters, no_observations, model) -> None:
        super().__init__(no_parameters, no_observations)
        self.model = model

    def __call__(self, datapoints: npt.NDArray):
        """Evaluates the surrogate model: ``(n, no_parameters) -> (n, no_observations)``.

        The ``reshape`` is required by ``Ridge`` (and not by the ``LinearRegression`` it
        replaced): for ``no_observations == 1`` it ravels the targets internally and its
        ``predict`` returns ``(n,)``, which would silently break the WS6 evaluator contract
        (``12_evaluator_contract_spec.md`` §2) and, through broadcasting, every caller.
        """
        datapoints = datapoints.reshape(-1, self.no_parameters)
        return np.asarray(self.model.predict(datapoints)).reshape(-1, self.no_observations)


# Maintainer notes (moved out of the class docstring on 2026-09-22):
# - Pipeline StandardScaler -> PolynomialFeatures -> Ridge. The degree grows automatically as
#   snapshots accumulate (get_evaluator() refits from scratch whenever the snapshot count
#   increased), capped by max_degree AND by the snapshot count: the degree used is the largest d
#   with num_polynomial_terms(no_parameters, d) < num_snapshots (finding 3.10), so the model never
#   has more terms than snapshots and falls back to a constant fit while there are too few. It
#   used to start at degree 1 unconditionally (a hyperplane through a single point).
# - Standardization + ridge keep the normal equations conditioned for degree >= 3 in more than a
#   couple of dimensions (finding 3.10). Numerical change only: the surrogate sits inside DAMH's
#   delayed-acceptance correction, which is exact whatever the surrogate returns.
# - alpha: Ridge minimizes ||y - Xw||^2 + alpha*||w||^2; with standardized features (all O(1))
#   the default 1e-6 shifts the eigenvalues of X^T X far below the data term for any
#   non-degenerate design, yet keeps the solve finite when the design is rank-deficient
#   (repeated snapshots, a degree the snapshots cannot separate). RidgeCV is deliberately not
#   used: a cross-validation at every collector update costs more than the fit and makes the
#   surrogate depend on the snapshot ordering.
# - Weighting: supports_sample_weights = True; "multiplicity" passes the multiplicities to the
#   Ridge step (fit(..., ridge__sample_weight=...)) and drops zero-multiplicity snapshots on
#   arrival; "uniform" fits unweighted. StandardScaler is always fitted unweighted (it only
#   fixes the feature scale).
@register_updater
class PolynomialSklearnUpdater(Updater):  # initiated by COLLECTOR
    """
    Polynomial regression surrogate: cheap, deterministic, good for smooth models in a few
    parameters::

        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(
            no_parameters=conf.no_parameters, no_observations=conf.no_observations, max_degree=3)

    The polynomial degree grows on its own as snapshots accumulate, never exceeding
    ``max_degree`` and never having more terms than there are snapshots. Provides no gradients.

    Args:
        no_parameters: number of parameters.
        no_observations: number of observations.

        max_degree: highest polynomial degree the model may grow to.
        alpha: ridge penalty on the (standardized) polynomial features; the tiny default only
            keeps the fit well-conditioned. Raise it if the surrogate visibly overfits.

        weighting: ``"uniform"`` fits every snapshot once; ``"multiplicity"`` drops rejected
            proposals and weights each state by how long the chain stayed there.
    """

    supports_sample_weights = True

    def __init__(self, no_parameters: int, no_observations: int,
                 # --- model ---
                 max_degree: int = 5,
                 alpha: float = DEFAULT_RIDGE_ALPHA,
                 # --- data ---
                 weighting: WeightingPolicy = "uniform") -> None:
        """See the class docstring for every argument."""
        super().__init__(no_parameters, no_observations, weighting=weighting)
        self.max_degree = max_degree
        self.alpha = float(alpha)
        # snapshots used for surrogate model construction:
        self.par = np.empty((0, self.no_parameters))
        self.obs = np.empty((0, self.no_observations))
        self.mul = np.empty((0, 1))
        self.degree = 0
        self.num_terms = num_polynomial_terms(self.no_parameters, self.degree)
        self.num_snapshots = 0
        self.num_snapshots_current = 0
        self.degree_current = -1
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

    def _affordable_degree(self) -> int:
        """
        Highest degree the current snapshot count affords: the largest ``d <= max_degree``
        with ``num_polynomial_terms(d) < num_snapshots``.

        This is the pre-existing escalation rule ("grow to ``d+1`` once there are more
        snapshots than a degree-``d+1`` polynomial has terms"), applied from degree 0 up
        instead of from a hard-coded starting degree of 1 -- which is exactly the
        minimum-snapshot rule (finding 3.10): the number of terms is then always strictly
        below the number of snapshots, at every degree including the first. The result is
        monotone in ``num_snapshots``, which only grows, so the degree never drops.
        """
        degree = 0
        while (degree < self.max_degree
               and self.num_snapshots > num_polynomial_terms(self.no_parameters, degree + 1)):
            degree += 1
        return degree

    def _make_model(self, degree: int):
        """``StandardScaler -> PolynomialFeatures(degree) -> Ridge``.

        ``PolynomialFeatures`` keeps its bias column (which also makes ``degree=0`` a plain
        constant fit); ``Ridge(fit_intercept=True)`` centres the design matrix first, so that
        column becomes exactly zero and the constant term is carried by the *unpenalized*
        intercept -- the penalty therefore never shrinks the mean of the observations.
        """
        return make_pipeline(StandardScaler(), PolynomialFeatures(degree), Ridge(alpha=self.alpha))

    def get_evaluator(self):
        self.degree = self._affordable_degree()
        self.num_terms = num_polynomial_terms(self.no_parameters, self.degree)
        if self.num_snapshots > self.num_snapshots_current:  # train the model if num_snapshots increased
            if self.degree > self.degree_current:  # create new model if degree changed
                self.model = self._make_model(self.degree)
                print("Polynomial surrogate model degree increased to ", self.degree, "- no_snapshots =", self.num_snapshots, flush=True)
                self.degree_current = self.degree
            sample_weight = self._training_weights(self.mul)
            if sample_weight is None:  # weighting="uniform": unweighted ridge regression
                self.model.fit(self.par, self.obs)
            else:
                self.model.fit(self.par, self.obs, ridge__sample_weight=sample_weight)
            self.num_snapshots_current = self.num_snapshots
        return PolynomialSklearnEvaluator(self.no_parameters, self.no_observations, self.model)
