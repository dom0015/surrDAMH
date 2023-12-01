#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from surrDAMH.surrogates.parent import Updater, Evaluator


class PolynomialEvaluator(Evaluator):
    def __init__(self, no_parameters: int, degree: int, poly: npt.NDArray, no_poly: int, system_result: npt.NDArray) -> None:
        self.no_parameters = no_parameters
        self.degree = degree
        self.poly = poly
        self.no_poly = no_poly
        self.coefficients_matrix = system_result

    def __call__(self, datapoints):
        # evaluates the surrogate model in datapoints
        datapoints = datapoints.reshape(-1, self.no_parameters)
        no_datapoints = datapoints.shape[0]
        hermite_evaluations = evaluate_Hermite_polynomials(self.degree, datapoints)
        products = np.ones((no_datapoints, self.no_poly))
        for j in range(self.no_parameters):
            products *= hermite_evaluations[:, j, self.poly[:, j]]
        evaluations = np.matmul(products, self.coefficients_matrix)
        return evaluations


class PolynomialProjectionUpdater(Updater):  # initiated by COLLECTOR
    def __init__(self, no_parameters: int, no_observations: int, max_degree: int = 5, solver: str = "pinv"):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_degree = max_degree
        self.solver = solver
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
                self.increase_degree()
            self.update_model()
            print("Polynomial surrogate model updated: degree =", self.current_degree, ", no_snapshots =", self.no_snapshots)
            self.no_included_snapshots = self.no_snapshots

        system_matrix = np.matmul(self.products.transpose(), self.products)
        system_rhs = np.matmul(self.products.transpose(), self.obs)
        # shape (no_poly,no_observations)
        if self.solver == "pinv":
            system_result = np.matmul(np.linalg.pinv(system_matrix), system_rhs)
        else:
            system_result = np.linalg.solve(system_matrix, system_rhs)
        return PolynomialEvaluator(self.no_parameters, self.current_degree, self.poly.copy(), self.no_poly, system_result)

    def increase_degree(self):
        self.poly = generate_polynomials(self.no_parameters, self.current_degree)
        self.no_poly = self.poly.shape[0]

    def update_model(self):
        hermite_eval = evaluate_Hermite_polynomials(self.current_degree, self.par)
        self.products = np.ones((self.no_snapshots, self.no_poly))
        for j in range(self.no_parameters):
            self.products *= hermite_eval[:, j, self.poly[:, j]]
        # self.products_weighted = (self.products * self.wei)

        # print("SURROGATE: polynomial degree increased to",
        #       self.current_degree, "i.e.", self.no_poly, "polynomials")


def generate_polynomials(dim, degree):
    poly = np.zeros([1, dim], dtype=int)

    if degree == 0:
        return poly

    if degree > 0:
        poly = np.vstack((poly, np.eye(dim, dtype=int)))

    if degree > 1:
        tmp0 = np.eye(dim, dtype=int)
        tmp1 = np.eye(dim, dtype=int)
        for i in range(degree-1):
            polynew = np.zeros((tmp0.shape[0]*tmp1.shape[0], dim), dtype=int)
            idx = 0
            for j in range(tmp0.shape[0]):
                for k in range(tmp1.shape[0]):
                    polynew[idx] = tmp0[j, :]+tmp1[k, :]
                    idx += 1
            tmp1 = np.unique(polynew, axis=0)
            poly = np.vstack((poly, tmp1))
    return poly


def evaluate_Hermite_polynomials(max_degree, points):
    # for Hermite polynomials
    # using three term recurrence
    # p ... each row = coefficients of univariate polynomial
    # points ... points of evaluation, shape (n1,n2)
    # TODO: add columns on degree change
    no_polynomials = max_degree + 1
    no_points, no_parameters = points.shape
    values = np.zeros((no_points, no_parameters, no_polynomials))
    values[:, :, 0] = 1/np.sqrt(np.sqrt(2*np.pi))  # H_0(x) = 1
    if max_degree == 0:
        return values
    values[:, :, 1] = points/np.sqrt(np.sqrt(2*np.pi))  # H_1(x) = x
    if max_degree == 1:
        return values
    for n in range(2, no_polynomials):  # loop over degrees higher than 1
        # H_n(x) = x*H_n-1(x)/sqrt(n) - H_n-2(x)*sqrt(n-1)/sqrt(n)
        values[:, :, n] = (points*values[:, :, n-1] - values[:, :, n-2]*np.sqrt(n-1))/np.sqrt(n)
    return values
