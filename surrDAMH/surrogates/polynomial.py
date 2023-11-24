#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from surrDAMH.surrogates.parent import Trainer, Evaluator


class PolynomialEvaluator(Evaluator):
    def __init__(self, no_parameters: int, hermite_coefs: npt.NDArray, poly: npt.NDArray, no_poly: int, system_result: npt.NDArray) -> None:
        self.no_parameters = no_parameters
        self.hermite_coefs = hermite_coefs
        self.poly = poly
        self.no_poly = no_poly
        self.coefficients_matrix = system_result

    def __call__(self, datapoints):
        # evaluates the surrogate model in datapoints
        datapoints = datapoints.reshape(-1, self.no_parameters)
        no_datapoints = datapoints.shape[0]
        hermite_evaluations = evaluate_polynomials(self.hermite_coefs, datapoints)
        products = np.ones((no_datapoints, self.no_poly))
        for j in range(self.no_parameters):
            products *= hermite_evaluations[:, j, self.poly[:, j]]
        evaluations = np.matmul(products, self.coefficients_matrix)
        return evaluations


class PolynomialTrainer(Trainer):  # initiated by COLLECTOR
    def __init__(self, no_parameters, no_observations, max_degree=5, solver="pinv"):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_degree = max_degree
        self.solver = solver
        # processed data (used for surrogate model construction):
        self.processed_par = np.empty((0, self.no_parameters))
        self.processed_obs = np.empty((0, self.no_observations))
        self.processed_wei = np.empty((0, 1))
        # self.no_processed = self.processed_par.shape[0]
        # data not yet used for surrogate construction:
        self.new_par = np.empty((0, self.no_parameters))
        self.new_obs = np.empty((0, self.no_observations))
        self.new_wei = np.empty((0, 1))
        self.no_processed = 0
        self.no_snapshots = 0
        self.current_degree = 1
        self.on_degree_change()

    def add_data(self, parameters, observations, weights=None):
        # add new data to matrices of non-processed data
        # TODO: polynomial surrogate weights
        weights = None  # ALL WEIGHTS WILL BE SET TO ONES
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)

        if weights is None:
            no_new_snapshots = parameters.shape[0]
            weights = np.ones((no_new_snapshots, 1))
        self.new_par = np.vstack((self.new_par, parameters))
        self.new_obs = np.vstack((self.new_obs, observations))
        self.new_wei = np.vstack((self.new_wei, weights))

    def get_evaluator(self):
        no_non_processed = self.new_par.shape[0]
        # both processed and non-processed
        self.no_snapshots += no_non_processed
        # TODO: polynomial degree formula
        degree = int(np.floor(np.log(self.no_snapshots)/np.log(self.no_parameters)))
        degree = min(degree, self.max_degree)
        if degree > self.current_degree:
            self.current_degree = degree
            self.on_degree_change()  # recalculates self.products_weighted
        products_new = np.ones((no_non_processed, self.no_poly))
        hermite_eval = evaluate_polynomials(self.hermite_coefs, self.new_par)
        for j in range(self.no_parameters):
            products_new *= hermite_eval[:, j, self.poly[:, j]]
        products_new_weighted = products_new * self.new_wei

        self.no_processed += no_non_processed
        self.processed_par = np.vstack((self.processed_par, self.new_par))
        self.processed_obs = np.vstack((self.processed_obs, self.new_obs))
        self.processed_wei = np.vstack((self.processed_wei, self.new_wei))
        self.products = np.vstack((self.products, products_new))
        self.products_weighted = np.vstack((self.products_weighted, products_new_weighted))

        self.new_par = np.empty((0, self.no_parameters))
        self.new_obs = np.empty((0, self.no_observations))
        self.new_wei = np.empty((0, 1))

        system_matrix = np.matmul(self.products_weighted.transpose(), self.products)
        system_rhs = np.matmul(self.products_weighted.transpose(), self.processed_obs)
        # shape (no_poly,no_observations)
        if self.solver == "pinv":
            system_result = np.matmul(np.linalg.pinv(system_matrix), system_rhs)
        else:
            system_result = np.linalg.solve(system_matrix, system_rhs)
        return PolynomialEvaluator(self.no_parameters, self.hermite_coefs.copy(), self.poly.copy(), self.no_poly, system_result)

    def on_degree_change(self):
        # TODO: reuse previous degree (old self.products)
        self.poly = generate_polynomials(self.no_parameters, self.current_degree)
        self.no_poly = self.poly.shape[0]
        self.hermite_coefs = calculate_hermite_coefs(self.current_degree)
        print("HERMITE:")
        d = self.hermite_coefs.diagonal()
        print()

        # recalculate products and products_weighted for already processed data
        self.products = np.ones((self.no_processed, self.no_poly))
        hermite_eval = evaluate_polynomials(self.hermite_coefs, self.processed_par)
        for j in range(self.no_parameters):
            self.products *= hermite_eval[:, j, self.poly[:, j]]
        self.products_weighted = (self.products * self.processed_wei)

        print("SURROGATE: polynomial degree increased to",
              self.current_degree, "i.e.", self.no_poly, "polynomials")


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


def calculate_hermite_coefs(degree):
    # coefficients of normalized Hermite polynomials
    n = degree + 1
    coefs = np.zeros((n, n))
    coefs[0, 0] = 1
    if degree == 0:
        return coefs
    coefs[1, 1] = 1
    diff = np.arange(1, n)
    for i in range(2, n):
        coefs[i, 1:] += coefs[i-1, :-1]
        coefs[i, :-1] -= diff*coefs[i-1, 1:]
    for i in range(n):
        coefs[i, :] = np.divide(coefs[i, :], np.sqrt(np.math.factorial(i)))
    return coefs


def evaluate_polynomials(p, points):
    # p ... each row = coefficients of univariate polynomial
    # points ... points of evaluation, shape (n1,n2)
    no_polynomials, n = p.shape
    n1, n2 = points.shape
    values = np.zeros((n1, n2, no_polynomials))
    for j in range(no_polynomials):  # loop over polynomials
        values[:, :, j] = p[j, 0]  # constant term
    points_pow = np.ones((n1, n2))
    for i in range(1, n):  # loop over degree+1
        points_pow *= points
        for j in range(no_polynomials):  # loop over polynomials
            values[:, :, j] += p[j, i]*points_pow
    return values  # shape (n1, n2, no_polynomials)
