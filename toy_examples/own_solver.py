#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 4 with required number of MPI processes):
mpirun -n 4 python3 -m mpi4py own_solver.py

(Here, one process will be used as collector,
and the remaining processes will be used as samplers.)
"""

import os

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage


# create instance of own local solver
class Own_solver(Solver):
    def __init__(self, solver_id: int = 0, output_dir: str | None = None) -> None:
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters: npt.NDArray) -> None:
        self.x = parameters[0]
        self.y = parameters[1]

    def get_observations(self) -> npt.NDArray:
        res = (self.x**2-self.y)*(np.log((self.x-self.y)**2+1))
        return np.array([res])  # shape (no_observations,), as the Solver contract asks


# solver instance, takes 2 parameters, returns 1 observation:
solver_instance = Own_solver()

# configuration (specification of basic settings of the sampling framework),
# since local solver instance was specified, solvers pool cannot be used,
# the solver will be evaluated directly on Samplers:
conf = surrDAMH.Configuration(output_dir="out_own_solver", use_solvers_pool=False)

# polynominal surrogate model updater (sizes come straight from the solver instance --
# Problem, built below, resolves the same numbers from it):
updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=solver_instance.no_parameters,
                                                        no_observations=solver_instance.no_observations)

# Gaussian prior distribution:
prior = surrDAMH.distributions.PriorIndependentComponents([
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
])

# likelihood (additive Gaussian noise):
observations = surrDAMH.solvers.calculate_artificial_observations(solver_instance=solver_instance, parameters=[-2, 2])
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)

# sampling process stages:
list_of_stages = []
# during MH stage, initial surrogate model is constructed:
list_of_stages.append(Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=500))
# during DAMH-SMU stage, surrogate model is further updated:
list_of_stages.append(Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=500, surrogate_model_updates=True))
# during DAMH stage, surrogate model is used but not updated:
list_of_stages.append(Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=500, surrogate_model_updates=False))

problem = surrDAMH.Problem(prior, likelihood, solver=solver_instance)
run = problem.run_sampling(conf, list_of_stages, surrogate_updater=updater)

# post processing:
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.get_summary()

    # save histograms grid to file:
    fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30)
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")
