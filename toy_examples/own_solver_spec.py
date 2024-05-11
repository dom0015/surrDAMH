#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpirun -n 4 python3 -m mpi4py own_solver_spec.py
"""

import os

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage
from surrDAMH.solver_specification import SolverSpec


# prepare solver specification
class Own_solver_specification(SolverSpec):
    def __init__(self, solver_id: int = 0, output_dir: str | None = None) -> None:
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters: npt.NDArray) -> None:
        self.x = parameters[0]
        self.y = parameters[1]

    def get_observations(self) -> npt.ArrayLike:
        res = (self.x**2-self.y)*(np.log((self.x-self.y)**2+1))
        return res


# solver instance, takes 2 parameters, returns 1 observation:
solver_instance = Own_solver()

# configuration (specification of basic settings of the sampling framework),
# since local solver instance was specified, solvers pool cannot be used,
# the solver will be evaluated directly on Samplers:
conf = surrDAMH.Configuration(output_dir="out_own_solver", no_parameters=2, no_observations=1,
                              use_solvers_pool=False)

# polynominal surrogate model updater:
updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)

# Gaussian prior distribution:
prior = surrDAMH.distributions.Normal(mean=[0.0, 0.0], sd=1.0)

# likelihood (additive Gaussian noise):
observations = surrDAMH.solvers.calculate_artificial_observations(solver_instance=solver_instance, parameters=[-2, 2])
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)

# sampling process stages:
list_of_stages = []
# during MH stage, initial surrogate model is constructed:
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=500))
# during DAMH-SMU stage, surrogate model is further updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5, max_evaluations=500, surrogate_model_updates=True))
# during DAMH stage, surrogate model is used but not updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5, max_evaluations=500, surrogate_model_updates=False))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_instance=solver_instance)
sam.run()

# post processing:
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.print_summary()

    # save histograms grid to file:
    fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30)
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")
