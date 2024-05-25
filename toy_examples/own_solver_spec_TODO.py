#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 2 with required number of MPI processes):
mpirun -n 4 python3 -m mpi4py own_solver_spec.py

(Here, one of the processes will be used as collector,
one as solvers pool, remaining processes will be used as samplers.
Additional MPI processes will be spawned by solvers pool.)
"""

import os

from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.solver_specification import SolverSpec
from surrDAMH.stages import Stage


# prepare solver specification
# solver takes 2 parameters, returns 1 observation
class Own_solver_specification(SolverSpec):
    def __init__(self, f: float = -0.1, length: float = 1.0, m: float = 0.5) -> None:
        self.solver_module_path = "solver_examples/solver_examples.py"
        self.solver_module_name = "solver_examples"
        self.solver_class_name = "Solver_linela2exp_local"
        self.solver_parameters = {"f": f, "length": length, "m": m}


solver_spec = Own_solver_specification()

# configuration (specification of basic settings of the sampling framework):
conf = surrDAMH.Configuration(output_dir="out_own_solver_spec", no_parameters=2, no_observations=1)

# polynominal surrogate model updater:
updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)

# Gaussian prior distribution:
prior = surrDAMH.distributions.Normal(mean=[0.0, 0.0], sd=1.0)

# likelihood (additive Gaussian noise):
observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=[-2, 2])
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
                                 list_of_stages=list_of_stages, solver_spec=solver_spec)
sam.run()

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
