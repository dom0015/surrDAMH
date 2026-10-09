#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpiexec -n 3 python3 -m mpi4py auto_mode_example.py

(2 samplers + 1 collector: use_solvers_pool=False, use_collector=True, so the solver runs
in-process on every sampler and the one remaining rank trains the surrogate.)

Demonstrates ``Problem.run_sampling_auto`` (``docs/running.md#automatic-mode-2026-10-08``):
instead of writing an explicit stage list, a budget of exact model evaluations and a mode
("robust"/"fast") are turned into a warm-up + DAMH-SMU chunk layout and a default neural-network
surrogate (``surrDAMH.auto.plan_auto``). Both modes sample the exact posterior; "fast" uses a
Hamiltonian proposal on the surrogate's gradients and may mix better on an easy problem like this
one, at the cost of being less robust on a harder one. The resolved plan is printed once at
start-up and recorded in ``run_manifest.json``'s ``"auto"`` entry / as ``run.auto`` -- every
number in it is a placeholder pending the validation plan of
``library_notes/25_robust_by_default_roadmap_2026-10-08.md`` §6.
"""

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

import surrDAMH
from surrDAMH.solvers import Solver


# same toy forward model as toy_examples/own_solver.py / continue_sampling_example.py
class Own_solver(Solver):
    def __init__(self, solver_id: int = 0, output_dir: str | None = None) -> None:
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters: npt.NDArray) -> None:
        self.x = parameters[0]
        self.y = parameters[1]

    def get_observations(self) -> npt.NDArray:
        res = (self.x**2 - self.y) * (np.log((self.x - self.y)**2 + 1))
        return np.array([res])  # shape (no_observations,)


solver_instance = Own_solver()

prior = surrDAMH.distributions.PriorIndependentComponents([
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
])
observations = surrDAMH.solvers.calculate_artificial_observations(solver_instance=solver_instance, parameters=[-2, 2])
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)

problem = surrDAMH.Problem(prior, likelihood, solver=solver_instance)

conf = surrDAMH.Configuration(output_dir="out_auto_example", use_solvers_pool=False, use_collector=True)

# --------------------------------------------------------------------------------------------
# automatic mode: a budget of exact model evaluations instead of an explicit stage list
# --------------------------------------------------------------------------------------------
run = problem.run_sampling_auto(conf, budget=6000, mode="robust")
# Alternatives (uncomment one, instead of the call above):
# run = problem.run_sampling_auto(conf, budget=6000, mode="fast")       # Hamiltonian on the
#                                                                        # surrogate's gradients
# run = problem.run_sampling_auto(conf, time_limit=60.0, mode="robust")  # wall-clock budget
#                                                                        # instead of a budget
# run = problem.run_sampling_local_auto(conf, budget=6000, mode="robust")  # one chain, no MPI

if MPI.COMM_WORLD.Get_rank() == 0:
    # run.auto == run_manifest.json's "auto" entry (surrDAMH.auto.AutoPlan.manifest_entry()):
    # mode, budget, time_limit, no_samplers, per_chain_budget, test_data_size, warm_up, chunks,
    # stage_names, proposal, surrogate, conf_settings, notes -- see docs/outputs.md.
    print("run.auto keys:", list(run.auto.keys()))

run.write_report(observations=observations)
# post_processing_output/report_extended.html under out_auto_example/ shows the resolved stage
# list (copyable into a plain run_sampling(conf, stages, ...) call) alongside the usual sections.
