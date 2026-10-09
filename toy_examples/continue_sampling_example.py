#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpiexec -n 3 python3 -m mpi4py continue_sampling_example.py

(2 samplers + 1 collector: use_solvers_pool=False, use_collector=True, so the solver runs
in-process on every sampler and the one remaining rank trains the surrogate.)

Demonstrates ``SamplingRun.continue_sampling`` (``docs/running.md#continuing-a-run``):

  run A: MH(adaptive RandomWalk) -> DAMH-SMU with a polynomial surrogate updater, into
         out_continue_example_a/
  run B: run_a.continue_sampling(conf_b, stages_b) -- a fresh Configuration/stage list, into
         out_continue_example_b/, reusing run A's chain states, tuned proposal scale and
         surrogate automatically.
"""

import os

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

import surrDAMH
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates import PolynomialSklearnUpdater


# same toy forward model as toy_examples/own_solver.py
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

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

# --------------------------------------------------------------------------------------------
# run A: adaptive MH -> DAMH-SMU, with a polynomial surrogate updater
# --------------------------------------------------------------------------------------------
conf_a = surrDAMH.Configuration(output_dir="out_continue_example_a", use_solvers_pool=False, use_collector=True)
updater_a = PolynomialSklearnUpdater(no_parameters=solver_instance.no_parameters,
                                    no_observations=solver_instance.no_observations)

stages_a = [
    # adaptive random walk: tunes its own scale while the first surrogate model is built
    Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=800),
    # DAMH-SMU: surrogate_model_updates defaults to True for a DAMH stage, so this keeps
    # retraining the polynomial as new snapshots arrive
    Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=1500, subchain_length=5),
]
run_a = problem.run_sampling(conf_a, stages_a, surrogate_updater=updater_a)

# A plain run does NOT save its surrogate state automatically -- only a continue_sampling(_local)
# run does that at the end (docs/running.md#continuing-a-run). To let run B restore run A's
# polynomial model (instead of refitting it from scratch), save it explicitly here, on the
# collector rank (the only rank that holds the trained Updater):
if rank_world == conf_a.rank_collector:
    surrDAMH.SurrogateRestart(state_dir=os.path.join(conf_a.output_dir, "sampling_output")).save(updater_a)
comm_world.Barrier()  # every rank waits until the checkpoint file exists before continuing

# --------------------------------------------------------------------------------------------
# run B: continue_sampling with a DAMH-SMU first stage and a fresh RandomWalk()
# --------------------------------------------------------------------------------------------
conf_b = surrDAMH.Configuration(output_dir="out_continue_example_b", use_solvers_pool=False, use_collector=True)
# Do NOT set initial_sample_type/continued_from_dir/stage_index_offset/no_stages_lineage/
# lineage_generation on conf_b -- continue_sampling sets all of them (ValueError otherwise).
stages_b = [
    # RandomWalk(scale=None): continue_sampling merges run_a's adaptive-stage carry-over into
    # this stage, so it starts from the tuned scale instead of the prior-derived default.
    Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=1500, subchain_length=5),
]
run_b = run_a.continue_sampling(conf_b, stages_b)
# What continue_sampling reused here, all by default (chains="continue"):
#  - every chain of run_b starts from run_a's last state (sampling_output/last_sample/);
#  - stages_b's RandomWalk() starts from run_a's pooled adaptive scale (carry_over/*.npz);
#  - the polynomial surrogate saved above is restored (surrogates.reuse.SurrogateReused), so
#    this DAMH-SMU stage needs no further MH warm-up;
#  - stage directories/seeds continue run_a's numbering (alg0002_... here, not alg0000_...).

run_b.write_report(observations=observations)
# post_processing_output/selection.json under out_continue_example_b/ now records, per stage and
# chain, which samples the report used and how much burn-in was dropped (created with the
# defaults on this first call: run_a's stages are included by default because include_previous is
# None and both runs sampled the same Problem). Edit that file and call
# surrDAMH.SamplingRun.load("out_continue_example_b").write_report() to re-run the
# post-processing with a different selection -- see the report's own "Selection and re-run"
# section and docs/outputs.md.
