#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note 32 (2026-10-13): a flattened-likelihood Hamiltonian component inside a ``Mixture`` under
real MPI, four sampler ranks, and the ``component`` column of ``subchain_stats`` (D-32-3).

Stage 0 ``RandomWalk()`` (adaptive warm-up) -> stage 1 DAMH with ``[Hamiltonian(dimension_robust,
T=1) adaptive, Hamiltonian(dimension_robust, T=3, adaptive=False, likelihood_weight=0.2)]`` with
probabilities ``[0.8, 0.2]`` on the linear-Gaussian problem, on the FIXED perturbed linear
evaluator ``G~(x) = 1.05 A x`` of ``test_mpi_tempered.py``. Checked: the run completes; the DAMH
stage matches the closed form within the suite's 4 SE with R-hat < 1.1; its ``subchain_stats``
has the last column ``component`` (values in {0, 1}) on every rank, the warm-up has none; the
manifest records the flattened component's weight.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from surrDAMH.proposals import Hamiltonian, Mixture, RandomWalk
from surrDAMH.stages import Stage, stage_name

from .conftest import requires_mpiexec
from .test_mpi_posterior import _check_problem_matches, _check_stage_posterior, _run_case

pytestmark = [pytest.mark.mpi, requires_mpiexec]

NO_SAMPLERS = 4
WARM_UP_EVALUATIONS = 1_000
DAMH_EVALUATIONS = 5_000
SUBCHAIN = 3

BODY = f"""
    from surrDAMH.proposals import Mixture
    from surrDAMH.surrogates.parent import Evaluator

    class PerturbedLinearGradientEvaluator(Evaluator):
        def __init__(self):
            super().__init__(no_parameters=2, no_observations=2)
            self.matrix = 1.05 * solver_module.A

        def __call__(self, datapoints):
            return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.matrix.T

        def jacobian(self, datapoints):
            return self.matrix, self.matrix @ np.asarray(datapoints, dtype=float).reshape(-1)

    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="lhs", use_surrogate_gradients=True)
    assert conf.no_samplers == {NO_SAMPLERS}, conf.no_samplers
    mixture = Mixture([Hamiltonian(integrator="dimension_robust", integration_time=1.0, mass=1.0),
                       Hamiltonian(integrator="dimension_robust", integration_time=3.0, mass=1.0,
                                   adaptive=False, likelihood_weight=0.2)],
                      probabilities=[0.8, 0.2])
    stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations={WARM_UP_EVALUATIONS}),
              Stage(algorithm="DAMH", proposal=mixture, max_evaluations={DAMH_EVALUATIONS},
                    subchain_length={SUBCHAIN})]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    sam = problem.run_sampling(conf, stages, surrogate_evaluator=PerturbedLinearGradientEvaluator())
    print("DRIVER FINISHED", flush=True)
    """


def _stage_names():
    stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=WARM_UP_EVALUATIONS),
              Stage(algorithm="DAMH", subchain_length=SUBCHAIN, max_evaluations=DAMH_EVALUATIONS,
                    proposal=Mixture([Hamiltonian(integrator="dimension_robust", integration_time=1.0),
                                      Hamiltonian(integrator="dimension_robust", integration_time=3.0,
                                                  adaptive=False, likelihood_weight=0.2)],
                                     probabilities=[0.8, 0.2]))]
    return [stage_name(stage, i) for i, stage in enumerate(stages)]


def test_damh_with_a_flattened_mixture_component(mpi_workspace, run_mpi, linear_gaussian_problem):
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "flattened", BODY, n=NO_SAMPLERS, timeout=300)
    warm_up, damh = _stage_names()
    sampling_output = os.path.join(str(output_dir), "sampling_output")

    statistics = _check_stage_posterior("note 32 DAMH with a flattened mixture component", output_dir, damh,
                                        result, linear_gaussian_problem, no_samplers=NO_SAMPLERS)
    assert statistics.counter_accepted > 0

    assert not os.path.exists(os.path.join(sampling_output, "subchain_stats", warm_up))
    for rank in range(NO_SAMPLERS):
        path = os.path.join(sampling_output, "subchain_stats", damh, f"rank{rank:04d}.csv")
        frame = pd.read_csv(path)
        assert list(frame.columns)[-2:] == ["kernel", "component"], (rank, list(frame.columns))
        assert set(frame["component"].unique()) == {0, 1}, (rank, frame["component"].value_counts().to_dict())
        assert (frame["rank_world"] == frame["rank_world"].iloc[0]).all()

    with open(os.path.join(sampling_output, "run_manifest.json")) as f:
        resolution = json.load(f)["proposal_resolution"]
    assert resolution[1]["component1.likelihood_weight"] == 0.2
    assert "component0.likelihood_weight" not in resolution[1]
    assert not any("likelihood_weight" in key for key in resolution[0])
