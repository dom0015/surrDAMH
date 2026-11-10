#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 (2026-10-12): an adaptive ``Mixture`` stage under real MPI, four sampler ranks.

Stage 0 ``Mixture([RandomWalk(scale=0.05), RandomWalk()], probabilities=[0.1, 0.9])`` (adaptive:
component 1) -> stage 1 ``RandomWalk(adaptive=False)`` consuming the pooled carried scale (frozen
consumer, as B5 does: it prints a warning if nothing was carried) -> stage 2 a FIXED
``Mixture([RandomWalk(scale=1.5), RandomWalk(adaptive=False)], probabilities=[0.3, 0.7])`` whose
second component again takes the carried scale. Checked: the run completes; every rank carries
the identical scale and it is not the prior default; the trace of stage 0 is the random walk's,
written for component 1 (``proposal_resolution[0]["adaptive_stats_component"] == 1``); stages 1
and 2 match the closed-form posterior within the suite's 4 SE with R-hat < 1.1.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from surrDAMH.modules.continuation import load_carry_over
from surrDAMH.modules.proposals import GaussRandomWalk_adaptive

from .conftest import requires_mpiexec
from .test_mpi_posterior import (_assert_identical_carry_over, _check_problem_matches, _check_stage_posterior,
                                 _run_case)

pytestmark = [pytest.mark.mpi, requires_mpiexec]

NO_SAMPLERS = 4
ADAPTIVE_EVALUATIONS = 4_000
EVALUATIONS = 12_000

BODY = f"""
    from surrDAMH.proposals import Mixture
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=False, use_solvers_pool=False,
                                  debug=True,  # prints the carry-over lines parsed below
                                  initial_sample_type="lhs")
    assert conf.no_samplers == {NO_SAMPLERS}, conf.no_samplers
    stages = [Stage(algorithm="MH", max_evaluations={ADAPTIVE_EVALUATIONS},
                    proposal=Mixture([RandomWalk(scale=0.05), RandomWalk()], probabilities=[0.1, 0.9])),
              Stage(algorithm="MH", proposal=RandomWalk(adaptive=False), max_evaluations={EVALUATIONS}),
              Stage(algorithm="MH", max_evaluations={EVALUATIONS},
                    proposal=Mixture([RandomWalk(scale=1.5), RandomWalk(adaptive=False)],
                                     probabilities=[0.3, 0.7]))]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    sam = problem.run_sampling(conf, stages)
    print("DRIVER FINISHED", flush=True)
    """


def test_adaptive_mixture_then_its_carried_scale(mpi_workspace, run_mpi, linear_gaussian_problem):
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "mixture", BODY, n=NO_SAMPLERS, timeout=300)
    assert "Message truncated" not in result.stdout + result.stderr, result.describe()
    # stage 1 / stage 2 found a carried scale (no "fixed default" warning)
    assert "without a scale and no earlier adaptive stage" not in result.stdout, result.describe()

    # (b) the pooled scale, identical on every rank, from the adaptive component, not the prior default
    scale = _assert_identical_carry_over(result, "alg0000_MH-adaptive", "scale", no_samplers=NO_SAMPLERS)
    carried, summary = load_carry_over(str(output_dir), "alg0000_MH-adaptive")
    np.testing.assert_allclose(np.asarray(carried["scale"]).ravel(), scale)
    prior_default = 2.38 ** 2 / 2 * np.eye(2)
    assert not np.allclose(carried["scale"], prior_default), carried["scale"]
    assert {"component1.base_cov", "component1.log_sigma", "component1.n_pooled"} <= set(summary), summary

    # (d) the adaptation trace is the random walk's, of component 1, on every rank
    with open(os.path.join(str(output_dir), "sampling_output", "run_manifest.json")) as f:
        resolution = json.load(f)["proposal_resolution"]
    assert resolution[0]["adaptive_stats_component"] == 1, resolution
    assert "adaptive_stats_component" not in resolution[1] and "adaptive_stats_component" not in resolution[2]
    for rank in range(NO_SAMPLERS):
        path = os.path.join(str(output_dir), "sampling_output", "adaptive_stats", "alg0000_MH-adaptive",
                            f"rank{rank:04d}.csv")
        with open(path) as f:
            rows = list(csv.reader(f))
        assert rows[0] == GaussRandomWalk_adaptive.adaptation_stats_header + ["rank_world"], rows[0]
        assert len(rows) > 10, f"{path}: only {len(rows) - 1} trace rows"

    # (c) the frozen consumer and the fixed mixture sample the closed-form posterior
    for label, stage_name in (("C2 MH with the mixture's carried scale", "alg0001_MH"),
                              ("C2 fixed mixture", "alg0002_MH")):
        statistics = _check_stage_posterior(label, output_dir, stage_name, result, linear_gaussian_problem,
                                            no_samplers=NO_SAMPLERS)
        acceptance = statistics.counter_accepted / (statistics.counter_accepted + statistics.counter_rejected)
        assert 0.1 < acceptance < 0.6, f"{label}: acceptance {acceptance:.3f}\n" + result.describe()
