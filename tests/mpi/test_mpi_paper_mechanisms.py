#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The two mechanisms of the paper's realisation section (2026-10-23, paper §6.5) under real MPI: an MH
warm-up, then a DAMH-SMU stage with the exact PRIOR-proposal step (``exact_step_probability=0.2``,
``exact_step_proposal="prior"``) and RANDOMISED installation
(``RandomisedInstallation(n_lambda=50, a_lambda=0.5)``), two samplers and a collector with a
degree-1 polynomial updater (``-n 3``, the layout of ``test_mpi_s2.py``). Checked: the stage's
posterior against the closed form (4 SE, R-hat); per rank, the ``notes`` counters against the
``installations`` CSV and ``subchain_stats``, every install at a coin success of the replayed stream
``RandomState(seed0 + 6)`` (one uniform per outer iteration ``i >= 1``); the manifest's seeds, stage
spec and ``proposal_resolution``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from .conftest import requires_mpiexec
from .test_mpi_posterior import _check_problem_matches, _check_stage_posterior, _read_manifest, _run_case

pytestmark = [pytest.mark.mpi, requires_mpiexec]

BOOTSTRAP_EVALUATIONS = 2_000
DAMH_EVALUATIONS = 15_000
N_LAMBDA, A_LAMBDA = 50, 0.5

BODY = f"""
    from surrDAMH.installation import RandomisedInstallation
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=False,
                                  initial_sample_type="lhs", save_snapshots_to_file=True,
                                  min_snapshots_initial=50, min_snapshots_to_update=300)
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2,
                                                           max_degree=1)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5),
                    max_evaluations={BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations={DAMH_EVALUATIONS},
                    subchain_length=1, surrogate_model_updates=True, exact_step_probability=0.2,
                    exact_step_proposal="prior",
                    installation=RandomisedInstallation(n_lambda={N_LAMBDA}, a_lambda={A_LAMBDA}))]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def _frame(output_dir, kind, stage, rank):
    return pd.read_csv(os.path.join(str(output_dir), "sampling_output", kind, stage, f"rank{rank:04d}.csv"))


def _probability(n):
    return 1.0 if n <= 0 else min(1.0, (n / N_LAMBDA) ** (-A_LAMBDA))


def test_prior_step_and_randomised_installation_with_collector(mpi_workspace, run_mpi, linear_gaussian_problem):
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "paper_mechanisms", BODY, n=3, timeout=400)
    damh = "alg0001_DAMH-SMU"
    _check_stage_posterior("paper §6.5 (DAMH-SMU K=1, prior step 0.2, randomised installation)", output_dir, damh,
                           result, linear_gaussian_problem)
    manifest = _read_manifest(output_dir)
    seeds = {(entry["rank"], entry["stage_index"]): entry for entry in manifest["seeds"]["per_rank"]}
    assert manifest["stages"][1]["installation"] == {"type": "RandomisedInstallation", "n_lambda": float(N_LAMBDA),
                                                     "a_lambda": A_LAMBDA}
    assert manifest["stages"][1]["exact_step_proposal"] == "prior"
    assert any(entry.get("name") == damh and entry.get("exact_step_proposal") == "prior"
               for entry in manifest["proposal_resolution"]), manifest["proposal_resolution"]
    for rank in range(2):
        entry = seeds[(rank, 1)]
        assert entry["installation_seed"] == entry["seed0"] + 6 == 10 * (2 * rank + 1) + 6
        notes = _frame(output_dir, "notes", damh, rank)
        stats = _frame(output_dir, "subchain_stats", damh, rank)
        log = _frame(output_dir, "installations", damh, rank)
        assert list(notes.columns[-2:]) == ["installations", "versions_received"]
        assert "installations" not in _frame(output_dir, "notes", "alg0000_MH", rank).columns
        installations, received = int(notes["installations"].iloc[0]), int(notes["versions_received"].iloc[0])
        exact = stats["kernel"] == 1
        print(f"\n[paper MPI] rank {rank}: iterations {len(stats)}, exact steps {int(exact.sum())} "
              f"(accepted {int(notes['exact_accepted'].iloc[0])}), versions received {received}, installed {installations}")
        assert int(notes["exact_steps"].iloc[0]) == int(exact.sum()) > 0
        assert installations == len(log) > 0 and received >= installations
        assert log["iteration"].is_monotonic_increasing and log["version"].is_monotonic_increasing
        assert int(log["version"].max()) <= received
        # every install happened at a coin success of the replayed stream (one uniform per i >= 1)
        generator = np.random.RandomState(entry["installation_seed"])
        uniforms = np.array([np.nan] + [generator.uniform() for _ in range(1, len(stats))])
        for iteration in log["iteration"]:
            assert 1 <= iteration < len(stats)
            assert uniforms[iteration] < _probability(iteration - 1), iteration
        assert f"installations: {installations}/{received} versions received" in result.stdout
