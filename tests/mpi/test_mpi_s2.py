#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2 (2026-10-09) under real MPI: a DAMH-SMU stage with the exact safety kernel and the audit of
pre-rejected proposals (``exact_step_probability=0.3``, ``audit_prerejected=0.3``), two samplers
and a collector with a degree-1 polynomial updater (``-n 3``, the B6 layout of
``test_mpi_posterior.py``). Checked: the stage's posterior against the closed form (4 SE, R-hat),
the ``notes`` counters against ``raw_data``/``subchain_stats``, and that the audited snapshots
reached the collector (its snapshot total in ``surrogate_quality.csv`` equals the number of
snapshots the samplers sent, audited ones included). Also S3 (Polyak averaging of the received
networks) and S1 (surrogate error model: every installed evaluator carries an ``error_model``, the
surrogate likelihood is corrected, ``surrogate_quality.csv`` has the two appended columns).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from .conftest import requires_mpiexec
from .test_mpi_posterior import _check_problem_matches, _check_stage_posterior, _run_case

pytestmark = [pytest.mark.mpi, requires_mpiexec]

S2_BOOTSTRAP_EVALUATIONS = 2_000
S2_DAMH_EVALUATIONS = 15_000
AUDIT_HIDDEN_NATS = 5.0  # surrDAMH.modules.algorithms.AUDIT_HIDDEN_NATS

S2_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=False,
                                  initial_sample_type="lhs", save_snapshots_to_file=True,
                                  min_snapshots_initial=50, min_snapshots_to_update=500)
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2,
                                                           max_degree=1)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5),
                    max_evaluations={S2_BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5),
                    max_evaluations={S2_DAMH_EVALUATIONS}, subchain_length=1,
                    surrogate_model_updates=True, exact_step_probability=0.3, audit_prerejected=0.3)]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def _frame(output_dir, kind, stage, rank):
    return pd.read_csv(os.path.join(str(output_dir), "sampling_output", kind, stage, f"rank{rank:04d}.csv"))


def test_s2_exact_steps_and_audits_with_collector(mpi_workspace, run_mpi, linear_gaussian_problem):
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "s2", S2_BODY, n=3, timeout=300)
    damh = "alg0001_DAMH-SMU"
    _check_stage_posterior("S2 (DAMH-SMU K=1, exact 0.3, audit 0.3)", output_dir, damh, result,
                           linear_gaussian_problem)

    lgp = linear_gaussian_problem
    snapshots_sent = 0
    for rank in range(2):
        for stage in ("alg0000_MH", damh):
            raw = _frame(output_dir, "raw_data", stage, rank)
            evaluated_ok = raw["solver_tag"] >= 0
            snapshots_sent += int((raw["state_type"] == "accepted").sum())
            snapshots_sent += int(((raw["state_type"] == "rejected") & evaluated_ok).sum())
            snapshots_sent += int(((raw["state_type"] == "audited") & evaluated_ok).sum())
        raw = _frame(output_dir, "raw_data", damh, rank)
        notes = _frame(output_dir, "notes", damh, rank)
        stats = _frame(output_dir, "subchain_stats", damh, rank)
        exact = stats["kernel"] == 1
        audited = raw[raw["state_type"] == "audited"]
        residual_exact = audited[["obs_0", "obs_1"]].to_numpy() - lgp.observed_data
        residual_approx = audited[["obs_approx_0", "obs_approx_1"]].to_numpy() - lgp.observed_data
        gap = 0.5 * np.sum((residual_approx ** 2 - residual_exact ** 2) / lgp.obs_sd ** 2, axis=1)
        counters = {name: int(notes[name].iloc[0]) for name in
                    ("exact_steps", "exact_accepted", "audited", "audit_hidden", "pre-rejected", "accepted", "rejected")}
        print(f"\n[S2 MPI] rank {rank}: {counters}, kernel rows {int(exact.sum())}/{len(stats)}")
        assert counters["exact_steps"] == int(exact.sum()) > 0
        assert counters["exact_accepted"] == int(stats.loc[exact, "outer_accepted"].sum())
        assert counters["audited"] == len(audited) > 0
        assert counters["audit_hidden"] == int(np.sum(gap > AUDIT_HIDDEN_NATS))
        assert counters["audited"] < counters["pre-rejected"]  # ~30 % of them
        assert len(raw) == len(stats) + counters["audited"]  # one row per iteration plus the audits
        assert counters["accepted"] + counters["rejected"] + counters["audited"] == S2_DAMH_EVALUATIONS
        assert audited[["obs_0", "obs_1", "obs_approx_0", "obs_approx_1"]].notna().all().all()
        assert f"Stage {damh} at MPI rank {rank} finished" in result.stdout

    # every snapshot reached the collector, the audited ones included: its last quality row's
    # snapshots_total + batch_size is the number of snapshots it received
    quality = pd.read_csv(os.path.join(str(output_dir), "sampling_output", "surrogate_quality.csv"))
    received = int(quality["snapshots_total"].iloc[-1] + quality["batch_size"].iloc[-1])
    print(f"[S2 MPI] snapshots sent {snapshots_sent}, received by the collector {received}")
    assert received == snapshots_sent


# ---------------------------------------------------------------------------------------------
# S3 (2026-10-09): Polyak averaging of the received networks, 2 samplers + collector
# ---------------------------------------------------------------------------------------------

S3_BOOTSTRAP_EVALUATIONS = 2_000
S3_DAMH_EVALUATIONS = 12_000

S3_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=False,
                                  initial_sample_type="lhs", save_snapshots_to_file=True,
                                  min_snapshots_initial=200, min_snapshots_to_update=500)
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.NeuralNetworkUpdater(
        no_parameters=2, no_observations=2, hidden_layer_sizes=(16, 16),
        iterations_batch=50, seed=0)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5),
                    max_evaluations={S3_BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5),
                    max_evaluations={S3_DAMH_EVALUATIONS}, subchain_length=1,
                    surrogate_model_updates=True, polyak_averaging=True)]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def test_s3_polyak_averaging_with_network_and_collector(mpi_workspace, run_mpi, linear_gaussian_problem):
    """
    ``notes.polyak_updates`` is NOT required to be equal on the two ranks: the collector sends its
    current network to a sampler only when that sampler has a request posted, so the two chains
    may receive different subsets of the retrained versions (printed for the record).
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "s3", S3_BODY, n=3, timeout=400)
    damh = "alg0001_DAMH-SMU"
    _check_stage_posterior("S3 (DAMH-SMU K=1, network, polyak_averaging)", output_dir, damh, result,
                           linear_gaussian_problem)
    updates = []
    for rank in range(2):
        notes = _frame(output_dir, "notes", damh, rank)
        updates.append(int(notes["polyak_updates"].iloc[0]))
        assert f"Stage {damh} at MPI rank {rank} finished" in result.stdout
        assert f"polyak updates: {updates[-1]}" in result.stdout
        assert int(_frame(output_dir, "notes", "alg0000_MH", rank)["polyak_updates"].iloc[0]) == 0
    print(f"\n[S3 MPI] polyak_updates per rank: {updates} (equal: {updates[0] == updates[1]})")
    assert all(k > 1 for k in updates)


# ---------------------------------------------------------------------------------------------
# S1 (2026-10-09): surrogate error model, 2 samplers + collector, polynomial updater, held-out set
# ---------------------------------------------------------------------------------------------

S1_BOOTSTRAP_EVALUATIONS = 2_000
S1_DAMH_EVALUATIONS = 12_000

S1_BODY = f"""
    from surrDAMH.modules.algorithms import AlgorithmBase
    from surrDAMH.modules.error_model import CorrectedNormal

    _original_commit = AlgorithmBase._commit_evaluator

    def _recording_commit(self, candidate, evaluator, pending):
        _original_commit(self, candidate, evaluator, pending)
        model = getattr(candidate, "error_model", None)
        print(f"S1 INSTALL rank={{self.rank_world}} stage={{self.stage.name}} has_model={{model is not None}} "
              f"corrected={{isinstance(self.likelihood_surrogate, CorrectedNormal)}} "
              f"residuals={{-1 if model is None else model.no_residuals}}", flush=True)

    AlgorithmBase._commit_evaluator = _recording_commit
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=False,
                                  initial_sample_type="lhs", save_snapshots_to_file=True,
                                  min_snapshots_initial=50, min_snapshots_to_update=500)
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2,
                                                           max_degree=1)
    solver = solver_module.LinearSolver()
    test_parameters = np.random.default_rng(7).normal(size=(40, 2))
    test_observations = []
    for parameters in test_parameters:
        solver.set_parameters(parameters)
        test_observations.append(solver.get_observations())
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5),
                    max_evaluations={S1_BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5),
                    max_evaluations={S1_DAMH_EVALUATIONS}, subchain_length=2,
                    surrogate_model_updates=True, surrogate_error_model=True)]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater,
                               surrogate_test_data=(test_parameters, np.array(test_observations)))
    print("DRIVER FINISHED", flush=True)
    """


def test_s1_error_model_with_polynomial_updater_and_collector(mpi_workspace, run_mpi, linear_gaussian_problem):
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "s1", S1_BODY, n=3, timeout=300)
    damh = "alg0001_DAMH-SMU"
    _check_stage_posterior("S1 (DAMH-SMU K=2, polynomial, surrogate_error_model)", output_dir, damh, result,
                           linear_gaussian_problem)
    installs = [line for line in result.stdout.splitlines() if line.startswith("S1 INSTALL") and damh in line]
    for rank in range(2):
        mine = [line for line in installs if f"rank={rank} " in line]
        print(f"\n[S1 MPI] rank {rank}: {len(mine)} installs in {damh}; last: {mine[-1] if mine else None}")
        assert mine, result.describe()
        assert all("has_model=True" in line and "corrected=True" in line for line in mine), mine
        # the held-out set is pooled with the snapshots: more residuals than the 40 held-out points
        assert all(int(line.rsplit("residuals=", 1)[1]) > 40 for line in mine), mine
    quality = pd.read_csv(os.path.join(str(output_dir), "sampling_output", "surrogate_quality.csv"))
    assert list(quality.columns) == ["snapshots_total", "batch_size", "rmse", "max_abs_error",
                                     "error_bias_norm", "error_cov_trace"]
    assert quality[["error_bias_norm", "error_cov_trace"]].notna().all().all()
    assert (quality["error_cov_trace"] >= 0).all()
    print(f"[S1 MPI] surrogate_quality.csv rows {len(quality)}, last error_bias_norm "
          f"{quality['error_bias_norm'].iloc[-1]:.3e}, error_cov_trace {quality['error_cov_trace'].iloc[-1]:.3e}")
