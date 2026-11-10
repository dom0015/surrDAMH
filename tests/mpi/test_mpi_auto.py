#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic mode under real MPI (``Problem.run_sampling_auto``; API_SPEC_auto §5, 2026-10-08).

Three short runs on the linear-Gaussian toy problem of ``tests/conftest.py`` (closed-form
posterior), in-process solver (``use_solvers_pool=False``):

* robust mode, 2 samplers + collector, polynomial updater override, ``budget=8000``: posterior of
  the pooled DAMH chunks vs the closed form (4 batch-means SE, R-hat < 1.1), the manifest's
  ``"auto"`` entry, the held-out set evaluated on the collector (file, size) while every sampler
  spent exactly the planned evaluations per stage;
* fast mode with the default network and a small budget: smoke + loose posterior check (B8 pattern);
* ``use_collector=False``: a single adaptive MH stage, completes.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

from helpers_statistics import (assert_matches_closed_form, gelman_rubin, max_deviation_in_se,
                                pooled_chain_statistics, read_weighted_samples)

from .conftest import requires_mpiexec, write_driver

pytestmark = [pytest.mark.mpi, requires_mpiexec]

STAGE_COUNTER_RE = re.compile(
    r"Stage (\S+) at MPI rank (\d+) finished - acc/rej/prerej samples: (\d+) (\d+) (\d+)")
# S2 (2026-10-09): the stage-end line also reports the audits, which spend evaluations too
AUDIT_RE = re.compile(r"Stage (\S+) at MPI rank (\d+) finished .*- audited: (\d+) \(hidden: (\d+)\)")
RHAT_LIMIT = 1.1
CHUNK_NAMES = [f"alg000{i}_DAMH-SMU" for i in range(1, 5)]
PLANNED_NAMES = ["alg0000_MH-adaptive"] + CHUNK_NAMES

DRIVER_BODY = """
    import json
    from mpi4py import MPI
    MODE = sys.argv[2]
    BUDGET = int(sys.argv[3])
    USE_COLLECTOR = sys.argv[4] == "collector"
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, use_collector=USE_COLLECTOR, use_solvers_pool=False)
    assert conf.no_samplers == 2, conf.no_samplers
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    updater = None
    if MODE == "robust":
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1)
    run = problem.run_sampling_auto(conf, budget=BUDGET, mode=MODE, surrogate_updater=updater)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("AUTO_ENTRY " + json.dumps(run.auto), flush=True)
        print("RUN_STAGES " + json.dumps([s.name for s in run.stages]), flush=True)
    print("DRIVER FINISHED", flush=True)
    """


def _run(mpi_workspace, run_mpi, name, mode, budget, collector=True, n=3, timeout=240.0):
    driver = write_driver(mpi_workspace, name, DRIVER_BODY)
    result = run_mpi(driver, n=n, timeout=timeout,
                     args=[str(mpi_workspace / f"out_{name}"), mode, str(budget),
                           "collector" if collector else "none"])
    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    entry = json.loads(re.search(r"^AUTO_ENTRY (.*)$", result.stdout, re.M).group(1))
    run_stages = json.loads(re.search(r"^RUN_STAGES (.*)$", result.stdout, re.M).group(1))
    output_dir = mpi_workspace / f"out_{name}"
    with open(os.path.join(output_dir, "sampling_output", "run_manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["auto"] == entry
    assert run_stages == entry["stage_names"] == [s["name"] for s in manifest["stages"]]
    return result, output_dir, manifest


def _counters(stdout):
    return {(m.group(1), int(m.group(2))): tuple(int(m.group(i)) for i in (3, 4, 5))
            for m in STAGE_COUNTER_RE.finditer(stdout)}


def _audited(stdout):
    """``{(stage, rank): audited}`` from the stage-end lines (S2)."""
    return {(m.group(1), int(m.group(2))): int(m.group(3)) for m in AUDIT_RE.finditer(stdout)}


def _pooled_chunks(output_dir, no_parameters=2, no_chains=2):
    """Per chain: the non-excluded stages (the DAMH chunks) concatenated in order."""
    return [np.vstack([read_weighted_samples(output_dir, name, no_parameters, rank) for name in CHUNK_NAMES])
            for rank in range(no_chains)]


def _report(label, statistics, rhat, problem, duration):
    print(f"\n[{label}] pooled {statistics.no_samples} samples, mean {np.array2string(statistics.mean, precision=4)} "
          f"vs {np.array2string(problem.posterior_mean, precision=4)} "
          f"(max |dev|/SE {max_deviation_in_se(statistics.mean, problem.posterior_mean, statistics.se_mean):.2f}), "
          f"cov {np.array2string(statistics.cov.ravel(), precision=4)} vs "
          f"{np.array2string(problem.posterior_cov.ravel(), precision=4)} "
          f"(max |dev|/SE {max_deviation_in_se(statistics.cov, problem.posterior_cov, statistics.se_cov):.2f}), "
          f"R-hat {np.array2string(rhat, precision=4)}, wall {duration:.1f} s", flush=True)


def test_auto_robust_two_samplers_collector(mpi_workspace, run_mpi, linear_gaussian_problem):
    budget = 8_000
    result, output_dir, manifest = _run(mpi_workspace, run_mpi, "auto_robust", "robust", budget)
    entry = manifest["auto"]
    # plan: d = 2, C = 2, test = min(256, max(4, 160)) = 160, B = 3920, n0 = clip(40, 196, 980) = 196
    assert entry["stage_names"] == PLANNED_NAMES
    assert (entry["no_samplers"], entry["test_data_size"], entry["per_chain_budget"]) == (2, 160, 3_920)
    assert entry["warm_up"] == {"evaluations": 196} and entry["chunks"] == 4
    assert manifest["configuration"]["initial_sample_type"] == "lhs"
    assert manifest["configuration"]["use_surrogate_gradients"] is False
    # the plan and the resolved stage list are printed once (rank 0), without conf.debug
    assert result.stdout.count("Auto plan (mode='robust'") == 1, result.describe()
    assert result.stdout.count("Stage 4 (alg0004_DAMH-SMU)") == 1, result.describe()
    planned = [196, 931, 931, 931, 931]
    assert [s["max_evaluations"] for s in manifest["stages"]] == planned

    # held-out set: generated on the collector, saved into the run
    with np.load(os.path.join(output_dir, "sampling_output", "surrogate_test_data.npz")) as loaded:
        assert loaded["parameters"].shape == (160, 2) and loaded["observations"].shape == (160, 2)
    assert os.path.exists(os.path.join(output_dir, "sampling_output", "surrogate_quality_test.csv"))
    # ... and the samplers spent exactly the planned evaluations (acc + rej + audited, S2) per stage
    counters = _counters(result.stdout)
    audited = _audited(result.stdout)
    for rank in range(2):
        assert [sum(counters[(name, rank)][:2]) + audited[(name, rank)] for name in PLANNED_NAMES] == planned, \
            (counters, audited)
    assert all(audited[(name, rank)] > 0 for name in CHUNK_NAMES for rank in range(2)), audited

    chains = _pooled_chunks(output_dir)
    statistics = pooled_chain_statistics(chains)
    rhat = gelman_rubin(chains)
    _report("auto robust MPI", statistics, rhat, linear_gaussian_problem, result.duration)
    assert_matches_closed_form(statistics, linear_gaussian_problem.posterior_mean,
                               linear_gaussian_problem.posterior_cov, "auto robust MPI")
    assert np.all(rhat < RHAT_LIMIT), rhat


def test_auto_fast_default_network_smoke(mpi_workspace, run_mpi, linear_gaussian_problem):
    budget = 3_000
    result, output_dir, manifest = _run(mpi_workspace, run_mpi, "auto_fast", "fast", budget, timeout=300.0)
    entry = manifest["auto"]
    assert entry["stage_names"] == PLANNED_NAMES and entry["mode"] == "fast"
    assert entry["surrogate"]["class"] == "NeuralNetworkUpdater" and entry["surrogate"]["default"] is True
    assert entry["proposal"].startswith("Hamiltonian(") and "dimension_robust" in entry["proposal"]
    assert manifest["configuration"]["use_surrogate_gradients"] is True
    assert all(s["proposal"]["type"] == "Hamiltonian" for s in manifest["stages"][1:])

    # loose posterior check (B8 pattern, short chains): mean within 5 SE with 16 batches, R-hat < 1.2
    chains = _pooled_chunks(output_dir)
    statistics = pooled_chain_statistics(chains, no_batches=16)
    rhat = gelman_rubin(chains)
    _report("auto fast MPI (loose)", statistics, rhat, linear_gaussian_problem, result.duration)
    assert max_deviation_in_se(statistics.mean, linear_gaussian_problem.posterior_mean, statistics.se_mean) < 5.0
    assert np.all(np.abs(np.diag(statistics.cov) / np.diag(linear_gaussian_problem.posterior_cov) - 1.0) < 0.5)
    assert np.all(rhat < 1.2), rhat


def test_auto_without_collector_runs_mh_only(mpi_workspace, run_mpi):
    result, output_dir, manifest = _run(mpi_workspace, run_mpi, "auto_nocoll", "robust", 1_000,
                                        collector=False, n=2)
    entry = manifest["auto"]
    assert entry["stage_names"] == ["alg0000_MH-adaptive"] and entry["chunks"] == 0
    assert entry["per_chain_budget"] == 500 and entry["test_data_size"] == 0 and entry["surrogate"] == "none"
    assert any("use_collector=False" in note for note in entry["notes"])
    counters = _counters(result.stdout)
    assert [sum(counters[("alg0000_MH-adaptive", rank)][:2]) for rank in range(2)] == [500, 500]
    assert not os.path.exists(os.path.join(output_dir, "sampling_output", "surrogate_test_data.npz"))
