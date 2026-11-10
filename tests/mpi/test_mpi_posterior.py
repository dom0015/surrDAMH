#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Posterior correctness **under real MPI**: is the sampled distribution still the exact
posterior when the chain is produced by several sampler ranks talking to a solvers pool and
a collector, and is the MPI transport itself lossless?

Complements the two existing layers:

* ``tests/mpi/test_mpi_basic.py`` / ``test_mpi_surrogate.py`` / ``test_mpi_hangs.py`` check the
  *mechanics* of an ``mpiexec`` run (exit codes, files, deadlocks) but never look at the numbers;
* ``tests/validation/test_gaussian_toy.py`` checks the *statistics* but only in process
  (``run_local``, a single chain, no MPI role at all).

Part A - transport exactness (bit-for-bit, no statistics involved)
    A1  the same MH configuration run with and without the solvers pool must produce
        byte-identical sample files for both chains (the pool ships float64 parameters one way
        and pickled observations back, so any loss shows up immediately);
    A2  a DAMH stage whose surrogate arrives as a **pickled evaluator from the collector** must
        reproduce, byte for byte, the in-process chain of
        ``Problem.run_sampling_local(..., surrogate_evaluator=...)`` fitted on the same snapshots;
    A3  both runs write a ``run_manifest.json`` with ``runner == "mpi"`` and the MPI layout that
        was actually used (checked inside the A1/A2 tests, they are the runs it is about).

Part B - posterior correctness (statistical, tolerance ``TOLERANCE_IN_SE`` = 4 batch-means SE)
    Every case samples the linear-Gaussian toy problem of ``tests/conftest.py``
    (``G(x) = A x``, ``A = [[1, 0.5], [0, 1]]``, ``obs_sd = 0.3``, prior ``N(0, I)``, data
    ``A @ [0.7, -0.4]``), whose posterior is known in closed form, with **two sampler chains**
    started from the deterministic ``lhs`` design (``initial_sample_type="lhs"``, seed 0 -
    chain 0 starts ~5 posterior sd away from the posterior mean, which is what makes the
    per-chain 10 % burn-in and the R-hat check meaningful). The pooled post-burn-in chains are
    compared with the closed form within 4 standard errors, and R-hat must stay below 1.1.

    | case | layout                                                             | what it exercises          |
    |------|--------------------------------------------------------------------|----------------------------|
    | B1   | MH, solvers pool (2 solvers), no collector, 2 samplers (-n 3)       | pool round trip            |
    | B2   | MH -> DAMH-SMU(K=5) -> DAMH(K=5), collector + pool (-n 4), poly     | snapshots, retrain, bcast  |
    | B3   | as B2 with a 3-NN kd-tree surrogate (stays wrong, keeps changing)   | sub-chain under retraining |
    | B4   | MH with a pCN proposal, no collector (-n 3)                         | pCN under MPI              |
    | B5   | adaptive MH -> MH with the reduced covariance (-n 3)                | the adaptive ``Allreduce`` |
    | B6   | local solver + collector, MH -> DAMH-SMU(K=1) (-n 3)                | collector without a pool   |
    | B7   | DAMH(K=20) as the FIRST stage with ``initial_snapshots`` (-n 4)     | start-up handshake         |
    | B8   | MH -> MH with a Hamiltonian proposal on NN surrogate gradients      | surrogate gradients        |
    | B9   | as B2 with a torch MLP trained on the fly by the collector (-n 4)   | NN surrogate in DAMH       |
    | B10  | MH -> MH/Hamiltonian with the WS7 surrogate-refresh opt-in -> the    | gradient surrogate         |
    |      | same stage WITHOUT the opt-in (-n 4)                                | refreshed mid-stage        |
    | B11  | adaptive pCN -> pCN with ``PCN(beta=None)`` (-n 3)                   | pooled beta hand-over      |
    | B12  | as B8 but the Hamiltonian stage is adaptive -> a fixed Hamiltonian   | pooled dual-averaged step  |
    |      | stage with ``Hamiltonian(step_size=None)`` (-n 4)                    | size hand-over             |

    Sizing: every run is kept to tens of seconds (a few 10^4 full-model evaluations per chain);
    the batch-means standard error is estimated inside each chain (batches never straddle a chain
    boundary) and the per-chain errors are combined with length weights, see
    ``tests/helpers_statistics.pooled_chain_statistics``.

Both parts use the driver-script machinery of ``tests/mpi/conftest.py``; run them with
``pytest tests/mpi/test_mpi_posterior.py -m mpi``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re

import numpy as np
import pytest

# tests/ is on sys.path (pytest imports tests/conftest.py from there)
from helpers_statistics import (BURN_IN_FRACTION, NO_BATCHES, TOLERANCE_IN_SE,
                                assert_matches_closed_form, gelman_rubin,
                                max_deviation_in_se, pooled_chain_statistics, read_all_chains)

from .conftest import requires_mpiexec, stage_sample_files, write_driver

pytestmark = [pytest.mark.mpi, requires_mpiexec]

assert TOLERANCE_IN_SE == 4.0  # fixed before the first run; never loosen it for a failing case

PROPOSAL_SD = 0.5  # ~1.6 posterior sd, MH acceptance ~34 % on this problem
RHAT_LIMIT = 1.1  # Gelman-Rubin limit across the sampler chains
NO_SAMPLERS = 2  # every Part B case uses two sampler chains
MIN_SAMPLES_PER_BATCH = 50  # sanity floor on the batch-means block size (see _check_stage_posterior)

STAGE_COUNTER_RE = re.compile(
    r"Stage (\S+) at MPI rank (\d+) finished - acc/rej/prerej samples: (\d+) (\d+) (\d+)")
# printed once per stage by Algorithm_MH.run for a gradient-based proposal (WS7)
GRADIENT_REFRESH_RE = re.compile(
    r"Stage (\S+) at MPI rank (\d+) - gradient surrogate refreshes: (\d+)")
# printed once per carried field per rank by process_SAMPLER after an adaptive stage (2026-09-20)
CARRY_OVER_RE = re.compile(
    r"Stage (\S+) at MPI rank (\d+) carry-over (\w+): (\[[^\]]*\])")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _load_solver_module(mpi_workspace):
    """Import the generated ``solver_module.py`` into this process (same file the drivers use)."""
    path = os.path.join(str(mpi_workspace), "solver_module.py")
    spec = importlib.util.spec_from_file_location("mpi_posterior_solver_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_problem_matches(mpi_workspace, problem) -> None:
    """The drivers' ``solver_module`` must define the same problem as the closed-form fixture."""
    module = _load_solver_module(mpi_workspace)
    assert np.array_equal(module.A, problem.A)
    assert np.array_equal(module.OBSERVED_DATA, problem.observed_data)
    assert np.array_equal(module.OBS_SD, problem.obs_sd)


def _stage_counters(stdout: str) -> dict[tuple[str, int], tuple[int, int, int]]:
    """``{(stage_name, rank): (accepted, rejected, prerejected)}`` parsed from the sampler output."""
    return {(m.group(1), int(m.group(2))): (int(m.group(3)), int(m.group(4)), int(m.group(5)))
            for m in STAGE_COUNTER_RE.finditer(stdout)}


def _summed_counters(stdout: str, stage_name: str) -> tuple[int, int, int]:
    """Accepted/rejected/prerejected of one stage, summed over the sampler ranks."""
    per_rank = [v for (name, _rank), v in _stage_counters(stdout).items() if name == stage_name]
    assert per_rank, f"no counter line for stage {stage_name!r} in the driver output"
    return tuple(int(sum(column)) for column in zip(*per_rank))  # type: ignore[return-value]


def _gradient_surrogate_refreshes(stdout: str, stage_name: str) -> dict[int, int]:
    """``{rank: number of gradient-surrogate refreshes}`` of one MH stage (WS7).

    Parsed from the line ``Algorithm_MH.run`` prints for every stage whose proposal needs
    gradients; 0 means the stage kept the single evaluator it fetched at stage start.
    """
    return {int(m.group(2)): int(m.group(3))
            for m in GRADIENT_REFRESH_RE.finditer(stdout) if m.group(1) == stage_name}


def _carry_over_values(stdout: str, stage_name: str, key: str) -> dict[int, list[float]]:
    """``{rank: carried values}`` of one field of one adaptive stage, parsed from the samplers'
    output (``process_SAMPLER`` prints one line per carried field per rank, flattened)."""
    return {int(m.group(2)): ast.literal_eval(m.group(4))
            for m in CARRY_OVER_RE.finditer(stdout)
            if m.group(1) == stage_name and m.group(3) == key}


def _assert_identical_carry_over(result, stage_name: str, key: str, no_samplers: int = NO_SAMPLERS):
    """Every sampler must report the SAME carried value: that is what pooling the adaptation
    statistics across ranks buys (before 2026-09-20 the ranks averaged per-rank covariances and
    the study measured them disagreeing by 6-17x, `15` §2.4)."""
    per_rank = _carry_over_values(result.stdout, stage_name, key)
    assert sorted(per_rank) == list(range(no_samplers)), (
        f"expected a '{key}' carry-over line from every sampler of stage {stage_name}, got "
        f"{sorted(per_rank)}\n" + result.describe())
    values = [np.asarray(v, dtype=float) for v in per_rank.values()]
    reference = values[0]
    assert reference.size > 0 and np.all(np.isfinite(reference)), per_rank
    for rank, value in zip(sorted(per_rank), values):
        np.testing.assert_array_equal(
            value, reference,
            err_msg=f"rank {rank} carried a different '{key}' than rank 0: {per_rank}")
    print(f"[carry-over] {stage_name} {key} (identical on {no_samplers} ranks): {reference}",
          flush=True)
    return reference


def _run_case(mpi_workspace, run_mpi, name: str, body: str, n: int, timeout: float = 240.0):
    """Write + run one driver; assert it finished cleanly. Returns ``(result, output_dir)``."""
    driver = write_driver(mpi_workspace, name, body)
    result = run_mpi(driver, n=n, timeout=timeout)
    assert not result.timed_out, result.describe()
    assert not result.orphans_killed, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    return result, mpi_workspace / f"out_{name}"


def _assert_manifest(output_dir, expected_mpi: dict) -> None:
    """A3: the run wrote an MPI run manifest describing the layout it really used."""
    path = os.path.join(str(output_dir), "sampling_output", "run_manifest.json")
    assert os.path.exists(path), f"missing run manifest {path}"
    with open(path) as f:
        manifest = json.load(f)
    assert manifest["runner"] == "mpi", manifest["runner"]
    assert {key: manifest["mpi"][key] for key in expected_mpi} == expected_mpi, manifest["mpi"]


def _read_manifest(output_dir) -> dict:
    """The run manifest of a finished run."""
    path = os.path.join(str(output_dir), "sampling_output", "run_manifest.json")
    assert os.path.exists(path), f"missing run manifest {path}"
    with open(path) as f:
        return json.load(f)


def _collector_retrainings(output_dir, min_snapshots_to_update: int) -> int:
    """
    How many times the collector called ``Updater.train()``, reconstructed from
    ``sampling_output/surrogate_quality.csv``.

    The collector writes no "trained" line, so this has to be derived from the only file it
    does write about the surrogate. ``surrogate_quality.csv`` has one row per collector loop
    iteration that received new snapshots **while a monitoring evaluator already existed**
    (``docs/outputs.md``: columns ``snapshots_total, batch_size, rmse, max_abs_error``), with
    ``snapshots_total`` recorded *before* the batch is added. The collector's retraining rule
    (``process_COLLECTOR.run_COLLECTOR``) is

        train whenever ``no_snapshots_total - no_snapshots_used >= min_snapshots_to_update``

    and ``no_snapshots_used`` is set to ``no_snapshots_total`` at every training, so replaying
    the rows reproduces the training times exactly:

    * the very first training produced the monitoring evaluator and happened *before* the first
      row, with ``no_snapshots_used == rows[0].snapshots_total`` (nothing was added in between);
    * every later training coincides with a row, because the totals only change when a batch
      arrives, and a batch that arrives always produces a row once monitoring is on.
    """
    path = os.path.join(str(output_dir), "sampling_output", "surrogate_quality.csv")
    assert os.path.exists(path), f"missing {path} (the collector never trained a surrogate)"
    with open(path) as f:
        header = f.readline().strip().split(",")
    assert header[:2] == ["snapshots_total", "batch_size"], header
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    assert len(rows) > 0, f"{path} has a header but no rows"
    snapshots_used = int(rows[0, 0])
    retrainings = 1  # the one that created the first (monitoring) evaluator
    for snapshots_before, batch_size in rows[:, :2]:
        snapshots_after = int(snapshots_before) + int(batch_size)
        if snapshots_after - snapshots_used >= min_snapshots_to_update:
            snapshots_used = snapshots_after
            retrainings += 1
    return retrainings


def _check_stage_posterior(label: str, output_dir, stage_name: str, result, problem,
                           no_samplers: int = NO_SAMPLERS, no_batches: int = NO_BATCHES):
    """
    The full Part B check of one stage: file count, pooled posterior vs closed form, R-hat.

    Returns the pooled ``ChainStatistics`` (the caller may assert more on the counters).
    """
    files = stage_sample_files(output_dir, stage_name)
    assert [f.name for f in files] == [f"rank{i:04d}.csv" for i in range(no_samplers)], (
        f"{label}: expected one sample file per sampler in stage {stage_name}\n" + result.describe())

    chains = read_all_chains(output_dir, stage_name, problem.no_parameters, no_samplers)
    for index, chain in enumerate(chains):
        # keeps the batch-means error estimate meaningful if anyone shrinks the evaluation counts
        assert len(chain) * (1 - BURN_IN_FRACTION) >= MIN_SAMPLES_PER_BATCH * no_batches, (
            f"{label}: chain {index} of stage {stage_name} has only {len(chain)} samples; "
            f"at least {MIN_SAMPLES_PER_BATCH} per batch-means block are needed")
    accepted, rejected, prerejected = _summed_counters(result.stdout, stage_name)
    statistics = pooled_chain_statistics(chains, counter_accepted=accepted, counter_rejected=rejected,
                                         counter_prerejected=prerejected, no_batches=no_batches)
    rhat = gelman_rubin(chains)

    print(f"\n[{label}] stage {stage_name}: chains {[len(c) for c in chains]} samples "
          f"(pooled after {BURN_IN_FRACTION:.0%} burn-in: {statistics.no_samples}), "
          f"acc/rej/prerej {accepted}/{rejected}/{prerejected}\n"
          f"    mean {np.array2string(statistics.mean, precision=4)} "
          f"vs closed form {np.array2string(problem.posterior_mean, precision=4)} "
          f"(max |dev|/SE {max_deviation_in_se(statistics.mean, problem.posterior_mean, statistics.se_mean):.2f})\n"
          f"    cov  {np.array2string(statistics.cov.ravel(), precision=4)} "
          f"vs closed form {np.array2string(problem.posterior_cov.ravel(), precision=4)} "
          f"(max |dev|/SE {max_deviation_in_se(statistics.cov, problem.posterior_cov, statistics.se_cov):.2f})\n"
          f"    R-hat {np.array2string(rhat, precision=4)}, wall time {result.duration:.1f} s",
          flush=True)

    assert_matches_closed_form(statistics, problem.posterior_mean, problem.posterior_cov, label)
    assert np.all(rhat < RHAT_LIMIT), (
        f"{label}: Gelman-Rubin R-hat {rhat} is not below {RHAT_LIMIT}\n" + result.describe())
    return statistics


# ===========================================================================
# Part A - MPI transport exactness
# ===========================================================================

A1_BODY = """
    USE_POOL = sys.argv[2] == "pool"
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=False, use_solvers_pool=USE_POOL, no_solvers=1,
                                  initial_sample_type="lhs")
    assert conf.no_samplers == 2, conf.no_samplers
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=300)]
    solver = solver_spec() if USE_POOL else solver_module.LinearSolver()
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(), solver=solver)
    sam = problem.run_sampling(conf, stages)
    print("DRIVER FINISHED", flush=True)
    """


def test_a1_solvers_pool_chain_is_byte_identical_to_the_in_process_solver(mpi_workspace, run_mpi):
    """
    A1: the solvers pool must not change a single bit of the chain.

    The same MH configuration (two samplers, deterministic ``lhs`` initial samples, 300 full-model
    evaluations) is run twice: once with ``use_solvers_pool=True, no_solvers=1`` on ``-n 3``
    (2 samplers + pool) and once with ``use_solvers_pool=False`` on ``-n 2`` (2 samplers, solver
    inside each sampler). ``no_samplers`` is 2 in both, so the ``lhs`` design
    (``lhs_normal(n=no_samplers, seed=0)``) and the per-rank seeds
    (``seed0 = 10*(no_stages*rank + i)``) are identical and the two runs must produce the very
    same chains - in the pool case after a float64 parameter send and a pickled observation
    receive per evaluation.

    Compared as raw bytes of both ``samples/alg0000_MH/rank%04d.csv`` files, i.e. including the
    repr of every float; any transport rounding or reordering breaks it.
    """
    driver = write_driver(mpi_workspace, "a1", A1_BODY)

    dir_pool = str(mpi_workspace / "out_a1_pool")
    dir_local = str(mpi_workspace / "out_a1_local")
    result_pool = run_mpi(driver, n=3, timeout=180, args=[dir_pool, "pool"])
    assert not result_pool.timed_out and result_pool.returncode == 0, result_pool.describe()
    result_local = run_mpi(driver, n=2, timeout=180, args=[dir_local, "local"])
    assert not result_local.timed_out and result_local.returncode == 0, result_local.describe()

    files_pool = stage_sample_files(dir_pool, "alg0000_MH")
    files_local = stage_sample_files(dir_local, "alg0000_MH")
    assert [f.name for f in files_pool] == ["rank0000.csv", "rank0001.csv"], result_pool.describe()
    assert [f.name for f in files_local] == ["rank0000.csv", "rank0001.csv"], result_local.describe()

    for file_pool, file_local in zip(files_pool, files_local):
        bytes_pool = file_pool.read_bytes()
        bytes_local = file_local.read_bytes()
        if bytes_pool != bytes_local:  # give a useful message instead of a byte dump
            samples_pool = np.loadtxt(file_pool, delimiter=",", skiprows=1, ndmin=2)
            samples_local = np.loadtxt(file_local, delimiter=",", skiprows=1, ndmin=2)
            shape_info = f"shapes {samples_pool.shape} vs {samples_local.shape}"
            if samples_pool.shape == samples_local.shape:
                shape_info += f", max abs difference {np.max(np.abs(samples_pool - samples_local)):.3e}"
            raise AssertionError(
                f"{file_pool.name}: the solvers-pool chain differs from the in-process chain "
                f"({shape_info})\n" + result_pool.describe())

    # A3 for this run:
    _assert_manifest(dir_pool, {"size_world": 3, "no_samplers": 2, "rank_collector": None,
                                "rank_solvers_pool": 2, "no_solvers": 1})
    _assert_manifest(dir_local, {"size_world": 2, "no_samplers": 2, "rank_collector": None,
                                 "rank_solvers_pool": None})


A2_SNAPSHOTS = 40  # preloaded snapshots the collector's surrogate is fitted on
A2_SURROGATE_ERROR_FACTOR = 1.3  # the snapshots are generated with 1.3*A -> a wrong but fixed surrogate

A2_BODY = f"""
    N = {A2_SNAPSHOTS}
    rng = np.random.RandomState(7)
    parameters = rng.normal(size=(N, 2))
    observations = parameters @ ({A2_SURROGATE_ERROR_FACTOR} * solver_module.A).T
    weights = np.ones((N, 1))
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=True, no_solvers=1,
                                  min_snapshots_initial=N, min_snapshots_to_update=10**9,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=solver_module.FixedSample([0.3, -0.7]))
    assert conf.no_samplers == 1, conf.no_samplers
    updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2,
                                                           max_degree=1)
    stages = [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=300,
                    subchain_length=5, surrogate_model_updates=False)]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater,
                               surrogate_initial_training_data=[parameters, observations, weights])
    print("DRIVER FINISHED", flush=True)
    """


def test_a2_collector_provided_evaluator_chain_is_byte_identical_to_run_local(mpi_workspace, run_mpi):
    """
    A2: the evaluator that travels collector -> sampler (pickled) must give exactly the local chain.

    MPI side (``-n 3`` = 1 sampler + pool + collector): a single DAMH stage with
    ``surrogate_model_updates=False`` and ``subchain_length=5``; the collector is handed
    ``initial_snapshots`` of N=40 rows whose observations come from a deliberately **wrong**
    ``1.3*A`` map, fits a degree-1 ``PolynomialSklearnUpdater`` on exactly those rows
    (``min_snapshots_initial=N``, ``min_snapshots_to_update=10**9`` so it never refits) and sends
    the pickled evaluator to the sampler. The wrong surrogate is on purpose: it produces real
    outer rejections, so the DAMH accept/reject sequence actually depends on the transported
    surrogate values.

    In-process side: the same updater fitted on the same rows, handed to
    ``Problem.run_sampling_local(..., surrogate_evaluator=updater.get_evaluator())`` with the
    same configuration (collector and pool off), the same fixed initial sample and the same
    stage - ``run_sampling_local`` reproduces chain 0 of an MPI run by construction (same seed
    formula).

    Compared as raw bytes of ``samples/alg0000_DAMH/rank0000.csv``. Before comparing, the
    updater is fitted twice in process and the two evaluators are checked to agree exactly, so a
    failure cannot be blamed on sklearn nondeterminism without evidence.
    """
    result, output_dir = _run_case(mpi_workspace, run_mpi, "a2", A2_BODY, n=3, timeout=180)

    from surrDAMH.configuration import Configuration
    from surrDAMH.core import Problem
    from surrDAMH.proposals import RandomWalk
    from surrDAMH.stages import Stage
    from surrDAMH.surrogates import PolynomialSklearnUpdater

    solver_module = _load_solver_module(mpi_workspace)
    rng = np.random.RandomState(7)
    parameters = rng.normal(size=(A2_SNAPSHOTS, 2))
    observations = parameters @ (A2_SURROGATE_ERROR_FACTOR * solver_module.A).T
    weights = np.ones((A2_SNAPSHOTS, 1))

    def fit_evaluator():
        updater = PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1)
        updater.add_data(parameters, observations, weights)
        updater.train()
        return updater.get_evaluator()

    # sklearn determinism check first (so a byte difference below is attributable):
    probe = rng.normal(size=(16, 2))
    assert np.array_equal(fit_evaluator()(probe), fit_evaluator()(probe)), (
        "the degree-1 polynomial fit is not deterministic in process; the byte comparison below "
        "would be meaningless")

    local_dir = mpi_workspace / "out_a2_local"
    conf = Configuration(output_dir=str(local_dir), no_parameters=2, no_observations=2,
                         use_collector=False, use_solvers_pool=False,
                         min_snapshots_initial=A2_SNAPSHOTS, min_snapshots_to_update=10**9,
                         initial_sample_type="user_specified",
                         initial_samples_distribution=solver_module.FixedSample([0.3, -0.7]))
    stage = Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=300,
                  subchain_length=5, surrogate_model_updates=False)
    local_problem = Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                            solver=solver_module.LinearSolver())
    local_run = local_problem.run_sampling_local(conf, [stage], surrogate_evaluator=fit_evaluator())
    assert local_run.stage_results[0].counter_rejected > 0, (
        "the 1.3*A surrogate should cause outer rejections; otherwise this test does not depend "
        "on the transported surrogate values")

    mpi_file = stage_sample_files(output_dir, "alg0000_DAMH")[0]
    local_file = local_dir / "sampling_output" / "samples" / "alg0000_DAMH" / "rank0000.csv"
    if mpi_file.read_bytes() != local_file.read_bytes():
        mpi_samples = np.loadtxt(mpi_file, delimiter=",", skiprows=1, ndmin=2)
        local_samples = np.loadtxt(local_file, delimiter=",", skiprows=1, ndmin=2)
        shape_info = f"shapes {mpi_samples.shape} vs {local_samples.shape}"
        if mpi_samples.shape == local_samples.shape:
            shape_info += f", max abs difference {np.max(np.abs(mpi_samples - local_samples)):.3e}"
        raise AssertionError(
            "the DAMH chain driven by the collector-provided evaluator differs from the "
            f"in-process run_sampling_local chain ({shape_info})\n" + result.describe())

    # A3 for this run:
    _assert_manifest(output_dir, {"size_world": 3, "no_samplers": 1, "rank_collector": 2,
                                  "rank_solvers_pool": 1, "no_solvers": 1})


# ===========================================================================
# Part B - posterior correctness under MPI
# ===========================================================================

B1_EVALUATIONS = 25_000

B1_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=False, use_solvers_pool=True, no_solvers=2,
                                  initial_sample_type="lhs")
    assert conf.no_samplers == 2, conf.no_samplers
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B1_EVALUATIONS})]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages)
    print("DRIVER FINISHED", flush=True)
    """


def test_b1_mh_with_solvers_pool(mpi_workspace, run_mpi, linear_gaussian_problem):
    """B1: plain MH, two samplers served by a pool of two solvers, no collector (``-n 3``)."""
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b1", B1_BODY, n=3, timeout=240)
    statistics = _check_stage_posterior("B1 (MH, pool)", output_dir, "alg0000_MH", result,
                                        linear_gaussian_problem)
    acceptance_rate = statistics.counter_accepted / (statistics.counter_accepted + statistics.counter_rejected)
    assert 0.1 < acceptance_rate < 0.7, f"unreasonable MH acceptance rate {acceptance_rate:.3f}"


B2_BOOTSTRAP_EVALUATIONS = 2_000
B2_DAMH_EVALUATIONS = 12_000
B2_SUBCHAIN = 5

B2_BODY_TEMPLATE = """
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=True, no_solvers=2,
                                  initial_sample_type="lhs",
                                  min_snapshots_initial=50, min_snapshots_to_update=500)
    assert conf.no_samplers == 2, conf.no_samplers
    updater = {updater}
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale={sd}), max_evaluations={bootstrap}),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale={sd}), max_evaluations={damh},
                    subchain_length={subchain}, surrogate_model_updates=True),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale={sd}), max_evaluations={damh},
                    subchain_length={subchain}, surrogate_model_updates=False)]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def test_b2_mh_then_damh_smu_then_damh_with_polynomial_surrogate(mpi_workspace, run_mpi,
                                                                 linear_gaussian_problem):
    """
    B2: ``MH -> DAMH-SMU(K=5) -> DAMH(K=5)`` with a collector and a solvers pool (``-n 4``).

    The two DAMH stages are checked separately, each against the closed form: the first one while
    the collector keeps retraining the degree-1 polynomial and broadcasting new evaluators
    (snapshot transport + retraining + evaluator hand-over under real concurrency), the second one
    with the evaluator frozen (``surrogate_model_updates=False``).
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    body = B2_BODY_TEMPLATE.format(
        updater="surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1)",
        sd=PROPOSAL_SD, bootstrap=B2_BOOTSTRAP_EVALUATIONS, damh=B2_DAMH_EVALUATIONS,
        subchain=B2_SUBCHAIN)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b2", body, n=4, timeout=300)

    _check_stage_posterior("B2 (DAMH-SMU, poly, K=5)", output_dir, "alg0001_DAMH-SMU", result,
                           linear_gaussian_problem)
    _check_stage_posterior("B2 (DAMH, frozen poly, K=5)", output_dir, "alg0002_DAMH", result,
                           linear_gaussian_problem)


def test_b3_mh_then_damh_smu_then_damh_with_kdtree_surrogate(mpi_workspace, run_mpi,
                                                             linear_gaussian_problem):
    """
    B3: as B2 but with a 3-nearest-neighbour ``KDTreeUpdater``.

    A piecewise-constant interpolant never becomes the linear model, so the surrogate stays wrong
    while it keeps changing - the variant that stresses a sub-chain running across an evaluator
    refresh under MPI timing. Asserted in addition: the DAMH-SMU stage had outer rejections, i.e.
    the correction term was really exercised (with a surrogate that is exact, as the degree-1
    polynomial of B2 becomes, the outer ratio collapses to 1 and nothing is tested there).
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    body = B2_BODY_TEMPLATE.format(
        updater="surrDAMH.surrogates.KDTreeUpdater(no_parameters=2, no_observations=2, no_nearest_neighbors=3)",
        sd=PROPOSAL_SD, bootstrap=B2_BOOTSTRAP_EVALUATIONS, damh=B2_DAMH_EVALUATIONS,
        subchain=B2_SUBCHAIN)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b3", body, n=4, timeout=300)

    statistics = _check_stage_posterior("B3 (DAMH-SMU, kd-tree, K=5)", output_dir,
                                        "alg0001_DAMH-SMU", result, linear_gaussian_problem)
    assert statistics.counter_rejected > 0, (
        "the kd-tree surrogate should be wrong enough to cause outer rejections; otherwise this "
        "test does not stress the correction term")
    _check_stage_posterior("B3 (DAMH, frozen kd-tree, K=5)", output_dir, "alg0002_DAMH", result,
                           linear_gaussian_problem)


B4_EVALUATIONS = 30_000

B4_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=False, use_solvers_pool=True, no_solvers=2,
                                  initial_sample_type="lhs")
    assert conf.no_samplers == 2, conf.no_samplers
    stages = [Stage(algorithm="MH", proposal=PCN(beta=0.5),
                    max_evaluations={B4_EVALUATIONS})]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages)
    print("DRIVER FINISHED", flush=True)
    """


def test_b4_mh_with_pcn_proposal(mpi_workspace, run_mpi, linear_gaussian_problem):
    """
    B4: MH with a pCN proposal (``PCN(beta=0.5)``) on the ``PriorIndependentComponents`` prior,
    no collector, two samplers (``-n 3``).

    pCN drops the prior ratio from the acceptance probability (correct only for a Gaussian
    internal prior) and proposes with prior-scale steps, so it is a different acceptance path
    than RWMH; the sampled posterior must still be the closed-form one.
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b4", B4_BODY, n=3, timeout=240)
    _check_stage_posterior("B4 (MH, pCN)", output_dir, "alg0000_MH", result, linear_gaussian_problem)


B5_ADAPTIVE_EVALUATIONS = 4_000
B5_EVALUATIONS = 25_000

B5_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=False, use_solvers_pool=True, no_solvers=2,
                                  debug=True,  # the carry-over lines this test parses are printed only with debug (c518457)
                                  initial_sample_type="lhs")
    assert conf.no_samplers == 2, conf.no_samplers
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale={PROPOSAL_SD}, adaptive=True),
                    max_evaluations={B5_ADAPTIVE_EVALUATIONS}),
              Stage(algorithm="MH", proposal=RandomWalk(adaptive=False),  # frozen consumer (2026-09-21)
                    max_evaluations={B5_EVALUATIONS})]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages)
    print("DRIVER FINISHED", flush=True)
    """


def test_b5_adaptive_mh_then_mh_with_the_pooled_covariance(mpi_workspace, run_mpi,
                                                           linear_gaussian_problem):
    """
    B5 (rewritten 2026-09-20): an adaptive MH stage (``RandomWalk(adaptive=True)``, 4000
    evaluations) followed by a plain MH stage with a scale-less ``RandomWalk()``, i.e. with the covariance
    ``process_SAMPLER`` derives from the cross-sampler ``Allgather`` of the two chains'
    adaptation statistics at the end of the adaptive stage.

    Two things are checked, both observable only under real MPI:

    * every sampler prints the **same** carried covariance -- the point of pooling the
      sufficient statistics instead of averaging per-rank covariances (the ranks used to
      disagree by 6-17x, `15` §2.4), and proof that the fixed-length state buffer cannot
      produce the ``Message truncated`` shape mismatch of finding 2.5;
    * the frozen stage's posterior matches the closed form within the suite's 4 SE, with an
      acceptance rate in the band a d=2 chain targeting 0.234 should reach.

    Adaptive Block/Mixture proposals (C2, 2026-10-12) are covered by ``test_mpi_mixture.py``.
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b5", B5_BODY, n=3, timeout=300)
    assert [f.name for f in stage_sample_files(output_dir, "alg0000_MH-adaptive")] == \
        [f"rank{i:04d}.csv" for i in range(NO_SAMPLERS)], result.describe()
    assert "Message truncated" not in result.stdout + result.stderr, result.describe()

    covariance = _assert_identical_carry_over(result, "alg0000_MH-adaptive", "scale")
    assert covariance.size == linear_gaussian_problem.no_parameters ** 2, covariance

    # the adaptation trace of the adaptive stage is written and readable
    from surrDAMH.modules.run_data import read_run
    run = read_run(str(output_dir), load_raw_data=False)
    adaptive_stats = run.adaptive_stats[run.stage_index("alg0000_MH-adaptive")]
    assert adaptive_stats is not None and not adaptive_stats.empty, "no adaptive_stats written"
    assert set(adaptive_stats["rank_world"]) == set(range(NO_SAMPLERS)), adaptive_stats

    statistics = _check_stage_posterior("B5 (MH with the pooled adapted covariance)", output_dir,
                                        "alg0001_MH", result, linear_gaussian_problem)
    acceptance = statistics.counter_accepted / (statistics.counter_accepted
                                                + statistics.counter_rejected)
    # d=2: 0.234 is the high-dimensional limit, the optimum here is nearer 0.35, so the band
    # around the target is deliberately wide (same reasoning as the run_local unit test)
    assert 0.15 < acceptance < 0.45, (
        f"frozen-stage acceptance {acceptance:.3f} after the hand-over\n" + result.describe())


B6_BOOTSTRAP_EVALUATIONS = 2_000
B6_DAMH_EVALUATIONS = 15_000

B6_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=False,
                                  initial_sample_type="lhs",
                                  min_snapshots_initial=50, min_snapshots_to_update=500)
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2,
                                                           max_degree=1)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B6_BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B6_DAMH_EVALUATIONS}, subchain_length=1,
                    surrogate_model_updates=True)]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def test_b6_local_solver_with_collector_damh_smu(mpi_workspace, run_mpi, linear_gaussian_problem):
    """
    B6: no solvers pool (each sampler evaluates the model in process) but a collector
    (``-n 3`` = 2 samplers + collector), ``MH -> DAMH-SMU(K=1)`` with a degree-1 polynomial.

    This is the layout in which the sampler's snapshot/evaluator communicators are the only MPI
    traffic besides the barriers, so it isolates the collector path from the pool path.
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b6", B6_BODY, n=3, timeout=300)
    _check_stage_posterior("B6 (DAMH-SMU, K=1, local solver)", output_dir, "alg0001_DAMH-SMU",
                           result, linear_gaussian_problem)


B7_SNAPSHOTS = 60
B7_EVALUATIONS = 10_000  # see the note on chain length below B8_EVALUATIONS

B7_BODY = f"""
    N = {B7_SNAPSHOTS}
    rng = np.random.RandomState(3)
    parameters = rng.normal(size=(N, 2))
    observations = parameters @ solver_module.A.T
    weights = np.ones((N, 1))
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=True, no_solvers=2,
                                  initial_sample_type="lhs",
                                  min_snapshots_initial=N, min_snapshots_to_update=500)
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.KDTreeUpdater(no_parameters=2, no_observations=2,
                                                no_nearest_neighbors=3)
    stages = [Stage(algorithm="DAMH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B7_EVALUATIONS}, subchain_length=20,
                    surrogate_model_updates=True)]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater,
                               surrogate_initial_training_data=[parameters, observations, weights])
    print("DRIVER FINISHED", flush=True)
    """


def test_b7_damh_as_the_first_stage_with_initial_snapshots(mpi_workspace, run_mpi,
                                                           linear_gaussian_problem):
    """
    B7: DAMH-SMU (K=20) as the **first** stage (``-n 4``).

    ``initial_snapshots`` holds N=60 rows and ``min_snapshots_initial=N``, so the collector's
    start-up handshake reports "surrogate available" and the samplers start straight into a DAMH
    stage instead of raising (or, before WS8, deadlocking). A 3-NN kd-tree updater keeps the
    surrogate wrong, so the long K=20 sub-chains really depend on the outer correction.
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b7", B7_BODY, n=4, timeout=300)
    assert "surrogate" not in result.stderr.lower() or result.returncode == 0, result.describe()
    _check_stage_posterior("B7 (DAMH-SMU first stage, K=20)", output_dir, "alg0000_DAMH-SMU",
                           result, linear_gaussian_problem)


B8_BOOTSTRAP_EVALUATIONS = 2_000
B8_EVALUATIONS = 8_000

# Chain length vs Monte-Carlo noise (measured 2026-09-16): B7 and B8 are the shortest chains here
# and show the largest deviations - across four runs of this file the worst posterior-mean
# deviation was 2.66 SE (B7) and 2.42 SE (B8), the other cases stayed below 2 SE. Re-running the
# two of them once with 40 000 / 32 000 evaluations per chain brought those down to 0.86 SE and
# 1.32 SE, i.e. the deviation does not grow with the sample size: it is Monte-Carlo noise, not a
# bias. The short versions are what is committed, to keep the file within its runtime budget.

B8_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=True, no_solvers=2,
                                  initial_sample_type="lhs", use_surrogate_gradients=True,
                                  min_snapshots_initial=200, min_snapshots_to_update=2000)
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.NeuralNetworkUpdater(
        no_parameters=2, no_observations=2, hidden_layer_sizes=(16, 16),
        iterations_batch=50, seed=0)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B8_BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1, num_steps=5, mass=1.0),
                    max_evaluations={B8_EVALUATIONS})]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def test_b8_hamiltonian_proposal_on_neural_network_surrogate_gradients(mpi_workspace, run_mpi,
                                                                       linear_gaussian_problem):
    """
    B8 (best effort): an MH stage whose proposal is Hamiltonian, driven by the gradients of a
    small torch MLP surrogate trained by the collector (``use_surrogate_gradients=True``).

    The acceptance test still uses the full model (``process_SAMPLER`` only hands the evaluator to
    the proposal), so the chain is exact for *any* gradient field - this checks correctness of the
    gradient/evaluator transport, not the efficiency of the surrogate.
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b8", B8_BODY, n=4, timeout=300)
    _check_stage_posterior("B8 (MH, Hamiltonian proposal, NN surrogate gradients)", output_dir,
                           "alg0001_MH", result, linear_gaussian_problem)


B9_BOOTSTRAP_EVALUATIONS = 2_000
B9_DAMH_EVALUATIONS = 12_000
B9_SUBCHAIN = 5
B9_MIN_SNAPSHOTS_INITIAL = 200  # first surrogate is trained inside the MH bootstrap stage
B9_MIN_SNAPSHOTS_TO_UPDATE = 500  # ~24 retrainings inside the DAMH-SMU stage at 12 000 evaluations
B9_MIN_RETRAININGS = 3
B9_HIDDEN_LAYER_SIZES = (16, 16)
B9_ITERATIONS_BATCH = 50  # optimizer steps per train() call

B9_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=True, no_solvers=2,
                                  initial_sample_type="lhs",
                                  min_snapshots_initial={B9_MIN_SNAPSHOTS_INITIAL},
                                  min_snapshots_to_update={B9_MIN_SNAPSHOTS_TO_UPDATE})
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.NeuralNetworkUpdater(
        no_parameters=2, no_observations=2, hidden_layer_sizes={B9_HIDDEN_LAYER_SIZES},
        iterations_batch={B9_ITERATIONS_BATCH}, seed=0)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B9_BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B9_DAMH_EVALUATIONS}, subchain_length={B9_SUBCHAIN},
                    surrogate_model_updates=True),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B9_DAMH_EVALUATIONS}, subchain_length={B9_SUBCHAIN},
                    surrogate_model_updates=False)]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def test_b9_mh_then_damh_smu_then_damh_with_nn_surrogate(mpi_workspace, run_mpi,
                                                         linear_gaussian_problem):
    """
    B9: ``MH -> DAMH-SMU(K=5) -> DAMH(K=5, frozen)`` with a **neural-network surrogate trained
    on the fly by the collector** (``-n 4`` = 2 samplers + solvers pool + collector).

    The gap B2/B3/B8 leave open: B2/B3 run the full three-stage chain but with a polynomial /
    kd-tree surrogate, and B8 trains a torch MLP on the fly but only uses it for Hamiltonian
    *proposal gradients* inside an MH stage, where the chain is exact for any gradient field.
    Here the MLP is what delayed acceptance pre-screens with, so its values enter the sub-chain
    kernel and the outer correction ratio, and both DAMH stages are checked against the closed
    form (the first while the collector keeps retraining and broadcasting new evaluators, the
    second with the evaluator frozen).

    Surrogate: ``NeuralNetworkUpdater`` with two hidden layers of 16 units, AdamW,
    50 optimizer steps per ``train()``, ``weighting="uniform"`` and
    ``output_normalization="likelihood"`` (both defaults, and both asserted below - they decide
    what the network is fitted to). ``min_snapshots_initial=200`` puts the first training inside
    the MH bootstrap stage and ``min_snapshots_to_update=500`` retrains it every 500 snapshots,
    i.e. ~24 times during the DAMH-SMU stage; at least ``B9_MIN_RETRAININGS`` retrainings are
    asserted from ``surrogate_quality.csv``. That budget leaves the network clearly short of the
    exact linear model (unlike B2's degree-1 polynomial, which becomes exact), so the outer
    acceptance step has real work to do - asserted as outer rejections > 0 in the SMU stage.
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b9", B9_BODY, n=4, timeout=300)

    statistics = _check_stage_posterior("B9 (DAMH-SMU, NN, K=5)", output_dir, "alg0001_DAMH-SMU",
                                        result, linear_gaussian_problem)
    assert statistics.counter_rejected > 0, (
        "the on-the-fly neural-network surrogate should be wrong enough to cause outer "
        "rejections; otherwise this test does not stress the correction term\n" + result.describe())
    _check_stage_posterior("B9 (DAMH, frozen NN, K=5)", output_dir, "alg0002_DAMH", result,
                           linear_gaussian_problem)

    retrainings = _collector_retrainings(output_dir, B9_MIN_SNAPSHOTS_TO_UPDATE)
    print(f"[B9] collector retrainings (reconstructed): {retrainings}", flush=True)
    assert retrainings >= B9_MIN_RETRAININGS, (
        f"the surrogate was trained only {retrainings} time(s); the point of this case is a "
        f"network that is rebuilt repeatedly while the sub-chains run\n" + result.describe())

    manifest = _read_manifest(output_dir)
    assert manifest["surrogate"]["updater_class"] == "NeuralNetworkUpdater", \
        manifest["surrogate"]
    assert manifest["surrogate"]["weighting"] == "uniform", manifest["surrogate"]
    assert manifest["surrogate"]["output_normalization"] == "likelihood", manifest["surrogate"]
    assert manifest["surrogate"]["output_normalization_provenance"] == "likelihood", \
        manifest["surrogate"]
    # the collector's single start-up line stating what the network is fitted to (WS6):
    assert re.search(r"Collector - NeuralNetworkUpdater: weighting='uniform'.*"
                     r"output_normalization='likelihood'", result.stdout), result.describe()


B10_BOOTSTRAP_EVALUATIONS = 2_000
B10_EVALUATIONS = 8_000
B10_MIN_SNAPSHOTS_INITIAL = 200  # first surrogate is trained inside the MH bootstrap stage
B10_MIN_SNAPSHOTS_TO_UPDATE = 500  # budget-limited retraining, as in B9
B10_MIN_RETRAININGS = 3
B10_MIN_REFRESHES = 2  # gradient-surrogate refreshes asserted in the opted-in stage, per rank

B10_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=True, no_solvers=2,
                                  initial_sample_type="lhs", use_surrogate_gradients=True,
                                  min_snapshots_initial={B10_MIN_SNAPSHOTS_INITIAL},
                                  min_snapshots_to_update={B10_MIN_SNAPSHOTS_TO_UPDATE})
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.NeuralNetworkUpdater(
        no_parameters=2, no_observations=2, hidden_layer_sizes=(16, 16),
        iterations_batch=50, seed=0)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B10_BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1, num_steps=5, mass=1.0),
                    max_evaluations={B10_EVALUATIONS},
                    surrogate_model_updates=True),
              Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1, num_steps=5, mass=1.0),
                    max_evaluations={B10_EVALUATIONS})]
    assert stages[1].surrogate_model_updates is True, "the WS7 opt-in was refused"
    assert stages[2].surrogate_model_updates is False, "the default MH stage must not opt in"
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def test_b10_mh_hamiltonian_stage_refreshing_the_gradient_surrogate(mpi_workspace, run_mpi,
                                                                    linear_gaussian_problem):
    """
    B10 (WS7): B8 plus the new opt-in, and its own control stage.

    One run (``-n 4`` = 2 samplers + solvers pool + collector) with three stages: an MH
    bootstrap, then **two identical MH stages with a Hamiltonian proposal** on the gradients of
    a torch MLP the collector keeps retraining (``min_snapshots_to_update=500``, deliberately
    budget-limited as in B9, so the network stays clearly short of the exact linear model and
    new evaluators keep arriving throughout both stages). The two stages differ in exactly one
    field:

    * ``alg0001_MH`` sets ``surrogate_model_updates=True`` - it must pick up the newer
      evaluators *during* the stage (asserted: at least ``B10_MIN_REFRESHES`` refreshes on every
      sampler rank, reported by ``Algorithm_MH.run``);
    * ``alg0002_MH`` leaves the field alone - it must keep the single evaluator it fetched at
      stage start (asserted: exactly 0 refreshes on every rank), i.e. the pre-WS7 behaviour of
      every MH stage, proving the default path is untouched.

    Both stages are checked against the closed-form posterior within the suite's 4 SE: the MH
    accept/reject test uses the exact model only, so swapping the proposal's gradient field
    mid-stage cannot bias the chain - it is a valid MH kernel for any gradient field.
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b10", B10_BODY, n=4, timeout=480)

    refreshed = _gradient_surrogate_refreshes(result.stdout, "alg0001_MH")
    frozen = _gradient_surrogate_refreshes(result.stdout, "alg0002_MH")
    print(f"[B10] gradient surrogate refreshes per rank: opted-in {refreshed}, default {frozen}",
          flush=True)
    assert len(refreshed) == NO_SAMPLERS and len(frozen) == NO_SAMPLERS, (
        "both Hamiltonian stages must report their refresh count on every sampler rank\n"
        + result.describe())
    assert all(count >= B10_MIN_REFRESHES for count in refreshed.values()), (
        f"the opted-in stage refreshed its gradient surrogate {refreshed} times per rank; the "
        f"point of this case is a surrogate that is replaced repeatedly during the stage\n"
        + result.describe())
    assert all(count == 0 for count in frozen.values()), (
        f"the stage that did not opt in must keep one evaluator for its whole run, got {frozen}\n"
        + result.describe())

    _check_stage_posterior("B10 (MH, Hamiltonian, refreshed NN gradients)", output_dir,
                           "alg0001_MH", result, linear_gaussian_problem)
    _check_stage_posterior("B10 (MH, Hamiltonian, frozen NN gradients)", output_dir,
                           "alg0002_MH", result, linear_gaussian_problem)

    retrainings = _collector_retrainings(output_dir, B10_MIN_SNAPSHOTS_TO_UPDATE)
    print(f"[B10] collector retrainings (reconstructed): {retrainings}", flush=True)
    assert retrainings >= B10_MIN_RETRAININGS, (
        f"the surrogate was trained only {retrainings} time(s); the refresh path would not be "
        f"exercised\n" + result.describe())


B11_ADAPTIVE_EVALUATIONS = 3_000
B11_EVALUATIONS = 25_000

B11_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=False, use_solvers_pool=True, no_solvers=2,
                                  debug=True,  # the carry-over lines this test parses are printed only with debug (c518457)
                                  initial_sample_type="lhs")
    assert conf.no_samplers == 2, conf.no_samplers
    stages = [Stage(algorithm="MH", proposal=PCN(adaptive=True),
                    max_evaluations={B11_ADAPTIVE_EVALUATIONS}),
              Stage(algorithm="MH", proposal=PCN(adaptive=False),  # frozen consumer
                    max_evaluations={B11_EVALUATIONS})]
    assert stages[0].adaptive is True, "adaptive pCN was refused"
    assert stages[1].proposal.beta is None
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages)
    print("DRIVER FINISHED", flush=True)
    """


def test_b11_adaptive_pcn_then_pcn_with_the_pooled_beta(mpi_workspace, run_mpi,
                                                        linear_gaussian_problem):
    """
    B11 (new 2026-09-20): an adaptive pCN stage (Robbins-Monro on the logit of ``beta``,
    `16` §5 item 6) followed by a plain pCN stage with ``PCN(beta=None)``, two samplers
    (``-n 3``). Adaptive pCN did not exist before -- ``Stage.__post_init__`` forced
    ``adaptive=False`` for it -- so this is the first MPI coverage of that path.

    Checked: both samplers carry the SAME adapted ``beta`` (it is the mean of their logits,
    so the pooling really has to run on both), the carried value is a valid pCN step size,
    and the frozen stage samples the closed-form posterior within the suite's 4 SE.
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b11", B11_BODY, n=3, timeout=300)
    assert "Message truncated" not in result.stdout + result.stderr, result.describe()

    beta = _assert_identical_carry_over(result, "alg0000_MH-adaptive", "beta")
    assert beta.size == 1, beta
    assert 0.0 < float(beta[0]) < 1.0, beta
    # the adaptation really moved beta away from the 0.5 default it started at
    assert abs(float(beta[0]) - 0.5) > 0.02, f"beta barely moved: {beta}\n" + result.describe()

    _check_stage_posterior("B11 (pCN with the pooled adapted beta)", output_dir, "alg0001_MH",
                           result, linear_gaussian_problem)


B12_BOOTSTRAP_EVALUATIONS = 2_000
B12_EVALUATIONS = 8_000
B12_MIN_SNAPSHOTS_INITIAL = 200
B12_MIN_SNAPSHOTS_TO_UPDATE = 2_000

B12_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=True, no_solvers=2,
                                  debug=True,  # the carry-over lines this test parses are printed only with debug (c518457)
                                  initial_sample_type="lhs", use_surrogate_gradients=True,
                                  min_snapshots_initial={B12_MIN_SNAPSHOTS_INITIAL},
                                  min_snapshots_to_update={B12_MIN_SNAPSHOTS_TO_UPDATE})
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.NeuralNetworkUpdater(
        no_parameters=2, no_observations=2, hidden_layer_sizes=(16, 16),
        iterations_batch=50, seed=0)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale={PROPOSAL_SD}),
                    max_evaluations={B12_BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1, num_steps=5, mass=1.0, adaptive=True),
                    max_evaluations={B12_EVALUATIONS}),
              Stage(algorithm="MH", proposal=Hamiltonian(num_steps=5, mass=1.0, adaptive=False),  # frozen consumer
                    max_evaluations={B12_EVALUATIONS})]
    assert stages[1].adaptive is True, "adaptive Hamiltonian was refused"
    assert stages[2].proposal.step_size is None
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def test_b12_adaptive_hamiltonian_step_size_then_the_frozen_step_size(mpi_workspace, run_mpi,
                                                                      linear_gaussian_problem):
    """
    B12 (new 2026-09-20, best effort like B8): B8's layout with the Hamiltonian stage made
    ``Hamiltonian(adaptive=True)`` (dual-averaged leapfrog step size, `16` §5 item 4) followed by
    a fixed Hamiltonian stage with ``Hamiltonian(step_size=None)``, so it runs on the carried,
    dual-averaged ``eps_bar``.

    Checked: no ``Message truncated`` (the adapted-state buffer has the same length on every
    rank whatever each rank's ``adapt()`` count was), both samplers carry the SAME step size
    (the mean of their ``log eps_bar``), the carried step size really differs from the 0.1 the
    stage started at, the per-period ``adaptive_stats`` trace is written, and the **frozen**
    stage -- the one the hand-over produces, and the only one here whose proposal is constant --
    matches the closed-form posterior within the suite's 4 SE with an acceptance rate near the
    0.8 dual-averaging target. As in B8, the accept/reject test uses the exact model only, so the
    chain is exact for any step size and any gradient field; what is under test is the transport
    and the hand-over.

    Why the ADAPTIVE stage's own posterior is not checked (measured 2026-09-20, this run): its
    first half is a warm-up transient. Dual averaging starts at ``exp(mu) = 10*eps_0`` by
    construction (Hoffman & Gelman) and then has to find the step size the *surrogate's* gradient
    field can sustain -- the neural network is deliberately under-trained here, so the exact-model
    acceptance pays for its gradient error and the step size settles ~4x below the 0.1 the stage
    started at (eps_bar 0.024, i.e. an integration time of 0.12 at ``num_steps=5``), which mixes
    slowly. Over 8000 evaluations that leaves the stage's pooled covariance 43 % above the closed
    form (5.80 SE), shrinking monotonically along the chain -- its last 20 % gives 0.11/0.086
    against the closed form's 0.0996/0.0810, and the frozen stage that follows it gives
    0.095-0.098, i.e. the chain is not biased, it is not yet stationary. The same adaptation
    driven by an EXACT gradient is unbiased and lands at eps_bar 0.39 (checked in process on this
    problem; `tests/unit/test_proposals.py` pins the target rate and the stability bound).
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b12", B12_BODY, n=4, timeout=480)
    assert "Message truncated" not in result.stdout + result.stderr, result.describe()

    step_size = _assert_identical_carry_over(result, "alg0001_MH-adaptive", "step_size")
    assert step_size.size == 1 and step_size[0] > 0.0, step_size
    assert abs(float(step_size[0]) - 0.1) > 1e-6, (
        f"the dual averaging never moved the step size away from 0.1: {step_size}\n"
        + result.describe())

    from surrDAMH.modules.run_data import read_run
    run = read_run(str(output_dir), load_raw_data=False)
    adaptive_stats = run.adaptive_stats[run.stage_index("alg0001_MH-adaptive")]
    assert adaptive_stats is not None and not adaptive_stats.empty
    assert list(adaptive_stats.columns) == ["m", "mean_acceptance_probability", "log_step_size",
                                            "log_step_size_bar", "rank_world"]
    assert run.adaptive_stats[run.stage_index("alg0002_MH")] is None
    # the trace really tracks the target the dual averaging is driven to
    assert 0.6 < float(adaptive_stats["mean_acceptance_probability"].mean()) < 0.95, adaptive_stats

    statistics = _check_stage_posterior("B12 (MH, Hamiltonian with the carried step size)",
                                        output_dir, "alg0002_MH", result, linear_gaussian_problem)
    acceptance = statistics.counter_accepted / (statistics.counter_accepted
                                                + statistics.counter_rejected)
    assert 0.6 < acceptance < 0.95, (
        f"the frozen stage runs at the carried step size, so its acceptance should be near the "
        f"0.8 dual-averaging target, got {acceptance:.3f}\n" + result.describe())


B13_BOOTSTRAP_EVALUATIONS = 2_000
B13_CHUNK_EVALUATIONS = 6_000
B13_INTEGRATION_TIME = 1.0

B13_BODY = f"""
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=True, use_solvers_pool=False,
                                  debug=True,  # the carry-over lines this test parses are printed only with debug
                                  initial_sample_type="lhs", use_surrogate_gradients=True,
                                  min_snapshots_initial=200, min_snapshots_to_update=1000)
    assert conf.no_samplers == 2, conf.no_samplers
    updater = surrDAMH.surrogates.NeuralNetworkUpdater(
        no_parameters=2, no_observations=2, hidden_layer_sizes=(16, 16),
        iterations_batch=50, seed=0)
    chunk = lambda: Hamiltonian(integrator="dimension_robust", mass="covariance",
                                integration_time={B13_INTEGRATION_TIME})
    stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations={B13_BOOTSTRAP_EVALUATIONS}),
              Stage(algorithm="DAMH", proposal=chunk(), subchain_length=1, surrogate_model_updates=True,
                    max_evaluations={B13_CHUNK_EVALUATIONS}),
              Stage(algorithm="DAMH", proposal=chunk(), subchain_length=1, surrogate_model_updates=True,
                    max_evaluations={B13_CHUNK_EVALUATIONS})]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
    print("DRIVER FINISHED", flush=True)
    """


def test_b13_fast_mode_like_chunks_with_the_covariance_mass(mpi_workspace, run_mpi, linear_gaussian_problem):
    """
    B13 (C1, 2026-10-11): 2 samplers + collector (-n 3, in-process solver), a torch MLP trained on
    the fly, an adaptive random-walk MH warm-up and two DAMH-SMU chunks (K = 1) with Auto's fast-mode
    proposal ``Hamiltonian(integrator="dimension_robust", mass="covariance", integration_time=T)``.

    Checked: the pooled posterior statistics are identical on both samplers after every stage (one
    ``Allgather`` per stage end) and count every outer iteration of both chains exactly once
    (pooled ``n`` == the notes' cumulative ``sum`` over ranks and stages; 2026-10-11 fix), carried
    across the stages (``carry_over/``); each chunk resolved its mass from the statistics (manifest
    ``proposal_resolution``: ``covariance (n=...)`` with the warm-up's 4000 states for the first
    chunk); ``num_steps`` follows ``T / step`` in the trace; the last chunk matches the closed-form
    posterior within 4 SE.
    """
    _check_problem_matches(mpi_workspace, linear_gaussian_problem)
    result, output_dir = _run_case(mpi_workspace, run_mpi, "b13", B13_BODY, n=3, timeout=480)
    assert "Message truncated" not in result.stdout + result.stderr, result.describe()

    from surrDAMH.modules.continuation import load_carry_over
    names = ["alg0000_MH-adaptive", "alg0001_DAMH-SMU", "alg0002_DAMH-SMU"]
    counts = []
    for name in names:
        statistics_vector = _assert_identical_carry_over(result, name, "posterior_statistics")
        assert statistics_vector.size == 1 + 2 + 4
        carried, summary = load_carry_over(str(output_dir), name)
        np.testing.assert_array_equal(np.asarray(carried["posterior_statistics"]), statistics_vector)
        counts.append(int(summary["posterior_statistics_n"]))
    assert counts[0] == NO_SAMPLERS * B13_BOOTSTRAP_EVALUATIONS, counts
    assert counts[0] < counts[1] < counts[2], counts
    # the pooled count is the TRUE number of outer iterations of all chains so far (notes column
    # "sum" per rank and stage); before the 2026-10-11 fix every stage end counted the shared
    # history once per sampler
    import pandas as pd
    true_counts, total = [], 0
    for name in names:
        notes_dir = os.path.join(str(output_dir), "sampling_output", "notes", name)
        files = sorted(f for f in os.listdir(notes_dir) if f.startswith("rank"))
        assert len(files) == NO_SAMPLERS, files
        total += sum(int(pd.read_csv(os.path.join(notes_dir, f))["sum"].iloc[-1]) for f in files)
        true_counts.append(total)
    assert counts == true_counts, (counts, true_counts)
    for name, true_count in zip(names, true_counts):
        carried, _summary = load_carry_over(str(output_dir), name)
        assert int(carried["posterior_statistics_n_seen"]) == true_count  # no burn-in: n_seen == n

    resolution = _read_manifest(output_dir)["proposal_resolution"]
    assert [entry["name"] for entry in resolution] == names
    assert resolution[1]["mass_source"] == f"covariance (n={counts[0]})", resolution
    assert resolution[2]["mass_source"] == f"covariance (n={counts[1]})", resolution
    assert resolution[1]["integration_time"] == B13_INTEGRATION_TIME

    from surrDAMH.modules.run_data import read_run
    run = read_run(str(output_dir), load_raw_data=False)
    trace = run.adaptive_stats[run.stage_index("alg0002_DAMH-SMU")]
    assert trace is not None and list(trace.columns)[-2:] == ["num_steps", "rank_world"]
    expected = np.maximum(1, np.round(B13_INTEGRATION_TIME / np.exp(trace["log_step_size"].to_numpy())))
    np.testing.assert_array_equal(trace["num_steps"].to_numpy(), np.minimum(expected, 1000))

    _check_stage_posterior("B13 (DAMH-SMU, NN, covariance mass, T)", output_dir, "alg0002_DAMH-SMU", result,
                           linear_gaussian_problem)
