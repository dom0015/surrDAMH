#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI integration tests for collector / surrogate behaviour.

Items I5, I9, I11 of ``library_notes/07_testing_plan.md`` Phase 3.

I5 and I11 assert WS0 fixes (C1, A3); I9 asserts the WS8 fix of the ``initial_snapshots``
double counting (finding 2.8).

Run them with ``pytest -m mpi``; they are not part of the default fast suite.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from .conftest import requires_mpiexec, stage_sample_files, write_driver

pytestmark = [pytest.mark.mpi, requires_mpiexec]

INITIAL_SNAPSHOTS = 8  # N of item I9


def test_i5_surrogate_survives_into_a_use_only_surrogate_stage(mpi_workspace, run_mpi):
    """
    I5 (WS0 item C1, finding 2.6): stages ``[MH, DAMH(surrogate_model_updates=True),
    MH(use_only_surrogate=True)]`` with ``use_collector=True``.

    Before C1 the sampler terminated the collector communicators after the DAMH stage (the
    look-ahead only considered following DAMH stages), so the third stage found no evaluator.
    Asserted: exit 0 and a non-empty sample file for the third stage on every sampler.

    ``no_observations = 2`` since WS3 item 3 (finding A20): before that fix the
    ``use_only_surrogate`` path died for ``no_observations > 1`` in
    ``Normal.calculate_logpdf_uncorrelated`` with ``ValueError: shapes (1,2) and (1,2) not
    aligned``, because ``SurrogateAsSolver.get_observations`` returned the evaluator's
    ``(1, no_observations)`` array instead of the ``Solver`` contract's ``(no_observations,)``;
    the test had to use one observation. It now also covers the surrogate-observations
    hand-over of finding A11 (the fourth stage re-evaluates the carried sample exactly).

    The plan's second variant ``[DAMH, MH(Hamiltonian)]`` is not covered: it needs a
    gradient-capable (torch) updater plus initial snapshots to avoid the I4 deadlock.
    """
    driver = write_driver(mpi_workspace, "i5", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=6)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=40),
                  Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=30,
                        surrogate_model_updates=True),
                  Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_samples=20,
                        use_only_surrogate=True),
                  Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec("LinearSolver"))
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i5"

    result = run_mpi(driver, n=4, timeout=180)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()

    for stage_name in ("alg0000_MH", "alg0001_DAMH-SMU", "alg0002_MH", "alg0003_MH"):
        files = stage_sample_files(output_dir, stage_name)
        assert [f.name for f in files] == ["rank0000.csv", "rank0001.csv"], (
            f"stage {stage_name} is missing sample files\n" + result.describe())
        for f in files:
            assert f.stat().st_size > 0, f"{f} is empty\n" + result.describe()


def test_i11_test_data_object_as_surrogate_test_data(mpi_workspace, run_mpi):
    """
    I11 (WS0 item A3, finding 0.3): a ``TestData`` object passed as ``surrogate_test_data``.

    Before A3 the collector fed the object straight into ``len(...)``/unpacking, which raised a
    ``TypeError`` on the collector rank and left the samplers hanging. Asserted: exit 0,
    ``surrogate_quality.csv`` and ``surrogate_quality_test.csv`` written, and the fixed-test
    metrics reporting the ``n_test`` rows that were handed in.
    """
    driver = write_driver(mpi_workspace, "i11", """
        from surrDAMH.modules.test_data import TestData
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=6)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        prior = solver_module.make_prior()
        likelihood = solver_module.make_likelihood()
        problem = surrDAMH.Problem(prior, likelihood, solver=solver_spec())
        test_data = TestData.generate(problem, size=8,
                                      transform_before_surrogate=conf.transform_before_surrogate)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=40)]
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater,
                                   surrogate_test_data=test_data)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i11"

    result = run_mpi(driver, n=3, timeout=180)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()

    quality = output_dir / "sampling_output" / "surrogate_quality.csv"
    quality_test = output_dir / "sampling_output" / "surrogate_quality_test.csv"
    assert quality.exists() and quality.stat().st_size > 0, result.describe()
    assert quality_test.exists() and quality_test.stat().st_size > 0, result.describe()

    with open(quality_test, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["update_index", "snapshots_total", "n_test", "max_log_posterior", "rmse",
                       "max_abs_error", "weighted_rmse", "weighted_mean_abs_error", "weighted_ess"]
    assert len(rows) > 1, result.describe()
    assert all(int(row[2]) == 8 for row in rows[1:]), rows


def test_i9_initial_snapshots_are_counted_once(mpi_workspace, run_mpi):
    """
    I9 (finding 2.8, fixed in WS8): ``initial_snapshots`` with N rows handed to
    ``Problem.run_sampling`` together with ``use_collector=True``.

    ``run_COLLECTOR`` used to set ``no_snapshots_total = N`` from the preloaded snapshots and then
    add the very same batch through ``add_data``, adding ``num_new_snapshots`` (which still
    included those N rows) to the counter again - so the first ``snapshots_total`` recorded in
    ``sampling_output/surrogate_quality.csv`` was 2N (observed: N=8 -> 16), not N.

    Now the preloaded rows are counted only where they are actually added to the updater (and
    only counted up front in the ``training_data_loaded`` case, where the updater already holds
    them and ``add_data`` is skipped). Note this is a **behaviour change** for runs that pass
    ``initial_snapshots``: ``min_snapshots_initial`` / ``min_snapshots_to_update`` are now reached
    at different moments, so such runs retrain the surrogate at different times and produce a
    different DAMH-SMU sample stream than before.
    """
    driver = write_driver(mpi_workspace, "i9", f"""
        N = {INITIAL_SNAPSHOTS}
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=N)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        rng = np.random.RandomState(0)
        parameters = rng.normal(size=(N, 2))
        observations = np.array([solver_module.A @ p for p in parameters])
        weights = np.ones((N, 1))
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=40)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater,
                                   surrogate_initial_training_data=[parameters, observations, weights])
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i9"

    result = run_mpi(driver, n=3, timeout=180)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()

    quality = output_dir / "sampling_output" / "surrogate_quality.csv"
    assert quality.exists(), result.describe()
    with open(quality, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "snapshots_total", rows[0]
    first_total = int(rows[0 + 1][0])
    assert first_total == INITIAL_SNAPSHOTS, (
        f"first snapshots_total is {first_total}, expected {INITIAL_SNAPSHOTS} "
        f"(2N = {2 * INITIAL_SNAPSHOTS} means initial_snapshots are counted twice)")


def test_i9_initial_snapshots_reach_the_surrogate(mpi_workspace, run_mpi):
    """
    Companion to the test above: whatever the counting, the run itself must succeed and the
    preloaded snapshots must reach the surrogate. Asserted: exit 0 and a ``surrogate_quality.csv``
    whose first ``snapshots_total`` is at least N (exactly N since the WS8 fix of finding 2.8).
    """
    driver = write_driver(mpi_workspace, "i9b", f"""
        N = {INITIAL_SNAPSHOTS}
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=N)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        rng = np.random.RandomState(0)
        parameters = rng.normal(size=(N, 2))
        observations = np.array([solver_module.A @ p for p in parameters])
        weights = np.ones((N, 1))
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=40)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater,
                                   surrogate_initial_training_data=[parameters, observations, weights])
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i9b"

    result = run_mpi(driver, n=3, timeout=180)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    quality = output_dir / "sampling_output" / "surrogate_quality.csv"
    assert quality.exists(), result.describe()
    data = np.loadtxt(quality, delimiter=",", skiprows=1, ndmin=2)
    assert data[0, 0] >= INITIAL_SNAPSHOTS, result.describe()


UPDATER_SOURCE = """surrDAMH.surrogates.NeuralNetworkUpdater(
            no_parameters=2, no_observations=2, hidden_layer_sizes=(8, 8), solver="adamw",
            iterations_batch=5, device="cpu", verbose=False)"""


def test_torch_threads_effective_on_every_rank(mpi_workspace, run_mpi):
    """
    ``Configuration.torch_threads`` (WS4, author decision 2026-09-18): every rank that has
    torch loaded must end up with ``torch.get_num_threads() == conf.torch_threads`` after
    ``problem.run_sampling(...)`` returns, regardless of role (sampler / solvers pool / collector).

    ``torch_threads=2`` so the assertion is distinguishable from the launcher's own default
    (measured elsewhere: this MPICH ``mpiexec`` already starts every rank at 1 -- a launcher
    artefact, not something this library sets, see ``library_notes/10_manual_review_notes.md``
    §7b). The driver prints ``TORCH_THREADS rank=<r> value=<v>`` on every rank right after
    ``problem.run_sampling(...)`` (a plain rank check, no special hook needed: torch is unconditionally
    imported as soon as ``import surrDAMH`` runs, on every rank, by the time the driver gets
    this far -- see ``surrDAMH/modules/torch_threads.py``'s module docstring).
    """
    driver = write_driver(mpi_workspace, "torch_threads", f"""
        from mpi4py import MPI
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=6,
                                      torch_threads=2)
        updater = {UPDATER_SOURCE}
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20),
                  Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=15,
                        surrogate_model_updates=True)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec("LinearSolver"))
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
        import torch
        print(f"TORCH_THREADS rank={{MPI.COMM_WORLD.Get_rank()}} value={{torch.get_num_threads()}}",
              flush=True)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_torch_threads"

    result = run_mpi(driver, n=4, timeout=180)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()

    reported_ranks = set()
    for line in result.stdout.splitlines():
        if not line.startswith("TORCH_THREADS "):
            continue
        parts = dict(item.split("=") for item in line.removeprefix("TORCH_THREADS ").split())
        rank = int(parts["rank"])
        value = int(parts["value"])
        assert value == 2, f"rank {rank} reports torch.get_num_threads()={value}, expected 2\n" + result.describe()
        reported_ranks.add(rank)
    assert reported_ranks == {0, 1, 2, 3}, (
        f"expected a TORCH_THREADS line from every one of the 4 ranks, got {reported_ranks}\n"
        + result.describe())

    manifest_path = output_dir / "sampling_output" / "run_manifest.json"
    assert manifest_path.exists(), result.describe()
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["configuration"]["torch_threads"] == 2
    assert manifest["environment"]["torch_num_threads"] == 2


def test_surrogate_restart_lets_a_damh_first_stage_run(mpi_workspace, run_mpi):
    """
    WS5 ``SurrogateRestart``: run A trains a surrogate and saves its state, run B restores it
    with ``Problem.run_sampling(..., surrogate_restart=...)`` and starts straight into DAMH.

    Run B sets ``min_snapshots_initial = 1000``, far more snapshots than the restart holds, so
    the collector's start-up handshake can only report "evaluator available" through the
    ``pretrained_ready`` branch of ``process_COLLECTOR`` — i.e. purely because the restored
    checkpoint is already trained. Without the restart this is exactly the I4 configuration
    (``test_i4_damh_as_first_stage_deadlocks_without_a_surrogate``): the sampler raises and the
    run fails. Asserted: exit 0, the restore message, and a non-empty DAMH sample file.
    """
    body = f"""
        from mpi4py import MPI
        STATE_DIR = sys.argv[2] if len(sys.argv) > 2 else None
        RESTART = STATE_DIR is not None
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, use_solvers_pool=False,
                                      min_snapshots_initial=1000 if RESTART else 4,
                                      min_snapshots_to_update=1000)
        updater = {UPDATER_SOURCE}
        stages = [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=15)] if RESTART \\
            else [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30)]
        restart = surrDAMH.SurrogateRestart(STATE_DIR, mode="state") if RESTART else None
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_module.LinearSolver())
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater, surrogate_restart=restart)
        if not RESTART and MPI.COMM_WORLD.Get_rank() == conf.rank_collector:
            surrDAMH.SurrogateRestart(os.path.join(OUTPUT_DIR, "sampling_output")).save(updater)
        print("DRIVER FINISHED", flush=True)
        """
    driver = write_driver(mpi_workspace, "restart", body)
    dir_a = str(mpi_workspace / "out_restart_A")
    dir_b = str(mpi_workspace / "out_restart_B")

    result_a = run_mpi(driver, n=3, timeout=180, args=[dir_a])
    assert not result_a.timed_out and result_a.returncode == 0, result_a.describe()
    assert "saved surrogate state to" in result_a.stdout, result_a.describe()

    state_dir = os.path.join(dir_a, "sampling_output")
    result_b = run_mpi(driver, n=3, timeout=180, args=[dir_b, state_dir])

    assert not result_b.timed_out, result_b.describe()
    assert result_b.returncode == 0, result_b.describe()
    assert "DRIVER FINISHED" in result_b.stdout, result_b.describe()
    assert "restored surrogate state from" in result_b.stdout, result_b.describe()
    files = stage_sample_files(dir_b, "alg0000_DAMH-SMU")
    assert [f.name for f in files] == ["rank0000.csv", "rank0001.csv"], result_b.describe()
    assert all(f.stat().st_size > 0 for f in files), result_b.describe()


NONLINEAR_SOLVER_SOURCE = '''\
"""Nonlinear 2-parameter toy solver (toy_examples Solver_illustrative_local) with a small delay,
so that snapshots reach the collector one by one and its first surrogate is trained early."""
import time
import numpy as np
from surrDAMH.solvers import Solver


class SlowIllustrativeSolver(Solver):
    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters):
        self.x, self.y = float(parameters[0]), float(parameters[1])

    def get_observations(self):
        time.sleep(0.005)
        return np.array([(self.x ** 2 - self.y) * np.log((self.x - self.y) ** 2 + 1)])
'''


def test_frozen_damh_after_mh_uses_the_newest_evaluator(mpi_workspace, run_mpi):
    """
    Regression test for the stale-evaluator bug (fixed 2026-09-30, review notes §3 and
    ``library_notes/19_efficiency_study_2026-09-28.md`` §4.4): ``[MH warm-up,
    DAMH(surrogate_model_updates=False)]``, 2 samplers + collector.

    The frozen stage used to install the evaluator answering the request posted at start-up --
    the collector's FIRST surrogate, here a polynomial fitted to 4 snapshots -- and never poll
    again, although the collector retrained on hundreds of snapshots during the warm-up.
    Observed before the fix with this layout: surrogate-vs-exact RMSE 7.55 in the frozen
    stage and 0 accepted outer proposals; after: RMSE 0.02, ~290 of 300 accepted. Asserted:
    RMSE of ``obs_approx`` vs ``obs`` over the stage's exact rows below 0.5 and at least one
    accepted proposal per chain.
    """
    (mpi_workspace / "nonlinear_solver.py").write_text(NONLINEAR_SOLVER_SOURCE)
    driver = write_driver(mpi_workspace, "frozen_after_mh", """
        spec = SolverSpec(solver_module_path=os.path.join(SUPPORT_DIR, "nonlinear_solver.py"),
                          solver_module_name="nonlinear_solver", solver_class_name="SlowIllustrativeSolver",
                          solver_parameters={})
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=1,
                                      use_collector=True, use_solvers_pool=False,
                                      min_snapshots_initial=4, min_snapshots_to_update=10,
                                      save_snapshots_to_file=True)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=1)
        likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=0.2)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=300),
                  Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=100,
                        surrogate_model_updates=False)]
        # likelihood is Normal(mean=5.0, ...), a scalar/dimension-free mean (WS: "dimension-free
        # Normal"); SlowIllustrativeSolver declares no_observations=1 but spec is a SolverSpec
        # here (not an instance), so Problem cannot see that attribute -- give the size explicitly.
        problem = surrDAMH.Problem(solver_module.make_prior(), likelihood, solver=spec, no_observations=1)
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_frozen_after_mh"

    result = run_mpi(driver, n=3, timeout=180)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()

    stage = "alg0001_DAMH"
    raw_files = sorted((output_dir / "sampling_output" / "raw_data" / stage).glob("rank*.csv"))
    assert [f.name for f in raw_files] == ["rank0000.csv", "rank0001.csv"], result.describe()
    errors = []
    for f in raw_files:
        with open(f, newline="") as handle:
            for row in csv.DictReader(handle):
                exact, approx = float(row["obs_0"]), float(row["obs_approx_0"])
                if np.isfinite(exact) and np.isfinite(approx):
                    errors.append(exact - approx)
    assert len(errors) > 0, result.describe()
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    assert rmse < 0.5, (f"frozen DAMH stage surrogate RMSE {rmse:.3g} over {len(errors)} exact rows: "
                        "the stage uses a stale (first-trained) evaluator")

    for f in sorted((output_dir / "sampling_output" / "notes" / stage).glob("rank*.csv")):
        with open(f, newline="") as handle:
            notes = next(csv.DictReader(handle))
        assert int(notes["accepted"]) > 0, f"{f}: {notes}"
