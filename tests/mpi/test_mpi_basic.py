#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI integration tests: plain sampling runs and continuation.

Items I1, I2, I8 of ``library_notes/07_testing_plan.md`` Phase 3. Each test writes a
self-contained driver script into ``tmp_path`` and runs it with
``mpiexec -n k python -m mpi4py <driver>`` under a timeout (see ``tests/mpi/conftest.py``).

Run them with ``pytest -m mpi``; they are not part of the default fast suite.
"""

from __future__ import annotations

import importlib.util
import glob
import os

import numpy as np
import pytest

from .conftest import requires_mpiexec, stage_sample_files, write_driver

pytestmark = [pytest.mark.mpi, requires_mpiexec]


def _load_solver_module(mpi_workspace):
    """Import the generated ``solver_module.py`` into this process (same code as the drivers use)."""
    path = os.path.join(str(mpi_workspace), "solver_module.py")
    spec = importlib.util.spec_from_file_location("mpi_test_solver_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n", [2, 3, 4])
def test_i1_minimal_example_shape_runs_and_writes_one_file_per_sampler(mpi_workspace, run_mpi, n):
    """
    I1: ``minimal_example.py`` shape (MH only, no collector, solvers pool with ``no_solvers=1``).

    Expected (and asserted) behaviour: clean exit, ``no_samplers = n - 1`` non-empty
    ``samples/<stage>/rank%04d.csv`` files, and the spawned solver child reporting its
    evaluation count on stdout (``process_CHILD.py:65``).
    """
    driver = write_driver(mpi_workspace, "i1", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR,
                                      no_parameters=solver_module.NO_PARAMETERS,
                                      no_observations=solver_module.NO_OBSERVATIONS,
                                      use_collector=False, no_solvers=1)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=40)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i1"

    result = run_mpi(driver, n=n, timeout=120)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()

    expected_samplers = n - 1  # pool takes one rank, no collector (configuration.py:73-76)
    files = stage_sample_files(output_dir, "alg0000_MH")
    assert [f.name for f in files] == [f"rank{i:04d}.csv" for i in range(expected_samplers)], result.describe()
    for f in files:
        assert f.stat().st_size > 0
        assert np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2).shape[1] == 1 + 2 + 1  # multiplicity, u_1, u_2, log_posterior

    assert "Solver at spawned process - evaluations:" in result.stdout, result.describe()
    evaluations = int(result.stdout.split("Solver at spawned process - evaluations:")[1].split("\n")[0])
    assert evaluations > 0


def test_i2_single_process_local_solver(mpi_workspace, run_mpi):
    """
    I2: ``one_process_only.py`` shape - ``-n 1``, ``use_collector=False``,
    ``use_solvers_pool=False``: the solver runs inside the single sampler rank and no
    point-to-point MPI call is made. Expected: clean exit and exactly one sample file.

    Bonus (cheap): the produced chain is compared with ``surrDAMH.Problem.run_sampling_local``
    for the same configuration and the same fixed initial sample; the in-process version of
    this equivalence is ``tests/test_runner_local.py``.
    """
    driver = write_driver(mpi_workspace, "i2", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR,
                                      no_parameters=solver_module.NO_PARAMETERS,
                                      no_observations=solver_module.NO_OBSERVATIONS,
                                      use_collector=False, use_solvers_pool=False,
                                      initial_sample_type="user_specified",
                                      initial_samples_distribution=solver_module.FixedSample([0.3, -0.7]))
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_module.LinearSolver())
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i2"

    result = run_mpi(driver, n=1, timeout=120)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()

    files = stage_sample_files(output_dir, "alg0000_MH")
    assert [f.name for f in files] == ["rank0000.csv"], result.describe()
    mpi_samples = np.loadtxt(files[0], delimiter=",", skiprows=1, ndmin=2)
    assert mpi_samples.size > 0

    # same configuration, run in this process by the standalone local runner:
    import surrDAMH
    from surrDAMH.proposals import RandomWalk
    from surrDAMH.stages import Stage

    solver_module = _load_solver_module(mpi_workspace)
    local_dir = mpi_workspace / "out_i2_local"
    conf = surrDAMH.Configuration(output_dir=str(local_dir),
                                  no_parameters=solver_module.NO_PARAMETERS,
                                  no_observations=solver_module.NO_OBSERVATIONS,
                                  use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=solver_module.FixedSample([0.3, -0.7]))
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_module.LinearSolver())
    problem.run_sampling_local(
        conf, [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30)])
    local_samples = np.loadtxt(local_dir / "sampling_output" / "samples" / "alg0000_MH" / "rank0000.csv",
                               delimiter=",", skiprows=1, ndmin=2)
    assert np.array_equal(mpi_samples, local_samples), (
        "run_sampling_local() must reproduce chain 0 of the -n 1 MPI run bit-for-bit\n" + result.describe())


def test_i8_continuation_restarts_from_the_saved_last_sample_bit_for_bit(mpi_workspace, run_mpi):
    """
    I8 (WS0 items C9/C10): run A, then run B with ``initial_sample_type="continued"`` in pool
    mode with a solver echoing its input.

    What is compared: run A's ``sampling_output/last_sample/<stage>/rank0000.npz`` (written by
    ``modules/continuation.save_last_sample``, float64 since C10) against the **first row** of
    run B's ``sampling_output/samples/<stage>/rank0000.csv``. That first row is run B's initial
    sample, i.e. exactly the parameter vector the (echo) solver evaluates first
    (``AlgorithmBase._prepare_run`` evaluates ``self.current`` before the first proposal, and
    ``_record_current_sample`` writes it as the first row whatever the first acceptance decision
    is). Both runs use ``transform_before_saving=False``, so the CSV holds the raw internal
    parameters; ``csv.writer`` writes the round-trip repr of ``np.float64``, so the comparison
    is bit-exact.

    Before WS0 the last sample was stored as float32 (finding 2.3), so the restart silently
    lost precision.
    """
    body = """
        PREVIOUS_DIR = sys.argv[2] if len(sys.argv) > 2 else None
        kwargs = {}
        if PREVIOUS_DIR is not None:
            kwargs = dict(initial_sample_type="continued", continued_from_dir=PREVIOUS_DIR)
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR,
                                      no_parameters=solver_module.NO_PARAMETERS,
                                      no_observations=solver_module.NO_OBSERVATIONS,
                                      use_collector=False, no_solvers=1,
                                      transform_before_saving=False, **kwargs)
        max_evaluations = 25 if PREVIOUS_DIR is None else 5
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=max_evaluations)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec("EchoSolver"))
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """
    driver = write_driver(mpi_workspace, "i8", body)
    dir_a = str(mpi_workspace / "out_i8_A")
    dir_b = str(mpi_workspace / "out_i8_B")

    result_a = run_mpi(driver, n=2, timeout=120, args=[dir_a])
    assert not result_a.timed_out and result_a.returncode == 0, result_a.describe()

    result_b = run_mpi(driver, n=2, timeout=120, args=[dir_b, dir_a])
    assert not result_b.timed_out and result_b.returncode == 0, result_b.describe()

    last_sample_files = glob.glob(os.path.join(dir_a, "sampling_output", "last_sample", "*", "rank0000.npz"))
    assert len(last_sample_files) == 1, last_sample_files
    with np.load(last_sample_files[0]) as loaded:
        last_of_a = loaded["parameters"]
    assert last_of_a.dtype == np.float64, "C10: continuation files must be stored as float64"

    first_of_b = np.loadtxt(stage_sample_files(dir_b, "alg0000_MH")[0], delimiter=",", skiprows=1, ndmin=2)[0, 1:3]
    assert np.array_equal(last_of_a, first_of_b), (
        f"continuation is not bit-exact: A last sample {last_of_a!r} vs B first sample {first_of_b!r}\n"
        + result_b.describe())


def test_relative_solver_module_path_reaches_spawned_children(mpi_workspace, run_mpi):
    """
    WS5 / finding M18: ``SolverSpec`` resolves a **relative** ``solver_module_path`` against the
    launching process's working directory, at construction, so the spawned solver children
    receive an absolute path through the broadcast spec.

    Measured before the change (2026-09-17, MPICH/Hydra): a relative path already worked,
    because ``MPI_Comm_spawn`` handed the children the launcher's working directory. The
    driver below therefore also ``os.chdir("/")`` after building the spec and before
    ``problem.run_sampling(...)``, which is what an implementation/launcher that gives the children a
    different ``wdir`` looks like from the spec's point of view; with the pre-WS5 code the
    child then fails with ``FileNotFoundError: solver_module.py``.

    Asserted: the spec stores an absolute path, and the run (solvers pool + spawned child)
    completes and writes samples.
    """
    body = """
        from surrDAMH.solver_specification import SolverSpec
        spec = SolverSpec(solver_module_path="solver_module.py", solver_module_name="solver_module",
                          solver_class_name="LinearSolver", solver_parameters={})
        print("SPEC PATH:", spec.solver_module_path, flush=True)
        assert os.path.isabs(spec.solver_module_path), spec.solver_module_path
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=False, use_solvers_pool=True, no_solvers=1)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=15)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(), solver=spec)
        os.chdir("/")  # the spec must no longer depend on the working directory
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """
    driver = write_driver(mpi_workspace, "relpath", body)
    output_dir = mpi_workspace / "out_relpath"

    # cwd = mpi_workspace (run_mpi's default): that is what "solver_module.py" is relative to
    result = run_mpi(driver, n=2, timeout=120, cwd=str(mpi_workspace))

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    assert f"SPEC PATH: {mpi_workspace}/solver_module.py" in result.stdout, result.describe()
    files = stage_sample_files(output_dir, "alg0000_MH")
    assert [f.name for f in files] == ["rank0000.csv"], result.describe()
    assert files[0].stat().st_size > 0, result.describe()
