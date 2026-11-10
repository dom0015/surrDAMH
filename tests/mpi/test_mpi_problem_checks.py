#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI tests for the problem-size checks added with ``surrDAMH.Problem`` (2026-10-08):

- C2: at solvers-pool start-up every spawned child reports the sizes its solver declares
  (``parent_comm.gather`` in ``process_CHILD.py`` / ``CommunicationWithChild.__init__``); a
  mismatch with the configuration must abort the job with a message naming solver_id, declared
  and configured values -- not hang. A solver that declares nothing only prints a warning.
- C3: the first evaluation in a spawned child must return ``(no_observations,)``.

The solvers live in an extra module written next to the shared ``solver_module.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import requires_mpiexec, stage_sample_files, write_driver

pytestmark = [pytest.mark.mpi, requires_mpiexec]

CHECK_SOLVERS_SOURCE = '''\
import numpy as np
from surrDAMH.solvers import Solver

A = np.array([[1.0, 0.5], [0.0, 1.0]])


class DeclaresThreeObservations(Solver):
    """Declares no_observations=3 while the likelihood has 2 values (C2)."""

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = 2
        self.no_observations = 3

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float).reshape(-1)

    def get_observations(self):
        return A @ self.parameters


class ReturnsThreeObservations(Solver):
    """Declares the right sizes (2, 2) but returns 3 values (C3)."""

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = 2
        self.no_observations = 2

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float).reshape(-1)

    def get_observations(self):
        return np.append(A @ self.parameters, 0.0)


class DeclaresNothing(Solver):
    """Correct model without size declarations (C2 warning only)."""

    def __init__(self, solver_id=0, output_dir=None):
        pass

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float).reshape(-1)

    def get_observations(self):
        return A @ self.parameters
'''

DRIVER_BODY = """
    CHECK_SOLVERS_PATH = os.path.join(SUPPORT_DIR, "check_solvers.py")
    spec = SolverSpec(solver_module_path=CHECK_SOLVERS_PATH, solver_module_name="check_solvers",
                      solver_class_name=sys.argv[2], solver_parameters={{}})
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, use_collector=False, no_solvers=1)
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(), solver=spec)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations={max_evaluations})]
    problem.run_sampling(conf, stages)
    print("DRIVER FINISHED", flush=True)
    """


def _write(mpi_workspace, name, max_evaluations=20):
    (Path(mpi_workspace) / "check_solvers.py").write_text(CHECK_SOLVERS_SOURCE)
    return write_driver(mpi_workspace, name, DRIVER_BODY.format(max_evaluations=max_evaluations))


def test_c2_solver_declaring_wrong_sizes_aborts_with_the_message(mpi_workspace, run_mpi):
    driver = _write(mpi_workspace, "c2_mismatch")
    output_dir = mpi_workspace / "out_c2_mismatch"

    result = run_mpi(driver, n=2, timeout=90, args=[str(output_dir), "DeclaresThreeObservations"])

    assert not result.timed_out, "C2: a size mismatch must abort, not hang\n" + result.describe()
    assert result.returncode not in (0, None), result.describe()
    assert "DRIVER FINISHED" not in result.stdout, result.describe()
    output = result.stdout + result.stderr
    assert ("solver DeclaresThreeObservations (solver_id=0, child rank 0) declares no_observations=3, "
            "but the run is configured with no_observations=2") in output, result.describe()


def test_c3_spawned_solver_returning_the_wrong_shape_aborts_with_the_message(mpi_workspace, run_mpi):
    driver = _write(mpi_workspace, "c3_shape")
    output_dir = mpi_workspace / "out_c3_shape"

    result = run_mpi(driver, n=2, timeout=90, args=[str(output_dir), "ReturnsThreeObservations"])

    assert not result.timed_out, "C3: a wrong observation shape must abort, not hang\n" + result.describe()
    assert result.returncode not in (0, None), result.describe()
    assert "DRIVER FINISHED" not in result.stdout, result.describe()
    output = result.stdout + result.stderr
    assert ("ReturnsThreeObservations.get_observations() returned shape (3,), expected (2,)") in output, \
        result.describe()


def test_c2_undeclared_sizes_only_warn(mpi_workspace, run_mpi):
    driver = _write(mpi_workspace, "c2_undeclared")
    output_dir = mpi_workspace / "out_c2_undeclared"

    result = run_mpi(driver, n=2, timeout=90, args=[str(output_dir), "DeclaresNothing"])

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    assert result.stdout.count("does not declare") == 1, result.describe()
    assert [f.name for f in stage_sample_files(output_dir, "alg0000_MH")] == ["rank0000.csv"]
