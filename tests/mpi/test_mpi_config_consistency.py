#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-rank ``Configuration`` consistency check (finding 2.9, WS8 "Configuration consistency").

``Problem.run_sampling()`` starts with
``communication.check_configuration_consistency``: rank 0 broadcasts the *requested*
(as-constructed) values of ``Configuration.POSTERIOR_AFFECTING_FIELDS`` and every rank asserts
equality. Needs real multi-rank communication, hence ``tests/mpi`` rather than a unit test.

Two behaviours are pinned here:

* a driver that deliberately builds a different ``Configuration`` per rank must fail at
  start-up, on every rank, with the offending field named (before the fix such a run went on
  to sample - each chain with its own configuration - or hung/crashed later somewhere
  unrelated);
* a correctly-built configuration whose posterior-affecting fields include the awkward types
  (a numpy-array ``lhs_scale``, a duck-typed ``initial_samples_distribution`` object whose two
  instances are never ``==``) must *not* trip the check and must not raise
  ``ValueError: truth value of an array ... is ambiguous`` - see
  ``Configuration.normalize_for_comparison``.

Run them with ``pytest -m mpi``; they are not part of the default fast suite.
"""

from __future__ import annotations

import pytest

from .conftest import requires_mpiexec, stage_sample_files, write_driver

pytestmark = [pytest.mark.mpi, requires_mpiexec]


def test_config_mismatch_across_ranks_fails_fast(mpi_workspace, run_mpi):
    """Rank 1 asks for a different ``min_snapshots_initial`` -> the job aborts at start-up, naming it.

    Not ``no_parameters``/``no_observations`` any more: those are now resolved by ``Problem``
    (from the identically-built prior/likelihood/solver here) and ``Configuration.resolve_problem_sizes``
    raises its own, rank-local ``ValueError`` the moment a rank's explicit value disagrees with
    that resolution -- before the cross-rank broadcast this test is about ever runs. A field
    ``Problem`` does not touch, such as ``min_snapshots_initial``, still reaches
    ``check_configuration_consistency`` and is the one this test needs.
    """
    driver = write_driver(mpi_workspace, "config_mismatch", """
        from mpi4py import MPI

        # the bug this check exists for: a Configuration argument that depends on the rank
        extra = 1 if MPI.COMM_WORLD.Get_rank() == 1 else 0
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR,
                                      no_parameters=solver_module.NO_PARAMETERS,
                                      no_observations=solver_module.NO_OBSERVATIONS,
                                      use_collector=False, no_solvers=1,
                                      min_snapshots_initial=1 + extra)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """)

    result = run_mpi(driver, n=3, timeout=120)

    assert not result.timed_out, result.describe()
    assert result.returncode != 0, result.describe()
    assert "DRIVER FINISHED" not in result.stdout, result.describe()
    output = result.stdout + result.stderr
    assert "Configuration mismatch across MPI ranks" in output, result.describe()
    assert "min_snapshots_initial" in output, result.describe()
    # fail-loud machinery: the exception goes through _run_role, i.e. traceback + job-wide abort
    assert "CONFIGURATION CHECK" in output, result.describe()


def test_rank_dependent_no_parameters_still_aborts(mpi_workspace, run_mpi):
    """The scenario the test above used before 2026-10-08: rank 1 asks for a different
    ``no_parameters``. It is now caught on that rank by ``Configuration.resolve_problem_sizes``
    (against the size ``Problem`` resolved) before the cross-rank check; the job must still
    abort with a message naming the field, not run or hang."""
    driver = write_driver(mpi_workspace, "config_mismatch_sizes", """
        from mpi4py import MPI

        extra = 1 if MPI.COMM_WORLD.Get_rank() == 1 else 0
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR,
                                      no_parameters=solver_module.NO_PARAMETERS + extra,
                                      no_observations=solver_module.NO_OBSERVATIONS,
                                      use_collector=False, no_solvers=1)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """)

    result = run_mpi(driver, n=3, timeout=120)

    assert not result.timed_out, result.describe()
    assert result.returncode != 0, result.describe()
    assert "DRIVER FINISHED" not in result.stdout, result.describe()
    output = result.stdout + result.stderr
    assert "Configuration(no_parameters=3) does not match the problem's no_parameters=2" in output, result.describe()


def test_awkward_field_types_do_not_trip_the_check(mpi_workspace, run_mpi):
    """A correct configuration with an array ``lhs_scale`` and an object-valued
    ``initial_samples_distribution`` runs to completion: no spurious mismatch, no ambiguous
    array comparison."""
    driver = write_driver(mpi_workspace, "config_ok", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR,
                                      no_parameters=solver_module.NO_PARAMETERS,
                                      no_observations=solver_module.NO_OBSERVATIONS,
                                      use_collector=False, use_solvers_pool=False,
                                      lhs_scale=np.array([1.0, 2.0]),
                                      initial_sample_type="user_specified",
                                      initial_samples_distribution=solver_module.FixedSample([0.3, -0.7]))
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_module.LinearSolver())
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_config_ok"

    result = run_mpi(driver, n=2, timeout=120)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    assert "Configuration mismatch" not in (result.stdout + result.stderr), result.describe()
    assert [f.name for f in stage_sample_files(output_dir, "alg0000_MH")] == \
        ["rank0000.csv", "rank0001.csv"], result.describe()
