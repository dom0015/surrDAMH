#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI integration tests for the observation transport between the solvers pool, its spawned
children and the samplers.

Item I7 of ``library_notes/07_testing_plan.md`` Phase 3 (finding 2.4). Since WS8 (decision 5,
2026-09-17) there is only **one** transport: the observations always travel pickled together
with the solver tag (``send([observations, solver_tag], tag=<request tag>)`` child -> pool ->
sampler). The ``Configuration.pickled_observations`` switch and the raw-buffer path it selected
are gone, so the remaining matrix is just "solver returns a bare array" vs "solver returns
``(observations, tag)``" (the ``Configuration.solver_returns_tag`` flag was removed on 2026-09-21,
the child detects the tuple itself) plus the negative-tag (solver failure) case that used to
crash the raw path.

Run them with ``pytest -m mpi``; they are not part of the default fast suite.
"""

from __future__ import annotations

import csv

import pytest

from .conftest import requires_mpiexec, sampling_output, stage_sample_files, write_driver

pytestmark = [pytest.mark.mpi, requires_mpiexec]

DRIVER_BODY = """
    conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                  use_collector=False, no_solvers=1,
                                  save_snapshots_to_file={save_snapshots},
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=solver_module.FixedSample([-0.5, 0.0]))
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations={max_evaluations})]
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                               solver=solver_spec({solver!r}))
    sam = problem.run_sampling(conf, stages)
    print("DRIVER FINISHED", flush=True)
    """


def _raw_data_rows(output_dir, stage_name: str, rank: int = 0) -> list[dict]:
    """Rows of ``raw_data/<stage_name>/rank%04d.csv`` (format v2, header row included)."""
    path = sampling_output(output_dir) / "raw_data" / stage_name / f"rank{rank:04d}.csv"
    assert path.is_file(), f"missing {path}"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("returns_tag, solver", [
    (False, "LinearSolver"),
    (True, "TaggedLinearSolver"),
])
def test_i7_observation_transport(mpi_workspace, run_mpi, returns_tag, solver):
    """
    I7 (finding 2.4): the single pickled observation transport, for both solver conventions.

    The fixed initial sample keeps the chain deterministic. For ``LinearSolver`` (bare array)
    the child synthesizes ``solver_tag=0``; ``TaggedLinearSolver`` returns ``(obs, 0)`` itself.
    Either way the pool answers the sampler with a pickled ``[observations, solver_tag]``.
    """
    name = f"i7_{int(returns_tag)}_{solver}"
    driver = write_driver(mpi_workspace, name,
                          DRIVER_BODY.format(solver=solver,
                                             save_snapshots=False, max_evaluations=20))

    result = run_mpi(driver, n=2, timeout=120)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    assert [f.name for f in stage_sample_files(mpi_workspace / f"out_{name}", "alg0000_MH")] == ["rank0000.csv"]


def test_i7_negative_solver_tag_completes_and_rejects(mpi_workspace, run_mpi):
    """
    I7, the cell that used to crash (finding 2.4, fixed by the WS8 removal of the raw path).

    ``TagFailingSolver`` returns ``tag=-1`` for every proposal with a positive first parameter.
    With the removed raw-buffer transport that tag was used as the MPI tag of the child's
    ``Send``, which is invalid for a negative value: the child died with
    ``mpi4py.MPI.Exception: Invalid tag`` and aborted the whole job (before the WS8 fail-loud
    change it hung instead). Now the tag only ever travels inside the pickled payload, so:

    * the run completes with exit code 0, and
    * every failed proposal is rejected by the sampler (``_compute_log_posterior_terms``
      returns ``-inf`` for ``solver_tag < 0``, ``modules/algorithms.py``) and is not forwarded
      to the collector.
    """
    driver = write_driver(mpi_workspace, "i7_negative_tag",
                          DRIVER_BODY.format(solver="TagFailingSolver",
                                             save_snapshots=True, max_evaluations=60))

    result = run_mpi(driver, n=2, timeout=120)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()

    output_dir = mpi_workspace / "out_i7_negative_tag"
    assert [f.name for f in stage_sample_files(output_dir, "alg0000_MH")] == ["rank0000.csv"]

    rows = _raw_data_rows(output_dir, "alg0000_MH")
    failed = [row for row in rows if int(float(row["solver_tag"])) < 0]
    assert failed, "the failing solver never reported a negative tag - the test no longer exercises 2.4"
    assert all(row["state_type"] == "rejected" for row in failed), \
        [(row["state_type"], row["solver_tag"]) for row in failed]
    # the failing solver only fails for par_0 > 0, so no accepted state may have one
    accepted = [row for row in rows if row["state_type"] == "accepted"]
    assert all(float(row["par_0"]) <= 0.0 for row in accepted), \
        [row["par_0"] for row in accepted]
