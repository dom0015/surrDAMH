#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-rank protocol tests for ``surrDAMH.modules.communication`` (item I12 of
``library_notes/07_testing_plan.md`` Phase 3).

Unlike the other ``tests/mpi`` tests these do not run a sampling framework: they drive the
sampler-side and collector-side evaluator communicators against each other directly, rank 0
playing the sampler and rank 1 the collector, so a single message-bookkeeping rule can be
asserted on its own.

Covered so far: finding 2.10's sampler-side half — ``CommEvaluator_sampler
.get_evaluator_and_terminate()`` must complete (``Wait``) the ``TAG_UPDATE`` ``Isend`` posted
by the last ``request_evaluator()`` instead of dropping the reference to a still-pending
request. The assertion is that the request handle is ``MPI.REQUEST_NULL`` afterwards, which
mpi4py sets exactly when ``Wait``/``Test`` completed a non-persistent request; an abandoned
request is still a live handle.
"""

from __future__ import annotations

import pytest

from .conftest import requires_mpiexec, write_driver

pytestmark = [pytest.mark.mpi, requires_mpiexec]

# rank 0 = sampler side, rank 1 = collector side. The sampler asks twice but the collector
# answers only the first request, so at shutdown the second TAG_UPDATE Isend is still the
# "pending" one whose fate this test is about (collector-side current_idx != max_idx, i.e. the
# branch of CommEvaluator_collector.terminate() that still owes a final evaluator message).
DRIVER_BODY = """
    from mpi4py import MPI

    from surrDAMH.modules.communication import (CommEvaluator_collector,
                                                CommEvaluator_sampler)

    RANK = MPI.COMM_WORLD.Get_rank()
    assert MPI.COMM_WORLD.Get_size() == 2, "this driver needs exactly 2 ranks"

    if RANK == 0:  # sampler side
        comm_sampler = CommEvaluator_sampler(rank_collector=1, max_buffer_size=1 << 16)

        comm_sampler.request_evaluator()          # request #1, answered by the collector
        first = comm_sampler.get_evaluator()
        assert first == "evaluator-1", first

        comm_sampler.request_evaluator()          # request #2, deliberately left unanswered
        pending = comm_sampler.request_Isend      # keep our own reference to the MPI request
        assert pending != MPI.REQUEST_NULL, "request_evaluator() did not post an Isend"

        last = comm_sampler.get_evaluator_and_terminate()
        assert last == "evaluator-last", last
        assert pending == MPI.REQUEST_NULL, (
            "the pending TAG_UPDATE Isend was abandoned instead of being completed (finding 2.10)")
        print("PROTOCOL OK", flush=True)

    else:  # collector side
        comm_collector = CommEvaluator_collector(rank_sampler=0)

        while not comm_collector.sampler_requests_evaluator():
            pass
        comm_collector.send_evaluator("evaluator-1")

        # deliberately stop consuming TAG_UPDATE here, so the sampler's second request is
        # still outstanding when it terminates
        while not comm_collector.sampler_stops():
            pass
        comm_collector.terminate("evaluator-last")

    MPI.COMM_WORLD.Barrier()
    """


def test_i12_sampler_side_update_isend_is_completed(mpi_workspace, run_mpi):
    """Finding 2.10 (sampler half): the last TAG_UPDATE Isend is waited on, not abandoned."""
    driver = write_driver(mpi_workspace, "protocol_evaluator_isend", DRIVER_BODY)
    result = run_mpi(driver, n=2, timeout=60.0)
    assert not result.timed_out, f"the two-rank protocol driver hung\n{result.describe()}"
    assert result.returncode == 0, result.describe()
    assert "PROTOCOL OK" in result.stdout, result.describe()
