#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI integration tests for the deadlock / abort regressions.

Items I3, I4, I6 of ``library_notes/07_testing_plan.md`` Phase 3, plus the WS8 fail-loud tests.

I3 asserts a WS0 fix (B1), I4 a WS8 fix (finding 2.2). I6 pins that the finding-2.5 buffer-shape
mismatch across sampler ranks cannot happen: first through G2's ``Allreduce`` normalisation
(2026-09-17) and, since 2026-09-20, structurally -- the ranks exchange the adaptation's
sufficient statistics, whose length depends only on ``no_parameters``.

Run them with ``pytest -m mpi``; they are not part of the default fast suite.
"""

from __future__ import annotations

import pytest

from .conftest import requires_mpiexec, stage_sample_files, write_driver

pytestmark = [pytest.mark.mpi, requires_mpiexec]

# process_SAMPLER.py, WS8 fail-fast for a surrogate-using first stage (finding 2.2):
FIRST_STAGE_SURROGATE_ERROR = "The first stage requires a surrogate model"
SOLVER_FAILURE = "deliberate solver failure"  # conftest.FailingSolver


def test_i3_collector_with_unreachable_min_snapshots_terminates(mpi_workspace, run_mpi):
    """
    I3 (regression for WS0 item B1, finding 2.1): ``use_collector=True``, a single MH stage
    and ``min_snapshots_initial=10**9``, i.e. a surrogate that can never be trained.

    Before B1 the collector polled the stop signal only once a surrogate existed, so this run
    hung forever. Asserted here: the job finishes cleanly (exit 0) well inside the timeout and
    the MH samples are written; no surrogate is produced.

    The testing plan wrote "expect a clear error" for this item - post-WS0 the run simply ends
    cleanly without a surrogate and without a diagnostic; emitting a diagnostic is WS8 work.
    """
    driver = write_driver(mpi_workspace, "i3", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=10**9)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i3"

    result = run_mpi(driver, n=3, timeout=60)

    assert not result.timed_out, "B1 regression: the run hung again\n" + result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    assert [f.name for f in stage_sample_files(output_dir, "alg0000_MH")] == ["rank0000.csv"]
    # no surrogate could ever be trained, so no quality metrics are written:
    assert not (output_dir / "sampling_output" / "surrogate_quality.csv").exists()


def test_i3b_damh_stage_before_min_snapshots_initial_trains_instead_of_deadlocking(mpi_workspace, run_mpi):
    """
    2026-09-22: a DAMH stage that starts before ``min_snapshots_initial`` snapshots exist used to
    deadlock -- every sampler blocked in ``get_evaluator()`` for the first surrogate, the collector
    waiting for snapshots only those samplers could produce (found by the GRF scheme study's smoke
    run). Samplers now announce the blocking wait (``TAG_EVALUATOR_NEEDED``); once every active
    sampler has, the collector trains on the snapshots it has and says so.
    """
    driver = write_driver(mpi_workspace, "i3b", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=10**6)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30),
                  Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=20, subchain_length=2)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i3b"

    result = run_mpi(driver, n=4, timeout=90)

    assert not result.timed_out, "deadlock regression: the DAMH stage never got a surrogate\n" + result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    assert "training the initial surrogate on them now instead of deadlocking" in result.stdout, result.describe()
    assert [f.name for f in stage_sample_files(output_dir, "alg0001_DAMH-SMU")] == ["rank0000.csv", "rank0001.csv"]


def test_i4_damh_as_first_stage_fails_fast(mpi_workspace, run_mpi):
    """
    I4 (finding 2.2, WS8): DAMH as the **first** stage with ``use_collector=True`` and no
    ``initial_snapshots``.

    Before WS8 this deadlocked: the sampler blocked in ``get_evaluator()`` while the collector
    waited for the snapshots that only samplers can produce, and the sampler had no way to know
    whether the collector had been handed a pretrained surrogate (WS0 item B2 only printed a
    warning on rank 0). WS8 added a one-message start-up handshake - the collector tells every
    sampler whether it can produce an evaluator before receiving any snapshot - so the sampler
    now raises a ``RuntimeError`` which ``Problem.run_sampling``'s ``core._run_role`` turns into
    an ``MPI_Abort`` of the whole job.

    Asserted: no timeout, nonzero exit, and the new error text on stdout or stderr.
    """
    driver = write_driver(mpi_workspace, "i4", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=4)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        stages = [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=20)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
        print("DRIVER FINISHED", flush=True)
        """)

    result = run_mpi(driver, n=3, timeout=60)

    assert not result.timed_out, ("the DAMH-first-stage deadlock is back (finding 2.2)\n"
                                  + result.describe())
    assert result.returncode not in (0, None), result.describe()
    assert "DRIVER FINISHED" not in result.stdout, result.describe()
    assert FIRST_STAGE_SURROGATE_ERROR in (result.stdout + result.stderr), result.describe()


def test_i4_damh_as_first_stage_with_initial_snapshots_runs(mpi_workspace, run_mpi):
    """
    Positive counterpart of I4 (WS8): the same DAMH first stage, but with ``initial_snapshots``
    holding exactly ``min_snapshots_initial`` rows.

    The collector can then train a surrogate before the first stage starts, so its start-up
    handshake reports "evaluator available", the sampler does not raise, and the run completes.
    This also exercises the ``initial_snapshots`` accounting of finding 2.8: the preloaded rows
    must reach ``min_snapshots_initial`` on their own (they are counted once).
    """
    driver = write_driver(mpi_workspace, "i4_ok", """
        N = 6
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=N)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        rng = np.random.RandomState(0)
        parameters = rng.normal(size=(N, 2))
        observations = np.array([solver_module.A @ p for p in parameters])
        weights = np.ones((N, 1))
        stages = [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=20)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater,
                                   surrogate_initial_training_data=[parameters, observations, weights])
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i4_ok"

    result = run_mpi(driver, n=3, timeout=180)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    files = stage_sample_files(output_dir, "alg0000_DAMH-SMU")
    assert [f.name for f in files] == ["rank0000.csv"], result.describe()
    assert files[0].stat().st_size > 0, result.describe()


def test_solver_exception_in_spawned_child_aborts_the_job(mpi_workspace, run_mpi):
    """
    WS8 fail-loud, solvers-pool variant: the solver runs in an ``MPI_Comm_spawn``-ed child
    (``process_CHILD.py``) and raises on its 5th evaluation.

    Measured before WS8: the child died with a traceback, the solvers-pool rank kept polling
    ``Iprobe`` forever and the sampler waited for observations that never came - the job hung
    until ``run_mpi`` killed the process tree (exit -9 after the full timeout). The child is
    spawned as a plain ``python process_CHILD.py``, so ``mpi4py.run``'s abort-on-exception hook
    does not apply to it.

    Asserted now: no timeout, nonzero exit, and the solver traceback visible in the output.
    """
    driver = write_driver(mpi_workspace, "fail_pool", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=False, no_solvers=1)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec("FailingSolver"))
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """)

    result = run_mpi(driver, n=2, timeout=60)

    assert not result.timed_out, ("a failing solver child must not hang the job\n"
                                  + result.describe())
    assert result.returncode not in (0, None), result.describe()
    assert "DRIVER FINISHED" not in result.stdout, result.describe()
    assert SOLVER_FAILURE in (result.stdout + result.stderr), result.describe()
    assert not result.orphans_killed  # nothing left for run_mpi to kill


def test_solver_exception_on_sampler_rank_aborts_the_job(mpi_workspace, run_mpi):
    """
    WS8 fail-loud, local-solver variant: ``use_solvers_pool=False`` (the solver instance lives on
    the sampler rank) with ``use_collector=True``, so the collector is the rank left waiting.

    Measured before WS8: under ``mpiexec ... python -m mpi4py driver.py`` this already exited 1
    after ~2 s, because ``mpi4py.run`` installs an abort-on-exception hook; under plain
    ``mpiexec python driver.py`` the identical run hung forever (verified manually). The WS8
    wrapper in ``Problem.run_sampling``'s ``core._run_role`` makes the abort a property of the
    library rather than of the launcher, and adds the rank and the role to the message.

    Asserted: no timeout, nonzero exit, the solver traceback, and the library's own FATAL line.
    """
    driver = write_driver(mpi_workspace, "fail_local", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, use_solvers_pool=False, no_solvers=1,
                                      min_snapshots_initial=6)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec("FailingSolver"))
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
        print("DRIVER FINISHED", flush=True)
        """)

    result = run_mpi(driver, n=2, timeout=60)

    assert not result.timed_out, ("a failing local solver must not hang the job\n"
                                  + result.describe())
    assert result.returncode not in (0, None), result.describe()
    assert "DRIVER FINISHED" not in result.stdout, result.describe()
    output = result.stdout + result.stderr
    assert SOLVER_FAILURE in output, result.describe()
    assert "FATAL: unhandled exception on MPI rank 0 (SAMPLER role)" in output, result.describe()
    assert not result.orphans_killed  # nothing left for run_mpi to kill


def test_i6_adaptive_mh_shorter_than_adaptation_period_does_not_break_allreduce(mpi_workspace, run_mpi):
    """
    I6, part 1 (finding 2.5 / decision G2): an adaptive MH stage with
    ``max_evaluations=5 < GaussRandomWalk_adaptive.period`` (default 10) on 2 samplers, i.e. a
    stage that ends before a single adaptation period completes.

    The testing plan predicted an ``Allreduce`` shape mismatch here. **Observed: there is none,
    and this test pins that.** ``Algorithm_MH.run`` iterates exactly
    ``min(max_samples, max_evaluations)`` times on every rank, so every rank calls ``adapt()``
    the same number of times. Since 2026-09-20 the ranks no longer exchange covariances at all
    but the adaptation's sufficient statistics (``adapted_state()``), whose length depends only
    on ``no_parameters`` -- so a shape mismatch is structurally impossible, whatever each rank's
    ``adapt()`` count was; see ``test_i6_damh_adaptive_allreduce_shape_mismatch`` for the case
    where the counts really do differ.
    """
    driver = write_driver(mpi_workspace, "i6_mh", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=False, no_solvers=1)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5, adaptive=True), max_evaluations=5)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """)
    output_dir = mpi_workspace / "out_i6_mh"

    result = run_mpi(driver, n=3, timeout=60)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    assert len(stage_sample_files(output_dir, "alg0000_MH-adaptive")) == 2, result.describe()


def test_i6_adaptive_block_proposal_pools_and_finishes(mpi_workspace, run_mpi):
    """
    I6, part 2 (finding 2.5, second half): a ``Block`` proposal with an adaptive sub-proposal.

    Before G2: ``build_proposal`` returned a ``BlockProposal`` without ``sd_or_cov``/``adapted_state``
    while ``process_SAMPLER`` read them after every adaptive stage -- ``AttributeError`` on every
    sampler, ``MPI_Abort``, exit code 1, after the whole stage had been sampled. G2 (2026-09-17)
    then refused the combination up front with a ``ValueError``.

    Since C2 (2026-10-12) the combination is SUPPORTED: ``BlockProposal`` implements
    ``adapted_state``/``set_pooled_state``/``carry_over``/``adapted_summary`` (the adaptive
    sub-proposal is pooled across the two sampler ranks at stage end), so the job must finish with
    exit code 0 and no ``AttributeError``; a Block hands nothing over (no ``carry_over__scale``),
    its summary keys are ``component<k>.*`` and the trace is the first adaptive sub-proposal's
    (``adaptive_stats_component == 0``).
    """
    driver = write_driver(mpi_workspace, "i6_block", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=False, no_solvers=1)
        stages = [Stage(algorithm="MH", max_evaluations=5,
                        proposal=Block(groups=[[0], [1]],
                                      proposals=[RandomWalk(), RandomWalk(scale=0.5)]))]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages)
        print("DRIVER FINISHED", flush=True)
        """)

    result = run_mpi(driver, n=3, timeout=60)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert result.stdout.count("DRIVER FINISHED") == 3, result.describe()
    output = result.stdout + result.stderr
    assert "AttributeError" not in output and "Traceback" not in output, result.describe()
    assert "adaptive covariance reduction is not defined for block proposals" not in output, result.describe()
    output_dir = mpi_workspace / "out_i6_block"
    assert len(stage_sample_files(output_dir, "alg0000_MH-adaptive")) == 2, result.describe()
    import json
    import numpy as np
    with open(output_dir / "sampling_output" / "run_manifest.json") as f:
        resolution = json.load(f)["proposal_resolution"]
    assert resolution[0]["adaptive_stats_component"] == 0, resolution
    with np.load(output_dir / "sampling_output" / "carry_over" / "alg0000_MH-adaptive.npz") as z:
        keys = set(z.files)
    assert "carry_over__scale" not in keys, keys          # a Block hands nothing over
    assert "summary__component0.log_sigma" in keys, keys  # ... but its adaptive sub-proposal was pooled


def test_i6_damh_adaptive_allreduce_shape_mismatch(mpi_workspace, run_mpi):
    """
    I6, part 3 (finding 2.5, first half): an adaptive DAMH stage on 2 samplers.

    ``Algorithm_DAMH.run`` used to call ``adapt()`` only on iterations whose sub-chain moved
    (since 2026-09-20 it is called on every outer iteration, pre-rejected ones scored as
    acceptance 0), so the number of adaptations differed per rank; it can still differ because
    the outer loop stops on ``max_evaluations`` after a rank-dependent number of iterations.
    When one rank crossed the adaptation period (10)
    and another did not, ``sd_or_cov`` used to be a 2-D covariance on one rank and a 1-D sd
    vector on the other, and the ``Allreduce`` in ``process_SAMPLER`` aborted with
    ``Fatal error in internal_Allreduce_c: Message truncated`` (observed exit code 9). That
    made this test ``xfail(strict=False)`` -- it XPASSed whenever the ranks happened to agree.

    G2 (2026-09-17) fixed it by normalising ``sd_or_cov`` to a 2-D covariance on every rank
    before the reduction. Since 2026-09-20 the reduction is gone: the ranks ``Allgather``
    ``Proposal.adapted_state()``, a vector whose length is fixed by ``no_parameters`` alone, and
    combine the pooled statistics. Either way the shapes can no longer disagree and the run must
    always finish cleanly.
    """
    driver = write_driver(mpi_workspace, "i6_damh", """
        conf = surrDAMH.Configuration(output_dir=OUTPUT_DIR, no_parameters=2, no_observations=2,
                                      use_collector=True, no_solvers=1, min_snapshots_initial=6)
        updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=2, no_observations=2)
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30),
                  Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5, adaptive=True), max_samples=12,
                        subchain_length=3)]
        problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(),
                                   solver=solver_spec())
        sam = problem.run_sampling(conf, stages, surrogate_updater=updater)
        print("DRIVER FINISHED", flush=True)
        """)

    result = run_mpi(driver, n=4, timeout=60)

    assert not result.timed_out, result.describe()
    assert result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    assert "Message truncated" not in (result.stdout + result.stderr), result.describe()
