#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback
import warnings
from typing import Any, Callable, List, Literal
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

import surrDAMH.process_COLLECTOR
import surrDAMH.process_SAMPLER
import surrDAMH.process_SOLVER
from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules.communication import (ABORT_GRACE_SECONDS,
                                            check_configuration_consistency,
                                            check_tag_upper_bound)
from surrDAMH.modules.surrogate_restart import SurrogateRestart
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.modules.torch_threads import apply_torch_threads
from surrDAMH.solver_specification import SolverSpec
from surrDAMH.solvers import Solver, get_solver_from_spec
from surrDAMH.stages import Stage, stage_name
from surrDAMH.surrogates.parent import (Evaluator, Updater,
                                        apply_output_normalization_from_likelihood)
from surrDAMH.modules.test_data import TestData


def identity(sample):
    return sample


def _insert_best_fit_visualization_note(html_file_path: str, pool_mode_note: str | None,
                                        image_filenames: list[str]) -> None:
    """
    Finding 2.7: ``html_report_extended`` has no section for the best-fit solver
    visualization (its PNG files are only ever saved to disk by the caller, never
    embedded), so pool mode silently dropped it with no mention anywhere. Adds a small
    "Best-fit Solver Visualization" block right before ``</body>`` of the already-written
    report: ``pool_mode_note`` when the run had no live Solver on the reporting rank,
    embedded images (same directory, plain relative ``src``) when some were produced,
    or a neutral "none generated" note otherwise. Purely additive: every other section of
    the file, written by ``html_report_extended`` itself, is left untouched.
    """
    from html import escape
    if pool_mode_note:
        body = f'        <p class="description" style="color: orange;">{escape(pool_mode_note)}</p>'
    elif image_filenames:
        body = "\n".join(f'        <img src="{escape(name)}" alt="Best-fit solver visualization">'
                         for name in image_filenames)
    else:
        body = '        <p class="description">No best-fit solver visualization was generated for this report.</p>'
    # a collapsed <details> block like every other section of the report (2026-09-21)
    section = (
        '    <details class="section" id="best_fit_visualization">\n'
        '        <summary><h2>Best-fit Solver Visualization</h2></summary>\n'
        '        <p class="description">Solver-produced visualization(s) of the best-fit sample '
        '(see Best-fit Analysis above), from Solver.visualize_solution().</p>\n'
        f'{body}\n'
        '    </details>\n'
    )
    with open(html_file_path, "r") as f:
        html = f.read()
    if "</body>" in html:
        html = html.replace("</body>", section + "</body>", 1)
    else:
        html += section
    with open(html_file_path, "w") as f:
        f.write(html)


class SamplingFramework:
    """
    Entry point of the MPI-based sampler: construct one instance identically on every
    rank of ``MPI.COMM_WORLD`` (except spawned solver children, which never see this
    class), then call ``run()``. ``rank_world`` decides which role
    (sampler/collector/solvers pool) each instance actually plays; see
    ``docs/running.md`` for the process-count table.

    The forward model must be reachable on every rank that may need it:
    - if ``use_solvers_pool=True``, ``solver_spec`` must be given (the pool rank loads
      it via ``get_solver_from_spec`` and spawns ``conf.no_solvers`` children from it);
    - if ``use_solvers_pool=False``, either ``solver_spec`` (loaded once per sampler
      rank) or ``solver_instance`` (an already-constructed solver, reused as-is) may be
      given.
    """

    def __init__(self, conf: Configuration, prior: Distribution, likelihood: Distribution,
                 list_of_stages: List[Stage], solver_spec: SolverSpec | None = None, solver_instance: Solver | None = None,
                 surrogate_updater: Updater | None = None, surrogate_evaluator: Evaluator | None = None,
                 initial_snapshots: List[npt.NDArray] | None = None,
                 surrogate_test_data: TestData | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray] | None = None,
                 surrogate_restart: SurrogateRestart | None = None):
        """
        Args:
            conf: run configuration; must be identical on every rank -- its
                posterior-affecting fields are checked against rank 0's in ``run()``.
            prior: prior distribution (internal space, see ``docs/concepts.md``).
            likelihood: likelihood distribution, evaluated on solver/surrogate output.
            list_of_stages: sampling stages, run in order on every sampler rank.
            solver_spec: how to construct the forward-model solver; required when
                ``conf.use_solvers_pool=True``, optional otherwise (see class docstring).
            solver_instance: a pre-built solver, used only when ``use_solvers_pool=False``;
                mutually substitutable with ``solver_spec`` in that case.
            surrogate_updater: trains the surrogate on the collector rank; required for
                any DAMH stage or Hamiltonian-family proposal when ``use_collector=True``.
            surrogate_evaluator: initial/fixed surrogate evaluator, used directly by
                samplers when ``use_collector=False`` (no in-run retraining).
            initial_snapshots: ``(parameters, observations, multiplicity)`` arrays to
                preload into the collector before sampling starts, so a first
                DAMH/Hamiltonian stage does not need to wait for samplers to generate them
                (see ``docs/running.md`` on the start-up handshake). Defaults to
                ``surrogate_updater.get_initial_snapshots()`` if not given.
            surrogate_test_data: fixed test set for surrogate-quality monitoring
                (``surrogate_quality_test.csv``); either a ``TestData`` instance (its
                posterior weights are computed here if missing) or the raw
                ``(parameters, observations, log_posterior, weights)`` tuple.
            surrogate_restart: where to restore the surrogate from, see
                ``surrDAMH.modules.surrogate_restart.SurrogateRestart``. Applied on the
                collector rank only, immediately before ``run_COLLECTOR``; the restored
                snapshots become ``initial_snapshots`` unless the updater reports them
                itself via ``get_initial_snapshots()`` or ``initial_snapshots`` was given
                explicitly.

        Notes:
            ``run()`` checks the posterior-affecting fields of ``conf`` across ranks
            (finding 2.9, see ``communication.check_configuration_consistency``); the rest
            -- the stage list, prior, likelihood, surrogate objects -- is still neither
            broadcast nor validated, and an inconsistency there is not detected until it
            causes a mismatched collective call.
        """
        self.conf = conf
        # Snapshotted by Configuration.__post_init__ (together with every other posterior-affecting
        # field, for the cross-rank check below), i.e. before _configure_surrogate_gradients() can
        # flip conf.use_surrogate_gradients, so the run manifest can record both the requested and
        # the effective value:
        self._use_surrogate_gradients_requested = conf.use_surrogate_gradients_requested
        self.prior = prior
        self.likelihood = likelihood
        self.solver_spec = solver_spec
        if solver_spec is not None and hasattr(solver_spec, "resolve_module_path"):
            # M18/WS5: make the solver module path absolute here, on the launching rank, while
            # the launching working directory is still in effect -- process_SOLVER broadcasts
            # this very object to the spawned children, which do not inherit sys.path and are
            # not guaranteed to inherit the working directory.
            solver_spec.resolve_module_path()
        self.list_of_stages = list_of_stages
        self.surrogate_updater = surrogate_updater
        self.surrogate_evaluator = surrogate_evaluator
        self.solver_instance = solver_instance
        self.initial_snapshots = initial_snapshots
        self.surrogate_test_data = surrogate_test_data
        self.surrogate_restart = surrogate_restart

        self.comm_world = MPI.COMM_WORLD
        self.rank_world = self.comm_world.Get_rank()

        # WS6: an updater configured with output_normalization="likelihood" gets its
        # per-observation statistics (observed data / noise sd) from the likelihood here,
        # once, on every rank that holds an updater (the collector trains it; samplers only
        # keep the reference). A likelihood that cannot supply them warns and leaves the
        # updater at the identity normalization.
        apply_output_normalization_from_likelihood(self.surrogate_updater, self.likelihood)

    def _configure_surrogate_gradients(self) -> None:
        should_warn = self.rank_world == 0

        if self.surrogate_updater is not None:
            self.surrogate_updater.set_use_gradients(self.conf.use_surrogate_gradients)
        if self.surrogate_evaluator is not None and hasattr(self.surrogate_evaluator, "set_use_gradients"):
            self.surrogate_evaluator.set_use_gradients(self.conf.use_surrogate_gradients)

        if not self.conf.use_surrogate_gradients:
            return

        if self.conf.transform_before_surrogate:
            if should_warn:
                warnings.warn(
                    "Surrogate gradients require transform_before_surrogate=False. "
                    "Disabling use_surrogate_gradients. The combination with "
                    "transform_before_surrogate=True is deprecated for gradient-based surrogate use.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.conf.use_surrogate_gradients = False
        elif self.surrogate_updater is not None and not self.surrogate_updater.supports_gradients():
            if should_warn:
                warnings.warn(
                    f"Surrogate updater {type(self.surrogate_updater).__name__} does not implement gradients. "
                    "Disabling use_surrogate_gradients.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.conf.use_surrogate_gradients = False
        elif self.surrogate_evaluator is not None and not self.surrogate_evaluator.supports_gradients():
            if should_warn:
                warnings.warn(
                    f"Surrogate evaluator {type(self.surrogate_evaluator).__name__} does not implement gradients. "
                    "Disabling use_surrogate_gradients.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.conf.use_surrogate_gradients = False

        if self.surrogate_updater is not None:
            self.surrogate_updater.set_use_gradients(self.conf.use_surrogate_gradients)
        if self.surrogate_evaluator is not None and hasattr(self.surrogate_evaluator, "set_use_gradients"):
            self.surrogate_evaluator.set_use_gradients(self.conf.use_surrogate_gradients)

    def _run_role(self, role_callable: Callable[[], Any], role_name: str):
        """
        Run one MPI role body and turn any uncaught exception into a job-wide abort
        (WS8, ``library_notes/09_improvement_plan.md``).

        Without this, an exception on a single rank terminates only that rank; every other
        rank stays blocked in a matching MPI call and the job has to be killed by the user
        or the batch system. (Measured: a solver raising inside ``run_SAMPLER`` under plain
        ``mpiexec python driver.py`` hangs forever. Runs launched with ``python -m mpi4py``
        already abort, because ``mpi4py.run`` installs its own excepthook - that hook does
        not cover ``mpiexec python ...`` nor the spawned solver children.)

        ``KeyboardInterrupt`` and ``SystemExit`` are deliberately re-raised untouched: they
        are not failures of the role body but a requested interruption / exit, mpiexec
        forwards the interrupt signal to every rank itself, and converting them into
        ``Abort(1)`` would only hide why the job stopped.
        """
        try:
            return role_callable()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            print(f"FATAL: unhandled exception on MPI rank {self.rank_world} ({role_name} role);"
                  " aborting the whole job.", file=sys.stderr, flush=True)
            traceback.print_exc()
            sys.stderr.flush()
            sys.stdout.flush()
            time.sleep(ABORT_GRACE_SECONDS)  # let the launcher forward the traceback before it kills the job
            MPI.COMM_WORLD.Abort(1)
            raise  # not reached (Abort does not return), kept so the exception is never swallowed

    def run(self):
        """
        Dispatch this rank to its role (sampler / solvers pool / collector) and run it
        to completion; must be called on every rank of ``MPI.COMM_WORLD`` (it is a
        collective operation: it starts with the cross-rank configuration check below and
        role dispatch ends in a ``comm_world.Barrier()``).

        Before anything else, ``communication.check_configuration_consistency`` compares the
        requested values of ``Configuration.POSTERIOR_AFFECTING_FIELDS`` against rank 0's
        (finding 2.9) and aborts the whole job if a rank was built with a different one.

        On rank 0, also writes ``sampling_output/run_manifest.json`` before dispatch and
        finalizes it (``finished_at`` + per-stage counters) after the sampler role
        returns; manifest failures are printed as warnings, never raised (writing the
        manifest must not be able to abort an otherwise-successful run).

        Returns:
            The sampler role's return value (currently unused) on sampler ranks;
            ``None`` (via ``_run_role``'s wrapped callables) is typical. Every rank
            returns *something* only because ``optional_output`` is always assigned
            before the trailing ``return`` — callers should not rely on its value.

        Raises:
            Nothing directly: any exception raised inside a role body is caught by
            ``_run_role``, printed with a traceback, and turned into
            ``MPI.COMM_WORLD.Abort(1)`` for the whole job (``KeyboardInterrupt`` and
            ``SystemExit`` are re-raised instead of being turned into an abort).
        """
        # Cross-rank configuration check (finding 2.9), first thing on every rank: rank 0
        # broadcasts the *requested* (as-constructed, pre-mutation) values of
        # Configuration.POSTERIOR_AFFECTING_FIELDS and every rank asserts equality. Collective,
        # and it raises on every rank at once; _run_role turns that into a job-wide Abort with a
        # traceback, like any other fatal error in a role body.
        self._run_role(lambda: check_configuration_consistency(self.conf), "CONFIGURATION CHECK")

        # Configuration.torch_threads (WS4/2026-09-18): on every rank, before anything
        # evaluates/trains a network and before the run manifest is built below (so
        # manifest["environment"]["torch_num_threads"] records the effective value).
        apply_torch_threads(self.conf)

        self._configure_surrogate_gradients()

        # check if prior has the "transform" method:
        if not hasattr(self.prior, "transform"):
            self.prior.transform = identity

        # stage names are the output-directory keys; assign them here (identically on every rank,
        # the sampler re-derives the same names) so the run manifest records them:
        for i, stage in enumerate(self.list_of_stages):
            stage.name = stage_name(stage, i)

        if self.rank_world == 0:
            # effective settings, once, on rank 0 (WS5): everything that was silently corrected
            # by __post_init__ or by _configure_surrogate_gradients above is visible here.
            print(self.conf.describe(
                use_surrogate_gradients_requested=self._use_surrogate_gradients_requested), flush=True)
            for i, stage in enumerate(self.list_of_stages):
                print(stage.describe(i), flush=True)
            if self.surrogate_restart is not None:
                print(f"  {self.surrogate_restart.describe()}", flush=True)

            # start-up diagnostic (finding 2.11): MPI tags grow by one per full-model evaluation
            # and are never reused, while MPI_TAG_UB is only guaranteed to be >= 32767. Warns on
            # rank 0 only; never raises and never changes control flow.
            check_tag_upper_bound(self.list_of_stages)

        if self.rank_world == 0:
            # run manifest (WS4, library_notes/09_improvement_plan.md §0 principle 4): must never
            # abort a run, so any failure here is a printed warning, not an exception.
            try:
                from surrDAMH.modules.manifest import build_run_manifest, write_run_manifest
                mpi_layout = {
                    "size_world": self.comm_world.Get_size(),
                    "no_samplers": self.conf.no_samplers,
                    "rank_collector": self.conf.rank_collector,
                    "rank_solvers_pool": self.conf.rank_solvers_pool,
                    "no_solvers": self.conf.no_solvers,
                    "solver_maxprocs": self.conf.solver_maxprocs,
                }
                manifest = build_run_manifest(
                    self.conf, self.list_of_stages, self.prior, self.likelihood, runner="mpi",
                    solver_spec=self.solver_spec, solver_instance=self.solver_instance,
                    surrogate_updater=self.surrogate_updater, surrogate_evaluator=self.surrogate_evaluator,
                    mpi_layout=mpi_layout,
                    use_surrogate_gradients_requested=self._use_surrogate_gradients_requested)
                write_run_manifest(self.conf.output_dir, manifest)
            except Exception as exc:
                print(f"WARNING: failed to write run manifest: {exc}", flush=True)

        if self.rank_world == self.conf.rank_solvers_pool:
            def _solver_pool_role():
                assert self.solver_spec is not None, "solver_spec must be given"
                return surrDAMH.process_SOLVER.run_SOLVER(self.conf, self.solver_spec)

            optional_output = self._run_role(_solver_pool_role, "SOLVER")
        elif self.rank_world == self.conf.rank_collector:
            def _collector_role():
                assert self.surrogate_updater is not None
                if isinstance(self.surrogate_test_data, TestData):
                    td = self.surrogate_test_data
                    if td.log_posterior is None or td.weights is None:
                        td.compute_log_posterior_and_weights(self.prior, self.likelihood)
                    self.surrogate_test_data = td.as_surrogate_test_data()
                # surrogate restart (WS5): the collector rank is the only one that owns an
                # Updater, so this is where a previous run's state is restored.
                restored_snapshots = None
                if self.surrogate_restart is not None:
                    restored_snapshots = self.surrogate_restart.apply(self.surrogate_updater)
                initial_snapshots = self.initial_snapshots
                if initial_snapshots is None:
                    # an updater that stored the restored snapshots itself reports them here
                    # (and sets training_data_loaded, so run_COLLECTOR does not re-add them)
                    initial_snapshots = self.surrogate_updater.get_initial_snapshots()
                if initial_snapshots is None:
                    initial_snapshots = restored_snapshots

                return surrDAMH.process_COLLECTOR.run_COLLECTOR(
                    self.conf,
                    surrogate_updater=self.surrogate_updater,
                    initial_snapshots=initial_snapshots,
                    surrogate_test_data=self.surrogate_test_data,
                )

            optional_output = self._run_role(_collector_role, "COLLECTOR")
        else:
            def _sampler_role():
                if self.conf.use_solvers_pool is False:
                    if self.solver_instance is None:
                        assert self.solver_spec is not None, "either solver_spec or solver_instance must be given"
                        solver_output_dir = ensure_dir(os.path.join(self.conf.output_dir, "solver_output", "rank{}".format(self.rank_world)))
                        self.solver_instance = get_solver_from_spec(self.solver_spec, solver_id=self.rank_world, solver_output_dir=solver_output_dir)
                else:
                    self.solver_instance = None
                return surrDAMH.process_SAMPLER.run_SAMPLER(
                    self.conf, self.prior, self.likelihood, self.list_of_stages,
                    solver_instance=self.solver_instance, surrogate_evaluator=self.surrogate_evaluator)

            optional_output = self._run_role(_sampler_role, "SAMPLER")

            if self.rank_world == 0:
                # rank 0 is always a sampler rank (ranks 0..no_samplers-1); finalize here is the
                # natural end point for it, with no extra MPI communication or barrier.
                try:
                    from surrDAMH.modules.manifest import finalize_run_manifest
                    finalize_run_manifest(self.conf.output_dir, extra={})
                except Exception as exc:
                    print(f"WARNING: failed to finalize run manifest: {exc}", flush=True)

        self.comm_world.Barrier()

        return optional_output


    def write_report(self, stages_to_disp: list[int | str] | None = None,
                     observations: np.ndarray | None = None,
                     par_names: list[str] | None = None, 
                     bins1d: int = 20, bins2d: int = 20, 
                     no_best_fits: int = 10,
                     ranking_mode: Literal["l2", "posterior", "likelihood"] = "l2",
                     parameters_to_disp: list[int] | None = None,
                     observations_to_disp: np.ndarray | None = None,
                     include_expensive_sections = True,
                     grid = None, grid_interp = None, obs_grid = None,
                     no_sensors = None, cmap = "viridis_r", chains_to_disp = None,
                     field_statistics_max_samples: int | None = None) -> surrDAMH.post_processing.Samples | None:
        """
        Writes a HTML report and ``post_processing_output/summary.csv`` to the output
        directory. Reads samples from ``conf.output_dir`` on disk (output format v2,
        see ``docs/outputs.md``), it does not use any in-memory state from ``run()``.

        Rank-0-only: every other rank only participates in the two barriers (call this
        on every rank, like ``run()``, or the collective will hang). Report generation on
        rank 0 runs inside ``_run_role``, so a failure there prints a traceback and aborts
        the whole job instead of leaving the other ranks blocked in the trailing barrier
        (WS9b).

        Args:
            stages_to_disp: list of stage indices and/or stage names (as produced by
                ``surrDAMH.stages.stage_name()``, e.g. ``"alg0000_MH"``) to include in the
                report; may mix both. If None, all stages are included.
            observations: reference observations, if None, no reference observations are used
            par_names: list of parameter names, if None, default names are used
            bins1d: number of bins for 1D histograms
            bins2d: number of bins for 2D histograms
            no_best_fits: number of best fit samples to display
            ranking_mode: how to rank the best fit samples, one of "l2", "posterior", "likelihood"
            parameters_to_disp: list of parameter indices to include in the report, if None, first 10 parameters are included
            observations_to_disp: list of observation indices to include in the report, if None, all observations are included
            include_expensive_sections: whether to include sections that are expensive to compute
            grid: grid for 2D histograms, if None, a default grid is used
            field_statistics_max_samples: the posterior field statistics (see Notes) call the
                solver once per decompressed posterior state; with ``None`` every state is used,
                which for a fast solver and long chains can take longer than the sampling itself
                (measured: 6e6 states -> 37 min). An integer draws that many states at random
                (fixed seed) instead; the statistics then carry Monte-Carlo error of order
                ``1/sqrt(field_statistics_max_samples)`` relative to the posterior spread.

        Returns:
            ``surrDAMH.post_processing.Samples`` on rank 0 (already used to write the
            report/summary); ``None`` on every other rank.

        Raises:
            ValueError: if ``stages_to_disp`` resolves to an empty list, or contains a
                stage name not present in this run. This (like every other failure of
                report generation) is turned into ``MPI.COMM_WORLD.Abort(1)`` with a
                printed traceback by ``_run_role``.

        Notes:
            If ``solver_instance`` was given to ``__init__`` and exposes
            ``field_builder``/``coords``/``measurement_points``, posterior field
            statistics are added to the report; the best-fit sample is also
            re-evaluated with the solver for ``visualize_solution()`` figures (solver
            errors here are caught and only printed, they do not fail the report).

            If ``conf.use_solvers_pool=True``, rank 0 (the reporting rank) never holds a
            live ``Solver`` instance (it lives in the spawned solver pool instead): a
            ``RuntimeWarning``-style message is printed here, and the report marks the
            parameter-names, posterior-field-statistics and best-fit-solver-visualization
            sections "Not available" instead of silently omitting them (finding 2.7).
        """

        if self.rank_world != 0:
            self.comm_world.Barrier()
            return None

        # Every other rank is already waiting in the barrier at the end of this method, so an
        # exception here would hang the job; _run_role turns it into a job-wide abort instead.
        samples = self._run_role(
            lambda: self._write_report_rank0(
                stages_to_disp=stages_to_disp, observations=observations, par_names=par_names,
                bins1d=bins1d, bins2d=bins2d, no_best_fits=no_best_fits, ranking_mode=ranking_mode,
                parameters_to_disp=parameters_to_disp, observations_to_disp=observations_to_disp,
                include_expensive_sections=include_expensive_sections, grid=grid,
                grid_interp=grid_interp, obs_grid=obs_grid, no_sensors=no_sensors, cmap=cmap,
                chains_to_disp=chains_to_disp, field_statistics_max_samples=field_statistics_max_samples),
            "REPORT")

        self.comm_world.Barrier()
        return samples

    def _write_report_rank0(self, stages_to_disp, observations, par_names, bins1d, bins2d,
                            no_best_fits, ranking_mode, parameters_to_disp, observations_to_disp,
                            include_expensive_sections, grid, grid_interp, obs_grid, no_sensors,
                            cmap, chains_to_disp, field_statistics_max_samples=None) -> surrDAMH.post_processing.Samples:
        """Body of :meth:`write_report`, run on rank 0 only (see that method for the arguments)."""
        # Finding 2.7: use_solvers_pool=True means the actual Solver lives in the spawned
        # solver pool, never on rank 0 (the reporting rank); self.solver_instance is then
        # always None here, so the sections below that need a live Solver silently had
        # nothing to show. Warn now (stdout) and pass a note through so the report itself
        # says so too, instead of just omitting those sections.
        pool_mode_note = None
        if self.conf.use_solvers_pool and self.solver_instance is None:
            pool_mode_note = ("Not available: pool mode (`use_solvers_pool=True`) has no live "
                              "Solver object on the reporting rank.")
            print(f"RuntimeWarning: {pool_mode_note} Affected report sections: parameter "
                  "names, posterior field statistics, best-fit solver visualization.",
                  flush=True)

        samples = surrDAMH.post_processing.Samples(self.conf.no_parameters, self.conf.output_dir)
        # WS9 bullet 5: stages_to_disp accepts stage NAMES (as produced by
        # surrDAMH.stages.stage_name()) alongside the existing positional indices;
        # Samples._resolve_stages does the lookup (against this run's own stage names, read
        # from output_dir) and also covers the pre-existing None-means-"all stages" default,
        # so an int-only or None caller takes exactly the old path.
        stages_to_disp = samples._resolve_stages(stages_to_disp)

        if par_names is None:
            par_names = getattr(self.solver_instance, "par_names", None)
        if parameters_to_disp is None:
            parameters_to_disp = list(range(min(self.conf.no_parameters, 10)))
        if not stages_to_disp:
            raise ValueError("No report stages are available.")

        post_processing_dir_path = os.path.join(self.conf.output_dir, "post_processing_output")
        ensure_dir(post_processing_dir_path)

        samples.calculate_CpUS([[i] for i in stages_to_disp])
        setattr(samples, "summary", samples.summary.iloc[stages_to_disp, :].copy())
        samples.get_summary().to_csv(os.path.join(post_processing_dir_path, "summary.csv"))

        output_html_file_path = os.path.join(post_processing_dir_path, "report_extended.html")

        if self.solver_instance and hasattr(self.solver_instance, 'field_builder') and hasattr(self.solver_instance, 'coords') and hasattr(self.solver_instance, 'measurement_points'):
            field_mean, field_std = samples.compute_posterior_field_statistics(
                self.solver_instance.field_builder, n_max_samples=field_statistics_max_samples)
            observation_field_mean, observation_field_std = samples.compute_posterior_field_statistics(
                self.solver_instance.set_parameters_and_get_observations, n_max_samples=field_statistics_max_samples)
            field_statistics = [
                {
                    "mean": field_mean,
                    "std": field_std,
                    "coordinates": self.solver_instance.coords,
                    "name": "Posterior diffusion coefficient field",
                },
                {
                    "mean": observation_field_mean,
                    "std": observation_field_std,
                    "coordinates": self.solver_instance.measurement_points[:, :2],
                    "name": "Posterior solution at measurement points",
                },
            ]
        else:
            field_statistics = None

        samples.html_report_extended(
            no_observations=self.conf.no_observations,
            stages_to_disp=stages_to_disp,
            observations=observations,
            output_file=output_html_file_path,
            bins1d=bins1d,
            bins2d=bins2d,
            parameters_to_disp=parameters_to_disp,
            prior=self.prior,
            no_best_fits=no_best_fits,
            include_expensive_sections=include_expensive_sections,
            field_statistics=field_statistics,
            configuration=self.conf,
            ranking_mode=ranking_mode,
            par_names=par_names,
            observations_to_disp=observations_to_disp,
            grid=grid,
            grid_interp=grid_interp,
            obs_grid=obs_grid,
            no_sensors=no_sensors,
            cmap=cmap,
            chains_to_disp=chains_to_disp,
            pool_mode_note=pool_mode_note,
            stages=self.list_of_stages,  # the "Sampling Stages" section (else the manifest copy)
        )

        # Finding 2.7: best-fit solver visualization figures are saved as separate PNG
        # files next to report_extended.html (below, unchanged); this note/image list is
        # inserted into the already-written HTML so the report says what happened instead
        # of leaving the gap silent -- additive only, no existing section is touched.
        best_fit_visualization_images: list[str] = []
        if self.solver_instance is not None:
            best_fit_parameters = np.asarray(getattr(samples, "best_fit_parameters", []), dtype=float)
            if best_fit_parameters.size:
                if best_fit_parameters.ndim == 1:
                    best_fit_parameters = best_fit_parameters.reshape(1, -1)
                self.solver_instance.set_parameters(best_fit_parameters[0])
                self.solver_instance.get_observations()
                try:
                    visualizations = self.solver_instance.visualize_solution(show=False)
                except Exception as exc:
                    print(f"Solver visualization skipped for best fit: {exc}", flush=True)
                    visualizations = []
                for fig_idx, (fig, _) in enumerate(visualizations, start=1):
                    fig_name = f"best_fit_solver_visualization_{fig_idx}.png"
                    fig_path = os.path.join(post_processing_dir_path, fig_name)
                    try:
                        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
                        best_fit_visualization_images.append(fig_name)
                    finally:
                        plt.close(fig)

        _insert_best_fit_visualization_note(
            output_html_file_path, pool_mode_note=pool_mode_note,
            image_filenames=best_fit_visualization_images)

        return samples
