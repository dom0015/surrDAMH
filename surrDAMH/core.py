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
from surrDAMH.distributions.normal import standardize_prior
from surrDAMH.distributions.parent import Distribution, distribution_dimension
from surrDAMH.modules.communication import (ABORT_GRACE_SECONDS,
                                            check_configuration_consistency,
                                            check_tag_upper_bound)
from surrDAMH.modules.surrogate_restart import SurrogateRestart
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.modules.torch_threads import apply_torch_threads
from surrDAMH.solver_specification import SolverSpec
from surrDAMH.solvers import Solver, get_solver_from_spec
from surrDAMH.stages import Stage, check_stage_list, stage_name, wasted_snapshot_notes
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


def _run_role(role_callable: Callable[[], Any], role_name: str):
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
        print(f"FATAL: unhandled exception on MPI rank {MPI.COMM_WORLD.Get_rank()} ({role_name} role);"
              " aborting the whole job.", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        time.sleep(ABORT_GRACE_SECONDS)  # let the launcher forward the traceback before it kills the job
        MPI.COMM_WORLD.Abort(1)
        raise  # not reached (Abort does not return), kept so the exception is never swallowed


def _configure_surrogate_gradients(conf: Configuration, surrogate_updater: Updater | None,
                                   surrogate_evaluator: Evaluator | None, should_warn: bool) -> None:
    """Disable ``conf.use_surrogate_gradients`` (with a warning) for an incompatible surrogate."""
    if surrogate_updater is not None:
        surrogate_updater.set_use_gradients(conf.use_surrogate_gradients)
    if surrogate_evaluator is not None and hasattr(surrogate_evaluator, "set_use_gradients"):
        surrogate_evaluator.set_use_gradients(conf.use_surrogate_gradients)

    if not conf.use_surrogate_gradients:
        return

    if conf.transform_before_surrogate:
        if should_warn:
            warnings.warn(
                "Surrogate gradients require transform_before_surrogate=False. "
                "Disabling use_surrogate_gradients. The combination with "
                "transform_before_surrogate=True is deprecated for gradient-based surrogate use.",
                RuntimeWarning,
                stacklevel=3,
            )
        conf.use_surrogate_gradients = False
    elif surrogate_updater is not None and not surrogate_updater.supports_gradients():
        if should_warn:
            warnings.warn(
                f"Surrogate updater {type(surrogate_updater).__name__} does not implement gradients. "
                "Disabling use_surrogate_gradients.",
                RuntimeWarning,
                stacklevel=3,
            )
        conf.use_surrogate_gradients = False
    elif surrogate_evaluator is not None and not surrogate_evaluator.supports_gradients():
        if should_warn:
            warnings.warn(
                f"Surrogate evaluator {type(surrogate_evaluator).__name__} does not implement gradients. "
                "Disabling use_surrogate_gradients.",
                RuntimeWarning,
                stacklevel=3,
            )
        conf.use_surrogate_gradients = False

    if surrogate_updater is not None:
        surrogate_updater.set_use_gradients(conf.use_surrogate_gradients)
    if surrogate_evaluator is not None and hasattr(surrogate_evaluator, "set_use_gradients"):
        surrogate_evaluator.set_use_gradients(conf.use_surrogate_gradients)


def _resolve_size(name: str, explicit: int | None, solver_instance: Any, dist: Any,
                  dist_label: str) -> tuple[int | None, str | None]:
    """
    One problem size (``no_parameters`` or ``no_observations``) and the source it came from.

    Sources, first one wins, all present ones must agree: the explicit ``Problem`` argument,
    the solver instance's declared attribute, the distribution's dimension
    (``distribution_dimension``). ``(None, None)`` when no source knows it.
    """
    sources: list[tuple[str, int]] = []
    if explicit is not None:
        sources.append((f"Problem({name}=...)", int(explicit)))
    if solver_instance is not None:
        declared = getattr(solver_instance, name, None)
        if declared is not None:
            sources.append((f"solver {type(solver_instance).__name__}.{name}", int(declared)))
    dimension = distribution_dimension(dist)
    if dimension is not None:
        sources.append((f"{dist_label} ({type(dist).__name__}) dimension", dimension))
    if not sources:
        return None, None
    if len({value for _, value in sources}) > 1:
        raise ValueError(f"inconsistent {name}: " + ", ".join(f"{label} = {value}" for label, value in sources))
    return sources[0][1], sources[0][0]


class Problem:
    """
    The Bayesian inverse problem: prior, likelihood (noise model around the observed data)
    and forward model. Build it identically on every MPI rank, then start a run::

        problem = surrDAMH.Problem(prior, likelihood, solver=my_solver)   # sizes found automatically
        conf = surrDAMH.Configuration(output_dir="out_x", use_solvers_pool=False, use_collector=False)
        stages = [surrDAMH.Stage(max_evaluations=1000)]
        run = problem.run_sampling(conf, stages)         # mpiexec -n 4 python3 -m mpi4py script.py
        run = problem.run_sampling_local(conf, stages)   # or: one chain in one process, no MPI
        run.write_report(observations=observed_data)

    Sizes. ``no_parameters`` is taken from, in this order (all present sources must agree,
    else ``ValueError`` listing them): the explicit argument, the ``no_parameters`` the solver
    instance declares, the prior's dimension. ``no_observations`` likewise from the explicit
    argument, the solver instance, the likelihood's dimension (a ``Normal`` likelihood with a
    vector ``mean`` = the data). A dimension-free ``Normal`` (scalar ``mean``, no ``dim``) is
    broadcast to the size the solver or the explicit argument gives. ``describe()`` says where
    each size came from. ``Configuration.no_parameters``/``no_observations`` may be left out;
    if given they must agree.
    """

    def __init__(self, prior: Distribution, likelihood: Distribution, solver: Solver | SolverSpec,
                 no_parameters: int | None = None, no_observations: int | None = None) -> None:
        """
        Args:
            prior: prior of the parameters, an object from ``surrDAMH.distributions``. The
                chain runs in the prior's *internal* space and ``prior.transform`` maps a
                sample to the physical parameters the solver receives. That internal space is
                the standard normal N(0, I) for every shipped prior: a ``Normal`` is
                standardized automatically (``StandardizedNormal``: ``transform(z) = mean +
                L z``), ``PriorIndependentComponents`` maps each coordinate to its component.
                Consequences: ``Configuration.lhs_scale`` and ``initial_samples_distribution``
                refer to internal (standardized) coordinates, and the surrogate is trained on
                them unless ``transform_before_surrogate=True``. The object as given is kept
                as ``prior_physical``; ``prior`` is the standardized one.
            likelihood: noise model of the data, an object from ``surrDAMH.distributions``
                evaluated on the solver output. Typically
                ``Normal(mean=observed_data, sd=noise_sd)``.
            solver: the forward model, either a ``surrDAMH.Solver`` instance (runs inside every
                sampler process; needs ``Configuration(use_solvers_pool=False)``) or a
                ``surrDAMH.SolverSpec`` (file + class name; the solver is built where it runs:
                in the spawned solvers-pool children with ``use_solvers_pool=True``, in every
                sampler process otherwise).
            no_parameters: explicit number of parameters (optional, see the class docstring).
            no_observations: explicit number of observations (optional).

        Raises:
            ValueError: no forward model; inconsistent sizes; a size no source supplies.
        """
        if solver is None:
            raise ValueError("give the forward model as solver=<a surrDAMH.Solver instance> or solver=SolverSpec(...)")
        self.solver_spec: SolverSpec | None = None
        self.solver_instance: Solver | None = None
        if isinstance(solver, SolverSpec):
            self.solver_spec = solver
            if hasattr(solver, "resolve_module_path"):
                # M18/WS5: make the solver module path absolute here, on the launching rank, while
                # the launching working directory is still in effect -- process_SOLVER broadcasts
                # this very object to the spawned children, which do not inherit sys.path and are
                # not guaranteed to inherit the working directory.
                solver.resolve_module_path()
        else:
            if not (hasattr(solver, "set_parameters") and hasattr(solver, "get_observations")):
                raise TypeError(f"solver must be a surrDAMH.Solver instance or a SolverSpec, got "
                                f"{type(solver).__name__} (no set_parameters/get_observations)")
            self.solver_instance = solver

        self.no_parameters, self.no_parameters_source = self._resolve(
            "no_parameters", no_parameters, prior, "prior",
            "give dim= on the prior, a Solver instance that declares no_parameters, or Problem(no_parameters=...)")
        self.no_observations, self.no_observations_source = self._resolve(
            "no_observations", no_observations, likelihood, "likelihood",
            "give the observed data as a vector (Normal(mean=data, ...)) or dim= on the likelihood, "
            "a Solver instance that declares no_observations, or Problem(no_observations=...)")
        if getattr(prior, "dimension_free", False):
            prior = prior.with_dimension(self.no_parameters)
        if getattr(likelihood, "dimension_free", False):
            likelihood = likelihood.with_dimension(self.no_observations)

        self.prior_physical = prior
        # every prior is sampled in the standard-normal internal space (2026-09-22): a Normal is
        # wrapped in StandardizedNormal, other priors are internal-space by design already
        self.prior = standardize_prior(prior)
        self.likelihood = likelihood

    def _resolve(self, name: str, explicit: int | None, dist: Any, dist_label: str, fixes: str) -> tuple[int, str]:
        value, source = _resolve_size(name, explicit, self.solver_instance, dist, dist_label)
        if value is None:
            if getattr(dist, "dimension_free", False):
                raise ValueError(f"cannot determine {name}: the {dist_label} {type(dist).__name__}(mean=<scalar>) has "
                                 f"no dimension and nothing else supplies one; {fixes}")
            raise ValueError(f"cannot determine {name} from the {dist_label} ({type(dist).__name__}) or the solver; "
                             f"{fixes}")
        assert source is not None
        if getattr(dist, "dimension_free", False):
            source += f" (dimension-free {dist_label} broadcast to it)"
        return value, source

    def describe(self) -> str:
        """One paragraph: the problem sizes and where each came from."""
        if self.solver_spec is not None:
            solver = f"SolverSpec({self.solver_spec.solver_class_name})"
        else:
            solver = f"solver instance {type(self.solver_instance).__name__}"
        return (f"Problem: no_parameters={self.no_parameters} (from {self.no_parameters_source}), "
                f"no_observations={self.no_observations} (from {self.no_observations_source}); "
                f"prior {type(self.prior_physical).__name__}, likelihood {type(self.likelihood).__name__}, "
                f"{solver}.")

    def _prepare_conf(self, conf: Configuration) -> None:
        conf.resolve_problem_sizes(self.no_parameters, self.no_observations)
        conf.load_continuation()

    def run_sampling(self, conf: Configuration, stages: List[Stage],
                     surrogate_updater: Updater | None = None, surrogate_evaluator: Evaluator | None = None,
                     surrogate_initial_training_data: List[npt.NDArray] | None = None,
                     surrogate_test_data: TestData | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray] | None = None,
                     surrogate_restart: SurrogateRestart | None = None) -> "SamplingRun":
        """
        Run the MCMC sampling with MPI: dispatch this rank to its role (sampler / solvers pool /
        collector) and run it to completion. Collective: call it on every rank of
        ``MPI.COMM_WORLD`` with identically built arguments (it starts with a cross-rank
        configuration check and ends in a ``Barrier``). Launch with
        ``mpiexec -n <N> python3 -m mpi4py script.py``; process counts in ``docs/running.md``.

        Before anything else, the problem sizes are written into ``conf``
        (``Configuration.resolve_problem_sizes``) and continued samples are loaded
        (``load_continuation``); then ``communication.check_configuration_consistency``
        compares the requested values of ``Configuration.POSTERIOR_AFFECTING_FIELDS`` against
        rank 0's (finding 2.9) and aborts the whole job if a rank was built with a different one.
        On rank 0, ``sampling_output/run_manifest.json`` is written before dispatch and
        finalized after the sampler role returns (manifest failures are printed warnings).

        Args:
            conf: run settings; construct it identically on every rank.
            stages: ``surrDAMH.stages.Stage`` objects, run in this order by every sampler.
            surrogate_updater: a ``surrDAMH.surrogates.*Updater``, trained during the run on
                the collector rank and re-sent to the samplers as it improves. Required when
                ``conf.use_collector=True`` and any stage is DAMH or uses a ``Hamiltonian``
                proposal; leave ``None`` for a plain-MH run or when ``use_collector=False``.
            surrogate_evaluator: a fixed, already trained surrogate (``Updater.get_evaluator()``
                of an earlier run, or ``surrDAMH.surrogates.reuse``). Used by the samplers
                directly when ``conf.use_collector=False``; ignored otherwise.
            surrogate_initial_training_data: training data the surrogate starts from, as a list
                ``[parameters, observations, multiplicity]`` with shapes ``(n, no_parameters)``,
                ``(n, no_observations)``, ``(n, 1)`` (multiplicity = weight of each point,
                usually ones). ``None`` (default): only snapshots produced during the run
                are used, so the first stage must be MH or wait for
                ``conf.min_snapshots_initial`` snapshots. Parameters are internal-space
                coordinates (or physical ones if ``conf.transform_before_surrogate=True``). To
                reuse a previous run's data prefer ``surrogate_restart`` (below).
            surrogate_test_data: fixed held-out points for monitoring surrogate accuracy
                (written to ``sampling_output/surrogate_quality_test.csv``). The set is used
                as given for the whole run, never extended or trained on. Create with
                ``surrDAMH.TestData.generate(problem, size=...)`` or load an earlier set with
                ``surrDAMH.TestData.reuse(output_dir)``. ``None`` = no monitoring.
            surrogate_restart: warm-start the surrogate from a previous run:
                ``surrDAMH.SurrogateRestart(state_dir="<previous_output_dir>/sampling_output")``
                reloads its weights and training data (``mode="data"`` reloads the data only).
                Missing files just print a message and the run starts cold. ``None`` (default)
                = start from scratch. Only meaningful with a collector and an updater.

        Returns:
            A ``SamplingRun`` (on every rank) for ``write_report()``.

        Raises:
            ValueError: on every rank, before any MPI communication, for sizes that disagree
                with ``conf``, a bad stage list, or a solver that does not fit
                ``conf.use_solvers_pool``. Any exception inside a role body is caught by
                ``_run_role``, printed with a traceback, and turned into
                ``MPI.COMM_WORLD.Abort(1)`` for the whole job.
        """
        # Maintainer notes: the stage list, prior, likelihood and surrogate objects are neither
        # broadcast nor validated across ranks, so an inconsistency there surfaces only as a
        # mismatched collective call. initial snapshots default to
        # surrogate_updater.get_initial_snapshots() and then to the snapshots restored by
        # surrogate_restart (see _collector_role). A TestData instance gets its posterior
        # weights computed on the collector if missing.
        self._prepare_conf(conf)
        check_stage_list(stages)  # fail here, with a hint, not deep inside a role
        # solver hand-over rule (2026-09-22): the message names what to change
        if conf.use_solvers_pool and self.solver_spec is None:
            raise ValueError(
                "use_solvers_pool=True needs solver=SolverSpec(...): the solvers pool spawns child "
                "processes that import and construct the solver from a file, so a ready Solver instance "
                "cannot be used there. Either give Problem(solver=SolverSpec(...)), or set "
                "Configuration(use_solvers_pool=False) to run the Solver instance inside every sampler process.")
        # Snapshotted by Configuration.__post_init__ (posterior-affecting fields, for the cross-rank
        # check), i.e. before _configure_surrogate_gradients() can flip conf.use_surrogate_gradients,
        # so the run manifest can record both the requested and the effective value:
        use_surrogate_gradients_requested = conf.use_surrogate_gradients_requested

        # WS6: an updater configured with output_normalization="likelihood" gets its
        # per-observation statistics (observed data / noise sd) from the likelihood here,
        # once, on every rank that holds an updater (the collector trains it; samplers only
        # keep the reference). A likelihood that cannot supply them warns and leaves the
        # updater at the identity normalization.
        apply_output_normalization_from_likelihood(surrogate_updater, self.likelihood)

        comm_world = MPI.COMM_WORLD
        rank_world = comm_world.Get_rank()
        prior = self.prior
        likelihood = self.likelihood
        solver_spec = self.solver_spec
        solver_instance = self.solver_instance

        # Cross-rank configuration check (finding 2.9), first collective on every rank: rank 0
        # broadcasts the *requested* (as-constructed, pre-mutation, sizes resolved) values of
        # Configuration.POSTERIOR_AFFECTING_FIELDS and every rank asserts equality. Collective,
        # and it raises on every rank at once; _run_role turns that into a job-wide Abort with a
        # traceback, like any other fatal error in a role body.
        _run_role(lambda: check_configuration_consistency(conf), "CONFIGURATION CHECK")

        # Configuration.torch_threads (WS4/2026-09-18): on every rank, before anything
        # evaluates/trains a network and before the run manifest is built below (so
        # manifest["environment"]["torch_num_threads"] records the effective value).
        apply_torch_threads(conf)

        _configure_surrogate_gradients(conf, surrogate_updater, surrogate_evaluator, should_warn=rank_world == 0)

        # check if prior has the "transform" method:
        if not hasattr(prior, "transform"):
            prior.transform = identity

        # stage names are the output-directory keys; assign them here (identically on every rank,
        # the sampler re-derives the same names) so the run manifest records them:
        for i, stage in enumerate(stages):
            stage.name = stage_name(stage, i)

        if rank_world == 0 and conf.debug:
            # effective settings, once, on rank 0 (WS5): everything that was silently corrected
            # by __post_init__ or by _configure_surrogate_gradients above is visible here.
            print(conf.describe(use_surrogate_gradients_requested=use_surrogate_gradients_requested), flush=True)
            print(self.describe(), flush=True)
            for i, stage in enumerate(stages):
                print(stage.describe(i), flush=True)
            if conf.use_collector:
                for note in wasted_snapshot_notes(stages):
                    print(f"  [note] {note}", flush=True)
            if surrogate_restart is not None:
                print(f"  {surrogate_restart.describe()}", flush=True)

            # start-up diagnostic (finding 2.11): MPI tags grow by one per full-model evaluation
            # and are never reused, while MPI_TAG_UB is only guaranteed to be >= 32767. Warns on
            # rank 0 only; never raises and never changes control flow.
            check_tag_upper_bound(stages)

        if rank_world == 0:
            # run manifest (WS4, library_notes/09_improvement_plan.md §0 principle 4): must never
            # abort a run, so any failure here is a printed warning, not an exception.
            try:
                from surrDAMH.modules.manifest import build_run_manifest, write_run_manifest
                mpi_layout = {
                    "size_world": comm_world.Get_size(),
                    "no_samplers": conf.no_samplers,
                    "rank_collector": conf.rank_collector,
                    "rank_solvers_pool": conf.rank_solvers_pool,
                    "no_solvers": conf.no_solvers,
                    "solver_maxprocs": conf.solver_maxprocs,
                }
                manifest = build_run_manifest(
                    conf, stages, prior, likelihood, runner="mpi",
                    solver_spec=solver_spec, solver_instance=solver_instance,
                    surrogate_updater=surrogate_updater, surrogate_evaluator=surrogate_evaluator,
                    mpi_layout=mpi_layout,
                    use_surrogate_gradients_requested=use_surrogate_gradients_requested)
                manifest["problem"] = self._manifest_entry()
                write_run_manifest(conf.output_dir, manifest)
            except Exception as exc:
                print(f"WARNING: failed to write run manifest: {exc}", flush=True)

        # the rank-local Solver of this rank, if it has one (report: par_names, field statistics)
        local_solver: dict[str, Any] = {"instance": solver_instance}
        if rank_world == conf.rank_solvers_pool:
            def _solver_pool_role():
                assert solver_spec is not None, "solver_spec must be given"
                return surrDAMH.process_SOLVER.run_SOLVER(conf, solver_spec)

            _run_role(_solver_pool_role, "SOLVER")
        elif rank_world == conf.rank_collector:
            def _collector_role():
                assert surrogate_updater is not None
                test_data = surrogate_test_data
                if isinstance(test_data, TestData):
                    if test_data.log_posterior is None or test_data.weights is None:
                        test_data.compute_log_posterior_and_weights(prior, likelihood)
                    test_data = test_data.as_surrogate_test_data()
                # surrogate restart (WS5): the collector rank is the only one that owns an
                # Updater, so this is where a previous run's state is restored.
                restored_snapshots = None
                if surrogate_restart is not None:
                    restored_snapshots = surrogate_restart.apply(surrogate_updater)
                initial_snapshots = surrogate_initial_training_data
                if initial_snapshots is None:
                    # an updater that stored the restored snapshots itself reports them here
                    # (and sets training_data_loaded, so run_COLLECTOR does not re-add them)
                    initial_snapshots = surrogate_updater.get_initial_snapshots()
                if initial_snapshots is None:
                    initial_snapshots = restored_snapshots

                return surrDAMH.process_COLLECTOR.run_COLLECTOR(
                    conf,
                    surrogate_updater=surrogate_updater,
                    initial_snapshots=initial_snapshots,
                    surrogate_test_data=test_data,
                )

            _run_role(_collector_role, "COLLECTOR")
        else:
            def _sampler_role():
                if conf.use_solvers_pool is False:
                    if local_solver["instance"] is None:
                        assert solver_spec is not None, "either solver_spec or solver_instance must be given"
                        # path only; Solver.output_dir creates the directory on first use (2026-09-22)
                        solver_output_dir = os.path.join(conf.output_dir, "solver_output", "rank{}".format(rank_world))
                        local_solver["instance"] = get_solver_from_spec(solver_spec, solver_id=rank_world,
                                                                        solver_output_dir=solver_output_dir)
                else:
                    local_solver["instance"] = None
                return surrDAMH.process_SAMPLER.run_SAMPLER(
                    conf, prior, likelihood, stages,
                    solver_instance=local_solver["instance"], surrogate_evaluator=surrogate_evaluator)

            _run_role(_sampler_role, "SAMPLER")

            if rank_world == 0:
                # rank 0 is always a sampler rank (ranks 0..no_samplers-1); finalize here is the
                # natural end point for it, with no extra MPI communication or barrier.
                try:
                    from surrDAMH.modules.manifest import finalize_run_manifest
                    finalize_run_manifest(conf.output_dir, extra={})
                except Exception as exc:
                    print(f"WARNING: failed to finalize run manifest: {exc}", flush=True)

        comm_world.Barrier()

        return SamplingRun(problem=self, conf=conf, stages=stages, runner="mpi",
                           solver_instance=local_solver["instance"])

    def run_sampling_local(self, conf: Configuration, stages: List[Stage],
                           surrogate_updater: Updater | None = None,
                           surrogate_evaluator: Evaluator | None = None) -> "SamplingRun":
        """
        Run all stages of a single chain in this process, without MPI roles (no collector,
        no solvers pool). The chain reproduces chain 0 of an MPI run with the same
        configuration (same seeds, same output files).

        Args:
            conf: configuration; ``use_collector`` and ``use_solvers_pool`` must be ``False``.
            stages: sampling stages, executed in the given order.
            surrogate_updater: optional surrogate updater, trained in process on the run's
                snapshots; the resulting evaluator is handed to DAMH stages.
            surrogate_evaluator: optional fixed surrogate evaluator, used when no updater is
                given (read-only, never retrained).

        Returns:
            A ``SamplingRun`` with ``runner="local"`` and ``stage_results`` (one
            ``runner_local.StageResult`` per stage).

        A ``SolverSpec`` is instantiated here, in this process (``solver_id=0``, solver output
        directory ``<output_dir>/solver_output/rank0``).
        """
        from surrDAMH.runner_local import run_local
        self._prepare_conf(conf)
        solver = self.solver_instance
        if solver is None:
            assert self.solver_spec is not None
            solver = get_solver_from_spec(self.solver_spec, solver_id=0,
                                          solver_output_dir=os.path.join(conf.output_dir, "solver_output", "rank0"))
        result = run_local(conf, self.prior, self.likelihood, stages, solver,
                           updater=surrogate_updater, evaluator=surrogate_evaluator, problem=self)
        return SamplingRun(problem=self, conf=conf, stages=stages, runner="local",
                           solver_instance=solver, stage_results=list(result.stage_results))

    def _manifest_entry(self) -> dict[str, Any]:
        return {"no_parameters": self.no_parameters, "no_parameters_source": self.no_parameters_source,
                "no_observations": self.no_observations, "no_observations_source": self.no_observations_source}


class SamplingRun:
    """
    A finished (or, via :meth:`load`, an earlier) sampling run: what ``Problem.run_sampling``
    and ``Problem.run_sampling_local`` return. Its main use is :meth:`write_report`.

    Attributes:
        problem: the ``Problem`` that was sampled.
        conf: the run's ``Configuration`` (sizes resolved).
        stages: the stage list, ``.name`` set.
        output_dir: ``conf.output_dir``.
        runner: ``"mpi"`` or ``"local"``.
        solver_instance: the Solver living on this rank, or ``None`` (pool mode, collector).
        stage_results: list of ``runner_local.StageResult`` for a local run, ``None`` for MPI.
        manifest_path: ``<output_dir>/sampling_output/run_manifest.json``.
    """

    def __init__(self, problem: Problem, conf: Any, stages: List[Stage] | None, runner: str,
                 solver_instance: Solver | None = None, stage_results: list | None = None,
                 _report_configuration: bool = True) -> None:
        self.problem = problem
        self.conf = conf
        self.stages = stages
        self.output_dir = conf.output_dir
        self.runner = runner
        self.solver_instance = solver_instance
        self.stage_results = stage_results
        self.manifest_path = os.path.join(conf.output_dir, "sampling_output", "run_manifest.json")
        # False for a run rebuilt by load(): the report then takes the configuration from the manifest
        self._report_configuration = _report_configuration

    @classmethod
    def load(cls, output_dir: str, problem: Problem) -> "SamplingRun":
        """
        Rebuild a report-only run object from ``<output_dir>/sampling_output/run_manifest.json``
        of a finished run, e.g. to write the report again with other options. The report's
        configuration and stage sections then come from the manifest. ``problem`` supplies the
        prior (best-fit section) and, if it holds a Solver instance, the solver-dependent sections.

        Raises:
            FileNotFoundError: no manifest under ``output_dir``.
        """
        import json
        from types import SimpleNamespace
        path = os.path.join(output_dir, "sampling_output", "run_manifest.json")
        with open(path) as f:
            manifest = json.load(f)
        recorded = manifest.get("configuration") or {}
        conf = SimpleNamespace(
            output_dir=output_dir,
            no_parameters=int(recorded.get("no_parameters", problem.no_parameters)),
            no_observations=int(recorded.get("no_observations", problem.no_observations)),
            debug=bool(recorded.get("debug", False)),
            use_solvers_pool=bool(recorded.get("use_solvers_pool", False)),
        )
        return cls(problem=problem, conf=conf, stages=None, runner=str(manifest.get("runner", "local")),
                   solver_instance=problem.solver_instance, _report_configuration=False)

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
        see ``docs/outputs.md``), it does not use any in-memory state of the run.

        MPI run (``runner="mpi"``): rank-0-only, every other rank only participates in the
        two barriers (call this on every rank, like ``run_sampling()``, or the collective will
        hang). Report generation on rank 0 runs inside ``_run_role``, so a failure there prints
        a traceback and aborts the whole job instead of leaving the other ranks blocked in the
        trailing barrier (WS9b). Local run: the same report, no MPI calls.

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
            ``surrDAMH.post_processing.Samples`` on rank 0 / in a local run (already used
            to write the report/summary); ``None`` on every other rank.

        Raises:
            ValueError: if ``stages_to_disp`` resolves to an empty list, or contains a
                stage name not present in this run. This (like every other failure of
                report generation) is turned into ``MPI.COMM_WORLD.Abort(1)`` with a
                printed traceback by ``_run_role``.

        Notes:
            If a Solver instance is available (``self.solver_instance``) and exposes
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

        report_kwargs = dict(
            stages_to_disp=stages_to_disp, observations=observations, par_names=par_names,
            bins1d=bins1d, bins2d=bins2d, no_best_fits=no_best_fits, ranking_mode=ranking_mode,
            parameters_to_disp=parameters_to_disp, observations_to_disp=observations_to_disp,
            include_expensive_sections=include_expensive_sections, grid=grid,
            grid_interp=grid_interp, obs_grid=obs_grid, no_sensors=no_sensors, cmap=cmap,
            chains_to_disp=chains_to_disp, field_statistics_max_samples=field_statistics_max_samples)
        if self.runner != "mpi":
            return self._write_report_rank0(**report_kwargs)

        comm_world = MPI.COMM_WORLD
        if comm_world.Get_rank() != 0:
            comm_world.Barrier()
            return None

        # Every other rank is already waiting in the barrier at the end of this method, so an
        # exception here would hang the job; _run_role turns it into a job-wide abort instead.
        samples = _run_role(
            lambda: self._write_report_rank0(
                stages_to_disp=stages_to_disp, observations=observations, par_names=par_names,
                bins1d=bins1d, bins2d=bins2d, no_best_fits=no_best_fits, ranking_mode=ranking_mode,
                parameters_to_disp=parameters_to_disp, observations_to_disp=observations_to_disp,
                include_expensive_sections=include_expensive_sections, grid=grid,
                grid_interp=grid_interp, obs_grid=obs_grid, no_sensors=no_sensors, cmap=cmap,
                chains_to_disp=chains_to_disp, field_statistics_max_samples=field_statistics_max_samples),
            "REPORT")

        comm_world.Barrier()
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

        samples = surrDAMH.post_processing.Samples(self.conf.no_parameters, self.conf.output_dir,
                                                    debug=self.conf.debug)
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
            prior=self.problem.prior,
            no_best_fits=no_best_fits,
            include_expensive_sections=include_expensive_sections,
            field_statistics=field_statistics,
            configuration=self.conf if self._report_configuration else None,
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
            stages=self.stages,  # the "Sampling Stages" section (None: the manifest copy)
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
