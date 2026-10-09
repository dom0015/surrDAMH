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
from surrDAMH.auto import AutoTestDataRequest, plan_auto


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
    selection_marker = '    <details class="section" id="selection">'
    if selection_marker in html:  # the "Selection and re-run" section stays the last one (2026-10-08)
        html = html.replace(selection_marker, section + selection_marker, 1)
    elif "</body>" in html:
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


#: Values of ``SamplingRun.continue_sampling(chains=...)``.
CONTINUATION_CHAINS: tuple[str, ...] = ("continue", "prior", "lhs")


def _save_surrogate_state_of(updater: Updater | None, output_dir: str) -> None:
    """
    End of a continued run (collector rank / local process): write the updater's state to
    ``<output_dir>/sampling_output`` so that the next continuation can restore it by default.
    Only for an updater with state persistence; a failure is a printed warning.
    """
    supports = getattr(updater, "supports_state_persistence", None)
    if updater is None or not callable(supports) or not supports():
        return
    try:
        SurrogateRestart(state_dir=os.path.join(output_dir, "sampling_output")).save(updater)
    except Exception as exc:
        print(f"WARNING: failed to save the surrogate state to {output_dir!r}: {exc}", flush=True)


def _read_run_manifest(output_dir: str) -> dict:
    """``<output_dir>/sampling_output/run_manifest.json`` of a finished run."""
    import json
    path = os.path.join(output_dir, "sampling_output", "run_manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no run manifest {path!r}: {output_dir!r} is not the output directory of a "
                                "finished surrDAMH run")
    with open(path) as f:
        return json.load(f)


def _lineage_of(manifest: dict, output_dir: str) -> dict:
    """The manifest's ``"lineage"`` entry; a manifest written before it existed is generation 0."""
    entry = manifest.get("lineage")
    if entry:
        return entry
    return {"continued_from": None, "generation": 0, "stage_index_offset": 0,
            "no_stages_lineage": len(manifest.get("stages") or []), "same_problem": True,
            "same_problem_lineage": True, "chains": None, "initial_sample_is_carried_over": False,
            "dirs": [os.path.abspath(output_dir)]}


def _merged_carry_over(dirs: list[str]) -> dict:
    """
    Proposal carry-over of a whole lineage: every stage of every run, oldest first, merged
    with later stages winning -- the same ``carried.update(...)`` the stage loop does within a run.
    """
    from surrDAMH.modules.continuation import load_carry_over
    carried: dict = {}
    for directory in dirs:
        try:
            names = [stage.get("name") for stage in (_read_run_manifest(directory).get("stages") or [])]
        except FileNotFoundError:
            print(f"WARNING: lineage directory {directory!r} has no run manifest; its proposal carry-over "
                  "is not used", flush=True)
            continue
        for name in names:
            loaded = load_carry_over(directory, name) if name else None
            if loaded is not None:
                carried.update(loaded[0])
    return carried


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
                     surrogate_restart: SurrogateRestart | None = None, *,
                     _initial_carry_over: dict | None = None, _initial_sample_is_carried_over: bool = False,
                     _lineage: dict | None = None, _save_surrogate_state: bool = False,
                     _auto: dict | None = None) -> "SamplingRun":
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
            A ``SamplingRun`` (on every rank) for ``write_report()`` and
            ``continue_sampling()``.

        The keywords starting with ``_`` are set by ``SamplingRun.continue_sampling`` only
        (carried proposal state, A30 flag of the first stage, the manifest's ``"lineage"``
        entry, saving the surrogate state at the end) and by :meth:`run_sampling_auto`
        (``_auto``: the manifest's ``"auto"`` entry); leave them at their defaults.

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
        # the sampler re-derives the same names) so the run manifest records them. A continued
        # run numbers its stages after the earlier runs of its lineage (conf.stage_index_offset).
        offset = int(conf.stage_index_offset or 0)
        for i, stage in enumerate(stages):
            stage.name = stage_name(stage, offset + i)

        if rank_world == 0 and conf.debug:
            # effective settings, once, on rank 0 (WS5): everything that was silently corrected
            # by __post_init__ or by _configure_surrogate_gradients above is visible here.
            print(conf.describe(use_surrogate_gradients_requested=use_surrogate_gradients_requested), flush=True)
            print(self.describe(), flush=True)
            for i, stage in enumerate(stages):
                print(stage.describe(offset + i), flush=True)
            if conf.use_collector:
                for note in wasted_snapshot_notes(stages):
                    print(f"  [note] {note}", flush=True)
            if surrogate_restart is not None:
                print(f"  {surrogate_restart.describe()}", flush=True)
        elif rank_world == 0 and _auto is not None:
            # automatic runs print the resolved stage list unconditionally (the plan itself was
            # printed by run_sampling_auto)
            for i, stage in enumerate(stages):
                print(stage.describe(offset + i), flush=True)

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
                if _lineage is not None:  # continued runs only; a plain run's manifest is unchanged
                    manifest["lineage"] = _lineage
                if _auto is not None:  # automatic runs only
                    manifest["auto"] = _auto
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
                if isinstance(test_data, AutoTestDataRequest):
                    # automatic mode: the held-out set is evaluated here, on the collector rank,
                    # so that no sampler spends an evaluation on it; saved into the run
                    test_data = test_data.generate(self, conf.transform_before_surrogate)
                    test_data.save(conf.output_dir)
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

                result = surrDAMH.process_COLLECTOR.run_COLLECTOR(
                    conf,
                    surrogate_updater=surrogate_updater,
                    initial_snapshots=initial_snapshots,
                    surrogate_test_data=test_data,
                )
                if _save_surrogate_state:
                    _save_surrogate_state_of(surrogate_updater, conf.output_dir)
                return result

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
                    solver_instance=local_solver["instance"], surrogate_evaluator=surrogate_evaluator,
                    initial_carry_over=_initial_carry_over,
                    initial_sample_is_carried_over=_initial_sample_is_carried_over)

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
                           solver_instance=local_solver["instance"], lineage_entry=_lineage,
                           auto_entry=_auto)

    def run_sampling_local(self, conf: Configuration, stages: List[Stage],
                           surrogate_updater: Updater | None = None,
                           surrogate_evaluator: Evaluator | None = None, *,
                           _initial_carry_over: dict | None = None, _initial_sample_is_carried_over: bool = False,
                           _lineage: dict | None = None, _save_surrogate_state: bool = False,
                           _auto: dict | None = None) -> "SamplingRun":
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
        directory ``<output_dir>/solver_output/rank0``). The keywords starting with ``_`` are set
        by ``SamplingRun.continue_sampling_local`` and :meth:`run_sampling_local_auto` only;
        leave them at their defaults.
        """
        from surrDAMH.runner_local import run_local
        self._prepare_conf(conf)
        solver = self.solver_instance
        if solver is None:
            assert self.solver_spec is not None
            solver = get_solver_from_spec(self.solver_spec, solver_id=0,
                                          solver_output_dir=os.path.join(conf.output_dir, "solver_output", "rank0"))
        result = run_local(conf, self.prior, self.likelihood, stages, solver,
                           updater=surrogate_updater, evaluator=surrogate_evaluator, problem=self,
                           initial_carry_over=_initial_carry_over,
                           initial_sample_is_carried_over=_initial_sample_is_carried_over,
                           lineage=_lineage, auto=_auto)
        if _save_surrogate_state and surrogate_updater is not None:
            _save_surrogate_state_of(surrogate_updater, conf.output_dir)
        return SamplingRun(problem=self, conf=conf, stages=stages, runner="local",
                           solver_instance=solver, stage_results=list(result.stage_results),
                           lineage_entry=_lineage, auto_entry=_auto)

    # ------------------------------------------------------------------------------------------
    # automatic mode (2026-10-08, library_notes/25 §5)
    # ------------------------------------------------------------------------------------------

    def run_sampling_auto(self, conf: Configuration, budget: int | None = None, time_limit: float | None = None,
                          mode: str = "robust", surrogate_updater: Updater | None = None,
                          surrogate_test_data: TestData | tuple | None = None) -> "SamplingRun":
        """
        Run the sampling with an automatically chosen stage layout and surrogate, under MPI.
        Collective like :meth:`run_sampling` (call it on every rank)::

            run = problem.run_sampling_auto(conf, budget=80_000, mode="robust")
            run = problem.run_sampling_auto(conf, time_limit=3600.0, mode="fast")

        The plan (``surrDAMH.auto.plan_auto``; its numbers are placeholders until the validation
        of note 25 §6 tunes them): chains start from a Latin-hypercube design; an excluded
        adaptive random-walk MH warm-up of ``clip(20 d, 0.05 B, 0.25 B)`` evaluations per chain
        trains the first surrogate; then 4 DAMH-SMU chunks (sub-chain length 1) share the rest
        of the per-chain budget ``B``. With ``B < 50 d`` or ``conf.use_collector=False`` a
        single adaptive MH stage runs instead. ``mode="robust"``: the chunks propose with an
        adaptive ``RandomWalk()``; ``mode="fast"``: with ``Hamiltonian(num_steps=30,
        integrator="dimension_robust", mass=1.0)`` on the surrogate's gradients. Both modes are
        exact. The surrogate is a ``NeuralNetworkUpdater(hidden_layer_sizes=(64, 64),
        activation="silu", solver="adamw", seed=0)``. With an in-process ``Solver`` a held-out
        set of ``min(256, max(2 d, 0.02 budget))`` prior draws is evaluated on the collector
        rank (counted in the budget, written to ``sampling_output/surrogate_test_data.npz``).

        The plan is printed on rank 0, the explicit stage list it produced is what runs
        (``run.stages``; copy it into :meth:`run_sampling` to re-run by hand) and it is recorded
        in ``run_manifest.json`` under ``"auto"`` and as ``run.auto``.

        Configuration fields Auto sets (recorded in ``"auto"["conf_settings"]``):
        ``initial_sample_type="lhs"`` if left at ``"prior"``; ``use_surrogate_gradients``
        (``mode == "fast"``); ``min_snapshots_initial``/``min_snapshots_to_update`` unless the
        user changed them from their defaults (then the user's value is kept, with a note).

        Args:
            conf: as in :meth:`run_sampling`; leave the lineage fields at their defaults (an
                automatic continuation, ``continue_sampling_auto``, does not exist yet).
            budget: TOTAL exact-model evaluations over all chains, including the held-out set.
            time_limit: wall-clock seconds for the whole run instead of ``budget``.
            mode: ``"robust"`` (default) or ``"fast"``.
            surrogate_updater: expert override of the default network (``mode="fast"`` needs
                one with gradients).
            surrogate_test_data: expert override of the held-out set (used as given, not
                counted in the budget).

        Returns:
            The ``SamplingRun`` (``run.auto`` = the plan's manifest entry).

        Raises:
            ValueError: both or neither of ``budget``/``time_limit``; an unknown ``mode``;
                ``mode="fast"`` without a gradient surrogate; lineage fields set; everything
                :meth:`run_sampling` raises.
        """
        plan = plan_auto(self, conf, budget=budget, time_limit=time_limit, mode=mode, local=False,
                         surrogate_updater=surrogate_updater, surrogate_test_data=surrogate_test_data)
        conf._set_by_auto(**plan.conf_settings)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(plan.describe(), flush=True)
        return self.run_sampling(conf, plan.stages, surrogate_updater=plan.surrogate_updater,
                                 surrogate_test_data=plan.surrogate_test_data, _auto=plan.manifest_entry())

    def run_sampling_local_auto(self, conf: Configuration, budget: int | None = None,
                                time_limit: float | None = None, mode: str = "robust",
                                surrogate_updater: Updater | None = None,
                                surrogate_test_data: TestData | tuple | None = None) -> "SamplingRun":
        """
        :meth:`run_sampling_auto` for one chain in this process (:meth:`run_sampling_local`;
        ``conf`` needs ``use_collector=False, use_solvers_pool=False``). Same plan with one
        chain; the surrogate is trained in process. The held-out set (in-process ``Solver``
        only) is evaluated before the run and saved to ``sampling_output/surrogate_test_data.npz``;
        the local runner does not monitor surrogate quality with it.
        """
        plan = plan_auto(self, conf, budget=budget, time_limit=time_limit, mode=mode, local=True,
                         surrogate_updater=surrogate_updater, surrogate_test_data=surrogate_test_data)
        conf._set_by_auto(**plan.conf_settings)
        print(plan.describe(), flush=True)
        if isinstance(plan.surrogate_test_data, AutoTestDataRequest):
            plan.surrogate_test_data.generate(self, conf.transform_before_surrogate).save(conf.output_dir)
        return self.run_sampling_local(conf, plan.stages, surrogate_updater=plan.surrogate_updater,
                                       _auto=plan.manifest_entry())

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
        lineage: absolute output directories of the chain of runs this one belongs to, oldest
            first, this run last (``[output_dir]`` for a run that is not a continuation).
        generation: 0 for a plain run, ``k`` for the k-th ``continue_sampling`` of it.
        auto: the resolved automatic plan (the manifest's ``"auto"`` entry) of a run started
            with ``run_sampling_auto``/``run_sampling_local_auto``, else ``None``.
    """

    def __init__(self, problem: Problem, conf: Any, stages: List[Stage] | None, runner: str,
                 solver_instance: Solver | None = None, stage_results: list | None = None,
                 _report_configuration: bool = True, lineage_entry: dict | None = None,
                 auto_entry: dict | None = None) -> None:
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
        # lineage of continued runs (the manifest's "lineage" entry; a plain run is generation 0)
        entry = lineage_entry or {}
        self.lineage: list[str] = list(entry.get("dirs") or [os.path.abspath(conf.output_dir)])
        self.generation: int = int(entry.get("generation", 0))
        # the manifest's "auto" entry of an automatic run (run_sampling_auto), else None
        self.auto: dict | None = auto_entry

    @classmethod
    def load(cls, output_dir: str, problem: Problem | None = None) -> "SamplingRun":
        """
        Rebuild a report-only run object from ``<output_dir>/sampling_output/run_manifest.json``
        of a finished run, e.g. to write the report again with other options or an edited
        ``post_processing_output/selection.json``. The report's configuration and stage sections
        then come from the manifest(s).

        Args:
            output_dir: the run's output directory (for a lineage: the newest run's).
            problem: optional. It supplies the prior (histogram overlay) and, if it holds a
                Solver instance, the solver-dependent report sections (parameter names, posterior
                field statistics, best-fit solver visualization); without it these are marked
                "Not available: no Problem given" and the sizes come from the manifest.
                ``continue_sampling`` on a run loaded without a problem needs ``problem=``.

        Raises:
            FileNotFoundError: no manifest under ``output_dir``.
            ValueError: no ``problem`` and the manifest does not record ``no_parameters`` /
                ``no_observations``.
        """
        from types import SimpleNamespace
        path = os.path.join(output_dir, "sampling_output", "run_manifest.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no run manifest {path!r}")
        manifest = _read_run_manifest(output_dir)
        recorded = manifest.get("configuration") or {}
        sizes = {}
        for name in ("no_parameters", "no_observations"):
            value = recorded.get(name, getattr(problem, name, None))
            if value is None:
                raise ValueError(f"{path} does not record {name}; pass problem= to SamplingRun.load")
            sizes[name] = int(value)
        conf = SimpleNamespace(
            output_dir=output_dir,
            debug=bool(recorded.get("debug", False)),
            use_solvers_pool=bool(recorded.get("use_solvers_pool", False)),
            **sizes,
        )
        return cls(problem=problem, conf=conf, stages=None, runner=str(manifest.get("runner", "local")),
                   solver_instance=getattr(problem, "solver_instance", None), _report_configuration=False,
                   lineage_entry=_lineage_of(manifest, output_dir), auto_entry=manifest.get("auto"))

    # ------------------------------------------------------------------------------------------
    # continuation (2026-10-08)
    # ------------------------------------------------------------------------------------------

    def continue_sampling(self, conf: Configuration, stages: List[Stage], chains: str = "continue",
                          problem: Problem | None = None,
                          surrogate_updater: Updater | None = None, surrogate_evaluator: Evaluator | None = None,
                          surrogate_initial_training_data: List[npt.NDArray] | None = None,
                          surrogate_test_data: TestData | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray] | None = None,
                          surrogate_restart: SurrogateRestart | None = None) -> "SamplingRun":
        """
        Continue this run with new stages, under MPI: a new run in ``conf.output_dir`` whose
        chains, stage numbering, seeds, tuned proposals and surrogate pick up where this one
        stopped. Collective like ``Problem.run_sampling`` (call it on every rank); works on a
        run returned by ``run_sampling``/``run_sampling_local`` or rebuilt by :meth:`load`, and
        on a continued run again (a lineage A -> B -> C ...).

        What it does by default:

        - chains: ``chains="continue"`` starts every chain from the last state of this run
          (``sampling_output/last_sample``; this run must have had at least as many chains,
          extra ones are dropped with a warning). The first stage does not count that state
          again if this run's last stage already wrote it (``save_to_file``, A30).
          ``"prior"``/``"lhs"`` draw new initial states instead (with the generation-shifted
          seeds, so not A's starting points).
        - numbering and seeds: the stages are numbered after all earlier stages of the lineage
          (``alg0002_...`` after a two-stage run); the seed formula uses the global stage
          index, the lineage's stage count and a shift of ``1_000_000 * generation``, so no
          random stream of an earlier run of the lineage is reused (``modules/seeds.py``). Set
          through ``conf.stage_index_offset`` / ``no_stages_lineage`` / ``lineage_generation``.
        - proposals: the carry-over of every adaptive stage of the lineage (later wins) is
          handed to the first stage, so ``RandomWalk()`` / ``PCN(beta=None)`` /
          ``Hamiltonian(step_size=None)`` start from the tuned values -- with every ``chains``.
        - surrogate: with neither ``surrogate_updater`` nor ``surrogate_evaluator`` given, a
          surrogate state saved in this run's ``sampling_output`` (``surrogate_checkpoint.pt`` +
          ``surrogate_training_data.npz``, written by ``SurrogateRestart(...).save(updater)``
          or automatically at the end of every continued run) is restored with
          ``surrDAMH.surrogates.reuse.SurrogateReused``: a DAMH first stage then needs no MH
          warm-up. With ``conf.use_collector=False`` the restored surrogate is used as a fixed
          evaluator. Training data without a checkpoint raise ``ValueError`` (pass
          ``surrogate_updater=`` with ``surrogate_restart=SurrogateRestart(..., mode="data")``);
          nothing saved = no surrogate. Given objects are used exactly as ``run_sampling`` would.
        - held-out test data: ``TestData.reuse(<this run>)`` if a set was saved there and none is
          given; the set used is saved into the new run as well.
        - ``problem=``: a different ``Problem`` (e.g. new observed data, same forward model;
          sizes must match). Default: this run's ``problem``.

        Args:
            conf: a fresh ``Configuration`` with its own ``output_dir``. Leave its initial-sample
                fields (``initial_sample_type``, ``continued_from_dir``) and
                ``stage_index_offset``/``no_stages_lineage``/``lineage_generation`` at their
                defaults: they are set here.
            stages: the stages of the continuation.
            chains: ``"continue"`` (default), ``"prior"`` or ``"lhs"``.
            problem: optional replacement ``Problem``.
            surrogate_updater, surrogate_evaluator, surrogate_initial_training_data,
            surrogate_test_data, surrogate_restart: as in ``Problem.run_sampling``.

        Returns:
            The new ``SamplingRun`` (``generation`` = this one's + 1, ``lineage`` extended).

        Raises:
            ValueError: unknown ``chains``; ``conf`` sets the initial samples itself; the same
                ``output_dir`` as a run of the lineage; ``problem`` sizes differ from this run;
                surrogate training data without a checkpoint.
            FileNotFoundError: no run manifest in this run's directory; ``chains="continue"``
                and no ``last_sample/``.
        """
        target, surrogate_updater, surrogate_evaluator, keywords = self._prepare_continuation(
            conf, stages, chains, problem, surrogate_updater, surrogate_evaluator, local=False)
        if surrogate_test_data is None:
            try:
                surrogate_test_data = TestData.reuse(self.output_dir)
            except FileNotFoundError:
                surrogate_test_data = None
        if isinstance(surrogate_test_data, TestData) and MPI.COMM_WORLD.Get_rank() == 0:
            surrogate_test_data.save(conf.output_dir)  # keeps the lineage self-contained
        return target.run_sampling(conf, stages, surrogate_updater=surrogate_updater,
                                   surrogate_evaluator=surrogate_evaluator,
                                   surrogate_initial_training_data=surrogate_initial_training_data,
                                   surrogate_test_data=surrogate_test_data,
                                   surrogate_restart=surrogate_restart, **keywords)

    def continue_sampling_local(self, conf: Configuration, stages: List[Stage], chains: str = "continue",
                                problem: Problem | None = None,
                                surrogate_updater: Updater | None = None,
                                surrogate_evaluator: Evaluator | None = None) -> "SamplingRun":
        """
        :meth:`continue_sampling` for one chain in this process (``Problem.run_sampling_local``):
        same defaults, same rules; chain 0 of the previous run is continued. A restored
        surrogate is trained further in process (``surrogate_updater``); there is no held-out
        test data and no ``surrogate_restart`` here (load data into an updater yourself with
        ``updater.load_training_data(path)``).
        """
        target, surrogate_updater, surrogate_evaluator, keywords = self._prepare_continuation(
            conf, stages, chains, problem, surrogate_updater, surrogate_evaluator, local=True)
        return target.run_sampling_local(conf, stages, surrogate_updater=surrogate_updater,
                                         surrogate_evaluator=surrogate_evaluator, **keywords)

    def _prepare_continuation(self, conf: Configuration, stages: List[Stage], chains: str,
                              problem: Problem | None, surrogate_updater: Updater | None,
                              surrogate_evaluator: Evaluator | None, local: bool):
        """
        Everything :meth:`continue_sampling` decides before delegating to ``Problem.run_sampling``
        (identical on every rank, no MPI communication): checks, ``conf`` fields, carried
        proposal state, default surrogate, the manifest's lineage entry.
        """
        from surrDAMH.modules.manifest import lineage_entry
        from surrDAMH.surrogates.reuse import SurrogateReused, surrogate_state_paths

        if chains not in CONTINUATION_CHAINS:
            raise ValueError(f"chains must be one of {CONTINUATION_CHAINS}, got {chains!r}")
        if conf.initial_sample_type != "prior" or conf.continued_from_dir is not None:
            raise ValueError("continue_sampling sets the initial samples; use chains=... "
                             f"(got initial_sample_type={conf.initial_sample_type!r}, "
                             f"continued_from_dir={conf.continued_from_dir!r})")
        if (conf.stage_index_offset or 0) != 0 or conf.no_stages_lineage is not None or conf.lineage_generation:
            raise ValueError("stage_index_offset / no_stages_lineage / lineage_generation are set by continue_sampling; "
                             "leave them at their defaults")
        check_stage_list(stages)

        manifest = _read_run_manifest(self.output_dir)
        previous = _lineage_of(manifest, self.output_dir)
        previous_dirs = list(previous.get("dirs") or [os.path.abspath(self.output_dir)])
        previous_dirs[-1] = os.path.abspath(self.output_dir)  # the run itself, wherever it is now
        new_dir = os.path.realpath(conf.output_dir)
        for directory in previous_dirs:
            if os.path.realpath(directory) == new_dir:
                raise ValueError(f"conf.output_dir={conf.output_dir!r} is the output directory of a run of this "
                                 f"lineage ({directory!r}); a continuation needs its own output_dir")

        recorded = manifest.get("configuration") or {}
        previous_sizes = (recorded.get("no_parameters"), recorded.get("no_observations"))
        if problem is None and self.problem is None:
            raise ValueError("this SamplingRun was loaded without a problem (SamplingRun.load(output_dir)); "
                             "pass problem= to continue_sampling, or load it with SamplingRun.load(output_dir, problem)")
        same_problem = problem is None
        target = self.problem if problem is None else problem
        if previous_sizes != (target.no_parameters, target.no_observations):
            raise ValueError(f"the problem has no_parameters={target.no_parameters}, no_observations="
                             f"{target.no_observations}, but the previous run {self.output_dir!r} sampled "
                             f"no_parameters={previous_sizes[0]}, no_observations={previous_sizes[1]}; "
                             "a continuation needs the same sizes")

        initial_sample_is_carried_over = False
        if chains == "continue":
            last_sample_root = os.path.join(self.output_dir, "sampling_output", "last_sample")
            if not os.path.isdir(last_sample_root) or not os.listdir(last_sample_root):
                raise FileNotFoundError(
                    f"chains='continue' needs the last states of the previous run, but {last_sample_root!r} "
                    "is missing or empty; use chains='prior' or chains='lhs' to start new chains")
            previous_stages = manifest.get("stages") or []
            # A30 across runs: the previous run's last stage wrote the state the chains continue from
            initial_sample_is_carried_over = bool(previous_stages[-1].get("save_to_file", True)) \
                if previous_stages else False
            conf._set_by_continuation(initial_sample_type="continued", continued_from_dir=self.output_dir)
        else:
            conf._set_by_continuation(initial_sample_type=chains)

        offset = int(previous.get("stage_index_offset", 0)) + len(manifest.get("stages") or [])
        generation = int(previous.get("generation", 0)) + 1
        conf._set_by_continuation(stage_index_offset=offset, no_stages_lineage=offset + len(stages),
                                  lineage_generation=generation)

        initial_carry_over = _merged_carry_over(previous_dirs)

        if surrogate_updater is None and surrogate_evaluator is None:
            checkpoint_path, data_path = surrogate_state_paths(self.output_dir)
            if os.path.exists(checkpoint_path):
                surrogate_updater = SurrogateReused(self.output_dir)
                if not local and not conf.use_collector:
                    # no collector to train it: the restored surrogate is a fixed evaluator
                    surrogate_evaluator, surrogate_updater = surrogate_updater.get_evaluator(), None
            elif os.path.exists(data_path):
                if local:
                    fix = ("pass surrogate_updater=<an updater> after updater.load_training_data("
                           f"{data_path!r})")
                else:
                    fix = ("pass surrogate_updater=<an updater> together with surrogate_restart=surrDAMH."
                           f"SurrogateRestart(state_dir={os.path.dirname(data_path)!r}, mode=\"data\")")
                raise ValueError(f"{self.output_dir!r} holds surrogate training data but no surrogate checkpoint, "
                                 f"so the surrogate cannot be restored automatically; {fix}")

        entry = lineage_entry(conf, stages, continued_from=self.output_dir, previous_dirs=previous_dirs,
                              generation=generation, same_problem=same_problem,
                              same_problem_lineage=bool(previous.get("same_problem_lineage", True)) and same_problem,
                              chains=chains, initial_sample_is_carried_over=initial_sample_is_carried_over)
        keywords = dict(_initial_carry_over=initial_carry_over,
                        _initial_sample_is_carried_over=initial_sample_is_carried_over,
                        _lineage=entry, _save_surrogate_state=True)
        return target, surrogate_updater, surrogate_evaluator, keywords

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
                     field_statistics_max_samples: int | None = None,
                     include_previous: bool | None = None,
                     selection: str | os.PathLike | bool | None = None) -> surrDAMH.post_processing.Samples | None:
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
            include_previous: for a run made by ``continue_sampling``: ``None`` (default) reports
                the whole lineage (every earlier run it continues, oldest first) iff the problem
                never changed along it, else this run only; ``True`` the whole lineage regardless
                (the report says the problem changed); ``False`` this run only. See
                ``post_processing.Samples``.
            selection: the (stage, chain) mask with per-chain burn-in
                (``post_processing/selection.py``). ``None`` (default): this run's
                ``post_processing_output/selection.json`` -- created on the first report with every
                chain of every stage except ``is_excluded`` (burn-in) stages, then read and applied
                on every later report, so editing it and calling
                ``SamplingRun.load(output_dir).write_report()`` re-runs the post-processing with the
                edit. A path: that file. ``False``: no selection, every chain is used. Stages whose
                every chain is excluded are left out of the report with a note; the report ends with
                a "Selection and re-run" section.

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
            chains_to_disp=chains_to_disp, field_statistics_max_samples=field_statistics_max_samples,
            include_previous=include_previous, selection=selection)
        if self.runner != "mpi":
            return self._write_report_rank0(**report_kwargs)

        comm_world = MPI.COMM_WORLD
        if comm_world.Get_rank() != 0:
            comm_world.Barrier()
            return None

        # Every other rank is already waiting in the barrier at the end of this method, so an
        # exception here would hang the job; _run_role turns it into a job-wide abort instead.
        samples = _run_role(lambda: self._write_report_rank0(**report_kwargs), "REPORT")

        comm_world.Barrier()
        return samples

    def _write_report_rank0(self, stages_to_disp, observations, par_names, bins1d, bins2d,
                            no_best_fits, ranking_mode, parameters_to_disp, observations_to_disp,
                            include_expensive_sections, grid, grid_interp, obs_grid, no_sensors,
                            cmap, chains_to_disp, field_statistics_max_samples=None,
                            include_previous=None, selection=None) -> surrDAMH.post_processing.Samples:
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

        if self.problem is None:
            # SamplingRun.load(output_dir) without a problem (2026-10-08): no prior, no solver
            pool_mode_note = ("Not available: no Problem given (SamplingRun.load(output_dir) without problem=); "
                              "the prior overlay, parameter names, posterior field statistics and best-fit "
                              "solver visualization need it.")
        # write_report: None = this run's selection.json (created if missing), False = none
        samples_selection = True if selection is None or selection is True else (None if selection is False
                                                                                 else selection)
        samples = surrDAMH.post_processing.Samples(self.conf.no_parameters, self.conf.output_dir,
                                                    debug=self.conf.debug, include_previous=include_previous,
                                                    selection=samples_selection)
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
        # html_report_extended drops these itself (with a note in the report), so it gets the full list
        requested_stages = list(stages_to_disp)
        skipped_stages = samples._fully_excluded_stages(stages_to_disp)
        if skipped_stages:
            print("Report: stage(s) " + ", ".join(samples.stage_names[i] for i in skipped_stages)
                  + f" skipped, every chain is excluded by the selection ({samples.selection_path}).", flush=True)
            stages_to_disp = [stage for stage in stages_to_disp if stage not in skipped_stages]
            if not stages_to_disp:
                raise ValueError(f"every stage of the report is excluded by the selection {samples.selection_path!r}; "
                                 "set include to 1 for some chain")
        if not stages_to_disp:
            raise ValueError("No report stages are available.")

        post_processing_dir_path = os.path.join(self.conf.output_dir, "post_processing_output")
        ensure_dir(post_processing_dir_path)

        samples.calculate_CpUS([[i] for i in stages_to_disp])
        # summary.csv holds the displayed stages only, but ``samples.summary`` must keep ALL stages:
        # every later section indexes it by the ORIGINAL stage position (``summary.iloc[stages_to_disp]``
        # in plots/html_report/calculate_CpUS), which broke as soon as the first stage was not displayed
        # (an explicit stages_to_disp=[1, ...], or -- since the selection mask of 2026-10-08 -- an
        # ``is_excluded`` warm-up stage dropped by default; found by the auto-mode example).
        samples.get_summary().iloc[stages_to_disp, :].to_csv(os.path.join(post_processing_dir_path, "summary.csv"))

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
            stages_to_disp=requested_stages,
            observations=observations,
            output_file=output_html_file_path,
            bins1d=bins1d,
            bins2d=bins2d,
            parameters_to_disp=parameters_to_disp,
            prior=self.problem.prior if self.problem is not None else None,
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
            # the "Sampling Stages" section (None: the manifest copy, of every run of a lineage)
            stages=self.stages if len(samples.run_data.output_dirs) <= 1 else None,
            selection_section=True,
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
