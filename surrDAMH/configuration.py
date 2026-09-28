#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:32:40 2021

@author: simona
"""

import sys
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules.describe import describe_fields

#: Fields of ``Configuration`` that can change the posterior, the acceptance rate or
#: reproducibility (the list the class docstring spells out). ``describe()`` marks them
#: with a trailing ``*`` so the start-up log says which numbers matter for a result.
POSTERIOR_AFFECTING_FIELDS: frozenset[str] = frozenset({
    "no_parameters", "no_observations", "transform_before_surrogate",
    "min_snapshots_initial", "min_snapshots_to_update",
    "max_collected_snapshots_per_loop", "use_surrogate_gradients", "initial_sample_type",
    "initial_samples_distribution", "continued_from_dir", "lhs_scale", "use_collector",
})

#: How deep :func:`normalize_for_comparison` descends into containers/objects before it gives up
#: and uses the type name only (guards against cycles and against huge nested objects).
MAX_NORMALIZATION_DEPTH = 4


def _qualified_type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def normalize_for_comparison(value: Any, _depth: int = 0) -> Any:
    """
    Turn a configuration field value into picklable primitives that compare with plain ``==``.

    Used by the cross-rank configuration check
    (``surrDAMH.modules.communication.check_configuration_consistency``, finding 2.9): the
    normalized values are what rank 0 broadcasts and what every rank compares against, so this
    must avoid both spurious *differences* and spurious *exceptions*:

    - numpy arrays (``lhs_scale``) become ``("ndarray", shape, dtype, nested lists)`` instead of
      an element-wise comparison whose truth value is ambiguous;
    - an arbitrary object (``initial_samples_distribution`` is a duck-typed ``Distribution``,
      usually without ``__eq__``, whose instances on two ranks are never ``==``) becomes its
      qualified type name plus its normalized ``__dict__``, i.e. a structural fingerprint that
      is identical on two ranks that ran the same constructor call;
    - anything that cannot be normalized (recursion limit, exotic object) falls back to the
      qualified type name, which never raises and never differs between equally-built ranks.

    NaN is deliberately not special-cased: ``float("nan")`` is not equal to itself, so a NaN in a
    posterior-affecting field would be reported as a mismatch. None of these fields takes NaN.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return ("ndarray", tuple(value.shape), value.dtype.str, value.tolist())
    if _depth >= MAX_NORMALIZATION_DEPTH:
        return _qualified_type_name(value)
    try:
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(normalize_for_comparison(item, _depth + 1) for item in value))
        if isinstance(value, (set, frozenset)):
            return (type(value).__name__,
                    tuple(sorted(repr(normalize_for_comparison(item, _depth + 1)) for item in value)))
        if isinstance(value, dict):
            return ("dict", tuple(sorted(((str(key), normalize_for_comparison(item, _depth + 1))
                                          for key, item in value.items()), key=lambda pair: pair[0])))
        attributes = getattr(value, "__dict__", None)
        if attributes:
            return (_qualified_type_name(value), normalize_for_comparison(dict(attributes), _depth + 1))
        return _qualified_type_name(value)
    except Exception:  # never let a fingerprint break a run
        return _qualified_type_name(value)


# Maintainer notes (moved out of the Configuration docstring on 2026-09-21):
#
# Cross-rank consistency check: since 2026-09-18 (finding 2.9), SamplingFramework.run()
# broadcasts rank 0's *requested* values of POSTERIOR_AFFECTING_FIELDS and every rank
# asserts equality, so a script that builds a different Configuration per rank fails at
# start-up instead of silently sampling a different posterior per chain
# (modules.communication.check_configuration_consistency). Fields outside that set are
# not compared. __post_init__ derives the MPI role layout (no_samplers, rank_collector,
# rank_solvers_pool) from MPI.COMM_WORLD's size and the two topology flags
# (use_collector, use_solvers_pool); see docs/running.md for the process-count table and
# docs/configuration.md for the full field reference (generated from this dataclass).
#
# Posterior-, acceptance-rate- or reproducibility-affecting fields (see
# POSTERIOR_AFFECTING_FIELDS): transform_before_surrogate (which space the surrogate is
# trained on); min_snapshots_initial/min_snapshots_to_update/max_collected_snapshots_per_loop
# (control exactly when the surrogate is (re)trained, hence the DAMH-SMU accept/reject
# sequence for runs that use a collector); use_surrogate_gradients (may be silently
# disabled by SamplingFramework for an incompatible surrogate, see run_manifest.json's
# use_surrogate_gradients_requested); initial_sample_type (every type is reproducible
# since G4, 2026-09-17: "prior"/"user_specified" draw from
# np.random.default_rng(10*no_stages*rank_world + 3), see surrDAMH.modules.seeds); and
# everything under "invalid combinations" below. transform_before_saving and
# save_snapshots_to_file affect only what is written to disk, not the posterior itself;
# torch_threads is a performance-only knob (intra-op CPU thread count for torch) and does
# not affect the posterior, acceptance rate or surrogate accuracy.
#
# Ineffective for spawned children: paths_to_append is appended to sys.path in the
# process that constructs this Configuration only -- it does NOT reach solver-pool
# children spawned by MPI.Comm.Spawn, which unpickle conf without running __post_init__
# (finding M18). It is kept for the ranks that do run in this process, but a solver
# module must be reachable without it: since WS5, SolverSpec resolves
# solver_module_path to an absolute path on the launching rank, so the spawned children
# receive an absolute path (see docs/writing_a_solver.md). A solver module whose own
# *imports* need paths_to_append still fails in the child -- put those on PYTHONPATH
# instead.
#
# Removed field (2026-09-17, decision 5 / WS8): pickled_observations. Passing it now
# raises TypeError: Configuration.__init__() got an unexpected keyword argument
# 'pickled_observations' -- just delete the argument, there is no replacement.
# Observations always travel from the spawned solver child to the solvers pool and on to
# the sampler as a pickled [observations, solver_tag] payload, so the solver's status
# code never becomes an MPI tag (this removes finding 2.4/M6: a negative solver_tag used
# to be an invalid MPI tag, and the raw path's fixed-size/dtype buffer hazards of finding
# 2.3).
#
# Removed field (2026-09-21, author decision): solver_returns_tag. Passing it raises TypeError;
# no replacement needed -- a Solver.get_observations() may return either the observations or an
# (observations, tag) tuple, and both the sampler (algorithms.py) and the spawned child
# (process_CHILD.py) detect the tuple with isinstance.
#
# Removed field (2026-09-18, finding 1.1): state_dependent_approximation. Passing it now
# raises TypeError: Configuration.__init__() got an unexpected keyword argument
# 'state_dependent_approximation' -- just delete the argument, there is no replacement.
# The DAMH sub-chain always uses the surrogate posterior directly (no shift by the
# model-vs-surrogate error at the current state): the delayed-acceptance correction is
# only valid for a surrogate that is a fixed, state-independent density, so the shifted
# variant was unsound at every Stage.subchain_length, not only for > 1 as previously
# documented.
#
# Invalid combinations that raise (at Stage/proposal construction, i.e. at
# SamplingFramework.run() time, not eagerly at Configuration() time): an unknown
# Stage.algorithm; a DAMH or Hamiltonian-family first stage with no evaluator
# available yet (no surrogate_updater/preloaded snapshots when use_collector=True, or no
# fixed surrogate_evaluator when use_collector=False -- see use_collector below); a pCN
# stage with a non-Gaussian internal prior (FromScipy, GaussianMixture); a Hamiltonian
# proposal without use_surrogate_gradients=True in effect; a Block proposal with an
# explicit Stage.adaptive=True (adaptation is not defined for block proposals; the default
# adaptive=None resolves to False there and to True for every other proposal type).
@dataclass
class Configuration:
    """
    Run-wide sampling configuration: one ``Configuration`` object, built identically on
    every MPI rank (e.g. the same ``Configuration(...)`` call in the launch script run by
    every rank). ``use_solvers_pool`` and ``use_collector`` together decide which process
    role each rank plays (sampler, collector, solvers pool) -- see ``docs/running.md`` for
    the resulting process-count table. Fields whose description below ends with
    "(posterior-affecting)" can change the sampled posterior, the acceptance rate, or the
    run's reproducibility; changing them changes your results, not just performance or
    output layout. See ``docs/configuration.md`` for the full field reference.

    Args:
        no_parameters: Number of unknown parameters of the forward model (the dimension
            of the parameter space). Required. (posterior-affecting)
        no_observations: Number of observed values (the dimension of the observation
            space). Required. (posterior-affecting)
        output_dir: Root directory where samples and other outputs are written. Required.
        use_solvers_pool: ``True``: one MPI rank runs a solvers pool that spawns
            ``no_solvers`` child processes, each importing and constructing the solver from a
            ``SolverSpec`` -- so ``SamplingFramework`` must get ``solver_spec=``; a ready
            ``solver_instance`` cannot be used (it cannot be shipped to a spawned process).
            ``False``: the solver runs inside every sampler process; ``solver_instance=`` or
            ``solver_spec=`` both work. Default: ``True``.
        no_solvers: Number of child solver processes spawned by the solvers pool. Ignored
            when ``use_solvers_pool=False``. Default: ``2``.
        solver_maxprocs: Number of MPI processes used by each spawned solver. Ignored when
            ``use_solvers_pool=False``. Default: ``1``.
        use_collector: If ``False``, no surrogate model is trained during the run -- an
            ``Updater`` only ever runs on the collector rank. DAMH and Hamiltonian-family
            stages then need a pre-trained, fixed ``surrogate_evaluator`` passed to
            ``SamplingFramework`` instead of a ``surrogate_updater`` (no in-run updates,
            no DAMH-SMU). MH-only stages work with any combination of ``use_collector``
            and ``use_solvers_pool``. Default: ``True``. (posterior-affecting)
        paths_to_append: Directories appended to ``sys.path`` in the process that builds
            this ``Configuration`` only; spawned solver processes do not see them. For those,
            set the ``PYTHONPATH`` environment variable of the launching command, e.g.
            ``PYTHONPATH=/path/to/my_models mpiexec -n 4 python3 -m mpi4py script.py`` (with
            Open MPI across several nodes add ``-x PYTHONPATH``); spawned children inherit
            that environment. The solver module itself needs no path: ``SolverSpec`` stores
            its absolute file location. Only the module's own imports of other local code
            need ``PYTHONPATH``. Default: ``None``.
        initial_sample_type: How the first sample of each chain is drawn: ``"lhs"``,
            ``"prior"``, ``"user_specified"``, or ``"continued"``. Default: ``"prior"``. (posterior-affecting)
        initial_samples_distribution: Distribution to draw the initial sample from, in the
            INTERNAL (standardized) coordinates of the prior; used only when
            ``initial_sample_type="user_specified"``. Default: ``None``. (posterior-affecting)
        continued_from_dir: Directory of a previous run to continue from, used only when
            ``initial_sample_type="continued"``. Default: ``None``. (posterior-affecting)
        lhs_scale: Spread of the Latin-hypercube design in the INTERNAL (standardized)
            coordinates of the prior, i.e. in prior standard deviations; used only when
            ``initial_sample_type="lhs"``. Default: ``1.0``. (posterior-affecting)
        min_snapshots_initial: Minimum number of snapshots collected before the first
            surrogate model is trained. Default: ``1``. (posterior-affecting)
        min_snapshots_to_update: New snapshots that must have arrived since the last
            retraining before the surrogate is retrained again. With the default ``0`` the
            collector retrains whenever the updater says another pass would change the
            surrogate (``Updater.needs_retraining``): the neural network then trains
            continuously on the otherwise idle collector rank, while the polynomial, RBF and
            k-d-tree updaters are refitted only after new snapshots arrived, never on identical
            data. Raise it to retrain less often. Default: ``0``. (posterior-affecting)
        transform_before_surrogate: If ``False``, the surrogate is trained and evaluated
            on the internal (standardized) parameters the chain works with; if ``True``, on
            the physical parameters the solver receives. Default: ``False``. (posterior-affecting)
        use_surrogate_gradients: Whether Hamiltonian-family proposals may use surrogate
            autograd. May be silently disabled by ``SamplingFramework`` if the surrogate
            is incompatible -- check ``run_manifest.json`` for the effective value.
            Default: ``True``. (posterior-affecting)
        save_snapshots_to_file: If ``True``, write every exact-model evaluation -- the
            proposed parameters, the observations, the solver tag, the surrogate's prediction
            where one existed, log-likelihood and log-prior, and whether the proposal was
            accepted, rejected or pre-rejected -- to
            ``sampling_output/raw_data/<stage>/rank%04d.csv``, one file per chain. Default: ``False``.
        transform_before_saving: If ``False``, ``samples/*.csv`` stores internal-space
            samples instead of prior-transformed ones. Default: ``True``.
        max_collected_snapshots_per_loop: Maximum number of snapshots the collector
            gathers in one poll loop; also bounds how often the surrogate can be
            retrained. Default: ``1000``. (posterior-affecting)
        max_sampler_isend_requests: Size of the buffer for ``isend`` requests used to
            send snapshots from samplers to the collector (performance/buffering only). Default: ``100``.
        max_buffer_size: Size in bytes of the buffer pre-allocated to receive a pickled
            surrogate evaluator from the collector; raise it for a large surrogate
            (performance/buffering only). Default: ``1 << 30`` (1 GiB).
        torch_threads: Torch CPU threads per rank; only matters with ``NeuralNetworkUpdater``
            (the collector trains it, the samplers evaluate it). ``None`` leaves torch's own
            default (all cores per process), which oversubscribes the node once several ranks
            use the network. Default: ``1``.
        debug: If ``True``, the collector rank prints extra diagnostics about surrogate
            updates. No effect on the other ranks. Default: ``False``.

    Some field combinations are only checked when a run actually starts (at
    ``SamplingFramework.run()`` time, not when ``Configuration()`` is constructed) and
    raise there: an unknown stage algorithm type, a DAMH/Hamiltonian first stage with no
    surrogate available yet, a pCN stage with a non-Gaussian internal prior, or a
    Hamiltonian proposal without ``use_surrogate_gradients`` in effect. See
    ``docs/configuration.md`` for the details.
    """

    # --- problem ---
    no_parameters: int
    no_observations: int
    output_dir: str

    # --- MPI layout ---
    use_solvers_pool: bool = True
    no_solvers: int = 2
    solver_maxprocs: int = 1
    use_collector: bool = True

    # --- solver ---
    paths_to_append: list[str] | None = None

    # --- initial sample ---
    initial_sample_type: Literal["lhs", "prior", "user_specified", "continued"] = "prior"
    initial_samples_distribution: Distribution | None = None
    continued_from_dir: str | None = None
    lhs_scale: float | npt.NDArray = 1.0

    # --- surrogate training ---
    min_snapshots_initial: int = 1
    min_snapshots_to_update: int = 0
    transform_before_surrogate: bool = False
    use_surrogate_gradients: bool = True

    # --- outputs ---
    save_snapshots_to_file: bool = False
    transform_before_saving: bool = True

    # --- performance / advanced ---
    max_collected_snapshots_per_loop: int = 1000
    max_sampler_isend_requests: int = 100
    max_buffer_size: int = 1 << 30
    torch_threads: int | None = 1
    debug: bool = False

    def __post_init__(self) -> None:
        # Requested (as-passed-in) values of the posterior-affecting fields, captured before
        # anything -- this method, SamplingFramework._configure_surrogate_gradients() -- can
        # mutate them. Two users: the requested-vs-effective reporting of
        # use_surrogate_gradients (describe()/run_manifest.json), and the cross-rank consistency
        # check of finding 2.9 (communication.check_configuration_consistency). The *requested*
        # values are the right thing to compare across ranks: a correct script runs the same
        # Configuration(...) call on every rank, whereas the *effective* use_surrogate_gradients
        # may legitimately differ per rank (only the collector holds an Updater, and an updater
        # that cannot do gradients turns the flag off on that rank alone).
        self._requested_posterior_fields: dict[str, Any] = {
            name: getattr(self, name) for name in sorted(POSTERIOR_AFFECTING_FIELDS)}

        if self.torch_threads is not None and (
                not isinstance(self.torch_threads, int) or isinstance(self.torch_threads, bool)
                or self.torch_threads <= 0):
            raise ValueError(f"torch_threads must be None or a positive int, got {self.torch_threads!r}")
        if self.paths_to_append is None:
            self.paths_to_append = []
        else:
            self._append_path()

        size_world = MPI.COMM_WORLD.Get_size()

        # ranks of samplers, collector, solvers pool:
        if self.use_collector and self.use_solvers_pool:
            self.no_samplers = size_world - 2
            self.rank_solvers_pool = size_world - 2
            self.rank_collector = size_world - 1
        elif self.use_collector:  # without solvers pool, solver is local
            self.no_samplers = size_world - 1
            self.rank_solvers_pool = None
            self.rank_collector = size_world - 1
        elif self.use_solvers_pool:  # without collector
            self.no_samplers = size_world - 1
            self.rank_solvers_pool = size_world - 1
            self.rank_collector = None
        else:  # no collector, no solvers pool
            self.no_samplers = size_world
            self.rank_collector = None
            self.rank_solvers_pool = None
        assert self.no_samplers > 0, ("number of MPI processes is too low: use at least 3 MPI processes with collector and "
                                      "solvers pool (1 sampler + pool + collector); 4 recommended")
        self.sampler_ranks = np.arange(self.no_samplers)  # ranks 0, 1, ..., no_samplers-1

        self.continued_samples: npt.NDArray | None = None
        if self.initial_sample_type == "continued":
            if self.continued_from_dir is None:
                raise ValueError("continued_from_dir must be set when initial_sample_type == 'continued'")
            from surrDAMH.modules.continuation import load_last_samples
            self.continued_samples = load_last_samples(
                experiment_dir=self.continued_from_dir,
                no_parameters=self.no_parameters,
                no_chains=self.no_samplers,
            )

    @property
    def use_surrogate_gradients_requested(self) -> bool:
        """``use_surrogate_gradients`` as passed to the constructor, i.e. before
        ``SamplingFramework._configure_surrogate_gradients()`` may have turned it off for an
        incompatible surrogate (see ``run_manifest.json`` and :meth:`describe`)."""
        return bool(self._requested_posterior_fields["use_surrogate_gradients"])

    def requested_posterior_fields(self) -> dict[str, Any]:
        """Copy of the constructor values of every field in :data:`POSTERIOR_AFFECTING_FIELDS`."""
        return dict(self._requested_posterior_fields)

    def requested_posterior_fields_for_comparison(self) -> dict[str, Any]:
        """:meth:`requested_posterior_fields` with every value passed through
        :func:`normalize_for_comparison`: picklable primitives that compare with plain ``==``,
        for the cross-rank check in ``modules.communication.check_configuration_consistency``."""
        return {name: normalize_for_comparison(value)
                for name, value in self._requested_posterior_fields.items()}

    def _append_path(self) -> None:
        assert self.paths_to_append is not None
        for path in self.paths_to_append:
            sys.path.append(path)

    def describe(self, use_surrogate_gradients_requested: bool | None = None) -> str:
        """
        Multi-line summary of the **effective** configuration, printed once on rank 0 by
        ``SamplingFramework.run()`` and by ``run_local()``.

        Every field of the dataclass appears as ``name=value``; a trailing ``*`` marks the
        posterior-/acceptance-rate-/reproducibility-affecting ones listed in the class
        docstring. The MPI role layout derived in ``__post_init__`` is appended, and so is
        the requested-versus-effective value of ``use_surrogate_gradients`` when they differ
        (``SamplingFramework`` may disable it for an incompatible surrogate).

        Args:
            use_surrogate_gradients_requested: the value the user passed in, before
                ``SamplingFramework._configure_surrogate_gradients`` possibly turned it off.
                Omit it (``None``) to show only the effective value.

        Returns:
            A string of about six lines, ready to ``print``.
        """
        extra = [f"[MPI layout] no_samplers={self.no_samplers}, rank_collector={self.rank_collector}, "
                 f"rank_solvers_pool={self.rank_solvers_pool}, size_world={MPI.COMM_WORLD.Get_size()}"]
        if use_surrogate_gradients_requested is not None and \
                bool(use_surrogate_gradients_requested) != bool(self.use_surrogate_gradients):
            extra.append(f"[effective] use_surrogate_gradients: requested="
                         f"{bool(use_surrogate_gradients_requested)}, in effect={bool(self.use_surrogate_gradients)}"
                         " (disabled by SamplingFramework, see the warning above)")
        if self.continued_samples is not None:
            extra.append(f"[continuation] loaded initial samples: shape={tuple(self.continued_samples.shape)}")
        return describe_fields(self, POSTERIOR_AFFECTING_FIELDS,
                               "Configuration (* = posterior-/acceptance-rate-/reproducibility-affecting):",
                               extra)
