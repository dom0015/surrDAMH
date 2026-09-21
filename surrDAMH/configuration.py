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


@dataclass
class Configuration:
    """
    Run-wide configuration, identical on every MPI rank. Since 2026-09-18 (finding 2.9) this is
    checked: ``SamplingFramework.run()`` broadcasts rank 0's *requested* values of
    ``POSTERIOR_AFFECTING_FIELDS`` and every rank asserts equality, so a script that builds a
    different ``Configuration`` per rank fails at start-up instead of sampling a different
    posterior per chain (``modules.communication.check_configuration_consistency``). Fields
    outside that set are not compared. ``__post_init__`` derives
    the MPI role layout (``no_samplers``, ``rank_collector``, ``rank_solvers_pool``)
    from ``MPI.COMM_WORLD``'s size and the two topology flags below; see
    ``docs/running.md`` for the process-count table and ``docs/configuration.md`` for
    the full field reference (generated from this dataclass).

    Posterior-, acceptance-rate- or reproducibility-affecting fields: ``transform_before_surrogate``
    (which space the surrogate is trained on),
    ``min_snapshots_initial``/``min_snapshots_to_update``/``max_collected_snapshots_per_loop``
    (control exactly when the surrogate is (re)trained, hence the DAMH-SMU accept/reject
    sequence for runs that use a collector), ``use_surrogate_gradients`` (may be silently
    disabled by ``SamplingFramework`` for an incompatible surrogate, see
    ``run_manifest.json``'s ``use_surrogate_gradients_requested``), ``initial_sample_type``
    (every type is reproducible since G4, 2026-09-17: ``"prior"``/``"user_specified"`` draw
    from ``np.random.default_rng(10*no_stages*rank_world + 3)``, see
    ``surrDAMH.modules.seeds``), and everything under "invalid combinations" below.
    ``transform_before_saving`` and ``save_snapshots_to_file`` affect only what is
    written to disk, not the posterior itself; ``torch_threads`` is a performance-only
    knob (intra-op CPU thread count for torch) and does not affect the posterior,
    acceptance rate or surrogate accuracy.

    Ineffective for spawned children: ``paths_to_append`` is appended to ``sys.path``
    in the process that constructs this ``Configuration`` only -- it does NOT reach
    solver-pool children spawned by ``MPI.Comm.Spawn``, which unpickle ``conf`` without
    running ``__post_init__`` (finding M18). It is kept for the ranks that do run in this
    process, but a solver module must be reachable without it: since WS5,
    ``SolverSpec`` resolves ``solver_module_path`` to an absolute path on the launching
    rank, so the spawned children receive an absolute path (see
    ``docs/writing_a_solver.md``). A solver module whose own *imports* need
    ``paths_to_append`` still fails in the child -- put those on ``PYTHONPATH`` instead.

    Removed field (2026-09-17, decision 5 / WS8): ``pickled_observations``. Passing it now
    raises ``TypeError: Configuration.__init__() got an unexpected keyword argument
    'pickled_observations'`` -- just delete the argument, there is no replacement. Observations
    always travel from the spawned solver child to the solvers pool and on to the sampler as a
    pickled ``[observations, solver_tag]`` payload, so the solver's status code never becomes an
    MPI tag (this removes finding 2.4/M6: a negative ``solver_tag`` used to be an invalid MPI
    tag, and the raw path's fixed-size/dtype buffer hazards of finding 2.3).

    Removed field (2026-09-18, finding 1.1): ``state_dependent_approximation``. Passing it now
    raises ``TypeError: Configuration.__init__() got an unexpected keyword argument
    'state_dependent_approximation'`` -- just delete the argument, there is no replacement.
    The DAMH sub-chain always uses the surrogate posterior directly (no shift by the
    model-vs-surrogate error at the current state): the delayed-acceptance correction is only
    valid for a surrogate that is a fixed, state-independent density, so the shifted variant was
    unsound at every ``Stage.subchain_max_length``, not only for ``> 1`` as previously documented.

    Invalid combinations that raise (at ``Stage``/proposal construction, i.e. at
    ``SamplingFramework.run()`` time, not eagerly at ``Configuration()`` time):
    an unknown ``Stage.algorithm_type``; a DAMH or Hamiltonian-family first stage with
    no evaluator available yet (no ``surrogate_updater``/preloaded snapshots when
    ``use_collector=True``, or no fixed ``surrogate_evaluator`` when
    ``use_collector=False`` -- see ``use_collector`` below); a pCN stage with a
    non-Gaussian internal prior (``FromScipy``, ``GaussianMixture``); a Hamiltonian
    proposal without ``use_surrogate_gradients=True`` in effect; ``Stage.adaptive=True``
    together with ``proposal_type="pCN"`` is not an error but is silently forced back to
    ``adaptive=False`` by ``Stage.__post_init__`` (a printed warning, not an exception).
    """

    no_parameters: int  # number of unknowns (i.e. parameters of the forward model)
    no_observations: int  # number of observed values (i.e. outputs of the forward model)
    output_dir: str  # directory where samples and other outputs will be saved
    use_solvers_pool: bool = True  # if False, the solver runs locally on each sampler process
    no_solvers: int = 2  # number of child solvers spawned by solvers pool
    solver_maxprocs: int = 1  # processes used by each spawned solver
    solver_returns_tag: bool = False  # if True, solver returns Tuple(observations, tag:int), negative tag indicates solver error
    use_collector: bool = True  # if False, no surrogate model is trained during the run: an Updater only ever runs on the collector rank, so DAMH/Hamiltonian stages then need a pre-trained, fixed surrogate_evaluator= passed to SamplingFramework (no in-run updates/DAMH-SMU); MH-only stages work with any combination of use_collector/use_solvers_pool
    save_snapshots_to_file: bool = False  # save all obtained snapshots to file
    transform_before_saving: bool = True  # if False, save samples based on internal distribution
    transform_before_surrogate: bool = False  # if False, construct surrogate on internal distribution; posterior-affecting: changes what the surrogate is trained/evaluated on
    initial_sample_type: Literal["lhs", "prior", "user_specified", "continued"] = "prior"  # specifies how to generate initial samples; all four are reproducible - "prior"/"user_specified" draw from a per-rank np.random.default_rng seeded from modules.seeds.initial_sample_seed (G4)
    initial_samples_distribution: Distribution | None = None  # only if initial_sample_type == "user_specified"
    continued_from_dir: str | None = None  # experiment directory to continue from (only if initial_sample_type == "continued")
    lhs_scale: float | npt.NDArray = 1.0  # only if initial_sample_type == "lhs"
    min_snapshots_initial: int = 1  # minimal number of snapshots for the construction of initial surrogate model; posterior-affecting for DAMH-SMU (controls when the first surrogate is trained)
    min_snapshots_to_update: int = 1  # how many snapshots (at least) have to be added to update the surrogate model; posterior-affecting for DAMH-SMU (controls when later retrains happen)
    max_collected_snapshots_per_loop: int = 1000  # maximal number of snapshots to be collected in one loop (collector-side batching/performance knob)
    max_sampler_isend_requests: int = 100  # size of the buffer for isend requests (sending snapshots from samplers to collector; performance/buffering knob)
    use_surrogate_gradients: bool = True  # whether to allow autograd in pytorch surrogate; may be silently forced to False by SamplingFramework if the surrogate/settings are incompatible (see run_manifest.json)
    paths_to_append: list[str] | None = None  # appended to sys.path in this process only; does NOT reach spawned solver-pool children (M18) -- ineffective for them, and no longer needed to find the solver module itself (SolverSpec stores an absolute path since WS5)
    max_buffer_size: int = 1 << 30  # size (bytes) of the pre-allocated irecv buffer used to receive a pickled Evaluator from the collector; performance/buffering knob, not posterior-affecting
    debug: bool = False  # collector-side: print extra diagnostics; not posterior-affecting
    torch_threads: int | None = 1  # number of torch intra-op CPU threads set on every rank that has torch loaded (samplers evaluating the NN surrogate; the collector when it trains on CPU -- irrelevant on GPU); None = leave torch's default (all cores per process), which oversubscribes the node when several ranks evaluate the NN

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
