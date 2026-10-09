#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic mode (2026-10-08): a stage layout and a surrogate chosen from a budget.

``Problem.run_sampling_auto(conf, budget=..., mode=...)`` / ``run_sampling_local_auto`` call
:func:`plan_auto`, apply the configuration fields it owns (``Configuration._set_by_auto``) and run
the explicit stage list it produced with ``run_sampling`` / ``run_sampling_local``. The resolved
plan is printed at start-up and recorded under ``"auto"`` in ``run_manifest.json``, so an expert
can copy the stage list into a plain ``run_sampling`` call.

Layout (``library_notes/25_robust_by_default_roadmap_2026-10-08.md`` §5). Every number below is a
PLACEHOLDER chosen before validation (note 25 §6 will tune them, the warm-up share first):

* ``d`` = number of parameters, ``C`` = number of chains (sampler ranks; 1 locally), ``K = 4``.
* evaluation budget: ``T = budget - test_data_size``, ``B = T // C`` exact evaluations per chain.
* ``B < 50 d`` (a surrogate trained on fewer points does not pay, note 19 §4), or an MPI run without
  a collector: one adaptive random-walk MH stage with the whole budget (no held-out set then).
* otherwise an excluded MH warm-up with ``n0 = clip(20 d, 0.05 B, 0.25 B)`` evaluations, then ``K``
  DAMH-SMU chunks (sub-chain length 1) of ``(B - n0) // K`` evaluations, the remainder added to the
  last chunk; no frozen stage.
* time budget: warm-up ``0.15 time_limit`` (excluded), chunks ``0.85 time_limit / K`` each.
* ``mode="robust"``: the chunks use an adaptive ``RandomWalk()``; ``mode="fast"``:
  ``Hamiltonian(num_steps=30, integrator="dimension_robust", mass=1.0)`` with a dual-averaged step.

Not covered here (later steps of note 25): the safety components S1/S2, the carry-over of the
Welford statistics across chunks, ``continue_sampling_auto`` (an automatic continuation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from surrDAMH.proposals import Hamiltonian, RandomWalk
from surrDAMH.stages import Stage, stage_name

if TYPE_CHECKING:
    from surrDAMH.configuration import Configuration
    from surrDAMH.core import Problem
    from surrDAMH.modules.test_data import TestData
    from surrDAMH.surrogates.parent import Updater

#: Values of ``mode``.
AUTO_MODES: tuple[str, ...] = ("robust", "fast")

# --- placeholder numbers of the layout rule (note 25 §5; to be tuned by the §6 validation) ---
#: number of DAMH chunks after the warm-up
AUTO_CHUNKS = 4
#: below ``MIN_EVALUATIONS_PER_DIMENSION * d`` evaluations per chain no surrogate is used
MIN_EVALUATIONS_PER_DIMENSION = 50
#: warm-up target ``WARM_UP_PER_DIMENSION * d``, clipped to [WARM_UP_MIN_SHARE, WARM_UP_MAX_SHARE] * B
WARM_UP_PER_DIMENSION = 20
WARM_UP_MIN_SHARE = 0.05
WARM_UP_MAX_SHARE = 0.25
#: share of a time budget spent in the warm-up
WARM_UP_TIME_SHARE = 0.15
#: held-out set: min(TEST_DATA_MAX, max(TEST_DATA_PER_DIMENSION * d, TEST_DATA_BUDGET_SHARE * budget))
TEST_DATA_MAX = 256
TEST_DATA_PER_DIMENSION = 2
TEST_DATA_BUDGET_SHARE = 0.02
#: held-out set size floor with a time budget (no evaluation count to take a share of)
TEST_DATA_TIME_BUDGET = 64
#: seed of the held-out prior draws (the default of ``TestData.generate``)
TEST_DATA_SEED = 25347
#: Hamiltonian proposal of the fast mode
FAST_NUM_STEPS = 30
#: default network of the automatic mode
DEFAULT_NETWORK_HPARAMS: dict[str, Any] = {"hidden_layer_sizes": (64, 64), "activation": "silu",
                                           "solver": "adamw", "seed": 0}

#: Configuration fields Auto refuses to touch: Auto is not a continuation.
_LINEAGE_DEFAULTS: dict[str, Any] = {"stage_index_offset": 0, "no_stages_lineage": None, "lineage_generation": 0}
#: Defaults of the surrogate-training fields Auto sets (a different value = set by the user).
_SNAPSHOT_DEFAULTS: dict[str, int] = {"min_snapshots_initial": 1, "min_snapshots_to_update": 0}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


@dataclass
class AutoTestDataRequest:
    """
    A deferred held-out set: ``size`` prior draws (seed ``seed``) evaluated with the exact model
    where the request is resolved -- on the collector rank inside ``Problem.run_sampling`` (so
    the samplers spend no evaluation on it), or before the run in ``run_sampling_local_auto``.
    The generated set is saved into the run (``TestData.save``).
    """
    size: int
    seed: int = TEST_DATA_SEED

    def generate(self, problem: "Problem", transform_before_surrogate: bool) -> "TestData":
        """``TestData.generate(problem, size, transform_before_surrogate, seed)``."""
        from surrDAMH.modules.test_data import TestData
        return TestData.generate(problem, size=int(self.size), transform_before_surrogate=bool(transform_before_surrogate),
                                 seed=int(self.seed))


@dataclass
class AutoPlan:
    """
    The resolved automatic layout (what :func:`plan_auto` returns).

    Attributes:
        stages: the explicit stage list that runs.
        surrogate_updater: the updater handed to the run (``None``: no surrogate).
        surrogate_test_data: a user-given set, an :class:`AutoTestDataRequest`, or ``None``.
        conf_settings: the ``Configuration`` fields Auto sets (``Configuration._set_by_auto``).
        per_chain_budget: ``B`` (exact evaluations per chain), ``None`` for a time budget.
        warm_up: ``{"evaluations": n0}`` or ``{"time_limit": t0}``, ``None`` for a single MH stage.
        chunks: number of DAMH chunks (0 for a single MH stage).
        test_data_size: auto-generated held-out points (subtracted from an evaluation budget).
        mode: ``"robust"`` or ``"fast"``.
        notes: human-readable reasons for every decision that is not the plain rule.
    """
    stages: list[Stage]
    surrogate_updater: Any
    surrogate_test_data: Any
    conf_settings: dict[str, Any]
    per_chain_budget: int | None
    warm_up: dict[str, Any] | None
    chunks: int
    test_data_size: int
    mode: str
    notes: list[str] = field(default_factory=list)
    budget: int | None = None
    time_limit: float | None = None
    no_samplers: int = 1
    local: bool = False
    proposal: Any = None
    surrogate: Any = "none"

    def stage_names(self) -> list[str]:
        """Directory names of the planned stages (``alg0000_MH-adaptive``, ...)."""
        return [stage_name(stage, i) for i, stage in enumerate(self.stages)]

    def manifest_entry(self) -> dict[str, Any]:
        """The manifest's ``"auto"`` entry (also ``SamplingRun.auto``); JSON-safe."""
        from surrDAMH.modules.manifest import _json_safe
        return {
            "mode": self.mode,
            "budget": self.budget,
            "time_limit": self.time_limit,
            "no_samplers": int(self.no_samplers),
            "per_chain_budget": self.per_chain_budget,
            "test_data_size": int(self.test_data_size),
            "warm_up": _json_safe(self.warm_up),
            "chunks": int(self.chunks),
            "stage_names": self.stage_names(),
            "proposal": repr(self.proposal) if self.proposal is not None else None,
            "surrogate": _json_safe(self.surrogate),
            "conf_settings": _json_safe(dict(self.conf_settings)),
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        """Multi-line summary of the plan, printed once at start-up (rank 0 / local)."""
        if self.budget is not None:
            what = f"budget={self.budget} exact evaluations in total"
        else:
            what = f"time_limit={self.time_limit:g} s per run"
        lines = [f"Auto plan (mode={self.mode!r}, {what}, {self.no_samplers} chain(s), "
                 f"{'local' if self.local else 'MPI'}):"]
        if self.test_data_size:
            where = "before the run" if self.local else "on the collector rank"
            counted = ", counted in the budget" if self.budget is not None else ""
            lines.append(f"  held-out test data: {self.test_data_size} prior draws evaluated {where}{counted}")
        if self.per_chain_budget is not None:
            lines.append(f"  per-chain budget B = {self.per_chain_budget} exact evaluations")
        if self.warm_up is None:
            lines.append("  layout: one MH stage, adaptive RandomWalk, no surrogate in the chain")
        else:
            if "evaluations" in self.warm_up:
                lines.append(f"  warm-up: MH, adaptive RandomWalk, {self.warm_up['evaluations']} evaluations per "
                             "chain, excluded (burn-in)")
            else:
                lines.append(f"  warm-up: MH, adaptive RandomWalk, {self.warm_up['time_limit']:g} s, excluded (burn-in)")
            chunk_sizes = [s.max_evaluations if self.budget is not None else s.time_limit for s in self.stages[1:]]
            unit = "evaluations" if self.budget is not None else "s"
            lines.append(f"  {self.chunks} DAMH-SMU chunks (sub-chain length 1), {unit} per chain: "
                         f"{', '.join(f'{c:g}' if isinstance(c, float) else str(c) for c in chunk_sizes)}")
            lines.append(f"  chunk proposal: {self.proposal!r}")
        if isinstance(self.surrogate, dict):
            hparams = ", ".join(f"{k}={v!r}" for k, v in self.surrogate.get("hparams", {}).items())
            origin = "default" if self.surrogate.get("default") else "given"
            lines.append(f"  surrogate ({origin}): {self.surrogate['class']}({hparams})")
        else:
            lines.append(f"  surrogate: {self.surrogate}")
        if self.conf_settings:
            lines.append("  configuration set by Auto: "
                         + ", ".join(f"{k}={v!r}" for k, v in self.conf_settings.items()))
        lines.append(f"  stages: {', '.join(self.stage_names())}")
        for note in self.notes:
            lines.append(f"  [note] {note}")
        return "\n".join(lines)


def _surrogate_summary(updater: Any, default: bool) -> Any:
    if updater is None:
        return "none"
    if default:
        return {"class": type(updater).__name__, "default": True,
                "hparams": {k: (list(v) if isinstance(v, tuple) else v) for k, v in DEFAULT_NETWORK_HPARAMS.items()}}
    # a given updater: its public scalar settings (arrays and training-state counters left out)
    try:
        hparams = {key: value for key, value in vars(updater).items()
                   if not key.startswith("_") and "snapshot" not in key
                   and isinstance(value, (bool, int, float, str))}
    except Exception:
        hparams = {}
    return {"class": type(updater).__name__, "default": False, "hparams": hparams}


def _default_updater(problem: "Problem") -> "Updater":
    from surrDAMH.surrogates.torch_perceptron_minibatches import NeuralNetworkUpdater
    return NeuralNetworkUpdater(no_parameters=problem.no_parameters, no_observations=problem.no_observations,
                                **DEFAULT_NETWORK_HPARAMS)


def plan_auto(problem: "Problem", conf: "Configuration", *, budget: int | None, time_limit: float | None,
              mode: str, local: bool, surrogate_updater: "Updater | None" = None,
              surrogate_test_data: Any = None) -> AutoPlan:
    """
    Resolve the automatic layout; pure apart from building the default network (no file, no MPI).
    Identical on every rank for identically built arguments.

    Args:
        problem: the ``Problem`` (``no_parameters``, ``no_observations``, ``solver_instance``).
        conf: the run's configuration (read only; the fields to set are returned in
            ``conf_settings``). Its ``no_samplers`` is the number of chains of an MPI run.
        budget: TOTAL exact evaluations over all chains, including the held-out set.
        time_limit: wall-clock seconds per run (instead of ``budget``).
        mode: ``"robust"`` (adaptive random walk in the DAMH chunks) or ``"fast"`` (Hamiltonian on
            the surrogate's gradients).
        local: plan for ``run_sampling_local`` (one chain, the surrogate trained in process).
        surrogate_updater: expert override of the default network.
        surrogate_test_data: expert override of the held-out set (used as given, not subtracted).

    Returns:
        An :class:`AutoPlan`.

    Raises:
        ValueError: both or none of ``budget``/``time_limit``; a non-positive one; an unknown
            ``mode``; lineage fields of ``conf`` not at their defaults; ``mode="fast"`` with an
            updater without gradients, with ``use_surrogate_gradients=False`` or with
            ``transform_before_surrogate=True``; a budget smaller than one evaluation per chain.
    """
    if (budget is None) == (time_limit is None):
        raise ValueError("give exactly one of budget= (total exact evaluations) or time_limit= (seconds)")
    if mode not in AUTO_MODES:
        raise ValueError(f"mode must be one of {AUTO_MODES}, got {mode!r}")
    if budget is not None:
        if isinstance(budget, bool) or int(budget) != budget or budget <= 0:
            raise ValueError(f"budget must be a positive int, got {budget!r}")
        budget = int(budget)
    if time_limit is not None:
        time_limit = float(time_limit)
        if not time_limit > 0.0 or time_limit == float("inf"):
            raise ValueError(f"time_limit must be a positive finite number of seconds, got {time_limit!r}")
    for name, default in _LINEAGE_DEFAULTS.items():
        if (getattr(conf, name, default) or default) != default:
            raise ValueError(f"{name}={getattr(conf, name)!r}: Auto is not a continuation (continue_sampling_auto does "
                             f"not exist yet); leave {name} at its default")

    d = int(problem.no_parameters)
    no_chains = 1 if local else int(conf.no_samplers)
    notes: list[str] = []
    has_collector = local or bool(conf.use_collector)

    # --- held-out set (planned first: it is subtracted from an evaluation budget) ---
    user_test_data = surrogate_test_data is not None
    if user_test_data:
        planned_test_size = 0
    elif problem.solver_instance is None:
        planned_test_size = 0
    elif budget is not None:
        planned_test_size = min(TEST_DATA_MAX, max(TEST_DATA_PER_DIMENSION * d, int(TEST_DATA_BUDGET_SHARE * budget)))
    else:
        planned_test_size = min(TEST_DATA_MAX, max(TEST_DATA_PER_DIMENSION * d, TEST_DATA_TIME_BUDGET))

    # --- single MH stage or warm-up + chunks ---
    per_chain_budget: int | None = None
    use_surrogate = True
    if not has_collector:
        use_surrogate = False
        notes.append("use_collector=False: no surrogate can be trained in an MPI run without a collector, so "
                     "one adaptive MH stage runs with the whole budget")
    if budget is not None:
        per_chain_budget = (budget - planned_test_size) // no_chains
        if use_surrogate and per_chain_budget < MIN_EVALUATIONS_PER_DIMENSION * d:
            use_surrogate = False
            notes.append(f"per-chain budget {per_chain_budget} < {MIN_EVALUATIONS_PER_DIMENSION}*d = "
                         f"{MIN_EVALUATIONS_PER_DIMENSION * d}: a surrogate trained on so few points does not pay, "
                         "so one adaptive MH stage runs with the whole budget")
        if not use_surrogate:
            per_chain_budget = budget // no_chains  # no held-out set without a surrogate
        if per_chain_budget < 1:
            raise ValueError(f"budget={budget} is smaller than the number of chains ({no_chains})")

    test_data_size = planned_test_size if use_surrogate else 0
    if use_surrogate and not user_test_data and problem.solver_instance is None:
        notes.append("no held-out test data: the problem has a SolverSpec, so no rank but the solver processes "
                     "holds a solver to evaluate it (pass surrogate_test_data= to monitor the surrogate)")
    if user_test_data and use_surrogate:
        notes.append("surrogate_test_data given: used as given, not subtracted from the budget")
    if local and use_surrogate and user_test_data:
        notes.append("the local runner does not monitor surrogate quality; the given held-out set is not used")
    elif local and use_surrogate and test_data_size:
        notes.append("the local runner does not monitor surrogate quality; the held-out set is only saved "
                     "(sampling_output/surrogate_test_data.npz) for a later run")

    # --- stages ---
    stages: list[Stage]
    warm_up: dict[str, Any] | None = None
    chunks = 0
    proposal: Any = RandomWalk()
    if not use_surrogate:
        if budget is not None:
            stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=per_chain_budget)]
        else:
            stages = [Stage(algorithm="MH", proposal=RandomWalk(), time_limit=time_limit)]
    else:
        chunks = AUTO_CHUNKS

        def chunk_proposal():
            if mode == "fast":
                return Hamiltonian(num_steps=FAST_NUM_STEPS, integrator="dimension_robust", mass=1.0)
            return RandomWalk()

        proposal = chunk_proposal()
        if budget is not None:
            assert per_chain_budget is not None
            n0 = int(_clip(WARM_UP_PER_DIMENSION * d, WARM_UP_MIN_SHARE * per_chain_budget,
                           WARM_UP_MAX_SHARE * per_chain_budget))
            n0 = max(1, n0)
            warm_up = {"evaluations": n0}
            per_chunk, remainder = divmod(per_chain_budget - n0, chunks)
            stages = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=n0, is_excluded=True)]
            for k in range(chunks):
                evaluations = per_chunk + (remainder if k == chunks - 1 else 0)
                stages.append(Stage(algorithm="DAMH", proposal=chunk_proposal(), subchain_length=1,
                                    surrogate_model_updates=True, max_evaluations=evaluations))
        else:
            assert time_limit is not None
            t0 = WARM_UP_TIME_SHARE * time_limit
            t_chunk = (1.0 - WARM_UP_TIME_SHARE) * time_limit / chunks
            warm_up = {"time_limit": t0}
            stages = [Stage(algorithm="MH", proposal=RandomWalk(), time_limit=t0, is_excluded=True)]
            for _ in range(chunks):
                stages.append(Stage(algorithm="DAMH", proposal=chunk_proposal(), subchain_length=1,
                                    surrogate_model_updates=True, time_limit=t_chunk))
            notes.append(f"time budget: the warm-up floor of {WARM_UP_PER_DIMENSION}*d = {WARM_UP_PER_DIMENSION * d} "
                         "evaluations per chain cannot be enforced; a too short time_limit leaves the first "
                         "surrogate undertrained")

    # --- surrogate ---
    updater = surrogate_updater
    default_updater = False
    needs_collector_updater = not local and bool(conf.use_collector)  # the collector rank requires an updater
    if use_surrogate or needs_collector_updater:
        if updater is None:
            updater = _default_updater(problem)
            default_updater = True
        if not use_surrogate:
            notes.append("the collector rank still receives the snapshots and trains the surrogate, but no stage "
                         "uses it (training data for a later run)")
    elif updater is not None:
        notes.append(f"surrogate_updater {type(updater).__name__} ignored: no stage uses a surrogate")
        updater = None
    if use_surrogate and mode == "fast" and not updater.supports_gradients():
        raise ValueError(f"mode='fast' needs a surrogate with gradients; {type(updater).__name__} has none "
                         "(use mode='robust', or a NeuralNetworkUpdater)")

    # --- configuration fields ---
    conf_settings: dict[str, Any] = {}
    if conf.initial_sample_type == "prior":
        conf_settings["initial_sample_type"] = "lhs"
    if use_surrogate:
        wanted_gradients = (mode == "fast")
        if mode == "fast":
            if not conf.use_surrogate_gradients:
                raise ValueError("mode='fast' needs Configuration(use_surrogate_gradients=True) (the default): the "
                                 "Hamiltonian proposal is driven by the surrogate's gradients")
            if conf.transform_before_surrogate:
                raise ValueError("mode='fast' needs Configuration(transform_before_surrogate=False) (the default): "
                                 "surrogate gradients are only available in the internal space")
        conf_settings["use_surrogate_gradients"] = wanted_gradients
        if budget is not None:
            assert warm_up is not None
            wanted_initial = max(1, min(10 * d, (warm_up["evaluations"] * no_chains) // 2))
        else:
            wanted_initial = max(1, 10 * d)
        wanted = {"min_snapshots_initial": wanted_initial, "min_snapshots_to_update": max(20, 2 * d)}
        for name, value in wanted.items():
            current = getattr(conf, name)
            if current != _SNAPSHOT_DEFAULTS[name]:
                notes.append(f"{name}={current} set by the user is kept (Auto would use {value})")
            else:
                conf_settings[name] = value

    surrogate_test: Any = surrogate_test_data
    if not use_surrogate:
        surrogate_test = None
        if user_test_data:
            notes.append("surrogate_test_data ignored: no stage uses a surrogate")
    elif test_data_size:
        surrogate_test = AutoTestDataRequest(size=test_data_size)

    return AutoPlan(stages=stages, surrogate_updater=updater, surrogate_test_data=surrogate_test,
                    conf_settings=conf_settings, per_chain_budget=per_chain_budget, warm_up=warm_up,
                    chunks=chunks, test_data_size=test_data_size, mode=mode, notes=notes, budget=budget,
                    time_limit=time_limit, no_samplers=no_chains, local=local,
                    proposal=proposal if use_surrogate else stages[0].proposal,
                    surrogate=_surrogate_summary(updater, default_updater))


__all__ = ["plan_auto", "AutoPlan", "AutoTestDataRequest", "AUTO_MODES"]
