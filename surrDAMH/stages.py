#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sampling stages: what the chain does, in which order.

``problem.run_sampling(conf, stages=[...])`` runs them in sequence on every chain; each
stage continues from the last sample of the previous one. A stage combines an algorithm
(plain MH, or surrogate-accelerated DAMH), a proposal (``surrDAMH.proposals``) and a
stopping rule::

    from surrDAMH.proposals import RandomWalk
    stages = [
        surrDAMH.Stage(max_evaluations=1000),                              # MH warm-up, proposal tuned online
        surrDAMH.Stage(algorithm="DAMH", subchain_length=5, max_evaluations=500),
    ]

See ``docs/stages.md`` for every field and ``docs/concepts.md`` for the algorithms.
"""

import sys
from dataclasses import dataclass, field
from typing import Literal as _Literal

import numpy as np

from surrDAMH.modules.describe import describe_fields
import surrDAMH.proposals as _proposals  # private alias: surrDAMH.stages must offer Stage only

#: Fields of ``Stage`` that change the sampled distribution or the acceptance rate.
#: ``describe()`` marks them with a trailing ``*``.
POSTERIOR_AFFECTING_FIELDS: frozenset[str] = frozenset({
    "algorithm", "proposal", "subchain_length", "surrogate_model_updates",
    "use_only_surrogate", "is_excluded", "burn_in", "exact_step_probability", "audit_prerejected",
})

#: Stage fields of the S2 safety components (2026-10-09); only meaningful on a DAMH stage.
_S2_FIELDS: tuple[str, ...] = ("exact_step_probability", "audit_prerejected")


@dataclass
class Stage:
    """
    One sampling stage: algorithm + proposal + stopping rule. Stages run in order on every
    chain; each continues from the last sample of the previous one::

        Stage(max_evaluations=1000)                                    # MH, random walk tuned online
        Stage(proposal=RandomWalk(scale=0.5), max_evaluations=1000)    # MH, fixed step
        Stage(algorithm="DAMH", subchain_length=5, max_evaluations=200)  # surrogate-accelerated
        Stage(proposal=PCN(beta=0.2), time_limit=60)

    with ``from surrDAMH.proposals import RandomWalk, PCN, Hamiltonian, Block``.

    Args:
        algorithm: ``"MH"`` (every proposal is evaluated with the exact model) or ``"DAMH"``
            (delayed acceptance: a surrogate pre-screens proposals, the exact model confirms). Default: ``"MH"``.
        proposal: how new samples are proposed; a ``RandomWalk`` (default), ``PCN``,
            ``Hamiltonian`` or ``Block`` from ``surrDAMH.proposals``. Each carries its own step
            and ``adaptive`` switch; by default the step is tuned online and handed to the
            following stages. Default: ``RandomWalk()``.
        max_evaluations: stop after this many exact-model evaluations. Default: unlimited.
        max_samples: stop after this many samples (chain states). Default: unlimited.
        time_limit: stop after this many seconds. Default: unlimited.
        subchain_length: DAMH only: surrogate-only MH steps between two exact-model evaluations. Default: ``1``.
        surrogate_model_updates: DAMH only (default ``True`` there): keep receiving retrained
            surrogates during the stage (DAMH-SMU); ``False`` freezes the surrogate for the
            stage. On an MH stage only a ``Hamiltonian`` proposal can opt in (gradient surrogate
            refreshed once per iteration). Default: ``None``.
        use_only_surrogate: evaluate the surrogate instead of the exact model (exploration
            only, the samples do not follow the posterior). Default: ``False``.
        save_to_file: write this stage's samples to ``sampling_output/``. Default: ``True``.
        send_snapshots_to_collector: use this stage's exact-model evaluations as surrogate
            training data. Set ``False`` on a final frozen-surrogate stage: nothing will use the
            retrained surrogate, and refitting an RBF or k-d tree on every new batch slows the
            samplers down (a start-up note says so). Default: ``True``.
        is_excluded: discarded exploration: the next stage RESTARTS from the sample this stage
            started at, and the stage's chains are left out of the posterior by default. Default: ``False``.
        burn_in: warm-up: the chain CONTINUES into the next stage (exactly as after a normal
            stage), but this stage's samples are left out of the posterior by default. Both flags
            only set the default ``include = 0`` of the report's ``selection.json``; every chain
            stays visible in the report. ``burn_in`` and ``is_excluded`` exclude each other. Default: ``False``.
        exact_step_probability: DAMH only, in ``[0, 1]``: probability that an outer iteration is
            an EXACT Metropolis-Hastings step instead of the delayed-acceptance step -- the
            proposal comes from a separate adaptive random walk (started from the carried
            ``scale`` when there is one, adapted on its own exact acceptances, never carried
            over) and is accepted with the exact likelihood; the surrogate only pre-scores it.
            The chain then follows the mixture of two kernels that both keep the exact posterior
            invariant, so it is exact for any surrogate, and it can move into regions the
            surrogate wrongly rules out (note 22's "blind spot"), also in a frozen stage. Cost:
            one exact evaluation per exact step (they count towards ``max_evaluations``). The
            stage proposal is not adapted on exact steps. Default: ``0.0`` (off).
        audit_prerejected: DAMH only, in ``[0, 1]``: probability that a proposal the surrogate
            pre-rejected is evaluated with the exact model anyway. The decision is not revisited
            (the chain's law is unchanged); the evaluation is written to ``raw_data`` as an
            ``audited`` row and sent to the collector as a training snapshot with multiplicity 0,
            and audits where the exact log-likelihood exceeds the surrogate's by more than
            ``AUDIT_HIDDEN_NATS`` (5 nats) are counted as ``audit_hidden`` in ``notes``. Cost: one
            exact evaluation per audit (counted towards ``max_evaluations``). On a frozen stage
            (``surrogate_model_updates=False``) the audit only measures: nothing retrains on it.
            Default: ``0.0`` (off).
        name: set by the library (``alg0000_MH``, ...); the stage's output-directory name. Default: ``None``.

    If no stopping rule is given, ``max_evaluations=10`` is used with a printed note.
    Fields marked ``*`` in ``describe()`` change the posterior or the acceptance rate.
    Full field table and the algorithms: ``docs/stages.md``, ``docs/concepts.md``.
    """

    algorithm: _Literal["MH", "DAMH"] = "MH"
    proposal: _proposals.ProposalSpec = field(default_factory=_proposals.RandomWalk)
    max_evaluations: int = sys.maxsize
    max_samples: int = sys.maxsize
    time_limit: float = np.inf
    subchain_length: int = 1
    # Tri-state on input, always a plain bool after __post_init__ (WS7, 2026-09-18):
    #   None  - True for DAMH (DAMH-SMU), False for MH;
    #   True  - keep picking up newer surrogate evaluators during the stage (DAMH sub-chain
    #           surrogate, or the gradient surrogate of an MH stage with a Hamiltonian proposal;
    #           refused with a warning on any other MH stage);
    #   False - one evaluator for the whole stage.
    surrogate_model_updates: bool | None = None
    use_only_surrogate: bool = False
    save_to_file: bool = True
    send_snapshots_to_collector: bool = True
    is_excluded: bool = False
    burn_in: bool = False
    exact_step_probability: float = 0.0
    audit_prerejected: float = 0.0
    name: str | None = None

    # ------------------------------------------------------------------ derived properties
    @property
    def adaptive(self) -> bool:
        """Whether this stage's proposal tunes itself (``proposal.adaptive``; a ``Block`` never does)."""
        return bool(getattr(self.proposal, "adaptive", False))

    def proposal_needs_gradients(self) -> bool:
        """
        True for a ``Hamiltonian`` proposal and for a ``Block`` containing one. Such a stage is
        handed a surrogate evaluator by both runners, and only such an MH stage may set
        ``surrogate_model_updates=True`` (WS7).
        """
        return bool(self.proposal.needs_gradients)

    def __post_init__(self):
        if not isinstance(self.proposal, (_proposals.RandomWalk, _proposals.PCN, _proposals.Hamiltonian, _proposals.Block)):
            raise TypeError(
                "Stage.proposal must be a RandomWalk, PCN, Hamiltonian or Block from surrDAMH.proposals, "
                f"got {type(self.proposal).__name__}")
        if self.burn_in and self.is_excluded:
            raise ValueError("Stage: burn_in=True and is_excluded=True exclude each other -- burn_in continues the "
                             "chain into the next stage, is_excluded restarts it from this stage's starting sample; "
                             "set only one of them")
        if self.max_samples == sys.maxsize and self.max_evaluations == sys.maxsize and self.time_limit == np.inf:
            self.max_evaluations = 10
            print(self.algorithm, ": No stopping condition specified, max_evaluations set to", self.max_evaluations)
        if self.use_only_surrogate:
            self.send_snapshots_to_collector = False
        self._resolve_surrogate_model_updates()
        self._validate_safety_fields()

    def _validate_safety_fields(self) -> None:
        """
        Check the S2 fields (``exact_step_probability``, ``audit_prerejected``): a real number in
        ``[0, 1]`` (``ValueError`` otherwise). On an MH stage a non-zero value is reset to 0 with a
        printed warning (same pattern as ``surrogate_model_updates``): every MH step is exact
        already and there is nothing pre-rejected to audit.
        """
        for name in _S2_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
                    or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Stage.{name} must be a number in [0, 1], got {value!r}")
            value = float(value)
            if self.algorithm == "MH" and value != 0.0:
                print(f"Warning: {name} is only used by a DAMH stage (an MH stage evaluates every proposal "
                      f"exactly), setting {name}=0")
                value = 0.0
            setattr(self, name, value)

    def _resolve_surrogate_model_updates(self) -> None:
        """
        Turn the tri-state ``surrogate_model_updates`` into a plain bool (WS7, 2026-09-18).

        * DAMH: ``None`` means the historical default ``True`` (DAMH-SMU); an explicit value
          is kept.
        * MH with a gradient-needing proposal (Hamiltonian, or a block proposal containing
          one): ``None`` means ``False``, i.e. the gradient evaluator is fetched once at stage
          start and kept for the whole stage. ``True`` opts in to polling the collector for a
          newer evaluator once per iteration; only the proposal's gradients change, the
          accept/reject test still uses the exact model, so the chain stays exact.
        * Any other MH stage: always ``False``. ``True`` is refused with a warning.
        """
        if self.algorithm != "MH":
            if self.surrogate_model_updates is None:
                self.surrogate_model_updates = True
            return
        if self.surrogate_model_updates and not self.proposal_needs_gradients():
            print("Warning: surrogate_model_updates is only supported in an MH stage whose proposal needs "
                  f"surrogate gradients (got proposal={type(self.proposal).__name__}), setting "
                  "surrogate_model_updates=False")
            self.surrogate_model_updates = False
        elif self.surrogate_model_updates is None:
            self.surrogate_model_updates = False

    def describe(self, index: int | None = None) -> str:
        """
        Multi-line summary of this stage's **effective** settings, printed once on rank 0 by
        ``Problem.run_sampling()`` and by ``Problem.run_sampling_local()``.

        Every field appears as ``name=value`` (the proposal spec with all its fields), with a
        trailing ``*`` on the ones that change the sampled distribution or the acceptance rate.
        The values shown are the ones after ``__post_init__`` (``surrogate_model_updates``
        resolved, the default ``max_evaluations=10``, the proposal's resolved ``adaptive``). A
        step left ``None`` is resolved only in ``build_proposal``, after this is printed; the
        ``[note]`` line says what it will resolve to.

        Args:
            index: position in ``list_of_stages``, used for the title (and the stage
                directory name via ``stage_name``); omit for a stand-alone stage.

        Returns:
            A string of a few lines, ready to ``print``.
        """
        title = "Stage" if index is None else f"Stage {index} ({stage_name(self, index)})"
        notes = []
        if self.algorithm == "MH":
            if self.surrogate_model_updates:
                notes.append("MH stage: subchain_length is not used; surrogate_model_updates refreshes "
                             "the proposal's gradient surrogate once per iteration (WS7)")
            else:
                notes.append("MH stage: subchain_length/surrogate_model_updates are not used")
        if self.algorithm == "DAMH" and self.audit_prerejected > 0.0:
            if not self.surrogate_model_updates:
                notes.append("audit_prerejected on a frozen stage (surrogate_model_updates=False): the audit only "
                             "measures the surrogate here, nothing retrains on the audited points")
            if not self.send_snapshots_to_collector:
                notes.append("audit_prerejected with send_snapshots_to_collector=False: audited points are written "
                             "to raw_data but not sent to the collector")
        spec = self.proposal
        step_field = _proposals.STEP_FIELD.get(type(spec))
        if step_field is not None and getattr(spec, step_field) is None:
            default = {"scale": "prior covariance * 2.38^2/d", "beta": "0.5", "step_size": "0.1"}[step_field]
            notes.append(f"{step_field}=None = carried from the last adaptive {type(spec).__name__} stage, else {default}"
                         + ("" if self.adaptive else " (fixed for the whole stage, with a warning)"))
        if step_field is not None and not self.adaptive:
            notes.append("proposal not adaptive: target_rate is not used")
        extra = [f"[note] {note}" for note in notes]
        return describe_fields(self, POSTERIOR_AFFECTING_FIELDS, f"{title} (* = posterior-/acceptance-rate-affecting):", extra)


def stage_name(stage: Stage, index: int) -> str:
    """Directory name of stage ``index`` under ``sampling_output/<data_name>/``, e.g. ``alg0001_DAMH-SMU``."""
    prefix = "alg" + str(index).zfill(4)
    if stage.algorithm == "MH":
        return prefix + ("_MH-adaptive" if stage.adaptive else "_MH")
    if stage.algorithm == "DAMH":
        return prefix + ("_DAMH-SMU" if stage.surrogate_model_updates else "_DAMH")
    raise ValueError(f"unknown algorithm {stage.algorithm!r} in stage {index} (expected 'MH' or 'DAMH')")

def check_stage_list(list_of_stages) -> None:
    """
    Refuse a stage list that holds something other than ``Stage`` objects, at construction
    time and with a message that names the fix (2026-09-22). Without it, a proposal spec
    appended in place of a stage surfaced only inside ``run()`` as
    ``AttributeError: 'RandomWalk' object has no attribute 'algorithm'``.
    """
    from surrDAMH import proposals as _proposals
    if isinstance(list_of_stages, Stage):
        raise TypeError("list_of_stages must be a list of Stage objects, got a single Stage; wrap it: [stage]")
    for i, item in enumerate(list_of_stages):
        if isinstance(item, Stage):
            continue
        hint = ""
        if isinstance(item, (_proposals.RandomWalk, _proposals.PCN, _proposals.Hamiltonian, _proposals.Block)):
            hint = f"; a proposal is not a stage, wrap it: Stage(proposal={item!r}, max_evaluations=...)"
        raise TypeError(f"list_of_stages[{i}] is a {type(item).__name__}, not a surrDAMH.Stage{hint}")


def _needs_surrogate(stage: Stage) -> bool:
    return stage.algorithm == "DAMH" or stage.use_only_surrogate or stage.proposal_needs_gradients()


def wasted_snapshot_notes(stages: list) -> list[str]:
    """
    One note per stage that still streams snapshots to the collector although no stage from
    there on will ever pick up a retrained surrogate (2026-09-22, found by the GRF scheme study:
    the frozen final DAMH stage of an RBF/KD-tree run took 2-3x longer than the retraining stage,
    because the collector kept refitting on every 2000 new snapshots and the samplers stalled on
    their snapshot queue). The snapshots are not useless in every case -- a later run may reuse
    them through ``SurrogateRestart(mode="data")`` -- so this only tells the user, it changes nothing.
    """
    notes = []
    for i, stage in enumerate(stages):
        if not stage.send_snapshots_to_collector:
            continue
        later_pickup = any(_needs_surrogate(s) and bool(s.surrogate_model_updates) for s in stages[i:])
        later_first_use = any(_needs_surrogate(s) for s in stages[i + 1:]) and not any(_needs_surrogate(s) for s in stages[:i + 1])
        if _needs_surrogate(stage) and not stage.surrogate_model_updates and not later_pickup and not later_first_use:
            notes.append(f"stage {i} ({stage.name or stage.algorithm}): surrogate frozen and no later stage retrains, yet "
                         "send_snapshots_to_collector=True -- the collector will keep retraining a surrogate nobody "
                         "uses (slow for RBF/KD-tree); set send_snapshots_to_collector=False unless you want these "
                         "snapshots for a later SurrogateRestart(mode='data')")
    return notes
