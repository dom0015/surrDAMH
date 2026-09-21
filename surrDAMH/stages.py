#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from surrDAMH.modules.describe import describe_fields
from surrDAMH.modules.proposals import Proposal

#: Fields of ``Stage`` that change the sampled distribution or the acceptance rate (the list
#: the class docstring spells out). ``describe()`` marks them with a trailing ``*``.
POSTERIOR_AFFECTING_FIELDS: frozenset[str] = frozenset({
    "algorithm_type", "proposal_type", "proposal_sd_or_cov", "pcn_beta",
    "hamiltonian_num_steps", "hamiltonian_step_size", "block_proposal_groups",
    "block_proposal_list", "adaptive", "subchain_max_length", "surrogate_model_updates",
    "use_only_surrogate", "is_excluded", "adaptive_target_rate",
})


@dataclass
class Stage:
    """
    One sampling stage: an algorithm + proposal + stopping rule, run in sequence by
    every sampler rank (or by ``run_local``). See ``docs/stages.md`` for the full field
    table and ``docs/concepts.md`` for the MH/DAMH/DAMH-SMU derivations.

    Posterior-/acceptance-rate-affecting: ``algorithm_type``, ``proposal_type`` and its
    parameters (``proposal_sd_or_cov``, ``pcn_beta``, ``hamiltonian_*``,
    ``block_proposal_*``), ``adaptive`` (retunes the proposal during the stage),
    ``subchain_max_length`` and ``surrogate_model_updates`` (DAMH/DAMH-SMU kernel),
    ``use_only_surrogate``, ``is_excluded`` (changes which sample the next stage starts
    from). ``max_samples``/``max_evaluations``/``time_limit`` only change how much of
    the (otherwise identical) chain is produced, not its distribution.
    ``send_snapshots_to_collector``/``save_to_file`` affect training data / on-disk
    output only.

    ``surrogate_model_updates`` is tri-state on input and a plain bool afterwards (WS7,
    2026-09-18): ``None`` (the default) means "whatever this stage type has always done",
    i.e. ``True`` for DAMH (DAMH-SMU) and ``False`` for MH. Setting it ``True`` explicitly is
    accepted on a DAMH stage (the sub-chain surrogate is refreshed once per sub-chain) and --
    new in WS7 -- on an MH stage whose proposal needs surrogate gradients (Hamiltonian family,
    or a ``"block"`` proposal containing such a sub-proposal, see ``proposal_needs_gradients``),
    where the gradient evaluator is then re-polled once per iteration instead of being fixed at
    stage start. On any other MH stage ``True`` is refused with a printed warning and resolved
    to ``False``, which is what happened silently before WS7. An MH stage that leaves the field
    alone therefore behaves exactly as it always did. Only the proposal's gradients change in
    the MH case: the accept/reject test uses the exact model only, so the stage samples the
    exact posterior for any gradient field.

    ``adaptive=True`` is supported for ``proposal_type`` ``"RWMH"``, ``"pCN"``,
    ``"Hamiltonian"`` and ``"HamiltonianInfinite"`` (2026-09-20); only ``"block"`` still
    raises. ``adaptive_target_rate`` is forwarded to the adaptive proposal only when it is not
    ``None``; ``None`` means that proposal's own default -- 0.234 for the random walk and for
    pCN (Robbins-Monro on the log-scale / on the logit of ``beta``), 0.8 for the Hamiltonian
    family (dual averaging of the leapfrog step size). It only has an effect together with
    ``adaptive=True``. In a DAMH stage the random walk and pCN are fed the *overall* outer
    acceptance probability (a pre-rejected iteration counts as 0), while the Hamiltonian step
    size is dual-averaged on the SUB-CHAIN's own acceptance against the surrogate -- see the
    proposal docstrings in ``modules/proposals.py`` and ``docs/stages.md``.

    ``pcn_beta`` and ``hamiltonian_step_size`` are tri-state: ``None`` (the default) means
    "the value carried over from the last adaptive stage of that proposal type, else 0.5 /
    0.1" -- the values that were the dataclass defaults before 2026-09-20, so the effective
    value of every existing configuration is unchanged.

    ``proposal_sd_or_cov=None`` on a random-walk stage means "the covariance carried over from
    the last adaptive random-walk stage"; when there is none, an ``adaptive=True`` stage starts
    from the prior covariance scaled by ``2.38^2 / d`` (unit sd if the prior has no
    ``get_covariance()``, with a printed warning) and a non-adaptive stage raises in
    ``build_proposal`` (2026-09-21). The adaptive random walk keeps its effective covariance
    continuous when its own estimate replaces the starting one, so the start only shapes the
    warm-up; no starting value has to be supplied.

    Currently ignored (accepted but not wired to any behaviour): ``proposal``
    (never read -- the proposal actually used is always constructed fresh by
    ``build_proposal`` from ``proposal_type`` and the other fields above).

    Invalid combinations that raise (at ``build_proposal``/``stage_name`` time, i.e.
    when ``SamplingFramework.run()``/``run_local()`` reaches this stage): unknown
    ``algorithm_type`` (anything other than ``"MH"``/``"DAMH"``); a DAMH or
    Hamiltonian-family first stage with no surrogate available yet (see
    ``Configuration.use_collector``); ``proposal_type="pCN"`` with a non-Gaussian
    internal prior; a Hamiltonian-family (or block-with-Hamiltonian-member) proposal
    without ``use_surrogate_gradients=True`` in effect; ``proposal_type="block"`` together
    with ``adaptive=True`` (adapt the sub-proposals instead).
    """

    algorithm_type: Literal["MH", "DAMH"] = "MH"  # DAMH uses delayed acceptance, MH does not
    proposal_type: Literal["RWMH", "pCN", "Hamiltonian", "HamiltonianInfinite", "block"] = "RWMH"  # RWMH = Gaussian random walk, pCN = preconditioned Crank-Nicolson
    proposal: Proposal | None = None  # NOT READ: the actual proposal is always built fresh by build_proposal() from proposal_type and the fields below; setting this field has no effect
    proposal_sd_or_cov: float | npt.ArrayLike | None = None
    # pCN step size in (0, 1), only used when proposal_type == "pCN"; None (the default since
    # 2026-09-20) = the beta carried over from the last adaptive pCN stage, else 0.5 -- the
    # former dataclass default, so effective values are unchanged.
    pcn_beta: float | None = None
    hamiltonian_num_steps: int = 10  # only used when proposal_type == "Hamiltonian"
    # Leapfrog step size, only used by the two Hamiltonian proposal types; None (the default
    # since 2026-09-20) = the step size carried over from the last adaptive Hamiltonian stage
    # (its dual-averaged eps_bar), else 0.1 -- the former dataclass default.
    hamiltonian_step_size: float | None = None
    block_proposal_groups: list[slice] | None = None  # only used when proposal_type == "block"
    block_proposal_list: list[Proposal] | None = None  # only used when proposal_type == "block"
    adaptive: bool = False
    max_samples: int = sys.maxsize  # termination condition - total number of samples
    max_evaluations: int = sys.maxsize  # termination condition - total number of full model evaluations
    time_limit: float = np.inf  # termination condition - total time
    send_snapshots_to_collector: bool = True  # use snapshots from this stage for surrogate updates
    subchain_max_length: int = 1  # only with DAMH, length of MH subchain using only surrogate
    # Tri-state on input, always a plain bool after __post_init__ (WS7, 2026-09-18):
    #   None  - "use the default for this stage": True for DAMH (DAMH-SMU), False for MH;
    #   True  - keep picking up newer surrogate evaluators during the stage. Allowed on a DAMH
    #           stage (the sub-chain surrogate) and on an MH stage whose proposal needs
    #           gradients (the gradient surrogate); on any other MH stage it is refused with a
    #           warning and resolved to False, exactly as before WS7;
    #   False - one evaluator for the whole stage.
    # See the class docstring and docs/stages.md.
    surrogate_model_updates: bool | None = None
    use_only_surrogate: bool = False  # if True, surrogate is used instead of full model
    save_to_file: bool = True  # samples are saved to file
    is_excluded: bool = False  # if True, the next stage starts from the same sample as this one
    # Target acceptance rate of the adaptive proposal; only used with adaptive=True and
    # forwarded only when not None. None = the proposal class's own default: 0.234 for
    # GaussRandomWalk_adaptive and PCN_adaptive, 0.8 for the dual-averaged Hamiltonian step
    # size. In a DAMH stage the random walk / pCN feedback is the OVERALL outer acceptance
    # probability (pre-rejected iterations count as 0) while the Hamiltonian step size is
    # dual-averaged on the sub-chain's own acceptance; see modules/proposals.py and
    # docs/stages.md.
    adaptive_target_rate: float | None = None
    name: str | None = None  # will be set later

    def proposal_needs_gradients(self) -> bool:
        """
        Does this stage's proposal need surrogate gradients?

        True for a Hamiltonian-family ``proposal_type`` and for a ``"block"`` proposal whose
        ``block_proposal_list`` contains at least one sub-proposal with ``needs_gradients``
        (the same rule ``BlockProposal.__init__`` applies to its own ``needs_gradients``).
        Such a stage is handed a surrogate evaluator by both runners, and only such an MH
        stage may set ``surrogate_model_updates=True`` (WS7).
        """
        if self.proposal_type in ("Hamiltonian", "HamiltonianInfinite"):
            return True
        if self.proposal_type == "block" and self.block_proposal_list is not None:
            return any(getattr(proposal, "needs_gradients", False) for proposal in self.block_proposal_list)
        return False

    def __post_init__(self):
        if self.max_samples == sys.maxsize and self.max_evaluations == sys.maxsize and self.time_limit == np.inf:
            self.max_evaluations = 10
            print(self.algorithm_type, ": No stopping condition specified, max_evaluations set to", self.max_evaluations)
        if self.use_only_surrogate:
            self.send_snapshots_to_collector = False
        self._resolve_surrogate_model_updates()

    def _resolve_surrogate_model_updates(self) -> None:
        """
        Turn the tri-state ``surrogate_model_updates`` into a plain bool (WS7, 2026-09-18).

        * DAMH: ``None`` means the historical default ``True`` (DAMH-SMU); an explicit value
          is kept.
        * MH with a gradient-needing proposal (Hamiltonian family, or a block proposal
          containing one): ``None`` means ``False``, i.e. exactly the behaviour of every MH
          stage before WS7 -- the gradient evaluator is fetched once at stage start and kept
          for the whole stage. ``True`` opts in to polling the collector for a newer
          evaluator once per iteration; only the proposal's gradients change, the accept /
          reject test still uses the exact model, so the chain stays exact.
        * Any other MH stage: always ``False``. ``True`` is refused with a warning, as it was
          silently before WS7.
        """
        if self.algorithm_type != "MH":
            if self.surrogate_model_updates is None:
                self.surrogate_model_updates = True
            return
        if self.surrogate_model_updates and not self.proposal_needs_gradients():
            print("Warning: surrogate_model_updates is only supported in an MH stage whose proposal needs "
                  f"surrogate gradients (got proposal_type={self.proposal_type!r}), setting "
                  "surrogate_model_updates=False")
            self.surrogate_model_updates = False
        elif self.surrogate_model_updates is None:
            self.surrogate_model_updates = False

    def describe(self, index: int | None = None) -> str:
        """
        Multi-line summary of this stage's **effective** settings, printed once on rank 0 by
        ``SamplingFramework.run()`` and by ``run_local()``.

        Every field appears as ``name=value``, with a trailing ``*`` on the ones that change
        the sampled distribution or the acceptance rate. The values shown are the ones after
        ``__post_init__``, so the corrections it makes (``surrogate_model_updates``
        resolved from ``None`` / refused on an MH stage without gradients, the default
        ``max_evaluations=10`` inserted when no stopping condition was given) are visible.
        Fields that only apply to another ``proposal_type`` are shown as well, at their
        unused default -- the ``[note]`` line names the ones that are ignored here.

        ``pcn_beta=None`` / ``hamiltonian_step_size=None`` are shown as ``None``, which means
        "carried over from the last adaptive stage of that proposal type, else 0.5 / 0.1";
        the value actually used is resolved in ``build_proposal``, after this is printed.

        Args:
            index: position in ``list_of_stages``, used for the title (and the stage
                directory name via ``stage_name``); omit for a stand-alone stage.

        Returns:
            A string of about five lines, ready to ``print``.
        """
        title = "Stage" if index is None else f"Stage {index} ({stage_name(self, index)})"
        notes = []
        if self.algorithm_type == "MH":
            if self.surrogate_model_updates:
                notes.append("MH stage: subchain_max_length is not used; surrogate_model_updates refreshes "
                             "the proposal's gradient surrogate once per iteration (WS7)")
            else:
                notes.append("MH stage: subchain_max_length/surrogate_model_updates are not used")
        if not self.adaptive:
            notes.append("adaptive=False: adaptive_target_rate is not used")
        if self.pcn_beta is None or self.hamiltonian_step_size is None:
            notes.append("pcn_beta/hamiltonian_step_size None = carried from the last adaptive "
                         "stage of that type, else 0.5/0.1")
        if self.proposal_sd_or_cov is None and self.proposal_type == "RWMH":
            notes.append("proposal_sd_or_cov None = carried from the last adaptive RWMH stage, else "
                         + ("prior covariance * 2.38^2/d (adaptive start)" if self.adaptive
                            else "an error (set it or use adaptive=True)"))
        if self.proposal is not None:
            notes.append("the 'proposal' field is never read (build_proposal always builds a fresh one)")
        extra = [f"[note] {note}" for note in notes]
        return describe_fields(self, POSTERIOR_AFFECTING_FIELDS, f"{title} (* = posterior-/acceptance-rate-affecting):", extra)


def stage_name(stage: Stage, index: int) -> str:
    """Directory name of stage ``index`` under ``sampling_output/<data_name>/``, e.g. ``alg0001_DAMH-SMU``."""
    prefix = "alg" + str(index).zfill(4)
    if stage.algorithm_type == "MH":
        return prefix + ("_MH-adaptive" if stage.adaptive else "_MH")
    if stage.algorithm_type == "DAMH":
        return prefix + ("_DAMH-SMU" if stage.surrogate_model_updates else "_DAMH")
    raise ValueError(f"unknown algorithm_type {stage.algorithm_type!r} in stage {index} (expected 'MH' or 'DAMH')")
