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
    "use_only_surrogate", "is_excluded", "adaptive_target_rate", "adaptive_corr_limit",
    "adaptive_sample_limit",
})


@dataclass
class Stage:
    """
    One sampling stage: an algorithm + proposal + stopping rule, run in sequence by
    every sampler rank (or by ``run_local``). See ``docs/stages.md`` for the full field
    table and ``docs/concepts.md`` for the MH/DAMH/DAMH-SMU derivations.

    Posterior-/acceptance-rate-affecting: ``algorithm_type``, ``proposal_type`` and its
    parameters (``proposal_sd_or_cov``, ``pcn_beta``, ``hamiltonian_*``,
    ``block_proposal_*``), ``adaptive`` (changes the proposal covariance over time),
    ``subchain_max_length`` and ``surrogate_model_updates`` (DAMH/DAMH-SMU kernel),
    ``use_only_surrogate``, ``is_excluded`` (changes which sample the next stage starts
    from). ``max_samples``/``max_evaluations``/``time_limit`` only change how much of
    the (otherwise identical) chain is produced, not its distribution.
    ``send_snapshots_to_collector``/``save_to_file`` affect training data / on-disk
    output only.

    Unverified: with ``Configuration.state_dependent_approximation=True`` (see there),
    the combination with ``subchain_max_length > 1`` is unchecked theory.

    ``adaptive_target_rate``, ``adaptive_corr_limit`` and ``adaptive_sample_limit`` are
    wired to ``GaussRandomWalk_adaptive`` since G1 (2026-09-17): each one is forwarded only
    when it is not ``None``, so a stage that leaves them unset keeps the previously
    hard-coded defaults (0.25 / 0.3 / unbounded history) and is bit-identical to before.
    They only have an effect together with ``adaptive=True``.

    Currently ignored (accepted but not wired to any behaviour): ``proposal``
    (never read -- the proposal actually used is always constructed fresh by
    ``build_proposal`` from ``proposal_type`` and the other fields above).

    Invalid combinations that raise (at ``build_proposal``/``stage_name`` time, i.e.
    when ``SamplingFramework.run()``/``run_local()`` reaches this stage): unknown
    ``algorithm_type`` (anything other than ``"MH"``/``"DAMH"``); a DAMH or
    Hamiltonian-family first stage with no surrogate available yet (see
    ``Configuration.use_collector``); ``proposal_type="pCN"`` with a non-Gaussian
    internal prior; a Hamiltonian-family (or block-with-Hamiltonian-member) proposal
    without ``use_surrogate_gradients=True`` in effect. Not an exception but a silent
    correction: ``proposal_type="pCN"`` with ``adaptive=True`` is forced back to
    ``adaptive=False`` by ``__post_init__`` (adaptive pCN is not supported), with a
    printed warning.
    """

    algorithm_type: Literal["MH", "DAMH"] = "MH"  # DAMH uses delayed acceptance, MH does not
    proposal_type: Literal["RWMH", "pCN", "Hamiltonian", "HamiltonianInfinite", "block"] = "RWMH"  # RWMH = Gaussian random walk, pCN = preconditioned Crank-Nicolson
    proposal: Proposal | None = None  # NOT READ: the actual proposal is always built fresh by build_proposal() from proposal_type and the fields below; setting this field has no effect
    proposal_sd_or_cov: float | npt.ArrayLike | None = None
    pcn_beta: float = 0.5  # pCN step size, only used when proposal_type == "pCN"
    hamiltonian_num_steps: int = 10  # only used when proposal_type == "Hamiltonian"
    hamiltonian_step_size: float = 0.1  # only used when proposal_type == "Hamiltonian"
    block_proposal_groups: list[slice] | None = None  # only used when proposal_type == "block"
    block_proposal_list: list[Proposal] | None = None  # only used when proposal_type == "block"
    adaptive: bool = False
    max_samples: int = sys.maxsize  # termination condition - total number of samples
    max_evaluations: int = sys.maxsize  # termination condition - total number of full model evaluations
    time_limit: float = np.inf  # termination condition - total time
    send_snapshots_to_collector: bool = True  # use snapshots from this stage for surrogate updates
    subchain_max_length: int = 1  # only with DAMH, length of MH subchain using only surrogate
    surrogate_model_updates: bool = True  # only with DAMH, surrogate changes during the stage (DAMH-SMU)
    use_only_surrogate: bool = False  # if True, surrogate is used instead of full model
    save_to_file: bool = True  # samples are saved to file
    is_excluded: bool = False  # if True, the next stage starts from the same sample as this one
    # Target acceptance rate of the adaptive proposal. In a DAMH stage the proposal only ever sees
    # sub-chain ENDPOINTS (``Proposal.adapt`` is called once per outer step, from the exact
    # acceptance test), so the rate being targeted is the SECOND-STAGE (outer) acceptance rate, not
    # the overall or sub-chain acceptance rate (09 §3 decision 2 is deferred; this documents the
    # current semantics only).
    # Passed to ``GaussRandomWalk_adaptive`` by ``build_proposal`` when not None (G1); None
    # means the proposal's own default of 0.25.
    adaptive_target_rate: float | None = None
    adaptive_corr_limit: float | None = None  # max |correlation| the adapted proposal covariance may have; None = GaussRandomWalk_adaptive's default 0.3 (G1)
    adaptive_sample_limit: int | None = None  # keep only the last N proposals when re-estimating the covariance; None = unbounded history (G1)
    name: str | None = None  # will be set later

    def __post_init__(self):
        if self.max_samples == sys.maxsize and self.max_evaluations == sys.maxsize and self.time_limit == np.inf:
            self.max_evaluations = 10
            print(self.algorithm_type, ": No stopping condition specified, max_evaluations set to", self.max_evaluations)
        if self.use_only_surrogate:
            self.send_snapshots_to_collector = False
        if self.algorithm_type == "MH":
            self.surrogate_model_updates = False
        if self.proposal_type == "pCN" and self.adaptive:
            print("Warning: adaptive mode is not supported with pCN proposal, setting adaptive=False")
            self.adaptive = False

    def describe(self, index: int | None = None) -> str:
        """
        Multi-line summary of this stage's **effective** settings, printed once on rank 0 by
        ``SamplingFramework.run()`` and by ``run_local()``.

        Every field appears as ``name=value``, with a trailing ``*`` on the ones that change
        the sampled distribution or the acceptance rate. The values shown are the ones after
        ``__post_init__``, so the silent corrections it makes (``surrogate_model_updates``
        forced off in an MH stage, ``adaptive`` forced off for pCN, the default
        ``max_evaluations=10`` inserted when no stopping condition was given) are visible.
        Fields that only apply to another ``proposal_type`` are shown as well, at their
        unused default -- the ``[note]`` line names the ones that are ignored here.

        Args:
            index: position in ``list_of_stages``, used for the title (and the stage
                directory name via ``stage_name``); omit for a stand-alone stage.

        Returns:
            A string of about five lines, ready to ``print``.
        """
        title = "Stage" if index is None else f"Stage {index} ({stage_name(self, index)})"
        notes = []
        if self.algorithm_type == "MH":
            notes.append("MH stage: subchain_max_length/surrogate_model_updates are not used")
        if not self.adaptive:
            notes.append("adaptive=False: adaptive_target_rate/corr_limit/sample_limit are not used")
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
