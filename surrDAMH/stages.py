#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from surrDAMH.modules.proposals import Proposal


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

    Currently ignored (accepted but not wired to any behaviour):
    ``adaptive_target_rate`` (the adaptive proposal always targets its own default,
    0.25 -- see the field comment below), ``adaptive_corr_limit``, and
    ``adaptive_sample_limit`` (``GaussRandomWalk_adaptive`` is built with its own
    hard-coded correlation limit and unbounded sample history regardless of these
    fields, finding G1 in ``library_notes/08_safe_changes_plan.md``); ``proposal``
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
    # NOTE: this field is currently NOT passed to the proposal - ``build_proposal`` constructs
    # ``GaussRandomWalk_adaptive`` with its own default target rate of 0.25 (see 08 §G1).
    adaptive_target_rate: float | None = None
    adaptive_corr_limit: float | None = None  # IGNORED: not passed to build_proposal; GaussRandomWalk_adaptive always uses its own default corr_limit=0.3 (finding G1)
    adaptive_sample_limit: int | None = None  # IGNORED: not passed to build_proposal; GaussRandomWalk_adaptive keeps an unbounded sample history regardless of this field (finding G1)
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


def stage_name(stage: Stage, index: int) -> str:
    """Directory name of stage ``index`` under ``sampling_output/<data_name>/``, e.g. ``alg0001_DAMH-SMU``."""
    prefix = "alg" + str(index).zfill(4)
    if stage.algorithm_type == "MH":
        return prefix + ("_MH-adaptive" if stage.adaptive else "_MH")
    if stage.algorithm_type == "DAMH":
        return prefix + ("_DAMH-SMU" if stage.surrogate_model_updates else "_DAMH")
    raise ValueError(f"unknown algorithm_type {stage.algorithm_type!r} in stage {index} (expected 'MH' or 'DAMH')")
