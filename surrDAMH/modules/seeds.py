#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The seed architecture of a sampling run, in one place.

Every random stream of a chain is a pure function of the chain index (``rank_world``,
0 for ``run_local``), the stage index and the number of stages, so that

- ``run_local`` reproduces chain 0 of an MPI run with the same configuration, and
- two chains of one run never share a stream.

Today's family (unchanged since the code was written, except for the ``+3`` offset added
by G4 on 2026-09-17)::

    seed0(rank, i)        = 10 * (no_stages * rank + i)
    proposal_seed         = seed0 + 1     # build_proposal
    algorithm_seed        = seed0 + 2     # AlgorithmBase._generator (acceptance draws)
    initial_sample_seed   = seed0(rank, 0) + 3   # initial sample of the chain (G4)
    exact_step_seed       = seed0 + 4     # S2: "which kernel" + audit draws (AlgorithmBase._generator_exact)
    exact_proposal_seed   = seed0 + 5     # S2: the exact step's own random walk

The stride of 10 leaves room for further per-stage streams; the initial-sample seed uses
stage 0's ``seed0`` because the initial sample is drawn once per chain, before the stage
loop, so it is a function of the rank only.

Continued runs (``SamplingRun.continue_sampling``, 2026-10-08): a lineage of runs is numbered
as one stage list. Run ``k`` passes its global stage index ``i_global = stage_index_offset + i``
as ``stage_index`` and ``no_stages_lineage = stage_index_offset + len(stages)`` as
``no_stages`` (:func:`seed_no_stages`); the initial-sample seed uses the same ``no_stages``.
Generation ``g`` of a lineage (``Configuration.lineage_generation``) adds
``generation_seed_offset(g) = GENERATION_SEED_STRIDE * g`` to every ``seed0`` (hence to the
proposal and algorithm seeds), to the initial-sample seed and to the LHS seed, so a continued
run never reuses a stream of an earlier run of its lineage, and ``chains="prior"``/``"lhs"``
draw new starting points. With offset 0, ``no_stages_lineage=None`` and generation 0 (every
plain run, and the old manual ``initial_sample_type="continued"`` path) that is exactly the
formula above.

Repeated experiments (``Configuration.seed_offset``, study 29, 2026-10-12): the integer
``seed_offset`` is added to every seed of the run on top of the generation shift
(:func:`run_seed_offset`): every ``seed0`` (hence all five per-stage streams), the
initial-sample seed and the LHS seed. ``0`` (default) is the formula above. Repetitions with
DISJOINT streams need offsets at least ``SEED_STRIDE * no_stages * no_samplers`` apart (e.g.
``seed_offset = 100_000 * k``) and below ``GENERATION_SEED_STRIDE`` if the run is continued.

This module is intentionally dependency-free (only ``no_stages``/``rank`` integers in,
integers out) so that both runners and ``modules/manifest.py`` can share it.
"""

from __future__ import annotations

SEED_STRIDE = 10  # seeds of consecutive (rank, stage) pairs are SEED_STRIDE apart
PROPOSAL_SEED_OFFSET = 1
ALGORITHM_SEED_OFFSET = 2
INITIAL_SAMPLE_SEED_OFFSET = 3
#: S2 (2026-10-09): per-(chain, stage) stream of the DAMH exact-step choice and of the audit of
#: pre-rejected proposals (``AlgorithmBase._generator_exact``). Drawn from only when
#: ``Stage.exact_step_probability > 0`` or ``Stage.audit_prerejected > 0``.
EXACT_STEP_SEED_OFFSET = 4
#: S2: the exact step's own adaptive random walk (built only when ``exact_step_probability > 0``).
EXACT_PROPOSAL_SEED_OFFSET = 5
#: Added (times the lineage generation) to every seed of a continued run, so its streams are
#: disjoint from those of all earlier runs of its lineage as long as every run has
#: ``SEED_STRIDE * no_stages_lineage * no_samplers < GENERATION_SEED_STRIDE`` (2026-10-08).
GENERATION_SEED_STRIDE = 1_000_000

SEED_FORMULA = ("seed0 = 10*(no_stages*rank_world + i) + 1000000*generation + seed_offset; proposal_seed = seed0+1; "
                "algorithm_seed = seed0+2; exact_step_seed = seed0+4; exact_proposal_seed = seed0+5; "
                "initial_sample_seed = 10*no_stages*rank_world + 3 + 1000000*generation + seed_offset; "
                "lhs_seed = 1000000*generation + seed_offset; continued runs: i = stage_index_offset + local index, "
                "no_stages = no_stages_lineage, generation = lineage generation (0 for a plain run)")


def stage_seed0(no_stages: int, rank_world: int, stage_index: int) -> int:
    """Base seed of one (chain, stage) pair."""
    return SEED_STRIDE * (no_stages * rank_world + stage_index)


def seed_no_stages(no_stages_lineage: int | None, no_stages: int) -> int:
    """``no_stages`` of the seed formula: the lineage's stage count if set, else the run's own."""
    return int(no_stages_lineage) if no_stages_lineage else int(no_stages)


def generation_seed_offset(generation: int | None) -> int:
    """Seed shift of lineage generation ``generation`` (0 for a plain run)."""
    return GENERATION_SEED_STRIDE * int(generation or 0)


def run_seed_offset(generation: int | None, seed_offset: int | None = 0) -> int:
    """Shift added to every seed of a run: ``generation_seed_offset(generation) + seed_offset``
    (``Configuration.lineage_generation`` and ``Configuration.seed_offset``; 0 for a plain run)."""
    return generation_seed_offset(generation) + int(seed_offset or 0)


def conf_seed_offset(conf) -> int:
    """:func:`run_seed_offset` of a ``Configuration`` (missing fields count as 0)."""
    return run_seed_offset(getattr(conf, "lineage_generation", 0), getattr(conf, "seed_offset", 0))


def initial_sample_seed(no_stages: int, rank_world: int) -> int:
    """Seed of the per-chain initial-sample generator (G4): stage 0's ``seed0`` + 3."""
    return stage_seed0(no_stages, rank_world, 0) + INITIAL_SAMPLE_SEED_OFFSET


__all__ = ["SEED_STRIDE", "PROPOSAL_SEED_OFFSET", "ALGORITHM_SEED_OFFSET",
           "INITIAL_SAMPLE_SEED_OFFSET", "EXACT_STEP_SEED_OFFSET", "EXACT_PROPOSAL_SEED_OFFSET", "SEED_FORMULA", "stage_seed0", "initial_sample_seed",
           "seed_no_stages", "GENERATION_SEED_STRIDE", "generation_seed_offset", "run_seed_offset",
           "conf_seed_offset"]
