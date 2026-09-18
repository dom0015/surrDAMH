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

The stride of 10 leaves room for further per-stage streams; the initial-sample seed uses
stage 0's ``seed0`` because the initial sample is drawn once per chain, before the stage
loop, so it is a function of the rank only.

This module is intentionally dependency-free (only ``no_stages``/``rank`` integers in,
integers out) so that both runners and ``modules/manifest.py`` can share it.
"""

from __future__ import annotations

SEED_STRIDE = 10  # seeds of consecutive (rank, stage) pairs are SEED_STRIDE apart
PROPOSAL_SEED_OFFSET = 1
ALGORITHM_SEED_OFFSET = 2
INITIAL_SAMPLE_SEED_OFFSET = 3

SEED_FORMULA = ("seed0 = 10*(no_stages*rank_world + i); proposal_seed = seed0+1; "
                "algorithm_seed = seed0+2; initial_sample_seed = 10*no_stages*rank_world + 3")


def stage_seed0(no_stages: int, rank_world: int, stage_index: int) -> int:
    """Base seed of one (chain, stage) pair."""
    return SEED_STRIDE * (no_stages * rank_world + stage_index)


def initial_sample_seed(no_stages: int, rank_world: int) -> int:
    """Seed of the per-chain initial-sample generator (G4): stage 0's ``seed0`` + 3."""
    return stage_seed0(no_stages, rank_world, 0) + INITIAL_SAMPLE_SEED_OFFSET


__all__ = ["SEED_STRIDE", "PROPOSAL_SEED_OFFSET", "ALGORITHM_SEED_OFFSET",
           "INITIAL_SAMPLE_SEED_OFFSET", "SEED_FORMULA", "stage_seed0", "initial_sample_seed"]
