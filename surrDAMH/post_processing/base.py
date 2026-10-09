#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common base of :class:`surrDAMH.post_processing.loading.Samples`: the instance
attributes ``Samples.__init__`` (``loading.py``) sets, declared here as class-level
annotations so a type checker can see them from every layer, plus the small resolver
helpers shared by ``statistics.py``, ``plots.py`` and ``html_report.py``.

``Samples`` used to be assembled from three independent mixins (``SamplesStatistics``,
``SamplesPlots``, ``SamplesReports``) that each read attributes/methods defined only on
a sibling mixin or on ``Samples`` itself -- correct at runtime (Python resolves ``self``
against the final MRO), but invisible to a type checker looking at one file at a time.
``SamplesBase`` breaks that: the three mixins now form a linear chain
(``SamplesStatistics(SamplesBase)`` -> ``SamplesPlots(SamplesStatistics)`` ->
``SamplesReports(SamplesPlots)`` -> ``Samples(SamplesReports)``), and everything a layer
needs from "below" is visible through ordinary single inheritance.

No behaviour changes here: the six methods below are moved verbatim from
``Samples`` in ``loading.py``; ``Samples.__init__`` (and the methods that only
``Samples`` itself needs -- ``load_notes``, ``load_subchain_stats``, ``load_adaptive_stats``, ``summarize``,
``get_summary``, ``load_snapshots``) stay in ``loading.py``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from surrDAMH.modules.run_data import RunData
    from surrDAMH.post_processing.loading import StageSamples


class SamplesBase:
    """
    Attribute contract of :class:`surrDAMH.post_processing.loading.Samples`, plus the
    stage/chain/burn-in resolvers every layer above it uses.

    The attributes below are all set by ``Samples.__init__`` or the ``summarize()``
    helper it calls (both in ``loading.py``); they are declared here, not assigned --
    ``SamplesBase`` is never instantiated on its own.
    """

    no_parameters: int
    no_stages: int
    stage_names: List[str]
    list_of_stages: List["StageSamples"]
    sampling_output_dir: str
    samples_dir: str
    run_data: "RunData"
    debug: bool
    summary: pd.DataFrame
    notes: List[pd.DataFrame]
    subchain_stats: List[pd.DataFrame]
    adaptive_stats: List[pd.DataFrame]  # empty DataFrame for a stage whose proposal did not adapt
    # lineage / selection (2026-10-08, see Samples.__init__)
    stage_sampling_dirs: List[str]      # [stage] -> sampling_output/ of the run that wrote it
    chain_indices: List[List[int]]      # [stage] -> original chain numbers kept by the selection
    no_chains_original: List[int]       # [stage] -> chains (rank files) before the selection
    lineage_note: str | None

    def _resolve_stages(self, stages_to_disp: Iterable | None) -> List[int]:
        """
        Stage indices to include; ``None`` means every stage of the run.

        Entries may be positional indices (``int``) or stage names as produced by
        ``surrDAMH.stages.stage_name()`` (``str``, e.g. ``"alg0000_MH"``) -- string
        entries are resolved against ``self.stage_names`` first (WS9 bullet 5), then the
        existing index-based validation below runs unchanged, so a caller that only ever
        passed integers sees no behaviour change.
        """
        if stages_to_disp is None:
            return list(range(self.no_stages))
        name_to_index = {name: index for index, name in enumerate(self.stage_names)}
        stages = []
        for stage in stages_to_disp:
            if isinstance(stage, str):
                if stage not in name_to_index:
                    raise ValueError(
                        f"stages_to_disp contains unknown stage name {stage!r}; the run in "
                        f"{self.sampling_output_dir} has stage(s) named {self.stage_names}."
                    )
                stages.append(name_to_index[stage])
            else:
                stages.append(int(stage))
        out_of_range = [stage for stage in stages if not 0 <= stage < self.no_stages]
        if out_of_range:
            raise IndexError(
                f"stages_to_disp={stages} requests stage(s) {out_of_range} but the run in "
                f"{self.sampling_output_dir} has {self.no_stages} stage(s): {self.stage_names}."
            )
        return stages

    def _burn_in_for(self, stages_to_disp: List[int],
                     burn_in: List[List[int]] | None) -> List[List[int]]:
        """
        Validated burn-in table for ``stages_to_disp``: ``burn_in[position][chain]``.

        ``None`` means no burn-in anywhere. A wrongly shaped table used to be swallowed by
        an ``except BaseException`` that printed "CHAIN ... NOT AVAILABLE" per chain
        (finding P6); it is now reported once, here, for what it is.

        Raises:
            ValueError: ``burn_in`` does not have one entry per displayed stage, or one
                entry per chain of that stage.
        """
        if burn_in is None:
            return [[0] * self.list_of_stages[stage].no_chains for stage in stages_to_disp]
        if len(burn_in) != len(stages_to_disp):
            raise ValueError(
                f"burn_in has {len(burn_in)} entries but {len(stages_to_disp)} stages are "
                f"displayed (one entry per DISPLAYED stage, in the order of stages_to_disp)."
            )
        for position, stage in enumerate(stages_to_disp):
            no_chains = self.list_of_stages[stage].no_chains
            if len(burn_in[position]) != no_chains:
                raise ValueError(
                    f"burn_in[{position}] has {len(burn_in[position])} entries but stage "
                    f"{self.stage_names[stage]!r} has {no_chains} chain(s)."
                )
        return [list(entry) for entry in burn_in]

    def _decompressed_samples(self, stage_index: int) -> List[np.ndarray]:
        """
        The decompressed (one row per iteration) chains of one stage.

        Raises:
            AttributeError: this ``Samples`` was built with ``decompress_samples=False``,
                so there are no decompressed chains. Before WS9b this surfaced as a
                per-chain "NOT AVAILABLE" print from an ``except BaseException`` that
                mis-reported a constructor choice as missing data (finding P6).
        """
        stage = self.list_of_stages[stage_index]
        samples = getattr(stage, "samples", None)
        if samples is None:
            raise AttributeError(
                "decompressed samples are not available because this Samples object was "
                "constructed with decompress_samples=False; re-create it with "
                "decompress_samples=True (the default) to use trace plots, histograms, "
                "autocorrelation, ESS, R-hat or correlation statistics."
            )
        return samples

    def _resolve_chains(self, stage_index: int, chains_to_disp: Iterable | None) -> List[int]:
        """
        POSITIONS (in ``list_of_stages[stage_index]``'s chain lists) of the chains of stage
        ``stage_index`` to include, honouring ``chains_to_disp`` (WS9b, finding P5: the
        argument is never silently ignored).

        ``chains_to_disp`` holds ORIGINAL chain numbers (the position of the chain's
        ``rank%04d`` file in its stage, 2026-10-08); a chain removed by the selection mask is
        simply not included. ``None`` means every kept chain of that stage. Without a selection
        positions and original numbers coincide, so this is the behaviour of WS9b unchanged.

        Raises:
            IndexError: a requested chain does not exist in this stage.
        """
        kept = self.chain_indices[stage_index]
        if chains_to_disp is None:
            return list(range(len(kept)))
        no_chains = self.no_chains_original[stage_index]
        chains = [int(chain) for chain in chains_to_disp]
        out_of_range = [chain for chain in chains if not 0 <= chain < no_chains]
        if out_of_range:
            raise IndexError(
                f"chains_to_disp={chains} requests chain(s) {out_of_range} but stage "
                f"{self.stage_names[stage_index]!r} has {no_chains} chain(s) (0..{no_chains - 1})."
            )
        position = {chain: index for index, chain in enumerate(kept)}
        return [position[chain] for chain in chains if chain in position]

    def _chains_across_stages(self, stages: List[int], chains_to_disp: Iterable | None
                              ) -> List[tuple[int, List[int | None]]]:
        """
        Chains followed across several stages (traces, autocorrelation, R-hat), identified by
        their ORIGINAL number: ``[(chain, [position in each of stages, or None]), ...]``.

        ``None`` = every chain kept in at least one of the stages, in increasing order;
        otherwise ``chains_to_disp`` in its own order, validated against the stage with the most
        chains. Without a selection and with equal chain counts (a single run) this is
        ``_resolve_chains(stages[0], chains_to_disp)`` with the same position in every stage.
        """
        if chains_to_disp is None:
            chains = sorted(set().union(*(self.chain_indices[stage] for stage in stages)))
        else:
            widest = max(stages, key=lambda stage: self.no_chains_original[stage])
            self._resolve_chains(widest, chains_to_disp)  # IndexError for an unknown chain
            chains = [int(chain) for chain in chains_to_disp]
        position_maps = [{chain: index for index, chain in enumerate(self.chain_indices[stage])}
                         for stage in stages]
        result = []
        for chain in chains:
            positions = [positions_of.get(chain) for positions_of in position_maps]
            if any(position is not None for position in positions):
                result.append((chain, positions))
        return result

    def _stage_data_dir(self, stage_index: int, data_name: str) -> str:
        """``<sampling_output of the stage's own run>/<data_name>/<stage name>``."""
        return os.path.join(self.stage_sampling_dirs[stage_index], data_name, self.stage_names[stage_index])

    def _raw_chain_numbers(self, stage_index: int, no_files: int,
                           requested_chains: List[int] | None) -> List[int]:
        """``raw_data`` file indices (= original chain numbers) to read for one stage: every file,
        or ``requested_chains``, minus the chains removed by the selection mask."""
        chains = list(range(no_files)) if requested_chains is None else list(requested_chains)
        excluded = self._excluded_chains(stage_index)
        if not excluded:
            return chains
        return [chain for chain in chains if chain not in excluded]

    def _excluded_chains(self, stage_index: int) -> set:
        """Original chain numbers of this stage removed by the selection (empty without one)."""
        return set(range(self.no_chains_original[stage_index])) - set(self.chain_indices[stage_index])

    def _adaptive_ranks(self, stage_index: int) -> List[int]:
        """Sorted ``rank_world`` values of the stage's UNFILTERED ``adaptive_stats``: chain
        position ``i`` of the adaptation plots/tables is ``ranks[i]``, also after a selection."""
        stats = self.run_data.adaptive_stats[stage_index]
        if stats is None or stats.empty or "rank_world" not in stats.columns:
            return [0]
        return sorted(int(r) for r in stats["rank_world"].unique())

    def _fully_excluded_stages(self, stages: Iterable[int]) -> List[int]:
        """Stages (of ``stages``) that have chains but whose every chain the selection removed."""
        return [stage for stage in stages
                if self.no_chains_original[stage] > 0 and not self.chain_indices[stage]]

    def _get_stage_names(self, stages_to_disp: List[int] | None = None):
        if stages_to_disp is None:
            return self.stage_names
        return [self.stage_names[i] for i in stages_to_disp]

    def _raw_data_available(self, stages_to_disp: List[int] | None = None):
        stage_indices = range(self.no_stages) if stages_to_disp is None else stages_to_disp
        for stage_index in stage_indices:
            dirname = self._stage_data_dir(stage_index, "raw_data")
            if not os.path.isdir(dirname):
                continue
            files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
            if len(files) > 0:
                return True
        return False
