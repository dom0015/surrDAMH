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
``Samples`` itself needs -- ``load_notes``, ``load_subchain_stats``, ``summarize``,
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
    summary: pd.DataFrame
    notes: List[pd.DataFrame]
    subchain_stats: List[pd.DataFrame]

    def _resolve_stages(self, stages_to_disp: Iterable | None) -> List[int]:
        """Stage indices to include; ``None`` means every stage of the run."""
        if stages_to_disp is None:
            return list(range(self.no_stages))
        stages = [int(stage) for stage in stages_to_disp]
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
        Chain indices of stage ``stage_index`` to include, honouring ``chains_to_disp``
        (WS9b, finding P5: the argument is never silently ignored).

        ``None`` means every chain of that stage -- the behaviour of every method before
        WS9b, and still the default everywhere.

        Raises:
            IndexError: a requested chain does not exist in this stage.
        """
        no_chains = self.list_of_stages[stage_index].no_chains
        if chains_to_disp is None:
            return list(range(no_chains))
        chains = [int(chain) for chain in chains_to_disp]
        out_of_range = [chain for chain in chains if not 0 <= chain < no_chains]
        if out_of_range:
            raise IndexError(
                f"chains_to_disp={chains} requests chain(s) {out_of_range} but stage "
                f"{self.stage_names[stage_index]!r} has {no_chains} chain(s) (0..{no_chains - 1})."
            )
        return chains

    def _get_stage_names(self, stages_to_disp: List[int] | None = None):
        if stages_to_disp is None:
            return self.stage_names
        return [self.stage_names[i] for i in stages_to_disp]

    def _raw_data_available(self, stages_to_disp: List[int] | None = None):
        stage_names = self._get_stage_names(stages_to_disp)
        for stage_name in stage_names:
            dirname = os.path.join(self.sampling_output_dir, "raw_data", stage_name)
            if not os.path.isdir(dirname):
                continue
            files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
            if len(files) > 0:
                return True
        return False
