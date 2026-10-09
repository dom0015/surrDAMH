#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construction of :class:`Samples` / :class:`StageSamples` from a
:class:`surrDAMH.modules.run_data.RunData` (output format v2), plus the per-run
bookkeeping that every other part of the package reads: stage names, ``notes``,
``subchain_stats``, ``adaptive_stats``, the acceptance ``summary`` table and the
raw-snapshot loader.

Split out of the former single-file ``surrDAMH/post_processing.py`` (WS9b). ``Samples``
is assembled here as the end of a linear mixin chain --
``SamplesStatistics(SamplesBase)`` (``statistics.py``) -> ``SamplesPlots`` (``plots.py``)
-> ``SamplesReports`` (``html_report.py``) -> ``Samples`` -- so that it stays ONE class
with the public API it always had, while each concern lives in its own module and every
layer sees what it uses from the layer(s) below it through ordinary inheritance.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

from surrDAMH.modules.run_data import (lineage_links, read_lineage, read_run,
                                       sampling_output_dir)
from surrDAMH.post_processing.html_report import SamplesReports
from surrDAMH.post_processing.selection import (default_selection, read_selection,
                                                selection_path, validate_selection,
                                                write_selection)
from surrDAMH.post_processing.statistics import (_raw_data_observation_columns,
                                                 _raw_data_parameter_columns,
                                                 _select_columns, decompress)


class StageSamples:
    """
    Per-stage view of the ``samples`` tables already loaded by
    ``surrDAMH.modules.run_data.read_run`` (output format v2).

    ``stage_samples`` is one ``(n, 2 + no_parameters)`` float array per chain, with the
    columns ``multiplicity, par_0..par_{p-1}, log_posterior``. ``self.weights`` keeps its
    historical attribute name but holds that ``multiplicity`` column.
    """

    def __init__(self, no_parameters, stage_samples: list[np.ndarray], decompress_samples: bool,
                 load_posterior: bool):
        self.no_chains = len(stage_samples)
        self.samples_compressed: list[Any] = [None] * self.no_chains
        self.weights: list[Any] = [None] * self.no_chains
        self.length: list[Any] = [None] * self.no_chains
        if decompress_samples:
            self.samples: list[Any] = [None] * self.no_chains
        if load_posterior:
            self.posterior: list[Any] = [None] * self.no_chains
        self.no_unique_samples = [0] * self.no_chains
        for i in range(self.no_chains):
            rows = stage_samples[i]
            # the multiplicity column is integral by construction; np.cov(fweights=...) and
            # decompress() both require integers, so keep it as one after the float read.
            self.weights[i] = rows[:, 0].astype(np.int64)
            # A30 (2026-09-17): the first row of a stage whose initial state was carried over
            # from the previous stage can have multiplicity 0 (it was already counted there).
            # Such a row occupies no iteration of the chain, so count_nonzero, not len --
            # otherwise the default histogram bin count (the only consumer) is inflated by
            # one per stage.
            self.no_unique_samples[i] = int(np.count_nonzero(self.weights[i]))
            self.length[i] = sum(self.weights[i])
            self.samples_compressed[i] = rows[:, 1:1 + no_parameters]
            if decompress_samples:
                self.samples[i] = decompress(self.samples_compressed[i], self.weights[i])
            if load_posterior:
                self.posterior[i] = rows[:, 1 + no_parameters]


class Samples(SamplesReports):
    """
    Facade over one run's output directory: statistics and plots on top of
    :func:`surrDAMH.modules.run_data.read_run` (output format v2).

    ``samples_dir`` is the run's ``Configuration.output_dir`` (the directory that CONTAINS
    ``sampling_output/``). Directories written in another format -- or with no
    ``run_manifest.json`` at all -- are refused with
    :class:`~surrDAMH.modules.manifest.RunFormatError`; there is no converter (decision 6).

    ``raw_data`` is deliberately NOT loaded eagerly (it is the largest output of a run):
    the snapshot consumers below stream it file by file, by column name.

    The class body is assembled from a linear chain of layers, one per concern --
    ``SamplesBase`` (``base.py``: shared attributes and resolvers), ``SamplesStatistics``
    (``statistics.py``), ``SamplesPlots`` (``plots.py``) and ``SamplesReports``
    (``html_report.py``, the direct base of ``Samples``) -- so ``Samples`` keeps exactly
    the public API it had as a single 2400-line module (WS9b).

    Note (WS9b): the former ``load_posterior_surrogate`` argument was REMOVED. The
    ``samples`` CSV has never had a surrogate-posterior column (only ``multiplicity``,
    ``par_*`` and ``log_posterior``), so the option could only ever raise ``IndexError``
    (finding P1).
    """

    def __init__(self, no_parameters: int, samples_dir: str,
                 decompress_samples: bool = True, load_posterior: bool = False,
                 debug: bool = False, include_previous: bool | None = None,
                 selection: str | os.PathLike | dict | bool | None = None):
        """
        Args:
            no_parameters: must equal the run's ``configuration.no_parameters``.
            samples_dir: the run's output_dir (for a lineage: the NEWEST run's).
            decompress_samples, load_posterior, debug: as before.
            include_previous: a run made by ``SamplingRun.continue_sampling`` continues earlier
                runs (its lineage, ``modules.run_data.read_lineage``). ``None`` (default): load the
                whole lineage iff every run of it sampled the same ``Problem``
                (``same_problem``), else this run only; ``True``: the whole lineage regardless
                (``lineage_note`` then says the problem changed); ``False``: this run only.
            selection: the (stage, chain) mask and per-chain burn-in to apply at load time
                (``post_processing.selection``). ``None``/``False``: none, every chain is used;
                ``True``: this run's ``post_processing_output/selection.json``, created with the
                defaults (every chain, burn-in stages ``is_excluded`` dropped) if missing; a path:
                that file (must exist); a dict: a selection in the file's layout. Excluded chains
                are removed from the stage's chain lists (``chain_indices[stage]`` keeps the
                original chain numbers, which ``chains_to_disp`` refers to everywhere), their
                ``notes``/``subchain_stats``/``adaptive_stats`` rows and ``raw_data`` files are
                left out too, and ``burn_in`` leading compressed rows are dropped from every kept
                chain (``raw_data`` snapshots have no burn-in).
        """
        self.debug = debug
        self.lineage_note: str | None = None
        if include_previous is None:
            dirs, same_problem, complete = lineage_links(samples_dir, strict=False)
            if len(dirs) > 1 and same_problem and complete:
                self.run_data = read_lineage(samples_dir, load_raw_data=False)
            else:
                self.run_data = read_run(samples_dir, load_raw_data=False)
                if not complete:
                    self.lineage_note = (f"This run continues {dirs[0]!r}'s lineage, but an earlier run could not "
                                         "be read; only this run is analysed.")
                    print(f"WARNING: {self.lineage_note}", flush=True)
                elif len(dirs) > 1:
                    self.lineage_note = ("This run continues earlier run(s) that sampled a different Problem (or "
                                         "were continued by hand, which does not record it); only this run is "
                                         "analysed. Pass include_previous=True to include them.")
        elif include_previous is True:
            self.run_data = read_lineage(samples_dir, load_raw_data=False)
            if not self.run_data.same_problem:
                self.lineage_note = ("The Problem changed within this lineage (a continuation with problem=, or "
                                     "one continued by hand): the stages below did not all sample the same "
                                     "posterior. They are combined because include_previous=True.")
        elif include_previous is False:
            self.run_data = read_run(samples_dir, load_raw_data=False)
        else:
            raise TypeError(f"include_previous must be None, True or False, got {include_previous!r}")
        if int(no_parameters) != self.run_data.no_parameters:
            raise ValueError(
                f"no_parameters={no_parameters} does not match the run in {samples_dir} "
                f"(run_manifest.json says {self.run_data.no_parameters})"
            )
        self.no_parameters = no_parameters
        self.sampling_output_dir = sampling_output_dir(samples_dir)
        self.samples_dir = os.path.join(self.sampling_output_dir, "samples")
        self.stage_names = list(self.run_data.stage_names)
        self.no_stages = len(self.stage_names)
        # per stage: the sampling_output/ of the run of the lineage that wrote it
        stage_dirs = self.run_data.stage_output_dir or [samples_dir] * self.no_stages
        self.stage_sampling_dirs = [sampling_output_dir(d) for d in stage_dirs]

        self._apply_selection(selection, samples_dir)
        self.list_of_stages = [
            StageSamples(no_parameters, stage_samples, decompress_samples, load_posterior)
            for stage_samples in self._selected_samples
        ]
        del self._selected_samples
        self.summarize()

    def _apply_selection(self, selection, samples_dir: str) -> None:
        """Resolve ``selection`` (see ``__init__``) into ``chain_indices``/``_selected_samples``."""
        self.selection_path: str | None = None
        self.selection_created = False
        self.selection_content: dict | None = None
        self.selection_mask: dict | None = None
        if selection is True:
            path = selection_path(samples_dir)
            if not os.path.isfile(path):
                write_selection(path, default_selection(self.run_data))
                self.selection_created = True
            self.selection_path = path
            self.selection_content, self.selection_mask = read_selection(path, self.run_data)
        elif isinstance(selection, dict):
            self.selection_content = selection
            self.selection_mask = validate_selection(selection, self.run_data, "the selection dict")
        elif selection is not None and selection is not False:
            path = os.path.abspath(os.fspath(selection))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"selection file {path!r} not found")
            self.selection_path = path
            self.selection_content, self.selection_mask = read_selection(path, self.run_data)

        self.no_chains_original = [self.run_data.no_chains(i) for i in range(self.no_stages)]
        self.chain_indices: list[list[int]] = []
        self.selection_burn_in: list[list[int]] = []
        self._selected_samples = []
        for index, name in enumerate(self.stage_names):
            rows = self.run_data.samples[index]
            if self.selection_mask is None:
                self.chain_indices.append(list(range(len(rows))))
                self.selection_burn_in.append([0] * len(rows))
                self._selected_samples.append(rows)
                continue
            include = self.selection_mask[name]["include"]
            burn_in = self.selection_mask[name]["burn_in"]
            kept = [chain for chain in range(len(rows)) if include[chain]]
            self.chain_indices.append(kept)
            self.selection_burn_in.append([burn_in[chain] for chain in kept])
            self._selected_samples.append([rows[chain][burn_in[chain]:] for chain in kept])

    def _kept_rows(self, frame: pd.DataFrame, stage_index: int, by_rank: bool) -> pd.DataFrame:
        """``frame`` without the rows of excluded chains: by ``rank_world`` (``by_rank``) or, for
        ``notes`` (one row per chain, in rank-file order), by position."""
        if not self._excluded_chains(stage_index) or frame.empty:
            return frame
        kept = self.chain_indices[stage_index]
        if by_rank:
            if "rank_world" not in frame.columns:
                return frame
            ranks = (self.run_data.chain_ranks[stage_index] if self.run_data.chain_ranks
                     else list(range(self.no_chains_original[stage_index])))
            kept_ranks = {int(ranks[chain]) for chain in kept}
            return frame[frame["rank_world"].astype(int).isin(kept_ranks)].reset_index(drop=True)
        if len(frame) != self.no_chains_original[stage_index]:
            print(f"WARNING: notes of stage {self.stage_names[stage_index]!r} have {len(frame)} row(s) for "
                  f"{self.no_chains_original[stage_index]} chain(s); the selection is not applied to them.",
                  flush=True)
            return frame
        return frame.iloc[kept].reset_index(drop=True)

    def load_notes(self):
        """``notes[stage]``: one row per chain, already concatenated by ``read_run``."""
        self.notes = [self._kept_rows(notes, i, by_rank=False).copy()
                      for i, notes in enumerate(self.run_data.notes)]

    def load_subchain_stats(self):
        """``subchain_stats[stage]``: all DAMH iterations of all chains (empty for MH stages)."""
        self.subchain_stats = [pd.DataFrame() if stats is None else self._kept_rows(stats, i, by_rank=True).copy()
                               for i, stats in enumerate(self.run_data.subchain_stats)]

    def load_adaptive_stats(self):
        """``adaptive_stats[stage]``: the per-period adaptation trace of every chain of an
        ``adaptive=True`` stage, empty for every stage whose proposal did not adapt
        (columns depend on the proposal class, see ``docs/outputs.md``)."""
        self.adaptive_stats = [pd.DataFrame() if stats is None else self._kept_rows(stats, i, by_rank=True).copy()
                               for i, stats in enumerate(self.run_data.adaptive_stats)]

    def load_carry_over(self):
        """``carry_over[stage]``: ``(carry_over, summary)`` dict pair handed from this stage to
        the next one (``modules.continuation.save_carry_over``/``load_carry_over``), ``None``
        for a non-adaptive stage or one written before this file existed (2026-09-21)."""
        self.carry_over = list(self.run_data.carry_over)

    def summarize(self):
        self.load_notes()
        self.load_subchain_stats()
        self.load_adaptive_stats()
        self.load_carry_over()
        # create pandas data frame containing sums of dataframes in self.notes:
        summary = pd.DataFrame()
        for notes in self.notes:
            summary = pd.concat([summary, notes.iloc[:, :-1].sum()], axis=1)
        # transpose data frame:
        summary = summary.T
        # name the rows with self.stage_names:
        summary = summary.set_axis(labels=self.stage_names, axis=0)

        subchain_acceptance_rate = []
        subchain_move_rate = []
        outer_acceptance_given_move = []
        for stats in self.subchain_stats:
            if stats.empty:
                subchain_acceptance_rate.append(np.nan)
                subchain_move_rate.append(np.nan)
                outer_acceptance_given_move.append(np.nan)
                continue
            subchain_acceptance_rate.append(float(stats["subchain_acceptance_rate"].mean()))
            subchain_move_rate.append(float(stats["outer_proposed_changed"].mean()))
            moved = stats["outer_proposed_changed"] > 0
            if moved.any():
                outer_acceptance_given_move.append(float(stats.loc[moved, "outer_accepted"].mean()))
            else:
                outer_acceptance_given_move.append(np.nan)

        summary["subchain_acc_rate"] = subchain_acceptance_rate
        summary["subchain_move_rate"] = subchain_move_rate
        summary["outer_acc_given_move"] = outer_acceptance_given_move
        self.summary = summary

    def get_summary(self, csv_filepath: str | None = None):
        print(self.summary)
        if csv_filepath:
            self.summary.to_csv(csv_filepath, index=True)
        return self.summary

    def load_snapshots(self, no_observations: int, chains_to_disp: Iterable | None = None,
                       stages_to_disp: List[int] | None = None):
        """
        Loads snapshots (parameters, observations).
        Observations can be loaded only if save_raw_data==True.

        Args:
            no_observations (int): number of observations
            chains_to_disp (list of int of length N): chains that should be included (otherwise all chains are included)
            stages_to_disp (list of int): stages that should be included (otherwise all stages are included)
        """
        requested_chains = None if chains_to_disp is None else [int(c) for c in chains_to_disp]
        stage_indices = list(range(self.no_stages)) if stages_to_disp is None else list(stages_to_disp)
        chosen_observations = np.arange(no_observations, dtype=np.int32)  # all

        weights_all = np.empty((0, 1))
        G_all = np.empty((0, no_observations))
        par_all = np.empty((0, self.no_parameters))
        for stage_index in stage_indices:
            stage_name = self.stage_names[stage_index]
            dirname = self._stage_data_dir(stage_index, "raw_data")
            files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
            files.sort()
            for i in self._raw_chain_numbers(stage_index, len(files), requested_chains):
                if i >= len(files):
                    print("FILE NOT AVAILABLE - chain:", i, "stage", stage_name, flush=True)
                    continue
                path_samples = os.path.join(dirname, files[i])
                df_samples = pd.read_csv(path_samples)
                types = df_samples["state_type"]
                idx = np.ones(len(types), dtype=bool)
                # prerejected proposals have no EXACT observations (their obs_* block is
                # NaN in format v2), so they are excluded here; rejected ones are kept.
                idx[types == "prerejected"] = 0
                idx[types == "rejected"] = 1
                temp = np.arange(len(types))
                weights = temp[idx]
                weights[1:] = weights[1:] - weights[:-1]
                if sum(idx) == 0:
                    print("EMPTY - chain:", i, "stage", stage_name, flush=True)
                else:
                    G_values = _select_columns(df_samples, _raw_data_observation_columns(chosen_observations))
                    G_values = G_values[idx]
                    G_all = np.vstack((G_all, G_values))
                    param = _select_columns(df_samples, _raw_data_parameter_columns(self.no_parameters))
                    param = param[idx]
                    par_all = np.vstack((par_all, param))
                    weights = weights.reshape((-1, 1))
                    weights_all = np.vstack((weights_all, weights))
            print("loaded - stage", stage_name, flush=True)
        return par_all, G_all, weights_all

