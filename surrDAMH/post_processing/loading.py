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

from surrDAMH.modules.run_data import read_run, sampling_output_dir
from surrDAMH.post_processing.html_report import SamplesReports
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
                 decompress_samples: bool = True, load_posterior: bool = False):
        self.run_data = read_run(samples_dir, load_raw_data=False)
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
        self.list_of_stages = [
            StageSamples(no_parameters, stage_samples, decompress_samples, load_posterior)
            for stage_samples in self.run_data.samples
        ]
        self.summarize()

    def load_notes(self):
        """``notes[stage]``: one row per chain, already concatenated by ``read_run``."""
        self.notes = [notes.copy() for notes in self.run_data.notes]

    def load_subchain_stats(self):
        """``subchain_stats[stage]``: all DAMH iterations of all chains (empty for MH stages)."""
        self.subchain_stats = [pd.DataFrame() if stats is None else stats.copy()
                               for stats in self.run_data.subchain_stats]

    def load_adaptive_stats(self):
        """``adaptive_stats[stage]``: the per-period adaptation trace of every chain of an
        ``adaptive=True`` stage, empty for every stage whose proposal did not adapt
        (columns depend on the proposal class, see ``docs/outputs.md``)."""
        self.adaptive_stats = [pd.DataFrame() if stats is None else stats.copy()
                               for stats in self.run_data.adaptive_stats]

    def summarize(self):
        self.load_notes()
        self.load_subchain_stats()
        self.load_adaptive_stats()
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
        if stages_to_disp is None:
            stage_names = self.stage_names
        else:
            stage_names = [self.stage_names[i] for i in stages_to_disp]
        chosen_observations = np.arange(no_observations, dtype=np.int32)  # all

        weights_all = np.empty((0, 1))
        G_all = np.empty((0, no_observations))
        par_all = np.empty((0, self.no_parameters))
        for stage_name in stage_names:
            dirname = os.path.join(self.sampling_output_dir, "raw_data", stage_name)
            files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
            files.sort()
            for i in (range(len(files)) if requested_chains is None else requested_chains):
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

