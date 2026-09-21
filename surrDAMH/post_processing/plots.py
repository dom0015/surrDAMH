#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib figures for one sampling run: chain traces, 1D/2D parameter histograms,
observation histograms, surrogate-quality diagnostics, autocorrelation, acceptance rates,
correlation heatmap, best-fit trajectories and derived-field maps.

Split out of the former single-file ``surrDAMH/post_processing.py`` (WS9b).
``SamplesPlots(SamplesStatistics)`` is the second link of the mixin chain combined into
:class:`surrDAMH.post_processing.Samples` in ``loading.py``
(``SamplesStatistics`` -> ``SamplesPlots`` -> ``SamplesReports`` -> ``Samples``), so it
can use the statistics layer's methods (``_collect_samples_matrix``, ``Autocorrelation``,
...) and the base attributes/resolvers (``_get_stage_names``, ``_raw_data_available``,
...) through ordinary inheritance.

Figure ownership: every method here that RETURNS a figure hands it to the caller, who
must close it; on the exception path the method closes the figure itself, so a failing
plot can no longer leak one (finding P8).
"""

from __future__ import annotations

import os
from typing import Iterable, List, Literal

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.stats import norm

from surrDAMH.post_processing.statistics import (Autocorrelation,
                                                 SamplesStatistics,
                                                 _raw_data_observation_columns,
                                                 _select_columns)


class SamplesPlots(SamplesStatistics):
    """Plotting layer of :class:`surrDAMH.post_processing.Samples` (see module docstring)."""

    @staticmethod
    def _placeholder_figure(message: str):
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        return fig

    def plot_chains(self, average=False, parameters_to_disp: Iterable | None = None,
                    stages_to_disp: Iterable | None = None, scale: List[Literal["linear", "log", "ln"]] | None = None,
                    par_names: List[str] | None = None, burn_in: List[List[int]] | None = None,
                    chains_to_disp: Iterable | None = None):
        """
        Plot generated chains.
        If average==True, serves to analyze the convergence af averages for several chains generated in parallel.

        Args:
            parameters_to_disp (list of int of length N): Parameters to display. If None, all parameters are displayed.
            stages_to_disp (list of int of length S): Stages to display. If None, all stages are displayed.
            scale (list of "linear", "log" of length N): Scale of the plots. If None, all set to "linear".
            par_names (list of str of length N): Parameter names. If None, "par. 0", "par. 1", ... is used.
            burn_in (list of list of int): Burn-in for each stage and each chain. If None, set to [[0] * no_chains] * S.
            chains_to_disp (list of int): Chains to display. If None, all chains are displayed.

        Returns:
            tuple[Figure, array of Axes]: the caller owns the figure and must close it.
        """
        if parameters_to_disp is None:
            parameters_to_disp = range(self.no_parameters)
        no_parameters_to_disp = len(parameters_to_disp)
        stages_to_disp = self._resolve_stages(stages_to_disp)
        if scale is None:
            scale = ["linear"] * no_parameters_to_disp
        burn_in = self._burn_in_for(stages_to_disp, burn_in)

        fig, axes = plt.subplots(no_parameters_to_disp, 1, sharex=False, sharey=False, figsize=(15, 15))
        try:
            plt.subplots_adjust(wspace=0.5, hspace=0.3)
            for idi, i in enumerate(parameters_to_disp):
                axis = axes[idi]
                # one trace per displayed chain, concatenated over the displayed stages
                all_x = {s: np.zeros((0,)) for s in self._resolve_chains(stages_to_disp[0], chains_to_disp)}
                for idj, j in enumerate(stages_to_disp):
                    for s in self._resolve_chains(j, chains_to_disp):
                        tmp = self._decompressed_samples(j)[s][burn_in[idj][s]:, i]
                        all_x[s] = np.concatenate((all_x.get(s, np.zeros((0,))), tmp))
                for s in sorted(all_x):
                    chain = all_x[s]
                    if average:
                        chain_cumsum = np.cumsum(chain)
                        indices = np.arange(1, len(chain) + 1)
                        axis.plot(chain_cumsum / indices)
                    else:
                        axis.plot(chain)
                    plt.yscale(scale[idi])
                # determine parameter name
                if par_names is not None:
                    par_name = par_names[i].replace('_', '\\_')
                    label = "${0}$".format(par_name)
                else:
                    label = "$par. {0}$".format(i)
                if scale[idi] == "log":
                    label += "\n(log10)"
                elif scale[idi] == "ln":
                    label += "\n(ln)"
                axis.set_title(label, x=1.05, multialignment='center')
                axis.grid(True)
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise
        return fig, axes

    def _plot_hist_1d(self, axis, burn_in: List[List[int]], param_no: int,
                      stages_to_disp: List[int], bins: int, show: bool, scale=str,
                      prior_component=None, chains_to_disp: Iterable | None = None):
        # use weighted data (compressed)
        all_x = np.zeros((0,))
        no_unique_samples_sum = 0
        for idj, j in enumerate(stages_to_disp):
            for i in self._resolve_chains(j, chains_to_disp):
                no_unique_samples_sum += self.list_of_stages[j].no_unique_samples[i]  # burn-in not excluded
                tmp = self._decompressed_samples(j)[i][burn_in[idj][i]:, param_no]
                if scale == "log":
                    all_x = np.concatenate((all_x, np.log10(tmp)))
                elif scale == "ln":
                    all_x = np.concatenate((all_x, np.log(tmp)))
                else:
                    all_x = np.concatenate((all_x, tmp))
        if bins is None:
            bins = np.floor(no_unique_samples_sum / 20)
            bins = min(bins, 100)
            bins = max(bins, 10)
        axis.hist(all_x, bins=int(bins), density=True)
        # Overlay marginal prior PDF if provided. A prior component is only required to be a
        # distribution of the library's own kind; anything without a scalar-vectorised .pdf
        # (e.g. a multivariate or discrete component) simply gets no overlay.
        if prior_component is not None and len(all_x) > 0:
            x_lo, x_hi = all_x.min(), all_x.max()
            margin = 0.15 * (x_hi - x_lo) if x_hi > x_lo else 1.0
            x_grid = np.linspace(x_lo - margin, x_hi + margin, 300)
            try:
                y_grid = prior_component.pdf(x_grid)
            except (AttributeError, TypeError, ValueError) as exc:
                print(f"prior overlay skipped for parameter {param_no}: "
                      f"{type(prior_component).__name__}.pdf failed ({exc})", flush=True)
            else:
                axis.plot(x_grid, y_grid, 'r-', linewidth=2, label='prior')
        axis.grid(True)
        if show:
            plt.show()

    def _plot_hist_2d(self, axis, burn_in: List[List[int]], param_no: List[int], stages_to_disp: List[int],
                      bins: int, show: bool, scale: List[str], colorbar: bool = False,
                      chains_to_disp: Iterable | None = None):
        # use weighted data
        all_x = np.zeros((0,))
        all_y = np.zeros((0,))
        no_unique_samples_sum = 0
        for idj, j in enumerate(stages_to_disp):
            for i in self._resolve_chains(j, chains_to_disp):
                no_unique_samples_sum += self.list_of_stages[j].no_unique_samples[i]  # burn-in not excluded
                chain = self._decompressed_samples(j)[i]
                tmp_x = chain[burn_in[idj][i]:, param_no[0]]
                tmp_y = chain[burn_in[idj][i]:, param_no[1]]
                if scale[0] == "log":
                    all_x = np.concatenate((all_x, np.log10(tmp_x)))
                elif scale[0] == "ln":
                    all_x = np.concatenate((all_x, np.log(tmp_x)))
                else:
                    all_x = np.concatenate((all_x, tmp_x))
                if scale[1] == "log":
                    all_y = np.concatenate((all_y, np.log10(tmp_y)))
                elif scale[1] == "ln":
                    all_y = np.concatenate((all_y, np.log(tmp_y)))
                else:
                    all_y = np.concatenate((all_y, tmp_y))
        if bins is None:
            bins = np.floor(np.sqrt(no_unique_samples_sum / 4))
            bins = min(bins, 100)
            bins = max(bins, 10)
        axis.hist2d(all_x, all_y, bins=int(bins), cmap="binary")
        axis.grid(True)
        if colorbar:
            axis.colorbar()
        if show:
            plt.show()

    def plot_hist_grid(self, bins1d: int = 20, bins2d: int = 20, parameters_to_disp: Iterable | None = None,
                       stages_to_disp: Iterable | None = None, scale: List[Literal["linear", "log", "ln"]] | None = None,
                       par_names: List[str] | None = None, burn_in: List[List[int]] | None = None,
                       prior=None, chains_to_disp: Iterable | None = None):
        """
        Plots a N x N grid of 1d and 2d histograms, where N is the number of displayed parameters.

        Args:
            bins1d (int): Number of bins in 1d histograms.
            bins2d (int): Number of bins in 2d histograms.
            parameters_to_disp (list of int of length N): Parameters to display. If None, all parameters are displayed.
            stages_to_disp (list of int of length S): Stages to display. If None, all stages are displayed.
            scale (list of "linear", "log", "ln" of length N): Scale of the histograms. If None, all set to "linear".
            par_names (list of str of length N): Parameter names. If None, "par. 0", "par. 1", ... is used.
            burn_in (list of list of int): Burn-in for each stage and each chain. If None, set to [[0] * no_chains] * S.
            prior (PriorIndependentComponents or None): Prior distribution. If provided, marginal prior PDFs
                are overlaid on the 1D diagonal histograms.
            chains_to_disp (list of int): Chains to display. If None, all chains are displayed.

        Returns:
            tuple[Figure, array of Axes]: the caller owns the figure and must close it.
        """
        if parameters_to_disp is None:
            parameters_to_disp = range(self.no_parameters)
        no_parameters_to_disp = len(parameters_to_disp)
        stages_to_disp = self._resolve_stages(stages_to_disp)
        if scale is None:
            scale = ["linear"] * no_parameters_to_disp
        burn_in = self._burn_in_for(stages_to_disp, burn_in)

        fig, axes = plt.subplots(no_parameters_to_disp, no_parameters_to_disp,
                                 sharex=False, sharey=False, figsize=(15, 15))
        try:
            plt.subplots_adjust(wspace=0.5, hspace=0.3)
            for idi, i in enumerate(parameters_to_disp):
                for idj, j in enumerate(parameters_to_disp):
                    axis = axes[idi, idj]
                    if idi == idj:
                        # Get the prior component for this parameter (if prior provided)
                        pc = None
                        if prior is not None and hasattr(prior, 'list_of_components'):
                            if i < len(prior.list_of_components):
                                pc = prior.list_of_components[i]
                        self._plot_hist_1d(
                            axis=axis,
                            param_no=i,
                            burn_in=burn_in,
                            stages_to_disp=stages_to_disp,
                            bins=bins1d,
                            show=False,
                            scale=scale[idi],
                            prior_component=pc,
                            chains_to_disp=chains_to_disp)
                    else:
                        self._plot_hist_2d(axis=axis, param_no=[j, i], burn_in=burn_in, stages_to_disp=stages_to_disp,
                                           bins=bins2d, show=False, scale=[scale[idj], scale[idi]],
                                           chains_to_disp=chains_to_disp)
                    if idi == 0:
                        # determine parameter name
                        if par_names is not None:
                            par_name = par_names[j].replace('_', '\\_')
                            label = "${0}$".format(par_name)
                        else:
                            label = "$par. {0}$".format(j)
                        if scale[idj] == "log":
                            label += "\n(log10)"
                        elif scale[idj] == "ln":
                            label += "\n(ln)"
                        axis.set_title(label, x=1.05, rotation=45, multialignment='center')
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise
        return fig, axes

    def plot_hist_marginals(self, bins1d: int = 20, parameters_to_disp: Iterable | None = None,
                            stages_to_disp: Iterable | None = None,
                            scale: List[Literal["linear", "log", "ln"]] | None = None,
                            par_names: List[str] | None = None,
                            burn_in: List[List[int]] | None = None,
                            prior=None, ncols: int = 4, chains_to_disp: Iterable | None = None):
        """
        Plot 1D marginal histograms for a selected set of parameters.

        This is intended for higher-dimensional problems where a full NxN
        histogram grid becomes too expensive or visually uninformative.

        Returns:
            tuple[Figure, array of Axes]: the caller owns the figure and must close it.
        """
        if parameters_to_disp is None:
            parameters_to_disp = range(self.no_parameters)
        parameters_to_disp = list(parameters_to_disp)
        no_parameters_to_disp = len(parameters_to_disp)
        stages_to_disp = self._resolve_stages(stages_to_disp)
        if scale is None:
            scale = ["linear"] * no_parameters_to_disp
        burn_in = self._burn_in_for(stages_to_disp, burn_in)

        ncols = max(1, int(ncols))
        nrows = int(np.ceil(no_parameters_to_disp / ncols))
        fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False,
                                 figsize=(4.5 * ncols, 3.2 * nrows))
        try:
            axes = np.atleast_1d(axes).reshape(-1)
            plt.subplots_adjust(wspace=0.35, hspace=0.5)

            for idi, i in enumerate(parameters_to_disp):
                axis = axes[idi]
                pc = None
                if prior is not None and hasattr(prior, 'list_of_components'):
                    if i < len(prior.list_of_components):
                        pc = prior.list_of_components[i]
                self._plot_hist_1d(
                    axis=axis,
                    param_no=i,
                    burn_in=burn_in,
                    stages_to_disp=stages_to_disp,
                    bins=bins1d,
                    show=False,
                    scale=scale[idi],
                    prior_component=pc,
                    chains_to_disp=chains_to_disp,
                )
                if par_names is not None:
                    par_name = par_names[i].replace('_', '\\_')
                    label = "${0}$".format(par_name)
                else:
                    label = "$par. {0}$".format(i)
                if scale[idi] == "log":
                    label += "\n(log10)"
                elif scale[idi] == "ln":
                    label += "\n(ln)"
                axis.set_title(label)

            for axis in axes[no_parameters_to_disp:]:
                axis.axis("off")
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise

        return fig, axes

    def hist_observations(self, no_observations: int, chosen_observations: np.ndarray | None = None, grid: np.ndarray | None = None,
                          grid_interp: np.ndarray | None = None, bins: List[int] | None = None, chains_to_disp: Iterable | None = None,
                          stages_to_disp: List[int] | None = None, observations: np.ndarray | None = None, cmap="viridis_r"):
        """
        Creates 2d histogram of observations.
        Observations can be loaded only if save_raw_data==True.
        Suitable for observations in the form of time series.

        Args:
            no_observations (int): number of observations
            chosen_observations (ndarray of int of length N): indices forming the time series (otherwise all are used)
            grid (ndarray of float of length N): time values for the time series (otherwise range(N) is used)
            grid_interp (ndarray of float): time grid for horizontal axis (otherwise grid_interp = grid)
            bins (list of int of length 2): [bins_x, bins_y] (optional)
            chains_to_disp (list of int of length N): chains that should be included (otherwise all chains are included)
            stages_to_disp (list of int): stages that should be included (otherwise all stages are included)
            observations (ndarray of shape (no_observations,)): vector of observations (optional)
        """
        if no_observations <= 0:
            return self._placeholder_figure(
                "Observation histograms are not available because no_observations <= 0."
            )

        if chosen_observations is None:
            chosen_observations = np.arange(no_observations, dtype=np.int32)
        if grid is None:
            grid = np.arange(len(chosen_observations), dtype=np.int32)
        if grid_interp is None:
            grid_interp = grid
        requested_chains = None if chains_to_disp is None else [int(c) for c in chains_to_disp]
        stage_names = self._get_stage_names(stages_to_disp)

        if not self._raw_data_available(stages_to_disp=stages_to_disp):
            return self._placeholder_figure(
                "Observation histograms are not available because raw snapshots were not saved.\n"
                "Run sampling with configuration save_snapshots_to_file=True to enable this plot."
            )

        len_grid = len(grid_interp)
        grid_interp = np.arange(len_grid)
        x_all = np.empty((0, len_grid))
        weights_all = np.empty((0, len_grid))
        G_all = np.empty((0, len_grid))
        # param_all = np.empty((0, self.no_parameters))
        for stage_name in stage_names:
            dirname = os.path.join(self.sampling_output_dir, "raw_data", stage_name)
            if not os.path.isdir(dirname):
                print("hist_G DIRECTORY NOT AVAILABLE:", dirname)
                continue
            files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
            files.sort()
            for i in (range(len(files)) if requested_chains is None else requested_chains):
                if i >= len(files):
                    print("hist_G FILE NOT AVAILABLE - chain:", i, "stage", stage_name)
                    continue
                path_samples = os.path.join(dirname, files[i])
                df_samples = pd.read_csv(path_samples)
                types = df_samples["state_type"]
                idx = np.ones(len(types), dtype=bool)
                idx[types == "prerejected"] = 0
                idx[types == "rejected"] = 0
                temp = np.arange(len(types))
                weights = temp[idx]
                no_accepted = len(weights)
                weights[1:] = weights[1:] - weights[:-1]
                if sum(idx) == 0:
                    print("hist_G EMPTY - chain:", i)
                else:
                    G_values = _select_columns(df_samples, _raw_data_observation_columns(chosen_observations))
                    G_values = G_values[idx]
                    G_interp = np.zeros((no_accepted, len_grid))
                    for i in range(no_accepted):
                        G_interp[i, :] = np.interp(grid_interp, grid, G_values[i, :])
                    G_all = np.vstack((G_all, G_interp))
                    weights = weights.reshape((-1, 1))
                    weights = np.repeat(weights, len_grid, 1)
                    x = np.repeat(grid_interp.reshape((1, -1)), no_accepted, 0)
                    x_all = np.vstack((x_all, x))
                    weights_all = np.vstack((weights_all, weights))

        if G_all.size == 0 or weights_all.size == 0:
            return self._placeholder_figure(
                "Observation histograms are not available because no accepted snapshot data were found."
            )

        fig = plt.figure()
        try:
            n_samples = G_all.shape[0]
            # (the block below used to be present twice, verbatim; the second copy only
            #  recomputed the same values and was removed in WS9b)
            if bins is None:
                nbins = (5 * 1.2 * np.sqrt(n_samples)).astype(int)
                bins = [len_grid, nbins]
            G_all = G_all.flatten()
            min_G = min(G_all)
            max_G = max(G_all)
            range_G = max_G - min_G
            if len_grid == 1:
                hist_range = None
            else:
                x_min = -0.5
                x_max = float(len_grid - 1 + 0.5)
                if grid is not None and len(grid) == len_grid:
                    x_min = float(np.min(grid)) - 0.5
                    x_max = float(np.max(grid)) + 0.5
                hist_range = [[x_min, x_max], [min_G - range_G / 10, max_G + range_G / 10]]
            if len_grid == 1:
                output = plt.hist(G_all, bins=bins[-1], weights=weights_all.flatten())
            else:
                output = plt.hist2d(x_all.flatten(), G_all, bins=bins, range=hist_range, weights=weights_all.flatten(), cmap=cmap)
                plt.colorbar(output[3])
            plt.grid()

            if observations is not None:
                if len_grid == 1:
                    plt.plot(observations[chosen_observations], 0, 'ro', linestyle='None', markersize=6, label="observation")
                else:
                    plt.plot(grid, observations[chosen_observations], 'r--', linewidth=1, marker='o', markersize=4, label="observations")
                plt.legend()
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise
        return fig

    def plot_surrogate_quality(self):
        """
        Plot surrogate model quality metrics (RMSE and max absolute error)
        over the course of the simulation.

        Reads the CSV file generated by the COLLECTOR process at
        ``<output_dir>/sampling_output/surrogate_quality.csv``.

        Returns:
            tuple[Figure, Axes] | tuple[Figure, None]: The figure and axes, or a placeholder
            figure with None axes if the data is not available.
        """
        return self._plot_surrogate_quality_from_csv(
            csv_filename="surrogate_quality.csv",
            title="Surrogate Model Quality",
            x_column="snapshots_total",
            x_label="Total Snapshots Collected",
            metric_columns=("rmse", "max_abs_error"),
            metric_labels=("RMSE", "Max Absolute Error"),
            missing_message=(
                "Surrogate quality plot unavailable:\n"
                "No surrogate_quality.csv file found.\n"
                "This file is generated only when a surrogate model (collector) is used."
            ),
        )

    def plot_surrogate_quality_test(self):
        """
        Plot surrogate model quality metrics on a fixed test set.

        Reads the CSV file generated by the COLLECTOR process at
        ``<output_dir>/sampling_output/surrogate_quality_test.csv``.

        Returns:
            tuple[Figure, Axes] | tuple[Figure, None]: The figure and axes, or a placeholder
            figure with None axes if the data is not available.
        """
        return self._plot_surrogate_quality_from_csv(
            csv_filename="surrogate_quality_test.csv",
            title="Surrogate Model Quality on Fixed Test Data",
            x_column="update_index",
            x_label="Surrogate Update Index",
            metric_columns=("rmse", "max_abs_error"),
            metric_labels=("RMSE", "Max Absolute Error"),
            missing_message=(
                "Fixed-test surrogate quality plot unavailable:\n"
                "No surrogate_quality_test.csv file found.\n"
                "This file is generated only when fixed surrogate test data are supplied to the collector."
            ),
        )

    def plot_surrogate_quality_test_weighted(self):
        """
        Plot posterior-weighted surrogate model quality metrics on a fixed test set.

        Reads the CSV file generated by the COLLECTOR process at
        ``<output_dir>/sampling_output/surrogate_quality_test.csv``.

        Returns:
            tuple[Figure, Axes] | tuple[Figure, None]: The figure and axes, or a placeholder
            figure with None axes if the data is not available.
        """
        return self._plot_surrogate_quality_from_csv(
            csv_filename="surrogate_quality_test.csv",
            title="Posterior-Weighted Surrogate Quality on Fixed Test Data",
            x_column="update_index",
            x_label="Surrogate Update Index",
            metric_columns=("weighted_rmse", "weighted_mean_abs_error"),
            metric_labels=("Weighted RMSE", "Weighted Mean Absolute Error"),
            missing_message=(
                "Posterior-weighted fixed-test surrogate quality plot unavailable:\n"
                "No surrogate_quality_test.csv file found.\n"
                "This file is generated only when fixed surrogate test data with weights are supplied to the collector."
            ),
        )

    def _plot_surrogate_quality_from_csv(self, csv_filename: str, title: str, x_column: str,
                                         x_label: str, metric_columns: tuple[str, str],
                                         metric_labels: tuple[str, str], missing_message: str):
        """Shared plotting helper for surrogate-quality CSV diagnostics."""
        csv_path = os.path.join(self.sampling_output_dir, csv_filename)
        if not os.path.isfile(csv_path):
            fig = self._placeholder_figure(missing_message)
            return fig, None

        df = pd.read_csv(csv_path)
        if df.empty:
            fig = self._placeholder_figure(f"{title} plot unavailable: CSV file is empty.")
            return fig, None

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        try:
            plt.subplots_adjust(hspace=0.15)

            x = df[x_column].values

            colors = ('b', 'r')
            for ax, metric_column, metric_label, color in zip(axes, metric_columns, metric_labels, colors):
                ax.plot(x, df[metric_column].values, f'{color}-o', markersize=4, linewidth=1.5, label=metric_label)
                ax.set_ylabel(metric_label, fontsize=12)
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)

            axes[0].set_title(title, fontsize=14, fontweight="bold")
            axes[1].set_xlabel(x_label, fontsize=12)
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise

        return fig, axes

    def plot_autocorr(self, stages_to_disp: Iterable | None = None, parameters_to_disp: Iterable | None = None,
                      par_names: List[str] | None = None, max_lag: int = 100,
                      chains_to_disp: Iterable | None = None):
        """
        Plot autocorrelation functions for parameters.

        Args:
            stages_to_disp (list of int): stages to include
            parameters_to_disp (list of int): parameters to display
            par_names (list of str): parameter names
            max_lag (int): maximum lag to display
            chains_to_disp (list of int): chains to include (None = every chain)

        Returns:
            tuple(Figure, Axes): axes is None if the autocorrelation could not be
            estimated; the figure is then a placeholder carrying the reason.
        """
        stages_to_disp = self._resolve_stages(stages_to_disp)
        if parameters_to_disp is None:
            parameters_to_disp = range(self.no_parameters)

        try:
            autocorr = Autocorrelation(self, stages_to_disp=stages_to_disp,
                                       chains_to_disp=chains_to_disp)
            autocorr.calculate_autocorr_function()
            autocorr_func_mean = autocorr.calculate_autocorr_function_mean()
        except (ValueError, IndexError, FloatingPointError) as exc:
            # chains too short for emcee's estimator, or an empty stage/chain selection
            print(f"Error plotting autocorrelation: {exc}")
            return self._placeholder_figure(f"Autocorrelation plot unavailable: {exc}"), None

        no_params = len(parameters_to_disp)
        fig, axes = plt.subplots(no_params, 1, figsize=(12, 3*no_params))
        try:
            plt.subplots_adjust(hspace=0.4)

            if no_params == 1:
                axes = [axes]

            for idi, param_idx in enumerate(parameters_to_disp):
                axis = axes[idi]
                lag_max = min(max_lag, autocorr_func_mean.shape[0])
                lags = np.arange(lag_max)
                acf = autocorr_func_mean[:lag_max, param_idx]

                axis.plot(lags, acf, 'b-', linewidth=2)
                axis.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                axis.axhline(y=0.05, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
                axis.axhline(y=-0.05, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
                axis.fill_between(lags, -0.05, 0.05, alpha=0.2, color='red')

                if par_names:
                    param_name = par_names[param_idx]
                else:
                    param_name = f"Parameter {param_idx}"
                axis.set_title(f"Autocorrelation: {param_name}")
                axis.set_xlabel("Lag")
                axis.set_ylabel("ACF")
                axis.grid(True, alpha=0.3)
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise

        return fig, axes

    def plot_acceptance_rates(self, stages_to_disp: Iterable | None = None):
        """
        Plot acceptance rates per stage.

        The counters come from ``summary``, i.e. they are whole-stage totals over ALL
        chains; this plot therefore takes no ``chains_to_disp`` (WS9b, finding P5: a
        parameter that could not be honoured is not offered).

        Args:
            stages_to_disp (list of int): stages to display

        Returns:
            Figure: the caller owns it and must close it.
        """
        stages_to_disp = self._resolve_stages(stages_to_disp)

        stage_names = self._get_stage_names(stages_to_disp)
        summary_subset = self.summary.iloc[stages_to_disp]

        # Calculate acceptance rates
        total = summary_subset['sum'].values
        accepted = summary_subset['accepted'].values
        acceptance_rates = (accepted / total) * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            x_pos = np.arange(len(stage_names))
            bars = ax.bar(x_pos, acceptance_rates, color='steelblue', alpha=0.8, edgecolor='navy')

            # Add value labels on bars
            for bar, rate in zip(bars, acceptance_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.axhline(y=23.4, color='green', linestyle='--', linewidth=2, label='Optimal MH (~23.4%)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(stage_names, rotation=45, ha='right')
            ax.set_ylabel('Acceptance Rate (%)', fontsize=12)
            ax.set_title('Acceptance Rates by Stage', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise

        return fig

    #: columns of ``adaptive_stats`` that are not adapted proposal parameters
    ADAPTATION_NON_PARAMETER_COLUMNS = ("n", "m", "mean_acceptance_probability", "rank_world")

    def plot_adaptation(self, stage: int | str, chains_to_disp: Iterable | None = None,
                        target_rate: float | None = None):
        """
        Per-period trace of an adaptive proposal: the acceptance probability it was fed and
        every adapted parameter it logged, one line per chain (2026-09-21).

        Reads ``self.adaptive_stats[stage]`` (``adaptive_stats/<stage>/rank%04d.csv``, one row
        per completed adaptation period, see ``docs/outputs.md``). The first panel is
        ``mean_acceptance_probability`` with ``target_rate`` as a dashed line when given; the
        remaining panels are the proposal's own columns (``log_sigma``, ``trace_C_over_d``,
        ``shrinkage_delta`` for the adaptive random walk; ``beta`` for adaptive pCN;
        ``log_step_size``, ``log_step_size_bar`` for the Hamiltonian family). The x axis is the
        proposal's own counter, ``n`` (``adapt()`` calls) or ``m`` (sub-chain steps). Columns are
        plotted as logged -- ``log_sigma`` stays on the log scale, nothing is transformed.

        Args:
            stage: stage index or stage name (resolved by ``_resolve_stages``).
            chains_to_disp: chain positions to include, in the same sense as everywhere else in
                the report (the i-th ``rank%04d`` file of the stage; the files are named by
                ``rank_world`` and sorted, so position i is the i-th smallest ``rank_world`` in
                the trace). ``None`` = every chain that wrote a file. A stage with
                ``save_to_file=False`` writes no trace and cannot be shown.
            target_rate: acceptance rate the adaptation drives towards, drawn as a reference
                line in the acceptance panel; ``None`` draws none.

        Returns:
            ``(fig, axes)``; ``(None, None)`` when the stage has no adaptation trace (a
            non-adaptive stage, a stage with ``save_to_file=False``, or ``chains_to_disp``
            excluding every chain). The caller owns the figure and must close it.
        """
        stage_idx = self._resolve_stages([stage])[0]
        stats = self.adaptive_stats[stage_idx]
        if stats is None or stats.empty:
            return None, None
        stats = stats.copy()
        if "rank_world" not in stats.columns:
            stats["rank_world"] = 0
        ranks = sorted(int(r) for r in stats["rank_world"].unique())
        chain_of_rank = {rank: position for position, rank in enumerate(ranks)}
        if chains_to_disp is not None:
            wanted = {ranks[int(c)] for c in chains_to_disp if 0 <= int(c) < len(ranks)}
            stats = stats[stats["rank_world"].isin(wanted)]
            if stats.empty:
                return None, None
        counter = next((c for c in ("n", "m") if c in stats.columns), None)
        parameter_columns = [c for c in stats.columns if c not in self.ADAPTATION_NON_PARAMETER_COLUMNS]
        panels = (["mean_acceptance_probability"] if "mean_acceptance_probability" in stats.columns else []) \
            + parameter_columns
        if not panels:
            return None, None

        stage_label = self.stage_names[stage_idx]
        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.6 * len(panels) + 0.8),
                                 sharex=True, squeeze=False)
        axes = axes[:, 0]
        try:
            for ax, column in zip(axes, panels):
                for rank, group in stats.groupby("rank_world", sort=True):
                    x = group[counter].to_numpy() if counter else np.arange(1, len(group) + 1)
                    ax.plot(x, group[column].to_numpy(), marker=".", markersize=3, linewidth=1,
                            label=f"chain {chain_of_rank[int(rank)]}")
                if column == "mean_acceptance_probability":
                    ax.set_ylabel("acceptance prob.\n(period mean)")
                    ax.set_ylim(-0.02, 1.02)
                    if target_rate is not None:
                        ax.axhline(target_rate, color="green", linestyle="--", linewidth=1.5,
                                   label=f"target {target_rate:g}")
                else:
                    ax.set_ylabel(column)
                ax.grid(True, alpha=0.3)
            axes[0].set_title(f"Proposal adaptation -- stage {stage_label}", fontsize=13, fontweight="bold")
            axes[0].legend(loc="best", fontsize=8, ncol=2)
            axes[-1].set_xlabel(f"{counter} (adaptation counter)" if counter else "adaptation period")
            fig.tight_layout()
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise
        return fig, axes

    def plot_parameter_correlation_heatmap(self, stages_to_disp: Iterable | None = None,
                                           par_names: List[str] | None = None,
                                           burn_in: List[List[int]] | None = None,
                                           chains_to_disp: Iterable | None = None):
        """
        Plot parameter correlation matrix heatmap for selected stages.

        Args:
            stages_to_disp (list of int): stages to include (None = all)
            par_names (list of str): parameter names
            burn_in (list of list of int): burn-in per stage/chain
            chains_to_disp (list of int): chains to include (None = every chain)

        Returns:
            tuple(Figure, Axes, ndarray): heatmap figure, axis, and correlation matrix;
            axis and matrix are None if there were fewer than two samples.
        """
        all_x = self._collect_samples_matrix(stages_to_disp=stages_to_disp, burn_in=burn_in,
                                             chains_to_disp=chains_to_disp)
        if all_x.shape[0] < 2:
            fig = self._placeholder_figure("Correlation heatmap unavailable: not enough samples.")
            return fig, None, None

        corr = np.corrcoef(all_x, rowvar=False)
        fig, ax = plt.subplots(figsize=(7, 6))
        try:
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Correlation", rotation=270, labelpad=15)

            labels = []
            for i in range(self.no_parameters):
                if par_names and i < len(par_names):
                    labels.append(par_names[i])
                else:
                    labels.append(f"Parameter {i}")

            ticks = np.arange(self.no_parameters)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            ax.set_title("Parameter Correlation Heatmap")

            if self.no_parameters <= 10:
                for i in range(self.no_parameters):
                    for j in range(self.no_parameters):
                        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

            fig.tight_layout()
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise
        return fig, ax, corr

    def plot_best_fits(self, best_obs: np.ndarray, best_scores: np.ndarray,
                       observations: np.ndarray | None = None,
                       obs_grid: np.ndarray | None = None,
                       n_sensors: int | None = None,
                       obs_label: str = "Observations",
                       score_label: str = "L2"):
        """
        Plot the best simulated observation series against the reference observations.
        """
        best_obs = np.asarray(best_obs, dtype=float)
        if best_obs.ndim == 1:
            best_obs = best_obs.reshape((1, -1))
        if best_obs.ndim != 2 or best_obs.shape[0] == 0:
            fig = self._placeholder_figure("Best-fit plot unavailable: no best-fit observations provided.")
            return fig, None

        best_scores = np.asarray(best_scores, dtype=float).reshape(-1)
        if best_scores.size != best_obs.shape[0]:
            raise ValueError("best_scores length must match the number of best-fit observation series.")

        n_best, no_observations = best_obs.shape
        if observations is not None:
            observations = np.asarray(observations, dtype=float).reshape(-1)
            if observations.size != no_observations:
                raise ValueError(
                    f"observations has length {observations.size}, expected {no_observations}."
                )

        colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.95, n_best))

        if n_sensors is not None and n_sensors > 0 and no_observations % n_sensors == 0:
            n_times = no_observations // n_sensors
            if obs_grid is None:
                obs_grid = np.arange(n_times)
            obs_grid = np.asarray(obs_grid)
            if obs_grid.size != n_times:
                raise ValueError(
                    f"obs_grid has length {obs_grid.size}, expected {n_times} for {n_sensors} sensors."
                )

            n_cols = 2 if n_sensors > 1 else 1
            n_rows = int(np.ceil(n_sensors / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.5 * n_rows), sharex=False, sharey=False)
            try:
                axes = np.atleast_1d(axes).reshape(-1)
                for sensor_idx in range(n_sensors):
                    axis = axes[sensor_idx]
                    sensor_slice = slice(sensor_idx * n_times, (sensor_idx + 1) * n_times)
                    if observations is not None:
                        axis.plot(
                            obs_grid,
                            observations[sensor_slice],
                            "k-o",
                            linewidth=2,
                            markersize=3,
                            label=obs_label,
                        )
                    for best_idx in range(n_best):
                        label = f"Best {best_idx + 1} ({score_label}={best_scores[best_idx]:.2f})"
                        axis.plot(
                            obs_grid,
                            best_obs[best_idx, sensor_slice],
                            color=colors[best_idx],
                            linewidth=1.6,
                            alpha=0.85,
                            label=label,
                        )
                    axis.set_title(f"Sensor {sensor_idx + 1}")
                    axis.set_xlabel("Observation grid")
                    axis.set_ylabel("Value")
                    axis.grid(True, alpha=0.3)

                for axis in axes[n_sensors:]:
                    axis.axis("off")
                axes[0].legend(loc="best", fontsize=8)
                fig.suptitle("Best-fit observation trajectories", fontsize=14, fontweight="bold")
                fig.tight_layout()
            except BaseException:
                plt.close(fig)  # P8: a failed plot must not leak its figure
                raise
            return fig, axes

        if obs_grid is None:
            obs_grid = np.arange(no_observations)
        obs_grid = np.asarray(obs_grid)
        if obs_grid.size != no_observations:
            raise ValueError(
                f"obs_grid has length {obs_grid.size}, expected {no_observations}."
            )

        fig, axis = plt.subplots(figsize=(12, 5))
        try:
            if observations is not None:
                axis.plot(obs_grid, observations, "k-o", linewidth=2, markersize=3, label=obs_label)
            for best_idx in range(n_best):
                axis.plot(
                    obs_grid,
                    best_obs[best_idx],
                    color=colors[best_idx],
                    linewidth=1.6,
                    alpha=0.85,
                    label=f"Best {best_idx + 1} ({score_label}={best_scores[best_idx]:.2f})",
                )
            axis.set_title("Best-fit observation trajectories")
            axis.set_xlabel("Observation index")
            axis.set_ylabel("Value")
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best", fontsize=8)
            fig.tight_layout()
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise
        return fig, axis

    def plot_posterior_field_statistics(self, mean_field: np.ndarray, std_field: np.ndarray,
                                        coordinates: np.ndarray | None = None,
                                        field_name: str = "Field"):
        """
        Plot posterior mean and standard deviation for a derived field.
        """
        mean_field = np.asarray(mean_field, dtype=float)
        std_field = np.asarray(std_field, dtype=float)
        if mean_field.shape != std_field.shape:
            raise ValueError("mean_field and std_field must have the same shape.")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        try:
            fields = [mean_field, std_field]
            titles = [f"{field_name} mean", f"{field_name} std"]
            cmaps = ["viridis", "magma"]

            if coordinates is not None:
                coordinates = np.asarray(coordinates, dtype=float)
                if coordinates.ndim == 2 and coordinates.shape[0] == mean_field.size and coordinates.shape[1] >= 2:
                    triangulation = None
                    if coordinates.shape[0] >= 3:
                        try:
                            triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
                        except (RuntimeError, ValueError) as exc:
                            # degenerate point set (collinear/duplicate points): fall back
                            # to a scatter plot instead of a filled contour
                            print(f"triangulation of {field_name!r} failed, using scatter: {exc}",
                                  flush=True)
                            triangulation = None

                    for axis, field_values, title, cmap in zip(axes, fields, titles, cmaps):
                        field_values = field_values.reshape(-1)
                        if triangulation is not None:
                            artist = axis.tricontourf(triangulation, field_values, levels=20, cmap=cmap)
                        else:
                            artist = axis.scatter(
                                coordinates[:, 0], coordinates[:, 1], c=field_values, cmap=cmap, s=12
                            )
                        fig.colorbar(artist, ax=axis)
                        axis.set_title(title)
                        axis.set_xlabel("x")
                        axis.set_ylabel("y")
                        axis.set_aspect("equal")
                    fig.tight_layout()
                    return fig, axes

            for axis, field_values, title, cmap in zip(axes, fields, titles, cmaps):
                if field_values.ndim == 2:
                    artist = axis.imshow(field_values, origin="lower", aspect="auto", cmap=cmap)
                    fig.colorbar(artist, ax=axis)
                else:
                    axis.plot(np.asarray(field_values).reshape(-1), color="tab:blue")
                    axis.grid(True, alpha=0.3)
                axis.set_title(title)
            fig.tight_layout()
        except BaseException:
            plt.close(fig)  # P8: a failed plot must not leak its figure
            raise
        return fig, axes


def add_normal_dist_grid(axes, mean: List[float], sd: List[float],
                         no_sigmas_to_show: int = 3, color: str = "red") -> None:
    """
    Adds visualizations of normal distribution to a N x N grid.
    Components are independent.
    In 1d visualized as pdf from mu-no_sigmas_to_show*sigma to mu+2*no_sigmas_to_show*sigma.
    In 2d visualized as mean, 1 sd, 2 sd, etc.

    Args:
        axes: N x N grid
        mean (list of float of length N): means of the normal distributions
        sd (list of float of length N): standard deviations of the normal distributions
        no_sigmas_to_show (int): number of sigmas to show
        color (str): color of all plots
    """
    N = len(mean)
    for i in range(N):
        mu_i = mean[i]
        sigma_i = sd[i]
        for j in range(N):
            mu_j = mean[j]
            sigma_j = sd[j]
            axis = axes[j, i]
            if i == j:  # pdf (2 sigmas)
                x = np.linspace(
                    mu_i - no_sigmas_to_show * sigma_i,
                    mu_i + no_sigmas_to_show * sigma_i,
                    no_sigmas_to_show * 50)
                pdf = norm.pdf(x, mu_i, sigma_i)
                axis.plot(x, pdf, color=color)
            else:  # mean, 1 sd, 2 sd
                for k in range(no_sigmas_to_show):
                    no_points = 100 * (k + 1)
                    theta = np.linspace(0, 2 * np.pi, no_points)
                    X = mu_i + (k + 1) * sigma_i * np.cos(theta)
                    Y = mu_j + (k + 1) * sigma_j * np.sin(theta)
                    axis.plot(X, Y, '-', color=color)  # plot ellipse (1 sigma)
                axis.plot(mu_i, mu_j, 'o', color=color, markersize=5, label='mean')  # plot mean as a circle
