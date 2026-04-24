#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Iterable, List, Literal

import emcee
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


class StageSamples:
    def __init__(self, no_parameters, samples_dir: str, decompress_samples: bool,
                 load_posterior: bool, load_posterior_surrogate: bool):
        filenames = [f for f in os.listdir(samples_dir) if os.path.isfile(os.path.join(samples_dir, f))]
        filenames.sort()
        self.no_chains = len(filenames)
        self.samples_compressed: list[Any] = [None] * self.no_chains
        self.weights: list[Any] = [None] * self.no_chains
        self.length: list[Any] = [None] * self.no_chains
        if decompress_samples:
            self.samples: list[Any] = [None] * self.no_chains
        if load_posterior:
            self.posterior: list[Any] = [None] * self.no_chains
        if load_posterior_surrogate:
            self.posterior_surrogate: list[Any] = [None] * self.no_chains
        self.no_unique_samples = [0] * self.no_chains
        for i in range(self.no_chains):
            file_path = os.path.join(samples_dir, filenames[i])
            try:
                df_samples = pd.read_csv(file_path, header=None)
            except pd.errors.EmptyDataError:
                print(file_path + "EMPTY")
                continue
            self.weights[i] = np.array(df_samples[0])
            self.length[i] = sum(self.weights[i])
            self.samples_compressed[i] = np.array(df_samples.iloc[:, 1:1 + no_parameters])
            if decompress_samples:
                self.samples[i] = decompress(self.samples_compressed[i], self.weights[i])
            if load_posterior:
                self.posterior[i] = np.array(df_samples.iloc[:, 1 + no_parameters])
            if load_posterior_surrogate:
                self.posterior_surrogate[i] = np.array(df_samples.iloc[:, 2 + no_parameters])


class Samples:
    def __init__(self, no_parameters: int, samples_dir: str,
                 decompress_samples: bool = True, load_posterior: bool = False, load_posterior_surrogate: bool = False):
        self.no_parameters = no_parameters
        self.sampling_output_dir = os.path.join(samples_dir, "sampling_output")
        self.samples_dir = os.path.join(samples_dir, "sampling_output", "samples")
        self.stage_names = [
            f for f in os.listdir(
                self.samples_dir) if not os.path.isfile(
                os.path.join(
                    self.samples_dir, f))]
        self.stage_names.sort()
        self.no_stages = len(self.stage_names)
        self.list_of_stages = []
        for stage in self.stage_names:
            stage_path = os.path.join(self.samples_dir, stage)
            self.list_of_stages.append(
                StageSamples(
                    no_parameters,
                    stage_path,
                    decompress_samples,
                    load_posterior,
                    load_posterior_surrogate))
        self.summarize()

    def get_mean_and_cov(self, stages_to_disp: Iterable | None = None, burn_in: List[List[int]] | None = None,
                         npy_filepath: str | None = None):
        if stages_to_disp is None:
            stages_to_disp = range(self.no_stages)
        if burn_in is None:
            burn_in = []
            for i in stages_to_disp:
                burn_in.append([0] * self.list_of_stages[i].no_chains)
        all_x = np.zeros((0, self.no_parameters))
        all_w = np.zeros((0, ))
        for idj, j in enumerate(stages_to_disp):
            for s in range(self.list_of_stages[j].no_chains):
                try:
                    tmp_x = self.list_of_stages[j].samples_compressed[s][burn_in[idj][s]:, :]
                    tmp_w = self.list_of_stages[j].weights[s][burn_in[idj][s]:]
                except BaseException:
                    print("CHAIN", self.stage_names[j], s, "NOT AVAILABLE")
                    continue
                all_x = np.concatenate((all_x, tmp_x))
                all_w = np.concatenate((all_w, tmp_w))
        mean = np.average(all_x, axis=0, weights=all_w)
        cov = np.cov(all_x, rowvar=False, fweights=all_w)
        if npy_filepath:
            # save to .npy file:
            np.save(npy_filepath, [mean, cov])
            print("Mean and covariance saved to", npy_filepath, flush=True)
        return mean, cov

    def load_notes(self):
        self.notes = [pd.DataFrame()] * self.no_stages
        for n, stage_name in enumerate(self.stage_names):
            dirname = os.path.join(self.sampling_output_dir, "notes", stage_name)
            files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
            files.sort()
            no_samplers = len(files)
            for i in range(no_samplers):
                filepath = os.path.join(dirname, files[i])
                data = pd.read_csv(filepath)
                self.notes[n] = pd.concat([self.notes[n], data])

    def summarize(self):
        self.load_notes()
        # create pandas data frame containing sums of dataframes in self.notes:
        summary = pd.DataFrame()
        for notes in self.notes:
            summary = pd.concat([summary, notes.iloc[:, :-1].sum()], axis=1)
        # transpose data frame:
        summary = summary.T
        # name the rows with self.stage_names:
        summary = summary.set_axis(labels=self.stage_names, axis=0)
        self.summary = summary

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

    @staticmethod
    def _placeholder_figure(message: str):
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        return fig

    def get_summary(self, csv_filepath: str | None = None):
        print(self.summary)
        if csv_filepath:
            self.summary.to_csv(csv_filepath, index=True)
        return self.summary

    def calculate_CpUS(self, list_of_stages_groups: List[List[int]], surrogate_cost_ratio: float = 0.0):
        """
        Calculates the cost per uncorrelated sample (CpUS)
        for all groups of stages. Returns extended summary.
        """
        ratio_evaluated = (self.summary["accepted"] + self.summary["rejected"]) / self.summary["sum"]
        # add column to summary:
        self.summary["ratio_evaluated"] = ratio_evaluated
        autocorr_stages = np.zeros((self.no_stages,))
        cpus_stages = -np.ones((self.no_stages,), dtype=float)
        for stages_to_disp in list_of_stages_groups:
            print("Stages:", stages_to_disp)
            try:
                autocorr = Autocorrelation(self, stages_to_disp=stages_to_disp)
                autocorr.calculate_autocorr_function()
                a, _ = autocorr.calculate_autocorr_time_mean()
                a = np.mean(a)
                autocorr_stages[stages_to_disp] = a
                ratio_loc = self.summary["ratio_evaluated"].iloc[stages_to_disp]
                cpus_stages[stages_to_disp] = a * (ratio_loc + surrogate_cost_ratio)
            except Exception as e:
                print(f"CpUS unavailable for stages {stages_to_disp}: {e}")
        # add columns to summary:
        self.summary["autocorr"] = autocorr_stages
        self.summary["CpUS"] = cpus_stages
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
        if chains_to_disp is None:
            chains_to_disp = range(self.list_of_stages[0].no_chains)
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
            for i in chains_to_disp:
                path_samples = os.path.join(dirname, files[i])
                df_samples = pd.read_csv(path_samples, header=None)
                types = df_samples.iloc[:, 0]
                idx = np.ones(len(types), dtype=bool)
                idx[types == "prerejected"] = 0
                idx[types == "rejected"] = 1
                temp = np.arange(len(types))
                weights = temp[idx]
                weights[1:] = weights[1:] - weights[:-1]
                if sum(idx) == 0:
                    print("EMPTY - chain:", i, "stage", stage_name, flush=True)
                else:
                    G_values = np.array(df_samples.iloc[:, 2 + self.no_parameters + chosen_observations])
                    G_values = G_values[idx]
                    G_all = np.vstack((G_all, G_values))
                    param = np.array(df_samples.iloc[:, 1:self.no_parameters + 1])
                    param = param[idx]
                    par_all = np.vstack((par_all, param))
                    weights = weights.reshape((-1, 1))
                    weights_all = np.vstack((weights_all, weights))
            print("loaded - stage", stage_name, flush=True)
        return par_all, G_all, weights_all

    def plot_chains(self, average=False, parameters_to_disp: Iterable | None = None,
                    stages_to_disp: Iterable | None = None, scale: List[Literal["linear", "log", "ln"]] | None = None,
                    par_names: List[str] | None = None, burn_in: List[List[int]] | None = None):
        """
        Plot generated chains.
        If average==True, serves to analyze the convergence af averages for several chains generated in parallel.

        Args:
            parameters_to_disp (list of int of length N): Parameters to display. If None, all parameters are displayed.
            stages_to_disp (list of int of length S): Stages to display. If None, all stages are displayed.
            scale (list of "linear", "log" of length N): Scale of the plots. If None, all set to "linear".
            par_names (list of str of length N): Parameter names. If None, "par. 0", "par. 1", ... is used.
            burn_in (list of list of int): Burn-in for each stage and each chain. If None, set to [[0] * no_chains] * S.

        Returns:
            tuple[Figure, array of Axes]
        """
        if parameters_to_disp is None:
            parameters_to_disp = range(self.no_parameters)
        no_parameters_to_disp = len(parameters_to_disp)
        if stages_to_disp is None:
            stages_to_disp = range(self.no_stages)
        if scale is None:
            scale = ["linear"] * no_parameters_to_disp
        if burn_in is None:
            burn_in = []
            for i in stages_to_disp:
                burn_in.append([0] * self.list_of_stages[i].no_chains)

        fig, axes = plt.subplots(no_parameters_to_disp, 1, sharex=False, sharey=False, figsize=(15, 15))
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        for idi, i in enumerate(parameters_to_disp):
            axis = axes[idi]
            all_x = []
            for _ in range(self.list_of_stages[0].no_chains):  # assumes equal number of chains in each stage
                all_x.append(np.zeros((0,)))
            for idj, j in enumerate(stages_to_disp):
                for s in range(self.list_of_stages[j].no_chains):
                    try:
                        tmp = self.list_of_stages[j].samples[s][burn_in[idj][s]:, i]
                    except BaseException:
                        print("HISTOGRAM 1D: CHAIN", self.stage_names[j], s, "NOT AVAILABLE")
                        continue
                    all_x[s] = np.concatenate((all_x[s], tmp))
            for chain in all_x:
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
        return fig, axes

    def _plot_hist_1d(self, axis, burn_in: List[List[int]], param_no: int,
                      stages_to_disp: List[int], bins: int, show: bool, scale=str,
                      prior_component=None):
        # use weighted data (compressed)
        all_x = np.zeros((0,))
        no_unique_samples_sum = 0
        for idj, j in enumerate(stages_to_disp):
            for i in range(self.list_of_stages[j].no_chains):
                no_unique_samples_sum += self.list_of_stages[j].no_unique_samples[i]  # burn-in not excluded
                try:
                    tmp = self.list_of_stages[j].samples[i][burn_in[idj][i]:, param_no]
                except BaseException:
                    print("HISTOGRAM 1D: CHAIN", self.stage_names[j], i, "NOT AVAILABLE")
                    continue
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
        # Overlay marginal prior PDF if provided
        if prior_component is not None and len(all_x) > 0:
            x_lo, x_hi = all_x.min(), all_x.max()
            margin = 0.15 * (x_hi - x_lo) if x_hi > x_lo else 1.0
            x_grid = np.linspace(x_lo - margin, x_hi + margin, 300)
            try:
                y_grid = prior_component.pdf(x_grid)
                axis.plot(x_grid, y_grid, 'r-', linewidth=2, label='prior')
            except Exception:
                pass
        axis.grid(True)
        if show:
            plt.show()

    def _plot_hist_2d(self, axis, burn_in: List[List[int]], param_no: List[int], stages_to_disp: List[int],
                      bins: int, show: bool, scale: List[str], colorbar: bool = False):
        # use weighted data
        all_x = np.zeros((0,))
        all_y = np.zeros((0,))
        no_unique_samples_sum = 0
        for idj, j in enumerate(stages_to_disp):
            for i in range(self.list_of_stages[j].no_chains):
                no_unique_samples_sum += self.list_of_stages[j].no_unique_samples[i]  # burn-in not excluded
                try:
                    tmp_x = self.list_of_stages[j].samples[i][burn_in[idj][i]:, param_no[0]]
                    tmp_y = self.list_of_stages[j].samples[i][burn_in[idj][i]:, param_no[1]]
                except BaseException:
                    print("HISTOGRAM 2D: CHAIN", self.stage_names[j], i, "NOT AVAILABLE")
                    continue
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
                       prior=None):
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

        Returns:
            tuple[Figure, array of Axes]
        """
        if parameters_to_disp is None:
            parameters_to_disp = range(self.no_parameters)
        no_parameters_to_disp = len(parameters_to_disp)
        if stages_to_disp is None:
            stages_to_disp = range(self.no_stages)
        if scale is None:
            scale = ["linear"] * no_parameters_to_disp
        if burn_in is None:
            burn_in = []
            for i in stages_to_disp:
                burn_in.append([0] * self.list_of_stages[i].no_chains)

        fig, axes = plt.subplots(no_parameters_to_disp, no_parameters_to_disp,
                                 sharex=False, sharey=False, figsize=(15, 15))
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
                        prior_component=pc)
                else:
                    self._plot_hist_2d(axis=axis, param_no=[j, i], burn_in=burn_in, stages_to_disp=stages_to_disp,
                                       bins=bins2d, show=False, scale=[scale[idj], scale[idi]])
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
            grid_interp (ndarray of float): time grid for horizontal axis (otherwise grid_inter = grid)
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
        if chains_to_disp is None:
            chains_to_disp = range(self.list_of_stages[0].no_chains)
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
            for i in chains_to_disp:
                if i >= len(files):
                    print("hist_G FILE NOT AVAILABLE - chain:", i, "stage", stage_name)
                    continue
                path_samples = os.path.join(dirname, files[i])
                df_samples = pd.read_csv(path_samples, header=None)
                types = df_samples.iloc[:, 0]
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
                    G_values = np.array(df_samples.iloc[:, 2 + self.no_parameters + chosen_observations])
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
        n_samples = G_all.shape[0]
        if bins is None:
            nbins = (5 * 1.2 * np.sqrt(n_samples)).astype(int)
            bins = [len_grid, nbins]
        G_all = G_all.flatten()
        min_G = min(G_all)
        max_G = max(G_all)
        range_G = max_G - min_G
        hist_range = [[min(grid_interp), max(grid_interp)], [min_G - range_G / 10, max_G + range_G / 10]]
        if len_grid == 1:
            output = plt.hist(G_all, bins=bins[-1], weights=weights_all.flatten())
        else:
            output = plt.hist2d(x_all.flatten(), G_all, bins=bins, range=hist_range, weights=weights_all.flatten(), cmap=cmap)  # , vmin=1, vmax=n_samples/10)
            plt.colorbar(output[3])
        plt.grid()

        if observations is not None:
            if len_grid == 1:
                plt.plot(observations[chosen_observations], 0, 'ro', label="observation",)
            else:
                plt.plot(grid, observations[chosen_observations], 'r', label="observations", linewidth=1)
            plt.legend()
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
        csv_path = os.path.join(self.sampling_output_dir, "surrogate_quality.csv")
        if not os.path.isfile(csv_path):
            fig = self._placeholder_figure(
                "Surrogate quality plot unavailable:\n"
                "No surrogate_quality.csv file found.\n"
                "This file is generated only when a surrogate model (collector) is used."
            )
            return fig, None

        df = pd.read_csv(csv_path)
        if df.empty:
            fig = self._placeholder_figure("Surrogate quality plot unavailable: CSV file is empty.")
            return fig, None

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.subplots_adjust(hspace=0.15)

        x = df["snapshots_total"].values

        # RMSE plot
        ax_rmse = axes[0]
        ax_rmse.plot(x, df["rmse"].values, 'b-o', markersize=4, linewidth=1.5, label="RMSE")
        ax_rmse.set_ylabel("RMSE", fontsize=12)
        ax_rmse.set_title("Surrogate Model Quality", fontsize=14, fontweight="bold")
        ax_rmse.set_yscale("log")
        ax_rmse.grid(True, alpha=0.3)
        ax_rmse.legend(fontsize=10)

        # Max absolute error plot
        ax_max = axes[1]
        ax_max.plot(x, df["max_abs_error"].values, 'r-o', markersize=4, linewidth=1.5, label="Max Absolute Error")
        ax_max.set_ylabel("Max Absolute Error", fontsize=12)
        ax_max.set_xlabel("Total Snapshots Collected", fontsize=12)
        ax_max.set_yscale("log")
        ax_max.grid(True, alpha=0.3)
        ax_max.legend(fontsize=10)

        return fig, axes

    def pdf_report(self, no_observations: int, chosen_observations: np.ndarray | None = None, grid: np.ndarray | None = None,
                   grid_interp: np.ndarray | None = None, bins: List[int] | None = None, chains_to_disp: Iterable | None = None,
                   stages_to_disp: List[int] | None = None, observations: np.ndarray | None = None, cmap="viridis_r"):
        """
        Creates a report in pdf format containing all of the above post-processing tools, 
        e.g. summary table, histograms of observations, chain traces.

        Args:
            no_observations (int): number of observations
            chosen_observations (ndarray of int of length N): indices forming the time series (otherwise all are used)
            grid (ndarray of float of length N): time values for the time series (otherwise range(N) is used)
            grid_interp (ndarray of float): time grid for horizontal axis (otherwise grid_inter = grid)
            bins (list of int of length 2): [bins_x, bins_y] (optional)
            chains_to_disp (list of int of length N): chains that should be included (otherwise all chains are included)
            stages_to_disp (list of int): stages that should be included (otherwise all stages are included)
            observations (ndarray of shape (no_observations,)): vector of observations (optional)
        """
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages('report.pdf') as pdf:
            # summary table:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=self.summary.values, colLabels=self.summary.columns, rowLabels=self.summary.index, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.2)
            pdf.savefig(fig)
            plt.close()

            # histograms of observations:
            fig = self.hist_observations(no_observations=no_observations, chosen_observations=chosen_observations,
                                         grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                         stages_to_disp=stages_to_disp, observations=observations, cmap=cmap)
            pdf.savefig(fig)
            plt.close()

            # chain traces:
            fig, _ = self.plot_chains(average=False, parameters_to_disp=None, stages_to_disp=stages_to_disp,
                                      scale=None, par_names=None, burn_in=None)
            pdf.savefig(fig)
            plt.close()

    def html_report(self, no_observations: int, chosen_observations: np.ndarray | None = None, grid: np.ndarray | None = None,
                    grid_interp: np.ndarray | None = None, bins: List[int] | None = None, chains_to_disp: Iterable | None = None,
                    stages_to_disp: List[int] | None = None, observations: np.ndarray | None = None, cmap="viridis_r"):
        """
        Creates a report in html format containing all of the above post-processing tools, 
        e.g. summary table, histograms of observations, chain traces.

        Args:
            no_observations (int): number of observations
            chosen_observations (ndarray of int of length N): indices forming the time series (otherwise all are used)
            grid (ndarray of float of length N): time values for the time series (otherwise range(N) is used)
            grid_interp (ndarray of float): time grid for horizontal axis (otherwise grid_inter = grid)
            bins (list of int of length 2): [bins_x, bins_y] (optional)
            chains_to_disp (list of int of length N): chains that should be included (otherwise all chains are included)
            stages_to_disp (list of int): stages that should be included (otherwise all stages are included)
            observations (ndarray of shape (no_observations,)): vector of observations (optional)
        """
        # summary table:
        html = self.summary.to_html()
        with open("report.html", "w") as f:
            f.write(html)
        # histograms of observations:
        fig = self.hist_observations(no_observations=no_observations, chosen_observations=chosen_observations, grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                     stages_to_disp=stages_to_disp, observations=observations, cmap=cmap)
        fig.savefig("hist_observations.png")
        plt.close()
        # chain traces:
        fig, _ = self.plot_chains(average=False, parameters_to_disp=None, stages_to_disp=stages_to_disp, scale=None, par_names=None, burn_in=None)
        fig.savefig("chain_traces.png")
        plt.close()
        # add images to html:
        with open("report.html", "a") as f:
            f.write("<h2>Histograms of observations</h2>")
            f.write('<img src="hist_observations.png" alt="hist_observations">')
            f.write("<h2>Chain traces</h2>")
            f.write('<img src="chain_traces.png" alt="chain_traces">')
        # surrogate quality:
        fig, axes = self.plot_surrogate_quality()
        if axes is not None:
            fig.savefig("surrogate_quality.png")
            plt.close(fig)
            with open("report.html", "a") as f:
                f.write("<h2>Surrogate Model Quality</h2>")
                f.write('<img src="surrogate_quality.png" alt="surrogate_quality">')
        else:
            plt.close(fig)  

    def calculate_effective_sample_size(self, stages_to_disp: Iterable | None = None, burn_in: List[List[int]] | None = None):
        """
        Calculate effective sample size (ESS) for each parameter.
        ESS = total_samples / autocorrelation_time
        
        Args:
            stages_to_disp (list of int): stages to include (None = all)
            burn_in (list of list of int): burn-in per stage/chain
            
        Returns:
            dict with ESS per parameter and overall statistics
        """
        if stages_to_disp is None:
            stages_to_disp = range(self.no_stages)
        if burn_in is None:
            burn_in = []
            for i in stages_to_disp:
                burn_in.append([0] * self.list_of_stages[i].no_chains)
        
        try:
            autocorr = Autocorrelation(self, stages_to_disp=stages_to_disp, burn_in=burn_in)
            autocorr.calculate_autocorr_function()
            autocorr_times, _ = autocorr.calculate_autocorr_time_mean()
            
            # Calculate total samples
            total_samples = 0
            for idj, j in enumerate(stages_to_disp):
                for s in range(self.list_of_stages[j].no_chains):
                    try:
                        tmp = self.list_of_stages[j].samples[s][burn_in[idj][s]:, 0]
                        total_samples += len(tmp)
                    except:
                        pass
            
            ess_per_param = {i: total_samples / float(autocorr_times[i]) for i in range(self.no_parameters)}
            ess_overall = np.mean(list(ess_per_param.values()))
            
            return {
                'ess_per_param': ess_per_param,
                'ess_overall': ess_overall,
                'total_samples': total_samples,
                'autocorr_times': autocorr_times
            }
        except Exception as e:
            print(f"Error calculating ESS: {e}")
            return None

    def plot_autocorr(self, stages_to_disp: Iterable | None = None, parameters_to_disp: Iterable | None = None,
                     par_names: List[str] | None = None, max_lag: int = 100):
        """
        Plot autocorrelation functions for parameters.
        
        Args:
            stages_to_disp (list of int): stages to include
            parameters_to_disp (list of int): parameters to display
            par_names (list of str): parameter names
            max_lag (int): maximum lag to display
            
        Returns:
            tuple(Figure, Axes)
        """
        if stages_to_disp is None:
            stages_to_disp = range(self.no_stages)
        if parameters_to_disp is None:
            parameters_to_disp = range(self.no_parameters)
        
        try:
            autocorr = Autocorrelation(self, stages_to_disp=stages_to_disp)
            autocorr.calculate_autocorr_function()
            autocorr_func_mean = autocorr.calculate_autocorr_function_mean()
            
            no_params = len(parameters_to_disp)
            fig, axes = plt.subplots(no_params, 1, figsize=(12, 3*no_params))
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
            
            return fig, axes
        except Exception as e:
            print(f"Error plotting autocorrelation: {e}")
            fig = self._placeholder_figure(f"Autocorrelation plot unavailable: {str(e)}")
            return fig, None

    def plot_acceptance_rates(self, stages_to_disp: Iterable | None = None):
        """
        Plot acceptance rates per stage.
        
        Args:
            stages_to_disp (list of int): stages to display
            
        Returns:
            Figure
        """
        if stages_to_disp is None:
            stages_to_disp = range(self.no_stages)
        
        stage_names = self._get_stage_names(stages_to_disp)
        summary_subset = self.summary.iloc[list(stages_to_disp)]
        
        # Calculate acceptance rates
        total = summary_subset['sum'].values
        accepted = summary_subset['accepted'].values
        acceptance_rates = (accepted / total) * 100
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x_pos = np.arange(len(stage_names))
        bars = ax.bar(x_pos, acceptance_rates, color='steelblue', alpha=0.8, edgecolor='navy')
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, acceptance_rates)):
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
        
        return fig

    def _collect_samples_matrix(self, stages_to_disp: Iterable | None = None,
                                burn_in: List[List[int]] | None = None):
        if stages_to_disp is None:
            stages_to_disp = range(self.no_stages)
        stages_to_disp = list(stages_to_disp)
        if burn_in is None:
            burn_in = []
            for i in stages_to_disp:
                burn_in.append([0] * self.list_of_stages[i].no_chains)

        all_x = np.zeros((0, self.no_parameters))
        for idj, j in enumerate(stages_to_disp):
            for s in range(self.list_of_stages[j].no_chains):
                try:
                    tmp_x = self.list_of_stages[j].samples[s][burn_in[idj][s]:, :]
                except BaseException:
                    continue
                all_x = np.concatenate((all_x, tmp_x))
        return all_x

    def calculate_gelman_rubin(self, stages_to_disp: Iterable | None = None,
                               burn_in: List[List[int]] | None = None):
        """
        Calculate Gelman-Rubin potential scale reduction factor (R-hat)
        for each parameter using parallel chains.

        Returns:
            dict with per-parameter and aggregate R-hat statistics, or None if unavailable.
        """
        if stages_to_disp is None:
            stages_to_disp = range(self.no_stages)
        stages_to_disp = list(stages_to_disp)
        if burn_in is None:
            burn_in = []
            for i in stages_to_disp:
                burn_in.append([0] * self.list_of_stages[i].no_chains)

        no_chains = self.list_of_stages[stages_to_disp[0]].no_chains
        chain_data = []
        for chain_idx in range(no_chains):
            chain_samples = np.zeros((0, self.no_parameters))
            for idj, stage_idx in enumerate(stages_to_disp):
                try:
                    tmp = self.list_of_stages[stage_idx].samples[chain_idx][burn_in[idj][chain_idx]:, :]
                except BaseException:
                    continue
                chain_samples = np.concatenate((chain_samples, tmp))
            if chain_samples.shape[0] >= 2:
                chain_data.append(chain_samples)

        if len(chain_data) < 2:
            return None

        n = min(chain.shape[0] for chain in chain_data)
        if n < 2:
            return None

        samples = np.stack([chain[:n, :] for chain in chain_data], axis=0)
        m = samples.shape[0]

        chain_means = np.mean(samples, axis=1)
        grand_mean = np.mean(chain_means, axis=0)
        B = (n / (m - 1)) * np.sum((chain_means - grand_mean) ** 2, axis=0)

        chain_vars = np.var(samples, axis=1, ddof=1)
        W = np.mean(chain_vars, axis=0)
        W_safe = np.maximum(W, 1e-15)

        var_hat = ((n - 1) / n) * W + B / n
        rhat = np.sqrt(var_hat / W_safe)

        return {
            "rhat_per_param": {i: float(rhat[i]) for i in range(self.no_parameters)},
            "rhat_max": float(np.max(rhat)),
            "rhat_mean": float(np.mean(rhat)),
            "n_per_chain": int(n),
            "m_chains": int(m)
        }

    def plot_parameter_correlation_heatmap(self, stages_to_disp: Iterable | None = None,
                                           par_names: List[str] | None = None,
                                           burn_in: List[List[int]] | None = None):
        """
        Plot parameter correlation matrix heatmap for selected stages.

        Returns:
            tuple(Figure, Axes, ndarray): heatmap figure, axis, and correlation matrix.
        """
        all_x = self._collect_samples_matrix(stages_to_disp=stages_to_disp, burn_in=burn_in)
        if all_x.shape[0] < 2:
            fig = self._placeholder_figure("Correlation heatmap unavailable: not enough samples.")
            return fig, None, None

        corr = np.corrcoef(all_x, rowvar=False)
        fig, ax = plt.subplots(figsize=(7, 6))
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

        for i in range(self.no_parameters):
            for j in range(self.no_parameters):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

        fig.tight_layout()
        return fig, ax, corr

    def html_report_extended(self, no_observations: int = 0, chosen_observations: np.ndarray | None = None, 
                            grid: np.ndarray | None = None, grid_interp: np.ndarray | None = None, 
                            bins: List[int] | None = None, chains_to_disp: Iterable | None = None,
                            stages_to_disp: List[int] | None = None, observations: np.ndarray | None = None, 
                            cmap="viridis_r", output_file: str = "report_extended.html",
                            bins1d: int = 20, bins2d: int = 20, par_names: List[str] | None = None,
                            prior=None):
        """
        Creates an extended report in HTML format containing all available post-processing tools,
        including visualizations and statistics for combined stages and individual stages separately.

        Args:
            no_observations (int): number of observations (default: 0, set to >0 if observations are available)
            chosen_observations (ndarray of int of length N): indices forming the time series (otherwise all are used)
            grid (ndarray of float of length N): time values for the time series (otherwise range(N) is used)
            grid_interp (ndarray of float): time grid for horizontal axis (otherwise grid_inter = grid)
            bins (list of int of length 2): [bins_x, bins_y] for observation histograms (optional)
            chains_to_disp (list of int of length N): chains that should be included (otherwise all chains are included)
            stages_to_disp (list of int): stages that should be included (otherwise all stages are included)
            observations (ndarray of shape (no_observations,)): vector of observations (optional)
            cmap (str): colormap for observation histograms (default: "viridis_r")
            output_file (str): name of the output HTML file (default: "report_extended.html")
            bins1d (int): Number of bins in 1d histograms (default: 20)
            bins2d (int): Number of bins in 2d histograms (default: 20)
            par_names (list of str of length N): Parameter names for plots (optional)
            prior (PriorIndependentComponents or None): Prior distribution. If provided, marginal prior PDFs
                are overlaid on 1D histograms in the histogram grids.
        """
        import base64
        from io import BytesIO
        
        # Initialize stages to display
        if stages_to_disp is None:
            stages_to_disp = list(range(self.no_stages))
        observation_data_available = self._raw_data_available(stages_to_disp=stages_to_disp)
        
        # Helper function to convert matplotlib figure to base64 image
        def fig_to_base64(fig):
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_base64
        
        # Start building HTML
        html_parts = []
        html_parts.append('<!DOCTYPE html>')
        html_parts.append('<html lang="en">')
        html_parts.append('<head>')
        html_parts.append('    <meta charset="UTF-8">')
        html_parts.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append('    <title>Extended Sampling Report</title>')
        html_parts.append('    <style>')
        html_parts.append('        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }')
        html_parts.append('        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }')
        html_parts.append('        h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 40px; }')
        html_parts.append('        h3 { color: #7f8c8d; margin-top: 30px; }')
        html_parts.append('        .section { background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }')
        html_parts.append('        .description { color: #555; margin: 10px 0; line-height: 1.6; font-style: italic; }')
        html_parts.append('        table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
        html_parts.append('        th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }')
        html_parts.append('        th { background-color: #3498db; color: white; font-weight: bold; }')
        html_parts.append('        tr:nth-child(even) { background-color: #f9f9f9; }')
        html_parts.append('        img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }')
        html_parts.append('        .stats-table { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 15px 0; }')
        html_parts.append('        .toc { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }')
        html_parts.append('        .toc ul { list-style-type: none; padding-left: 20px; }')
        html_parts.append('        .toc a { color: #2980b9; text-decoration: none; }')
        html_parts.append('        .toc a:hover { text-decoration: underline; }')
        html_parts.append('        pre { background-color: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; }')
        html_parts.append('    </style>')
        html_parts.append('</head>')
        html_parts.append('<body>')
        
        # Title
        html_parts.append('    <h1>Extended Sampling Report</h1>')
        html_parts.append('    <p class="description">This report contains comprehensive post-processing results including summary statistics, ')
        html_parts.append('    visualizations, and detailed analysis for all stages of the sampling process.</p>')
        
        # Table of Contents
        html_parts.append('    <div class="toc">')
        html_parts.append('        <h2>Table of Contents</h2>')
        html_parts.append('        <ul>')
        html_parts.append('            <li><a href="#summary">1. Summary Statistics</a></li>')
        html_parts.append('            <li><a href="#overall">2. Overall Analysis (Combined Stages)</a></li>')
        html_parts.append('            <li><a href="#individual">3. Individual Stage Analysis</a></li>')
        html_parts.append('            <li><a href="#diagnostics">4. Convergence Diagnostics & Autocorrelation</a></li>')
        html_parts.append('            <li><a href="#surrogate_quality">5. Surrogate Model Quality</a></li>')
        if no_observations > 0 and observation_data_available:
            html_parts.append('            <li><a href="#observations">6. Observation Histograms</a></li>')
        html_parts.append('        </ul>')
        html_parts.append('    </div>')
        
        # 1. SUMMARY STATISTICS
        html_parts.append('    <div class="section" id="summary">')
        html_parts.append('        <h2>1. Summary Statistics</h2>')
        html_parts.append('        <p class="description">This table summarizes the acceptance and rejection rates for all sampling stages. ')
        html_parts.append('        "Accepted" samples were accepted by the Metropolis-Hastings criterion, "rejected" samples were rejected, ')
        html_parts.append('        and "pre-rejected" samples (if any) were rejected by a surrogate model before evaluation.</p>')
        html_parts.append(self.summary.to_html(classes='summary-table'))
        html_parts.append('    </div>')
        
        # 2. OVERALL ANALYSIS (COMBINED STAGES)
        html_parts.append('    <div class="section" id="overall">')
        html_parts.append('        <h2>2. Overall Analysis (Combined Stages)</h2>')
        html_parts.append('        <p class="description">This section presents aggregated results from all selected sampling stages combined.</p>')
        
        # 2.1 Mean and Covariance
        html_parts.append('        <h3>2.1 Posterior Mean and Covariance Matrix</h3>')
        html_parts.append('        <p class="description">The posterior mean represents the expected value of each parameter, ')
        html_parts.append('        while the covariance matrix shows the variance and correlation structure among parameters.</p>')
        posterior_mean, cov = self.get_mean_and_cov(stages_to_disp=stages_to_disp)
        
        html_parts.append('        <div class="stats-table">')
        html_parts.append('            <h4>Posterior Mean:</h4>')
        html_parts.append('            <pre>')
        for i, val in enumerate(posterior_mean):
            param_name = par_names[i] if par_names and i < len(par_names) else f"Parameter {i}"
            html_parts.append(f'{param_name}: {val:.6f}')
        html_parts.append('            </pre>')
        html_parts.append('            <h4>Covariance Matrix:</h4>')
        html_parts.append('            <pre>')
        html_parts.append(np.array2string(cov, precision=6, suppress_small=True))
        html_parts.append('            </pre>')
        html_parts.append('        </div>')
        
        # 2.2 Histogram Grid
        html_parts.append('        <h3>2.2 Parameter Distribution Histograms</h3>')
        html_parts.append('        <p class="description">This grid shows 1D histograms (diagonal) and 2D joint histograms (off-diagonal) ')
        html_parts.append('        for all parameter combinations. 1D histograms show marginal distributions, while 2D histograms ')
        html_parts.append('        reveal correlations between parameter pairs.</p>')
        fig, _ = self.plot_hist_grid(bins1d=bins1d, bins2d=bins2d, stages_to_disp=stages_to_disp, par_names=par_names, prior=prior)
        img_base64 = fig_to_base64(fig)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Overall Histogram Grid">')
        
        # 2.3 Chain Traces
        html_parts.append('        <h3>2.3 Chain Traces</h3>')
        html_parts.append('        <p class="description">Chain traces show the evolution of parameter values throughout the sampling process. ')
        html_parts.append('        Multiple chains are overlaid to assess mixing and convergence. Good mixing shows chains exploring ')
        html_parts.append('        the parameter space uniformly without getting stuck.</p>')
        fig, _ = self.plot_chains(average=False, stages_to_disp=stages_to_disp, par_names=par_names)
        img_base64 = fig_to_base64(fig)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Overall Chain Traces">')
        
        # 2.4 Cumulative Averages
        html_parts.append('        <h3>2.4 Cumulative Averages</h3>')
        html_parts.append('        <p class="description">Cumulative averages show the running mean of each parameter over the sampling iterations. ')
        html_parts.append('        Convergence is indicated when these curves stabilize and become flat, suggesting that the chains ')
        html_parts.append('        have reached the stationary distribution.</p>')
        fig, _ = self.plot_chains(average=True, stages_to_disp=stages_to_disp, par_names=par_names)
        img_base64 = fig_to_base64(fig)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Overall Cumulative Averages">')
        
        html_parts.append('    </div>')
        
        # 3. INDIVIDUAL STAGE ANALYSIS
        html_parts.append('    <div class="section" id="individual">')
        html_parts.append('        <h2>3. Individual Stage Analysis</h2>')
        html_parts.append('        <p class="description">This section presents detailed analysis for each sampling stage separately. ')
        html_parts.append('        Different stages may use different proposal distributions or algorithm parameters.</p>')
        
        for stage_idx in stages_to_disp:
            stage_name = self.stage_names[stage_idx]
            html_parts.append(f'        <h3>Stage: {stage_name}</h3>')
            html_parts.append(f'        <p class="description">Analysis results for sampling stage "{stage_name}".</p>')
            
            # Stage-specific summary
            html_parts.append('        <h4>Stage Summary:</h4>')
            stage_summary = self.summary.iloc[stage_idx:stage_idx+1]
            html_parts.append(stage_summary.to_html(classes='stage-summary'))
            
            # Mean and Covariance for this stage
            html_parts.append('        <h4>Posterior Mean and Covariance (This Stage):</h4>')
            stage_mean, stage_cov = self.get_mean_and_cov(stages_to_disp=[stage_idx])
            html_parts.append('        <div class="stats-table">')
            html_parts.append('            <strong>Mean:</strong>')
            html_parts.append('            <pre>')
            for i, val in enumerate(stage_mean):
                param_name = par_names[i] if par_names and i < len(par_names) else f"Parameter {i}"
                html_parts.append(f'{param_name}: {val:.6f}')
            html_parts.append('            </pre>')
            html_parts.append('            <strong>Covariance:</strong>')
            html_parts.append('            <pre>')
            html_parts.append(np.array2string(stage_cov, precision=6, suppress_small=True))
            html_parts.append('            </pre>')
            html_parts.append('        </div>')
            
            # Histogram Grid for this stage
            html_parts.append('        <h4>Parameter Distribution Histograms:</h4>')
            html_parts.append('        <p class="description">1D and 2D histograms for this stage only.</p>')
            fig, _ = self.plot_hist_grid(bins1d=bins1d, bins2d=bins2d, stages_to_disp=[stage_idx], par_names=par_names, prior=prior)
            img_base64 = fig_to_base64(fig)
            html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Stage {stage_name} Histogram Grid">')
            
            # Chain Traces for this stage
            html_parts.append('        <h4>Chain Traces:</h4>')
            html_parts.append('        <p class="description">Evolution of parameter values during this stage.</p>')
            fig, _ = self.plot_chains(average=False, stages_to_disp=[stage_idx], par_names=par_names)
            img_base64 = fig_to_base64(fig)
            html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Stage {stage_name} Chain Traces">')
            
            # Cumulative Averages for this stage
            html_parts.append('        <h4>Cumulative Averages:</h4>')
            html_parts.append('        <p class="description">Running mean showing convergence behavior for this stage.</p>')
            fig, _ = self.plot_chains(average=True, stages_to_disp=[stage_idx], par_names=par_names)
            img_base64 = fig_to_base64(fig)
            html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Stage {stage_name} Cumulative Averages">')
        
        html_parts.append('    </div>')
        
        # 4. CONVERGENCE DIAGNOSTICS & AUTOCORRELATION
        html_parts.append('    <div class="section" id="diagnostics">')
        html_parts.append('        <h2>4. Convergence Diagnostics & Autocorrelation Analysis</h2>')
        html_parts.append('        <p class="description">Diagnostic measures to assess mixing quality, convergence, and sampling efficiency. ')
        html_parts.append('        Lower autocorrelation times and higher effective sample sizes indicate better sampling efficiency.</p>')
        
        # 4.1 Acceptance Rates
        html_parts.append('        <h3>4.1 Acceptance Rates</h3>')
        html_parts.append('        <p class="description">Acceptance rate visualized by stage. The optimal acceptance rate for Metropolis-Hastings is ~23.4%. ')
        html_parts.append('        Rates significantly lower (< 10%) or higher (> 50%) suggest issues with proposal distribution.</p>')
        fig_acc = self.plot_acceptance_rates(stages_to_disp=stages_to_disp)
        img_base64 = fig_to_base64(fig_acc)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Acceptance Rates">')
        
        # 4.2 Autocorrelation Analysis
        html_parts.append('        <h3>4.2 Autocorrelation Functions</h3>')
        html_parts.append('        <p class="description">Autocorrelation functions (ACF) show how correlated samples are at different lags. ')
        html_parts.append('        Rapid decay to near-zero indicates good mixing. The red shaded region (±0.05) represents ')
        html_parts.append('        the approximate 95% confidence interval under independence.</p>')
        try:
            fig_acf, _ = self.plot_autocorr(stages_to_disp=stages_to_disp, parameters_to_disp=range(self.no_parameters), par_names=par_names, max_lag=200)
            img_base64 = fig_to_base64(fig_acf)
            html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Autocorrelation Functions">')
        except Exception as e:
            html_parts.append(f'        <p class="description" style="color: orange;">Autocorrelation plot unavailable: {str(e)}</p>')
        
        # 4.3 Effective Sample Size
        html_parts.append('        <h3>4.3 Effective Sample Size (ESS)</h3>')
        html_parts.append('        <p class="description">ESS represents the equivalent number of independent samples drawn from the posterior. ')
        html_parts.append('        ESS = Total Samples / Autocorrelation Time. Higher ESS relative to total samples indicates efficient sampling.</p>')
        ess_results = self.calculate_effective_sample_size(stages_to_disp=stages_to_disp)
        if ess_results:
            html_parts.append('        <div class="stats-table">')
            html_parts.append('            <h4>Effective Sample Size Summary:</h4>')
            html_parts.append('            <pre>')
            html_parts.append(f'Total samples (all chains, all stages): {ess_results["total_samples"]:,}\n')
            html_parts.append(f'Overall ESS: {ess_results["ess_overall"]:.1f}\n')
            html_parts.append(f'ESS efficiency: {(ess_results["ess_overall"]/max(1,ess_results["total_samples"])*100):.2f}%\n\n')
            html_parts.append('ESS per parameter:\n')
            for param_idx, ess in sorted(ess_results['ess_per_param'].items()):
                param_name = par_names[param_idx] if par_names and param_idx < len(par_names) else f"Parameter {param_idx}"
                autocorr_time = ess_results['autocorr_times'][param_idx]
                html_parts.append(f'  {param_name}: ESS = {ess:.1f}, Autocorr. Time = {autocorr_time:.2f}\n')
            html_parts.append('            </pre>')
            html_parts.append('        </div>')
        else:
            html_parts.append('        <p class="description" style="color: orange;">ESS calculation not available for this run.</p>')

        # 4.4 Cost per Uncorrelated Sample (CpUS)
        html_parts.append('        <h3>4.4 Cost per Uncorrelated Sample (CpUS)</h3>')
        html_parts.append('        <p class="description">CpUS combines autocorrelation with evaluation cost. ') 
        html_parts.append('        Lower CpUS indicates a more efficient stage. Values are computed per stage using stage-wise autocorrelation.</p>')
        try:
            cpus_summary = self.calculate_CpUS(list_of_stages_groups=[[i] for i in stages_to_disp], surrogate_cost_ratio=0.0)
            cpus_df = cpus_summary.iloc[stages_to_disp][["accepted", "rejected", "pre-rejected", "sum", "ratio_evaluated", "autocorr", "CpUS"]].copy()
            cpus_df = cpus_df.rename(columns={"ratio_evaluated": "ratio_evaluated_exact", "autocorr": "autocorr_time"})
            html_parts.append(cpus_df.to_html(classes='summary-table', float_format=lambda x: f"{x:.4f}"))
        except Exception as e:
            html_parts.append(f'        <p class="description" style="color: orange;">CpUS calculation unavailable: {str(e)}</p>')
        
        # 4.5 Per-Stage Autocorrelation Analysis
        html_parts.append('        <h3>4.5 Per-Stage Autocorrelation Analysis</h3>')
        html_parts.append('        <p class="description">Detailed autocorrelation analysis for each sampling stage separately. ')
        html_parts.append('        This allows assessment of mixing quality at different stages of the algorithm.</p>')
        
        for stage_idx in stages_to_disp:
            stage_name = self.stage_names[stage_idx]
            html_parts.append(f'        <h4>Stage: {stage_name}</h4>')
            
            # ACF plot for this stage
            html_parts.append('        <p class="description"><strong>Autocorrelation Function:</strong></p>')
            try:
                fig_stage_acf, _ = self.plot_autocorr(stages_to_disp=[stage_idx], parameters_to_disp=range(self.no_parameters), par_names=par_names, max_lag=200)
                img_base64 = fig_to_base64(fig_stage_acf)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="ACF {stage_name}">')
            except Exception as e:
                html_parts.append(f'        <p class="description" style="color: orange;">ACF plot unavailable for {stage_name}: {str(e)}</p>')
            
            # ESS for this stage
            html_parts.append('        <p class="description"><strong>Effective Sample Size:</strong></p>')
            try:
                stage_ess = self.calculate_effective_sample_size(stages_to_disp=[stage_idx])
                if stage_ess:
                    html_parts.append('        <div class="stats-table">')
                    html_parts.append('            <pre>')
                    html_parts.append(f'Stage samples: {stage_ess["total_samples"]:,}\n')
                    html_parts.append(f'Stage ESS: {stage_ess["ess_overall"]:.1f}\n')
                    html_parts.append(f'ESS efficiency: {(stage_ess["ess_overall"]/max(1,stage_ess["total_samples"])*100):.2f}%\n\n')
                    html_parts.append('ESS per parameter:\n')
                    for param_idx, ess in sorted(stage_ess['ess_per_param'].items()):
                        param_name = par_names[param_idx] if par_names and param_idx < len(par_names) else f"Parameter {param_idx}"
                        autocorr_time = stage_ess['autocorr_times'][param_idx]
                        html_parts.append(f'  {param_name}: ESS = {ess:.1f}, Autocorr. Time = {autocorr_time:.2f}\n')
                    html_parts.append('            </pre>')
                    html_parts.append('        </div>')
            except Exception as e:
                html_parts.append(f'        <p class="description" style="color: orange;">ESS calculation unavailable for {stage_name}: {str(e)}</p>')

        # 4.6 Gelman-Rubin (R-hat)
        html_parts.append('        <h3>4.6 Gelman-Rubin Convergence (R-hat)</h3>')
        html_parts.append('        <p class="description">R-hat compares within-chain and between-chain variance. ') 
        html_parts.append('        Values close to 1.0 indicate convergence; values above 1.05 suggest insufficient mixing.</p>')
        rhat_results = self.calculate_gelman_rubin(stages_to_disp=stages_to_disp)
        if rhat_results:
            html_parts.append('        <div class="stats-table">')
            html_parts.append('            <pre>')
            html_parts.append(f'Chains used: {rhat_results["m_chains"]}\n')
            html_parts.append(f'Samples per chain (truncated): {rhat_results["n_per_chain"]}\n')
            html_parts.append(f'Max R-hat: {rhat_results["rhat_max"]:.4f}\n')
            html_parts.append(f'Mean R-hat: {rhat_results["rhat_mean"]:.4f}\n\n')
            html_parts.append('R-hat per parameter:\n')
            for param_idx, rhat in sorted(rhat_results['rhat_per_param'].items()):
                param_name = par_names[param_idx] if par_names and param_idx < len(par_names) else f"Parameter {param_idx}"
                html_parts.append(f'  {param_name}: R-hat = {rhat:.4f}\n')
            html_parts.append('            </pre>')
            html_parts.append('        </div>')
        else:
            html_parts.append('        <p class="description" style="color: orange;">Gelman-Rubin analysis unavailable (requires at least two non-empty chains).</p>')

        # 4.7 Parameter correlation heatmap
        html_parts.append('        <h3>4.7 Parameter Correlation Heatmap</h3>')
        html_parts.append('        <p class="description">Pairwise linear correlations among posterior parameters. ') 
        html_parts.append('        Values near ±1 indicate strong dependency; values near 0 indicate weak linear relationship.</p>')
        fig_corr, _, corr_matrix = self.plot_parameter_correlation_heatmap(stages_to_disp=stages_to_disp, par_names=par_names)
        img_base64 = fig_to_base64(fig_corr)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Parameter Correlation Heatmap">')

        # 4.8 Diagnostics summary
        html_parts.append('        <h3>4.8 Diagnostics Summary</h3>')
        html_parts.append('        <p class="description">Compact pass/warn/fail style summary for quick interpretation of chain quality.</p>')
        acceptance_rates = (self.summary.iloc[stages_to_disp]['accepted'] / self.summary.iloc[stages_to_disp]['sum']) * 100.0
        acc_min = float(np.min(acceptance_rates))
        acc_max = float(np.max(acceptance_rates))

        if ess_results:
            ess_eff = float((ess_results['ess_overall'] / max(1, ess_results['total_samples'])) * 100.0)
        else:
            ess_eff = None

        if rhat_results:
            rhat_max = float(rhat_results['rhat_max'])
        else:
            rhat_max = None

        if corr_matrix is not None and corr_matrix.shape[0] > 1:
            corr_tmp = corr_matrix.copy()
            np.fill_diagonal(corr_tmp, 0.0)
            max_abs_corr = float(np.max(np.abs(corr_tmp)))
        else:
            max_abs_corr = None

        cpus_value = None
        if "CpUS" in self.summary.columns:
            cpus_values = self.summary.iloc[stages_to_disp]["CpUS"].values
            cpus_values = cpus_values[cpus_values >= 0]
            if cpus_values.size > 0:
                cpus_value = float(np.mean(cpus_values))

        def grade(label, value, good_cond, warn_cond, fmt):
            if value is None:
                return f"{label}: unavailable"
            if good_cond(value):
                status = "PASS"
            elif warn_cond(value):
                status = "WARN"
            else:
                status = "FAIL"
            return f"{label}: {fmt(value)} ({status})"

        html_parts.append('        <div class="stats-table">')
        html_parts.append('            <pre>')
        html_parts.append(grade("Acceptance rate range [%]", (acc_min, acc_max),
                               lambda v: v[0] >= 10.0 and v[1] <= 50.0,
                               lambda v: v[0] >= 5.0 and v[1] <= 70.0,
                               lambda v: f"{v[0]:.2f} .. {v[1]:.2f}") + '\n')
        html_parts.append(grade("ESS efficiency [%]", ess_eff,
                               lambda v: v >= 10.0,
                               lambda v: v >= 3.0,
                               lambda v: f"{v:.2f}") + '\n')
        html_parts.append(grade("Max R-hat", rhat_max,
                               lambda v: v <= 1.01,
                               lambda v: v <= 1.05,
                               lambda v: f"{v:.4f}") + '\n')
        html_parts.append(grade("Max |correlation|", max_abs_corr,
                               lambda v: v <= 0.8,
                               lambda v: v <= 0.95,
                               lambda v: f"{v:.3f}") + '\n')
        html_parts.append(grade("Mean CpUS", cpus_value,
                               lambda v: v <= 50.0,
                               lambda v: v <= 200.0,
                               lambda v: f"{v:.4f}") + '\n')
        html_parts.append('            </pre>')
        html_parts.append('        </div>')
        
        html_parts.append('    </div>')

        # 5. SURROGATE MODEL QUALITY
        html_parts.append('    <div class="section" id="surrogate_quality">')
        html_parts.append('        <h2>5. Surrogate Model Quality</h2>')
        html_parts.append('        <p class="description">Out-of-sample quality metrics of the surrogate model, measured on newly arrived snapshots ')
        html_parts.append('        before they are added to the training set. Decreasing error indicates a surrogate model that improves as more data is collected.</p>')
        fig_sq, axes_sq = self.plot_surrogate_quality()
        img_base64 = fig_to_base64(fig_sq)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Surrogate Model Quality">')
        html_parts.append('    </div>')
        
        # 6. OBSERVATION HISTOGRAMS (if available)
        if no_observations > 0 and observation_data_available:
            html_parts.append('    <div class="section" id="observations">')
            html_parts.append('        <h2>6. Observation Histograms</h2>')
            html_parts.append('        <p class="description">These histograms show the distribution of model outputs (observations) ')
            html_parts.append('        generated by the sampled parameters. If actual observations are provided, they are overlaid ')
            html_parts.append('        in red for comparison.</p>')
            
            # Overall observation histogram
            html_parts.append('        <h3>6.1 Combined Stages</h3>')
            fig = self.hist_observations(no_observations=no_observations, chosen_observations=chosen_observations,
                                        grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                        stages_to_disp=stages_to_disp, observations=observations, cmap=cmap)
            img_base64 = fig_to_base64(fig)
            html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Overall Observation Histogram">')
            
            # Per-stage observation histograms
            html_parts.append('        <h3>6.2 Individual Stages</h3>')
            for stage_idx in stages_to_disp:
                stage_name = self.stage_names[stage_idx]
                html_parts.append(f'        <h4>Stage: {stage_name}</h4>')
                fig = self.hist_observations(no_observations=no_observations, chosen_observations=chosen_observations,
                                            grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                            stages_to_disp=[stage_idx], observations=observations, cmap=cmap)
                img_base64 = fig_to_base64(fig)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Stage {stage_name} Observation Histogram">')
            
            html_parts.append('    </div>')
        elif no_observations > 0:
            html_parts.append('    <div class="section" id="observations">')
            html_parts.append('        <h2>6. Observation Histograms</h2>')
            html_parts.append('        <p class="description">Observation histograms are not available for this run because raw snapshots were not saved. ')
            html_parts.append('        Enable <code>save_snapshots_to_file=True</code> in configuration to include this section.</p>')
            html_parts.append('    </div>')
        
        # Footer
        html_parts.append('    <div class="section">')
        html_parts.append('        <p style="text-align: center; color: #7f8c8d; margin-top: 40px;">')
        html_parts.append('        Report generated by surrDAMH post-processing module')
        html_parts.append('        </p>')
        html_parts.append('    </div>')
        
        html_parts.append('</body>')
        html_parts.append('</html>')
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(html_parts))
        
        print(f"Extended HTML report saved to: {output_file}")
        return output_file


# AUTOCORRELATION:
# Autocorrelation analysis using emcee, Foreman-Mackey,
# adapted for the needs of the DAMH-SMU framework.
# Considers "N = no_chains" chains of different lengths (l_1, ..., l_N),
# each of the chains has "n = no_parameters" components.
# The samples in one chain form a numpy array of shape (l_i, n).
# All samples form a python list of length N.

class Autocorrelation:
    def __init__(self, samples: Samples, stages_to_disp: Iterable, burn_in: List[List[int]] | None = None):
        """
        Autocorrelation analysis for a subset of stages.
        Using emcee (Foreman-Mackey).

        Args:
            samples (Samples): Samples object.
            stages_to_disp (list of int of length S): Stages to display. If None, all stages are displayed.
            burn_in (list of list of int): Burn-in for each stage and each chain. If None, set to [[0] * no_chains] * S.
        """
        self.samples = samples
        self.stages_to_disp = stages_to_disp
        self.stages = [samples.list_of_stages[i] for i in self.stages_to_disp]
        self.no_chains = self.samples.list_of_stages[0].no_chains
        if burn_in is None:
            burn_in = []
            for _ in stages_to_disp:
                burn_in.append([0] * self.no_chains)
        self.burn_in = burn_in

        chains_range = range(self.no_chains)
        self.samples_all_stages = []
        for i in range(self.no_chains):
            self.samples_all_stages.append(np.zeros((0, self.samples.no_parameters)))
        for idx_stage, stage in enumerate(self.stages):
            begin = self.burn_in[idx_stage]
            end = stage.length

            for idx, i in enumerate(chains_range):
                x = stage.samples[i][begin[idx]:end[idx], :]
                self.samples_all_stages[idx] = np.concatenate((self.samples_all_stages[idx], x))

    def calculate_autocorr_function(self):
        self.autocorr_function = [None] * self.no_chains

        for i in range(self.no_chains):
            self.autocorr_function[i] = np.zeros(self.samples_all_stages[i].shape)
            for j in range(self.samples.no_parameters):
                self.autocorr_function[i][:, j] = emcee.autocorr.function_1d(self.samples_all_stages[i][:, j])
        self.length = [x.shape[0] for x in self.autocorr_function]
        print("Autocorr. functions calculated, shapes:", [i.shape for i in self.autocorr_function])

    def calculate_autocorr_function_mean(self):
        chains_range = range(self.no_chains)
        max_length = max(self.length)
        autocorr_function_mean = np.zeros((max_length, self.samples.no_parameters))
        for j in range(self.samples.no_parameters):
            tmp = np.zeros(max_length)
            count = np.zeros(max_length)
            for idx, i in enumerate(chains_range):
                tmp[:self.length[idx]] += self.autocorr_function[i][:, j]
                count[:self.length[idx]] += 1
            autocorr_function_mean[:, j] = tmp / count

        return autocorr_function_mean

    def calculate_autocorr_time(self, c=5, tol=50, quiet=True):
        autocorr_time = [None] * self.no_chains
        for i in range(self.no_chains):
            tmp = np.zeros((self.samples.no_parameters))
            for j in range(self.samples.no_parameters):
                tmp[j] = emcee.autocorr.integrated_time(self.samples_all_stages[i][:, j], c=c, tol=tol, quiet=quiet)
            autocorr_time[i] = tmp
        return autocorr_time

    def calculate_autocorr_time_mean(self, c: int = 5):
        autocorr_time_mean = [None] * self.samples.no_parameters
        autocorr_time_mean_beta = [None] * self.samples.no_parameters
        length = min(self.length)
        autocorr_function_mean = self.calculate_autocorr_function_mean()
        for j in range(self.samples.no_parameters):
            f = autocorr_function_mean[:length, j]
            autocorr_time_mean[j] = autocorr_FM(f, c)
            f = autocorr_function_mean[:, j]
            autocorr_time_mean_beta[j] = autocorr_FM(f, c)
        return autocorr_time_mean, autocorr_time_mean_beta


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


def auto_window(taus, c):
    # Automated windowing procedure following Sokal (1989)
    # from https://dfm.io/posts/autocorr/ Foreman-Mackey
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr_FM(f, c: int = 5):
    # from https://dfm.io/posts/autocorr/ Foreman-Mackey
    # first calculates all autocorr. functions, than averages them
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def decompress(samples_compressed, weights):
    # samples_compressed ... "compressed" samples from DAMH-SMU
    # weights ... counts of consecutive identical samples
    sum_w = np.sum(weights)
    cumsum_w = np.append(0, np.cumsum(weights))
    no_unique, no_parameters = samples_compressed.shape
    samples_decompressed = np.zeros((sum_w, no_parameters))
    for i in range(no_unique):
        samples_decompressed[cumsum_w[i]:cumsum_w[i + 1], :] = samples_compressed[i, :]
    return samples_decompressed
