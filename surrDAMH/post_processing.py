#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from os import listdir
import os
from typing import Any, Iterable, List, Literal

import emcee
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

    def get_summary(self, csv_filepath: str | None = None):
        print(self.summary)
        if csv_filepath:
            self.summary.to_csv(csv_filepath, index=True)
        return self.summary

    def calculate_CpUS(self, list_of_stages_groups, surrogate_cost_ratio: float = 0.0):
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
            autocorr = Autocorrelation(self, stages_to_disp=stages_to_disp)
            autocorr.calculate_autocorr_function()
            a, _ = autocorr.calculate_autocorr_time_mean()
            a = np.mean(a)
            autocorr_stages[stages_to_disp] = a
            ratio_loc = self.summary["ratio_evaluated"][stages_to_disp]
            cpus_stages[stages_to_disp] = a * (ratio_loc + surrogate_cost_ratio)
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
                label = "$par. {0}$".format(j)
            if scale[idi] == "log":
                label += "\n(log10)"
            elif scale[idi] == "ln":
                label += "\n(ln)"
            axis.set_title(label, x=1.05, multialignment='center')
            axis.grid(True)
        return fig, axes

    def _plot_hist_1d(self, axis, burn_in: List[List[int]], param_no: int,
                      stages_to_disp: List[int], bins: int, show: bool, scale=str):
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
        axis.hist2d(all_x, all_y, bins=int(bins), cmap="binary")  # , density = True)
        axis.grid(True)
        if colorbar:
            axis.colorbar()
        if show:
            plt.show()

    def plot_hist_grid(self, bins1d: int = 20, bins2d: int = 20, parameters_to_disp: Iterable | None = None,
                       stages_to_disp: Iterable | None = None, scale: List[Literal["linear", "log", "ln"]] | None = None,
                       par_names: List[str] | None = None, burn_in: List[List[int]] | None = None):
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
                    self._plot_hist_1d(
                        axis=axis,
                        param_no=i,
                        burn_in=burn_in,
                        stages_to_disp=stages_to_disp,
                        bins=bins1d,
                        show=False,
                        scale=scale[idi])
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
        if chosen_observations is None:
            chosen_observations = np.arange(no_observations, dtype=np.int32)
        if grid is None:
            grid = np.arange(len(chosen_observations), dtype=np.int32)
        if grid_interp is None:
            grid_interp = grid
        if chains_to_disp is None:
            chains_to_disp = range(self.list_of_stages[0].no_chains)
        if stages_to_disp is None:
            stage_names = self.stage_names
        else:
            stage_names = [self.stage_names[i] for i in stages_to_disp]

        len_grid = len(grid_interp)
        grid_interp = np.arange(len_grid)
        x_all = np.empty((0, len_grid))
        weights_all = np.empty((0, len_grid))
        G_all = np.empty((0, len_grid))
        # param_all = np.empty((0, self.no_parameters))
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
                    # param = np.array(df_samples.iloc[:, 1:self.no_parameters+1])
                    # param = param[idx]
                    # param_all = np.vstack((param_all, param))
                    weights = weights.reshape((-1, 1))
                    weights = np.repeat(weights, len_grid, 1)
                    x = np.repeat(grid_interp.reshape((1, -1)), no_accepted, 0)
                    x_all = np.vstack((x_all, x))
                    weights_all = np.vstack((weights_all, weights))

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
        # img = np.flipud(output[0].transpose())
        # img[img > 0] = 1
        # print(img.shape)
        # xx = output[1]
        # yy = output[2]
        # plt.figure()
        # plt.imshow(img, extent=[xx[0], xx[-1], yy[0], yy[-1]], aspect='auto', cmap="viridis_r")

        plt.grid()
        # lbl_fontsize = "large"
        # plt.xlabel("time [d]", fontsize=lbl_fontsize)
        # plt.ylabel("pressure head [m]", fontsize=lbl_fontsize)
        if observations is not None:
            if len_grid == 1:
                plt.plot(observations[chosen_observations], 0, 'ro', label="observation",)
            else:
                plt.plot(grid, observations[chosen_observations], 'r', label="observations", linewidth=1)
            plt.legend()
        return fig

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

    # def plot_autocorr_function(self, length_disp, plot_mean=False, parameters_disp=None,
    #                            chains_disp=None, show_legend=False):
    #     if parameters_disp is None:
    #         parameters_disp = range(self.no_parameters)
    #     no_parameters_disp = len(parameters_disp)
    #     if chains_disp is None:
    #         chains_disp = range(self.no_chains)
    #     fig, axes = plt.subplots(1, no_parameters_disp, figsize=(12, 3), sharey=True)
    #     for idj, j in enumerate(parameters_disp):
    #         length_disp[idj] = min(max(self.length), length_disp[idj])
    #     for idj, j in enumerate(parameters_disp):
    #         for idi, i in enumerate(chains_disp):
    #             axes[idj].plot(self.autocorr_function[i][:length_disp[idj], j], label=i)
    #         if plot_mean:
    #             axes[idj].plot(self.autocorr_function_mean[:, j], label="mean")
    #         axes[idj].set_xlim(0, length_disp[idj] - 1)
    #         if show_legend:
    #             axes[idj].legend(loc=1)
    #         axes[idj].set_xlabel("$par. {0}$".format(j))
    #         axes[idj].grid(True)
    #         if self.known_autocorr_time:
    #             axes[idj].set_title("$\\tau_\\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]))
    #     axes[0].set_ylabel("autocorr. function")
    #     plt.show()


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
