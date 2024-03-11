#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from os import listdir
import os
from typing import List, Literal, Tuple
from scipy.stats import norm


class StageSamples:
    def __init__(self, no_parameters, samples_dir: str, decompress_samples: bool, load_posterior: bool, load_posterior_surrogate: bool):
        filenames = [f for f in os.listdir(samples_dir) if os.path.isfile(os.path.join(samples_dir, f))]
        filenames.sort()
        self.no_chains = len(filenames)
        self.samples_compressed = [None] * self.no_chains
        self.weights = [None] * self.no_chains
        if decompress_samples:
            self.samples = [None] * self.no_chains
        if load_posterior:
            self.posterior = [None] * self.no_chains
        if load_posterior_surrogate:
            self.posterior_surrogate = [None] * self.no_chains
        self.no_unique_samples = [0] * self.no_chains
        for i in range(self.no_chains):
            file_path = os.path.join(samples_dir, filenames[i])
            try:
                df_samples = pd.read_csv(file_path, header=None)
            except pd.errors.EmptyDataError:
                print(file_path + "EMPTY")
                continue
            self.weights[i] = np.array(df_samples[0])
            self.samples_compressed[i] = np.array(df_samples.iloc[:, 1:1+no_parameters])
            if decompress_samples:
                self.samples[i] = decompress(self.samples_compressed[i], self.weights[i])
            if load_posterior:
                self.posterior[i] = np.array(df_samples.iloc[:, 1+no_parameters])
            if load_posterior_surrogate:
                self.posterior_surrogate[i] = np.array(df_samples.iloc[:, 2+no_parameters])


class Samples:
    def __init__(self, no_parameters: int, samples_dir: str,
                 decompress_samples: bool = True, load_posterior: bool = False, load_posterior_surrogate: bool = False):
        # TODO: samples organized in folders by stages
        self.no_parameters = no_parameters
        self.sampling_output_dir = os.path.join(samples_dir, "sampling_output")
        self.samples_dir = os.path.join(samples_dir, "sampling_output", "samples")
        self.stage_names = [f for f in os.listdir(self.samples_dir) if not os.path.isfile(os.path.join(self.samples_dir, f))]
        self.stage_names.sort()
        self.no_stages = len(self.stage_names)
        self.list_of_stages = []
        for stage in self.stage_names:
            stage_path = os.path.join(self.samples_dir, stage)
            self.list_of_stages.append(StageSamples(no_parameters, stage_path, decompress_samples, load_posterior, load_posterior_surrogate))

    def _plot_hist_1d(self, axis, burn_in=None, param_no=0, stages_to_disp=None, bins=20, show=True, scale="linear"):
        # use weighted data (compressed)
        all_x = np.zeros((0,))
        no_unique_samples_sum = 0
        for idj, j in enumerate(stages_to_disp):
            for i in range(self.list_of_stages[j].no_chains):
                no_unique_samples_sum += self.list_of_stages[j].no_unique_samples[i]  # burn-in not excluded
                try:
                    tmp = self.list_of_stages[j].samples[i][burn_in[idj][i]:, param_no]
                except:
                    print("HISTOGRAM 1D: CHAIN", self.stage_names[j], i, "NOT AVAILABLE")
                    continue
                if scale == "log":
                    all_x = np.concatenate((all_x, np.log10(tmp)))
                elif scale == "ln":
                    all_x = np.concatenate((all_x, np.log(tmp)))
                else:
                    all_x = np.concatenate((all_x, tmp))
        if bins is None:
            bins = np.floor(no_unique_samples_sum/20)
            bins = min(bins, 100)
            bins = max(bins, 10)
        axis.hist(all_x, bins=int(bins), density=True)
        axis.grid(True)
        if show:
            plt.show()

    def _plot_hist_2d(self, axis, burn_in=None, param_no=[0, 1], stages_to_disp=None, bins=20, show=True, colorbar=False, scale=[False, False]):
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
                except:
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
            bins = np.floor(np.sqrt(no_unique_samples_sum/4))
            bins = min(bins, 100)
            bins = max(bins, 10)
        axis.hist2d(all_x, all_y, bins=int(bins), cmap="binary")  # , density = True)
        axis.grid(True)
        if colorbar:
            axis.colorbar()
        if show:
            plt.show()

    def plot_hist_grid(self, bins1d: int = 20, bins2d: int = 20, parameters_to_disp: List[int] | None = None,
                       stages_to_disp: List[int] | None = None, scale: List[Literal["linear", "log", "ln"]] | None = None,
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

        fig, axes = plt.subplots(no_parameters_to_disp, no_parameters_to_disp, sharex=False, sharey=False, figsize=(15, 15))
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        for idi, i in enumerate(parameters_to_disp):
            for idj, j in enumerate(parameters_to_disp):
                axis = axes[idi, idj]
                if idi == idj:
                    self._plot_hist_1d(axis=axis, param_no=i, burn_in=burn_in, stages_to_disp=stages_to_disp, bins=bins1d, show=False, scale=scale[i])
                else:
                    self._plot_hist_2d(axis=axis, param_no=[j, i], burn_in=burn_in, stages_to_disp=stages_to_disp,
                                       bins=bins2d, show=False, scale=[scale[j], scale[i]])
                if idi == 0:
                    # determine parameter name
                    if par_names is not None:
                        par_name = par_names[j].replace('_', '\_')
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
                          grid_interp: np.ndarray | None = None, bins: List[int] | None = None, chains_to_disp: List[int] | None = None,
                          stages_to_disp: List[int] | None = None, observations: np.ndarray | None = None, cmap="viridis_r"):
        """
        Creates 2d histogram of observations.
        Observations can be loaded only if save_raw_data==True.
        Suitable for observations in the form of time series.

        Args:
            no_observattions (int): number of observations
            chosen_observations (ndarray of int of length N): indices forming the time series (otherwise all are used)
            grid (ndarray of float of length N): time values for the time series (otherwise range(N) is used)
            grid_interp (ndarray of float): time grid for horizontal axis (otherwise grid_inter = grid)
            bins (list of int of length 2): [bins_x, bins_y] (optional)
            chains_disp (list of int of length N): chains that should be included (otherwise all chains are included)
            stages_disp (list of int): stages that should be included (otherwise all stages are included)
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
                    G_values = np.array(df_samples.iloc[:, 2+self.no_parameters+chosen_observations])
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

        plt.figure(figsize=(6, 6))
        n_samples = G_all.shape[0]
        if bins is None:
            nbins = (5*1.2*np.sqrt(n_samples)).astype(int)
            bins = [len_grid, nbins]
        G_all = G_all.flatten()
        min_G = min(G_all)
        max_G = max(G_all)
        range_G = max_G - min_G
        hist_range = [[min(grid_interp), max(grid_interp)], [min_G-range_G/10, max_G+range_G/10]]
        output = plt.hist2d(x_all.flatten(), G_all, bins=bins, range=hist_range, weights=weights_all.flatten(),
                            cmap=cmap)  # , vmin=1, vmax=n_samples/10)
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
        plt.plot(grid, observations[chosen_observations], label="observations", linewidth=1)
        plt.legend()


def add_normal_dist_grid(axes, mean: List[float], sd: List[float], no_sigmas_to_show: int = 3, color: str = "red") -> None:
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
                x = np.linspace(mu_i-no_sigmas_to_show*sigma_i, mu_i+no_sigmas_to_show*sigma_i, no_sigmas_to_show*50)
                pdf = norm.pdf(x, mu_i, sigma_i)
                axis.plot(x, pdf, color=color)
            else:  # mean, 1 sd, 2 sd
                for k in range(no_sigmas_to_show):
                    no_points = 100*(k+1)
                    theta = np.linspace(0, 2*np.pi, no_points)
                    X = mu_i + (k+1)*sigma_i * np.cos(theta)
                    Y = mu_j + (k+1)*sigma_j * np.sin(theta)
                    axis.plot(X, Y, '-', color=color)  # plot ellipse (1 sigma)
                axis.plot(mu_i, mu_j, 'o', color=color, markersize=5, label='mean')  # plot mean as a circle


def decompress(samples_compressed, weights):
    # samples_compressed ... "compressed" samples from DAMH-SMU
    # weights ... counts of consecutive identical samples
    sum_w = np.sum(weights)
    cumsum_w = np.append(0, np.cumsum(weights))
    no_unique, no_parameters = samples_compressed.shape
    samples_decompressed = np.zeros((sum_w, no_parameters))
    for i in range(no_unique):
        samples_decompressed[cumsum_w[i]:cumsum_w[i+1], :] = samples_compressed[i, :]
    return samples_decompressed
