#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from os import listdir
import os


class Samples:
    def __init__(self, no_parameters: int, samples_dir: str,
                 decompress_samples: bool = True, load_posterior: bool = False, load_posterior_surrogate: bool = False):
        # TODO: samples organized in folders by stages
        self.no_parameters = no_parameters
        self.samples_dir = os.path.join(samples_dir, "saved_samples", "data")
        filenames = [f for f in os.listdir(self.samples_dir) if os.path.isfile(os.path.join(self.samples_dir, f))]
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
            file_path = os.path.join(self.samples_dir, filenames[i])
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

    def _plot_hist_1d(self, axis, burn_in=None, param_no=0, chains_disp=None, bins=20, show=True, log=False):
        # use weighted data (compressed)
        if chains_disp is None:
            chains_disp = range(self.no_chains)
        if burn_in is None:
            burn_in = [0] * len(chains_disp)
        all_x = np.zeros((0,))
        no_unique_samples_sum = 0
        for i, chain in enumerate(chains_disp):
            no_unique_samples_sum += self.no_unique_samples[chain]  # burn-in not excluded
            try:
                tmp = self.samples[chain][burn_in[i]:, param_no]
            except:
                print("HISTOGRAM 1D: CHAIN", chain, "NOT AVAILABLE")
                continue
            if log:
                all_x = np.concatenate((all_x, np.log10(tmp)))
            else:
                all_x = np.concatenate((all_x, tmp))
        if bins is None:
            bins = np.floor(no_unique_samples_sum/20)
            bins = min(bins, 100)
            bins = max(bins, 10)
        print("ALL", all_x)
        axis.hist(all_x, bins=int(bins), density=True)
        axis.grid(True)
        if show:
            plt.show()

    def _plot_hist_2d(self, axis, burn_in=None, param_no=[0, 1], chains_disp=None, bins=20, show=True, colorbar=False, log=[False, False]):
        # use weighted data
        if chains_disp is None:
            chains_disp = range(self.no_chains)
        if burn_in is None:
            burn_in = [0] * len(chains_disp)
        all_x = np.zeros((0,))
        all_y = np.zeros((0,))
        no_unique_samples_sum = 0
        for i, chain in enumerate(chains_disp):
            no_unique_samples_sum += self.no_unique_samples[chain]  # burn-in not excluded
            try:
                tmp_x = self.samples[chain][burn_in[i]:, param_no[0]]
                tmp_y = self.samples[chain][burn_in[i]:, param_no[1]]
            except:
                print("HISTOGRAM 2D: CHAIN", chain, "NOT AVAILABLE")
                continue
            if log[0]:
                all_x = np.concatenate((all_x, np.log10(tmp_x)))
            else:
                all_x = np.concatenate((all_x, tmp_x))
            if log[1]:
                all_y = np.concatenate((all_y, np.log10(tmp_y)))
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

    def plot_hist_grid(self, bins1d: int = 20, bins2d: int = 20,
                       log_scale: list[bool] = None, par_names: list[str] = None, burn_in: list[int] = None,
                       parameters_to_disp: list[int] = None, chains_to_disp: list[int] = None):
        if parameters_to_disp is None:
            parameters_to_disp = range(self.no_parameters)
        if chains_to_disp is None:
            chains_to_disp = range(self.no_chains)
        if burn_in is None:
            burn_in = [0] * len(chains_to_disp)
        if log_scale is None:
            log_scale = [False] * len(parameters_to_disp)
        no_parameters_to_disp = len(parameters_to_disp)
        fig, axes = plt.subplots(no_parameters_to_disp, no_parameters_to_disp, sharex=False, sharey=False, figsize=(15, 15))
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        for idi, i in enumerate(parameters_to_disp):
            for idj, j in enumerate(parameters_to_disp):
                axis = axes[idi, idj]
                if idi == idj:
                    self._plot_hist_1d(axis=axis, param_no=i, burn_in=burn_in, chains_disp=chains_to_disp, bins=bins1d, show=False, log=log_scale[i])
                else:
                    self._plot_hist_2d(axis=axis, param_no=[j, i], burn_in=burn_in, chains_disp=chains_to_disp,
                                       bins=bins2d, show=False, log=[log_scale[j], log_scale[i]])
                if idi == 0:
                    # determine parameter name
                    if par_names is not None:
                        par_name = par_names[j].replace('_', '\_')
                        label = "${0}$".format(par_name)
                    else:
                        label = "$par. {0}$".format(j)
                    if log_scale[idj]:
                        label += "\n(log)"
                    axis.set_title(label, x=1.05, rotation=45, multialignment='center')
        return fig, axes


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
