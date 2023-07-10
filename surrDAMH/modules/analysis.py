#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:41 2023

@author: Pavel Exner
"""

import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, getsize
from surrDAMH.modules import Gaussian_process


class Sample:
    def __init__(self, raw_data, idx):
        self.raw_data = raw_data
        self.idx = idx
        self.type_names = ["accepted", "prerejected", "rejected"]

    def type(self):
        return self.raw_data.types[self.idx]

    def type_name(self):
        return self.type_names[self.raw_data.types[self.idx]]

    def stage(self):
        return self.raw_data.stages[self.idx]

    def chain(self):
        return self.raw_data.chains[self.idx]

    def weight(self):
        return self.raw_data.weights[self.idx]

    def parameters(self):
        return self.raw_data.parameters[self.idx, :]

    def observations(self):
        return self.raw_data.observations[self.idx, :]


class RawData:
    def __init__(self):
        self.types = None
        self.stages = None
        self.chains = None
        self.tags = None
        self.parameters = None
        self.observations = None
        self.weights = None

        self.no_chains = 0
        self.no_stages = 0
        self.no_samples = 0

    def load(self, folder_samples, no_parameters, no_observations):
        folder_samples = os.path.join(folder_samples, 'raw_data')
        file_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
        file_samples.sort()
        N = len(file_samples)

        self.types = np.empty((0, 1), dtype=np.int8)
        self.stages = np.empty((0, 1), dtype=np.int8)
        self.chains = np.empty((0, 1), dtype=np.int8)
        self.tags = np.empty((0, 1), dtype=np.int8)
        self.parameters = np.empty((0, no_parameters))
        self.observations = np.empty((0, no_observations))
        self.weights = np.empty((0, 1), dtype=np.int)

        for i in range(N):
            path_samples = folder_samples + "/" + file_samples[i]
            df_samples = pd.read_csv(path_samples, header=None)

            types = df_samples.iloc[:, 0]
            idx = np.zeros(len(types), dtype=np.int8)
            # idx[types == "accepted"] = 0
            idx[types == "prerejected"] = 1
            idx[types == "rejected"] = 2
            # print(np.shape(self.types))
            # print(np.shape(idx))

            stg = int(file_samples[i][3])
            self.no_stages = max(self.no_stages, stg+1)
            stages = stg * np.ones(len(types), dtype=np.int8)

            chain = int(file_samples[i][file_samples[i].find("rank")+4:file_samples[i].find(".")])
            self.no_chains = max(self.no_chains, chain+1)
            chains = chain * np.ones(len(types), dtype=np.int8)

            parameters = np.array(df_samples.iloc[:, 1:1 + no_parameters])
            tags = np.array(df_samples.iloc[:, 1 + no_parameters])
            observation = np.array(df_samples.iloc[:, 2 + no_parameters:])

            self.types = np.append(self.types, idx)
            self.stages = np.append(self.stages, stages)
            self.chains = np.append(self.chains, chains)
            self.tags = np.append(self.tags, tags)
            self.parameters = np.vstack((self.parameters, parameters))
            self.observations = np.vstack((self.observations, observation))

            # compute weights
            # prerejected and rejected have weight=0
            widx = np.ones(len(types), dtype=bool)
            widx[types == "prerejected"] = False
            widx[types == "rejected"] = False
            # not considering first sample
            # TODO get rid of last value
            temp = np.arange(len(types))[widx]
            temp = np.append(temp, len(types))
            temp = np.diff(temp)
            weights = np.zeros(len(types))
            weights[widx] = temp
            # if sum(widx) > 0:
            weights = weights.reshape((-1, 1))
            self.weights = np.vstack((self.weights, weights)).astype(int)

        self.no_samples = np.shape(self.types)[0]
        print("raw data: no_stages", self.no_stages)
        print("raw data: no_chains", self.no_chains)
        print("raw data: no_samples", self.no_samples)
        print("raw data: no_nonconverging", np.shape(self.types[self.tags < 0])[0])

        print("raw data: p", np.shape(self.parameters))
        print("raw data: w", np.shape(self.weights))
        # print(self.weights[:20])
        # print(self.weights[-20:])
        # print(self.weights[self.weights>0][:])
        print("raw_data: np.sum(weights):", np.sum(self.weights))

    def len(self):
        return len(self.types)

    def no_parameters(self):
        return np.shape(self.parameters)[1]

    def no_observations(self):
        return np.shape(self.observations)[1]

    def filter(self, types, stages):
        idx = np.zeros(len(self.types), dtype=bool)

        for t in types:
            for s in stages:
                idx[(self.types == t)*(self.stages == s)] = 1

        raw_data = RawData()
        raw_data.types = self.types[idx]
        raw_data.stages = self.stages[idx]
        raw_data.chains = self.chains[idx]
        raw_data.parameters = self.parameters[idx]
        raw_data.observations = self.observations[idx]
        raw_data.weights = self.weights[idx]

        raw_data.no_samples = np.shape(raw_data.types)[0]
        raw_data.no_stages = len(stages)

        print("filter raw data: p", np.shape(raw_data.parameters))
        return raw_data


class Analysis:
    def __init__(self, config, raw_data):
        self.config = config
        self.raw_data = raw_data

        self.par_names = [p["name"] for p in config["transformations"]]
        for i,p in enumerate(self.par_names):
            self.par_names[i] = p.replace('_', '\_')

    def compute_L2_norms(self, observations):
        diff2 = np.square(self.raw_data.observations - observations)
        G_norm = np.sqrt(np.sum(diff2, axis=1))
        return G_norm

    def compute_likelihood_norms(self, observations, noise_cov):
        diff = np.array(self.raw_data.observations - observations)
        invCv = np.linalg.solve(noise_cov, np.transpose(diff))
        # print(np.shape(diff), np.shape(invCv))
        G_norm = np.diag(-0.5 * np.matmul(diff, invCv))
        return G_norm

    def find_best_fit(self, observations, norm="L2", noise_cov=None):
        if norm == "L2":
            G_norms = self.compute_L2_norms(observations)
            idx = np.argmin(G_norms)
        elif norm == "likelihood":
            G_norms = self.compute_likelihood_norms(observations, noise_cov)
            idx = np.argmax(G_norms)
        else:
            raise Exception("Unknown norm type: " + norm)

        G_norm = G_norms[idx]

        est_noise = np.std(G_norm - observations)
        print("estimated noise std (best_fit-observations)", est_noise)

        return Sample(self.raw_data, idx), G_norm

    def find_n_best_fits(self, observations, count, norm="L2", noise_cov=None):
        if norm == "L2":
            G_norm = self.compute_L2_norms(observations)
        elif norm == "likelihood":
            G_norm = self.compute_likelihood_norms(observations, noise_cov)
        else:
            raise Exception("Unknown norm type: " + norm)

        sorted_idx = np.argsort(G_norm)
        if norm == "likelihood":
            sorted_idx = sorted_idx[::-1]

        samples = []
        for idx in sorted_idx[:count]:
            samples.append(Sample(self.raw_data, idx))

        est_noise = np.std(samples[0].observations() - observations)
        print("estimated noise std (best_fit-observations)", est_noise)

        return samples, G_norm[sorted_idx[:count]]

    def estimate_statistics(self):
        weights = self.raw_data.weights.reshape(-1,)
        average = np.average(self.raw_data.parameters, axis=0, weights=weights)
        # Fast and numerically precise:
        variance = np.average((self.raw_data.parameters - average) ** 2, axis=0, weights=weights)
        return average, np.sqrt(variance)

    def estimate_distributions(self, output_file = None):
        if output_file is not None:
            with open(output_file, 'w') as file:
                header ='N,' + ','.join(self.par_names)
                file.write(header + "\n")
                for idx in range(self.raw_data.len()):
                    s = Sample(self.raw_data, idx)
                    line = str(s.weight()) + ',' + ','.join([str(p) for p in s.parameters()])
                    file.write(line + "\n")

        # TODO weighted statistics!
        # print(np.shape(self.raw_data.parameters))
        param_all_log = np.log10(self.raw_data.parameters)
        mean = np.mean(self.raw_data.parameters, axis=0).tolist()
        mean_log = np.mean(param_all_log, axis=0).tolist()
        std = np.std(self.raw_data.parameters, axis=0).tolist()
        std_log = np.std(param_all_log, axis=0).tolist()

        output_list = []
        for i in range(self.raw_data.no_parameters()):
            d = dict()
            d["name"] = self.par_names[i]
            d["mu"] = mean[i]
            d["mu_log10"] = mean_log[i]
            d["sigma"] = std[i]
            d["sigma_log10"] = std_log[i]
            output_list.append(d)

        return output_list

class Visualization:
    def __init__(self, config, raw_data):
        self.config = config
        self.raw_data = raw_data
        self.no_parameters = config["no_parameters"]
        self.noise_cov = Gaussian_process.assemble_covariance_matrix(config["noise_model"])
        self.observations = np.array(config["problem_parameters"]["observations"])
        self.analysis = Analysis(config, raw_data)

        # Flow123d dependent
        self.time_axis = np.array(config["noise_model"][0]["time_grid"])

    def plot_likelihood_ij(self, axis, idp1, idp2, G_norm=None, vlimits=None):
        if G_norm is None:
            G_norm = self.analysis.compute_likelihood_norms(self.observations, self.noise_cov)
        xx = self.raw_data.parameters[:, idp1]
        yy = self.raw_data.parameters[:, idp2]

        trans = self.config["transformations"]
        if trans[idp1]["type"] == "normal_to_lognormal":
            xx = np.log10(xx)
        if trans[idp2]["type"] == "normal_to_lognormal":
            yy = np.log10(yy)

        if G_norm is None:
            vlimits = [np.min(G_norm), np.max(G_norm)]
        im = axis.scatter(xx, yy, s=1, c=G_norm, vmin=max(vlimits[0], -100), vmax=vlimits[1], cmap="viridis")
        # axes.title = "log likelihood"
        axis.grid()
        # axes.colorbar(extend="min")
        return im

    def plot_likelihood(self, fig, axes, parameters_disp=None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)

        G_norm = self.analysis.compute_likelihood_norms(self.observations, self.noise_cov)
        vlimits = [np.min(G_norm), np.max(G_norm)]

        n = len(parameters_disp)
        for idi,i in enumerate(parameters_disp):
            for idj,j in enumerate(parameters_disp):
                ax = axes[idi,idj]
                if idi==idj:
                    ax.set_axis_off()
                    continue
                else:
                    im = self.plot_likelihood_ij(ax, idj, idi, G_norm=G_norm, vlimits=vlimits)
                if idi==0:
                    label = "${0}$".format(self.analysis.par_names[j])

                    if self.config["transformations"][idj]["type"] == "normal_to_lognormal":
                        label += "\n(log)"
                    ax.set_title(label, x=1.05, rotation=45, multialignment='center')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        return fig, axes

    def plot_hist_1d(self, axis, burn_in=None, param_no=0, bins=20, color=None):
        # TODO burn_in for each chain (select from raw_data by chains??)
        # if burn_in == None:
        #     burn_in = [0] * self.raw_data.no_chains
        if color is None:
            color = "lightblue"

        trans = self.config["transformations"]
        xx = self.raw_data.parameters[:, param_no]
        if trans[param_no]["type"] == "normal_to_lognormal":
            xx = np.log10(xx)
        axis.hist(xx, bins=bins, density=True, weights=self.raw_data.weights, color=color)

    def plot_hist_2d(self, axis, burn_in=None, param_no=[0, 1], bins=20, colorbar=False, cmap=None):
        # if burn_in == None:
        #     burn_in = [0] * self.raw_data.no_chains
        if cmap is None:
            cmap = plt.cm.Binary

        trans = self.config["transformations"]
        xx = self.raw_data.parameters[:, param_no[0]]
        yy = self.raw_data.parameters[:, param_no[1]]
        if trans[param_no[0]]["type"] == "normal_to_lognormal":
            xx = np.log10(xx)
        if trans[param_no[1]]["type"] == "normal_to_lognormal":
            yy = np.log10(yy)

        # print(param_no[0], param_no[1], np.sum(self.raw_data.weights))
        # this keeps the maximal axis limits
        # when plotting multiple histograms
        # (otherwise hist2d sets limits according to last call)
        xlim1 = axis.get_xlim()
        ylim1 = axis.get_ylim()
        data = self.raw_data.weights.reshape((-1,))
        axis.hist2d(xx, yy, bins=bins, cmin=np.min(data), cmap=cmap, weights=data)  # , density = True)
        xlim2 = axis.get_xlim()
        ylim2 = axis.get_ylim()
        xlim = [min([xlim1[0], xlim2[0]]), max([xlim1[1], xlim2[1]])]
        ylim = [min([ylim1[0], ylim2[0]]), max([ylim1[1], ylim2[1]])]
        if xlim1[0] != 0:
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)

        axis.grid(True)
        if colorbar:
            axis.colorbar()

    def create_plot_grid(self, parameters_disp=None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)

        n = len(parameters_disp)
        fig, axes = plt.subplots(n, n, sharex=False, sharey=False, figsize=(15, 15))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        trans = self.config["transformations"]
        for idj, j in enumerate(parameters_disp):
            axis = axes[0,idj]
            # determine parameter name
            if self.analysis.par_names is not None:
                label = "${0}$".format(self.analysis.par_names[j])
            else:
                label = "$par. {0}$".format(j)
            if trans[idj]["type"] == "normal_to_lognormal":
                label += "\n(log)"
            axis.set_title(label, x=0.5, rotation=45, multialignment='center')

        return fig, axes

    def adjust_plot_grid_axes(self, no_parameters, axes):
        """
        Moves common ticks and ticklabels to the outside of the grid.
        """
        for i in range(no_parameters):
            for j in range(no_parameters):
                if j != no_parameters - 1:
                    # axes[j, i].sharex(axes[no_parameters - 1, i])
                    axes[j, i].set_xticklabels([])
                if i != j and i != 0 and i != no_parameters - 1:
                    axes[j, i].set_yticklabels([])
                if i != j and i == no_parameters - 1:
                    axes[j, i].tick_params(labelleft=False, labelright=True)

                if i == j and 0 < j < no_parameters - 1:
                    axes[j, i].tick_params(axis="y", direction="in", pad=-15)
                if i == j and j == no_parameters - 1:
                    axes[j, i].tick_params(labelleft=False, labelright=True)
                # if j != 0:
                #     axes[j, i].set_yticklabels([])

    def plot_hist_grid(self, fig, axes, burn_in=None, parameters_disp=None, bins1d=20, bins2d=20,
                       c_1d=None, cmap_2d=None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        # if burn_in == None:
        #     burn_in = [0] * self.raw_data.no_chains

        for idi, i in enumerate(parameters_disp):
            for idj, j in enumerate(parameters_disp):
                axis = axes[idi, idj]
                if idi == idj:
                    self.plot_hist_1d(axis=axis, param_no=i, burn_in=burn_in, bins=bins1d, color=c_1d)
                else:
                    self.plot_hist_2d(axis=axis, param_no=[j, i], burn_in=burn_in, bins=bins2d, cmap=cmap_2d)

    def plot_hist_grid_add_sample(self, fig, axes, sample, parameters_disp=None, color=None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        if color is None:
            color = "Red"

        trans = self.config["transformations"]
        for idi, i in enumerate(parameters_disp):
            for idj, j in enumerate(parameters_disp):
                axis = axes[idi, idj]
                if idi != idj:
                    y = sample.parameters()[idi]
                    x = sample.parameters()[idj]
                    if trans[idi]["type"] == "normal_to_lognormal":
                        y = np.log10(y)
                    if trans[idj]["type"] == "normal_to_lognormal":
                        x = np.log10(x)
                    # axis.plot(x, y, marker='.', mec=color, mfc=color)
                    axis.plot(x, y, marker='.', ms=11, markeredgewidth=2, mec="LimeGreen", mfc=color)

    def plot_observe_slice_ij(self, axis, idp1, idp2, values=None):
        xx = self.raw_data.parameters[:, idp1]
        yy = self.raw_data.parameters[:, idp2]

        trans = self.config["transformations"]
        if trans[idp1]["type"] == "normal_to_lognormal":
            xx = np.log10(xx)
        if trans[idp2]["type"] == "normal_to_lognormal":
            yy = np.log10(yy)

        vlimits = [np.min(values), np.max(values)]
        # vlimits = [-50,150]
        im = axis.scatter(xx, yy, s=1, c=values, vmin=vlimits[0], vmax=vlimits[1], cmap="viridis")
        # im = axis.scatter(xx, yy, s=1, c=values, cmap="viridis")
        # axes.title = "log likelihood"
        axis.grid()
        # axes.colorbar(extend="min")
        return im

    def plot_observe_slice(self, fig, axes, observe_idx, parameters_disp=None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)

        obs_slice = self.raw_data.observations[:, observe_idx]

        n = len(parameters_disp)
        for idi,i in enumerate(parameters_disp):
            for idj,j in enumerate(parameters_disp):
                ax = axes[idi,idj]
                if i==0:
                    label = "${0}$".format(self.analysis.par_names[j])

                    if self.config["transformations"][idj]["type"] == "normal_to_lognormal":
                        label += "\n(log)"
                    ax.set_title(label, x=1.05, rotation=45, multialignment='center')
                if idi==idj:
                    ax.set_axis_off()
                    continue
                else:
                    im = self.plot_observe_slice_ij(ax, idj, idi, values=obs_slice)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        return fig, axes

    def plot_observe_slice_1d(self, fig, axis, observe_idx, bins=None):
        # TODO burn_in for each chain (select from raw_data by chains??)
        # if burn_in == None:
        #     burn_in = [0] * self.raw_data.no_chains

        if bins is None:
            bins = (np.sqrt(self.raw_data.no_samples)).astype(int)

        obs_slice = self.raw_data.observations[:, observe_idx]
        # axis.hist(obs_slice, bins=bins)
        axis.hist(obs_slice, bins=bins, density=True, weights=self.raw_data.weights)

        axis.set_title("Pressure observation data slice histogram - t" + str(observe_idx), fontsize=36)
        # axis.set_xlabel(self.analysis.par_names[idp], fontsize=20)
        axis.set_xlabel("pressure t" + str(observe_idx), fontsize=20)
        axis.set_xlim(0,200)

    def plot_observe_sensitivity(self, fig, axis, idp, observe_idx):
        mean, std = self.analysis.estimate_statistics()

        min_bin = mean[idp] - std[idp]
        max_bin = mean[idp] + std[idp]
        no_bins = 10
        bin = (max_bin - min_bin)/no_bins
        bins_mean = np.arange(min_bin + bin/2, max_bin, bin)
        # print(min_bin, max_bin)
        # print(bins_mean)

        p = self.raw_data.parameters
        samples = []
        for i in range(no_bins):
            ref = mean.copy()
            ref[idp] = bins_mean[i]
            # print(ref)
            distance = (p[:,:] - ref[None,:]) / std[None,:]
            distance = np.sum(distance**2, axis=1)
            idx = np.argmin(distance)
            # print(idx)
            samples.append(Sample(self.raw_data, idx))

        obs_slice = [s.observations()[observe_idx] for s in samples]
        # print(obs_slice)
        # print(obs_slice)
        axis.bar(np.arange(no_bins), obs_slice, width=0.4)

        bins_labels = [f'{s:5.2g}' for s in bins_mean]
        axis.set_xticks(np.arange(no_bins))
        axis.set_xticklabels(bins_labels, fontsize=20)
        axis.yaxis.set_tick_params(labelsize=20)

        axis.set_title(self.analysis.par_names[idp], fontsize=36)
        axis.set_xlabel(self.analysis.par_names[idp], fontsize=20)
        axis.set_ylabel("pressure - t" + str(observe_idx), fontsize=20)


    def plot_obs_vs_par(self, fig, axis, idp, observe_idx, observations, norm="L2", noise_cov=None):
        """
        Scatter plot: parameter X observation(pressure at selected time)
        @param fig:
        @param axis:
        @param idp: parameter index
        @param observe_idx: observation index
        @param observations: vector of observations
        @param norm: norm type ('L2', 'likelihood')
        @param noise_cov: noise covariance matrix
        """
        if norm == "L2":
            G_norms = self.analysis.compute_L2_norms(observations)
        elif norm == "likelihood":
            G_norms = self.analysis.compute_likelihood_norms(observations, noise_cov)
        else:
            raise Exception("Unknown norm type: " + norm)

        xx = self.raw_data.parameters[:, idp]
        obs_slice = self.raw_data.observations[:, observe_idx]

        trans = self.config["transformations"]
        log = trans[idp]["type"] == "normal_to_lognormal"
        xlabel = self.analysis.par_names[idp]
        if log:
            xx = np.log10(xx)
            xlabel = xlabel + "(log)"

        # vlimits = [np.min(G_norms), np.max(G_norms)]
        vlimits = [0, 150]
        # axis.scatter(xx, obs_slice)
        # axis.scatter(xx, obs_slice, s=1, c=G_norms, vmin=vlimits[0], vmax=vlimits[1], cmap="viridis")
        im = axis.scatter(xx, obs_slice, s=10, c=G_norms, cmap="viridis")
                          # norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
                          # norm=matplotlib.colors.Normalize(vmin=vlimits[0], vmax=vlimits[1]))

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        axis.grid()
        axis.set_title(xlabel, fontsize=36)
        axis.set_xlabel(xlabel, fontsize=20)
        axis.set_ylabel("pressure - t" + str(observe_idx), fontsize=20)

        # axis.set_ylim(-50, 280)

    def linear_regression_analysis(self, fig, axis, observe_idx, count=100):
        """
        Use linear regression model to approximate the model in neighborhood of selected point of parameter space.
        - selected point is mean
        - neighborhood is defined by the number (count) of samples closest to the selected point
        - 'closest' is measured as: sqrt(sum((pars - mean) / stds)^2) / no_pars
        @param fig:
        @param axis:
        @param observe_idx: observation index
        @param count: number of samples
        @return:
        """
        mean, std = self.analysis.estimate_statistics()

        p = self.raw_data.parameters

        distance_par = (p[:,:] - mean[None,:]) / std[None,:]
        distance_par = np.sum(distance_par**2, axis=1)
        mean_sample_idx = np.argmin(distance_par)
        mean_sample = Sample(self.raw_data, mean_sample_idx)

        sort_indices = np.argsort(distance_par)
        sort_indices = sort_indices[:count]

        pressure = self.raw_data.observations[sort_indices, observe_idx]
        # params_samples = p[sort_indices]
        params_samples = (p[sort_indices,:] - mean[None,:]) / std[None,:]
        # params_samples = params_samples[:,[0,1,2,3,4,6]]

        max_dist = np.max(np.sqrt(distance_par[sort_indices] / self.raw_data.no_parameters()))
        # print("max of selected distances", max_dist)

        # axis.scatter(distance_obs[sort_indices], pressure, s=10)
        # axis.plot(distance_obs[mean_sample_idx], mean_sample_obs[observe_idx], marker=".", markersize=20, mfc='r')
        # axis.grid()
        # axis.set_title("Noise analysis", fontsize=36)
        # axis.set_xlabel("distance by observation ||h_i-h_0||_2", fontsize=20)
        # axis.set_ylabel("pressure - t" + str(observe_idx), fontsize=20)

        import statsmodels.api as sm

        # P1 model, linear depndency on parameters, tangent plane model
        y = pressure
        X = params_samples
        X = sm.add_constant(X)
        model_P1 = sm.OLS(y, X)
        results = model_P1.fit()
        # print(results.summary())

        # print("Overall significance of parameters (P-value of F statistics): ", results.f_pvalue)

        return results, max_dist
        # fpvalue
        # pvalues
        # fittedvalues
        # rsquared
        # bse

    def linear_regression_rsquared(self, fig, axis, observe_indices, counts):
        """
        Use function linear_regression_analysis() and plot R**2 over distance (or subsample count)
        around a selected point (currently parameter mean)
        @param fig:
        @param axis:
        @param observe_indices: list of selected observe indices
        @param counts: list of sample counts (closest to selected point)
        """
        rsquared = np.zeros((len(observe_indices), len(counts)))
        distances = np.zeros((len(observe_indices), len(counts)))
        for i, observe_idx in enumerate(observe_indices):
            for ci, count in enumerate(counts):
                res, md = self.linear_regression_analysis(fig, axis, observe_idx, count=count)
                rsquared[i,ci] = res.rsquared
                distances[i,ci] = md

        # self.time_axis[observe_indices]
        for i in range(np.shape(rsquared)[0]):
            # axis.plot(counts, rsquared[i,:], label="t = "+str(self.time_axis[i])+" d")
            axis.plot(distances[i,:], rsquared[i,:], label="t = " + str(self.time_axis[i]) + " d")
        axis.set_title("Linear regression analysis", fontsize=36)
        # axis.set_xlabel("selected sample size", fontsize=20)

        # xticks = [0.2,0.3,0.5,0.8,1.0,1.5,2,3,4]
        xticks = [0.2, 0.3, 0.5, 0.8, 1.0]
        xticks_labels = [f'{s:1.2f}' for s in xticks]
        axis.set_xscale("log")
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticks_labels, fontsize=20)
        axis.set_xlabel("sample distance to mean", fontsize=20)


        axis.set_ylabel("rqsuared", fontsize=20)

        # colormap = plt.cm.winter
        colormap = plt.cm.gist_rainbow
        colors = [colormap(i) for i in np.linspace(0, 1, len(axis.lines))]
        for i, j in enumerate(axis.lines):
            j.set_color(colors[i])

        axis.legend()

    def linear_regression_over_time(self, fig, axis, observe_indices, count=100):
        """
        Use function linear_regression_analysis() and plot R**2 and parameter p-values
        over time around a selected point (currently parameter mean)
        @param fig:
        @param axis:
        @param observe_indices: list of selected observe indices
        @param count: sample count (closest to selected point)
        """
        rsquared = np.zeros(len(observe_indices))
        pvalues = np.zeros((len(observe_indices), self.raw_data.no_parameters()))
        bse = np.zeros((len(observe_indices), self.raw_data.no_parameters()))
        for i, observe_idx in enumerate(observe_indices):
            res, md = self.linear_regression_analysis(fig, axis, observe_idx, count=count)
            rsquared[i] = res.rsquared
            pvalues[i,:] = res.pvalues[1:]
            bse[i, :] = res.bse[1:]

        axis.plot(self.time_axis[observe_indices], rsquared, color='red', linestyle='dashed', label="rsquared")
        axis.set_title("Linear regression analysis over time with N={}".format(count), fontsize=36)
        axis.set_xlabel("time [d]", fontsize=20)
        axis.set_ylabel("rqsuared", fontsize=20)
        axis.legend()

        ax2 = axis.twinx()
        ax2.set_ylabel('Y2-axis', color='blue')
        for i in range(np.shape(pvalues)[1]):
            ax2.plot(self.time_axis[observe_indices], pvalues[:,i], color='blue', label=self.analysis.par_names[i])
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_yscale("log")
        ax2.set_ylim([1e-3,1])
        ax2.set_ylabel("pvalues", fontsize=20)
        ax2.grid()

        # colormap = plt.cm.winter
        colormap = plt.cm.gist_rainbow
        colors = [colormap(i) for i in np.linspace(0, 1, len(ax2.lines))]
        for i, j in enumerate(ax2.lines):
            j.set_color(colors[i])

        ax2.legend()
