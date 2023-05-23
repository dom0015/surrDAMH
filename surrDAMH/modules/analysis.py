#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:41 2023

@author: Pavel Exner
"""

import warnings
import numpy as np
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

    def plot_hist_1d(self, axis, burn_in=None, param_no=0, bins=20):
        # TODO burn_in for each chain (select from raw_data by chains??)
        # if burn_in == None:
        #     burn_in = [0] * self.raw_data.no_chains

        trans = self.config["transformations"]
        xx = self.raw_data.parameters[:, param_no]
        if trans[param_no]["type"] == "normal_to_lognormal":
            xx = np.log10(xx)
        axis.hist(xx, bins=bins, density=True, weights=self.raw_data.weights)

    def plot_hist_2d(self, axis, burn_in=None, param_no=[0, 1], bins=20, colorbar=False):
        # if burn_in == None:
        #     burn_in = [0] * self.raw_data.no_chains

        trans = self.config["transformations"]
        xx = self.raw_data.parameters[:, param_no[0]]
        yy = self.raw_data.parameters[:, param_no[1]]
        if trans[param_no[0]]["type"] == "normal_to_lognormal":
            xx = np.log10(xx)
        if trans[param_no[1]]["type"] == "normal_to_lognormal":
            yy = np.log10(yy)

        # print(param_no[0], param_no[1], np.sum(self.raw_data.weights))
        axis.hist2d(xx, yy, bins=bins, cmap="binary", weights=self.raw_data.weights.reshape((-1,)))  # , density = True)
        axis.grid(True)
        if colorbar:
            axis.colorbar()

    def plot_hist_grid(self, burn_in=None, parameters_disp=None, bins1d=20, bins2d=20):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        # if burn_in == None:
        #     burn_in = [0] * self.raw_data.no_chains

        trans = self.config["transformations"]
        n = len(parameters_disp)
        fig, axes = plt.subplots(n, n, sharex=False, sharey=False, figsize=(15, 15))
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        for idi, i in enumerate(parameters_disp):
            for idj, j in enumerate(parameters_disp):
                axis = axes[idi, idj]
                if idi == idj:
                    self.plot_hist_1d(axis=axis, param_no=i, burn_in=burn_in, bins=bins1d)
                else:
                    self.plot_hist_2d(axis=axis, param_no=[j, i], burn_in=burn_in, bins=bins2d)
                if idi == 0:
                    # determine parameter name
                    if self.analysis.par_names is not None:
                        label = "${0}$".format(self.analysis.par_names[j])
                    else:
                        label = "$par. {0}$".format(j)
                    if trans[idj]["type"] == "normal_to_lognormal":
                        label += "\n(log)"
                    axis.set_title(label, x=1.05, rotation=45, multialignment='center')
        return fig, axes

    def plot_observe_slice_ij(self, axis, idp1, idp2, values=None):
        xx = self.raw_data.parameters[:, idp1]
        yy = self.raw_data.parameters[:, idp2]

        trans = self.config["transformations"]
        if trans[idp1]["type"] == "normal_to_lognormal":
            xx = np.log10(xx)
        if trans[idp2]["type"] == "normal_to_lognormal":
            yy = np.log10(yy)

        # vlimits = [np.min(values), np.max(values)]
        vlimits = [0,150]
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
        axis.hist(obs_slice, bins=bins)
        # axis.hist(obs_slice, bins=bins, density=True, weights=self.raw_data.weights)
