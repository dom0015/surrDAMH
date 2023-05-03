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
        self.parameters = None
        self.observations = None
        self.weights = None

    def load(self, folder_samples, no_parameters, no_observations):
        folder_samples = os.path.join(folder_samples, 'raw_data')
        file_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
        file_samples.sort()
        N = len(file_samples)

        self.types = np.empty((0, 1), dtype=np.int8)
        self.stages = np.empty((0, 1), dtype=np.int8)
        self.chains = np.empty((0, 1), dtype=np.int8)
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
            stages = stg * np.ones(len(types), dtype=np.int8)

            chain = int(file_samples[i][file_samples[i].find("rank")+4:file_samples[i].find(".")])
            chains = chain * np.ones(len(types), dtype=np.int8)

            parameters = np.array(df_samples.iloc[:, 1:1 + no_parameters])
            observation = np.array(df_samples.iloc[:, 2 + no_parameters:])

            self.types = np.append(self.types, idx)
            self.stages = np.append(self.stages, stages)
            self.chains = np.append(self.chains, chains)
            self.parameters = np.vstack((self.parameters, parameters))
            self.observations = np.vstack((self.observations, observation))

            # compute weights
            # prerejected and rejected have weight=0
            widx = np.ones(len(types), dtype=bool)
            widx[types == "prerejected"] = False
            widx[types == "rejected"] = False
            temp = np.arange(len(types))[widx]
            temp[1:] = temp[1:] - temp[:-1]
            weights = np.zeros(len(types))
            weights[widx] = temp
            # if sum(widx) > 0:
            weights = weights.reshape((-1, 1))
            self.weights = np.vstack((self.weights, weights)).astype(int)

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
        return raw_data

    def compute_L2_norms(self, observations):
        diff2 = np.square(self.observations - observations)
        G_norm = np.sqrt(np.sum(diff2, axis=1))
        return G_norm

    def compute_likelihood_norms(self, observations, noise_cov):
        diff = np.array(self.observations - observations)
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
        return Sample(self, idx), G_norm

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
            samples.append(idx)

        return samples, G_norm[sorted_idx[:count]]


class Visualization:
    def __init__(self, config, raw_data):
        self.config = config
        self.raw_data = raw_data
        self.no_parameters = config["no_parameters"]
        self.par_names = [p["name"] for p in config["transformations"]]
        self.noise_cov = Gaussian_process.assemble_covariance_matrix(config["noise_model"])
        self.observations = np.array(config["problem_parameters"]["observations"])

    def plot_likelihood_ij(self, axes, idp1, idp2, G_norm=None, vlimits=None):
        if G_norm is None:
            G_norm = self.raw_data.compute_likelihood_norms(self.observations, self.noise_cov)
        xx = self.raw_data.parameters[:, idp1]
        yy = self.raw_data.parameters[:, idp2]

        trans = self.config["transformations"]
        if trans[idp1]["type"] == "normal_to_lognormal":
            xx = np.log10(xx)
        if trans[idp2]["type"] == "normal_to_lognormal":
            yy = np.log10(yy)

        if G_norm is None:
            vlimits = [np.min(G_norm), np.max(G_norm)]
        im = axes.scatter(xx, yy, s=1, c=G_norm, vmin=max(vlimits[0], -100), vmax=vlimits[1], cmap="viridis")
        # axes.title = "log likelihood"
        axes.grid()
        # axes.colorbar(extend="min")
        return im

    def plot_likelihood(self, fig, axes, parameters_disp=None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)

        G_norm = self.raw_data.compute_likelihood_norms(self.observations, self.noise_cov)
        vlimits = [np.min(G_norm), np.max(G_norm)]

        n = len(parameters_disp)
        for idi,i in enumerate(parameters_disp):
            for idj,j in enumerate(parameters_disp):
                ax = axes[idi,idj]
                if idi==idj:
                    ax.set_axis_off()
                    continue
                else:
                    im = self.plot_likelihood_ij(ax, idi, idj, G_norm=G_norm, vlimits=vlimits)
                if idi==0:
                    par_name = self.par_names[j].replace('_', '\_')
                    label = "${0}$".format(par_name)

                    if self.config["transformations"][idj]["type"] == "normal_to_lognormal":
                        label += "\n(log)"
                    ax.set_title(label, x=1.05, rotation=45, multialignment='center')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        return fig, axes