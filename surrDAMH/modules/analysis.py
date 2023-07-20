#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:41 2023

@author: Pavel Exner
"""

import numpy as np
from surrDAMH.modules.raw_data import Sample, RawData


class Analysis:
    def __init__(self, config, raw_data):
        self.config = config
        self.raw_data = raw_data

        self.par_names = [p["name"] for p in config["transformations"]]
        self.par_names_latex = [p["name"] for p in config["transformations"]]
        for i,p in enumerate(self.par_names_latex):
            self.par_names_latex[i] = p.replace('_', '\_')

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
