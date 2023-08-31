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

    def sampling_count(self):
        counts = np.zeros((self.raw_data.no_stages, self.raw_data.no_chains, 3), dtype=int)
        for i in range(self.raw_data.no_stages):
            for j in range(self.raw_data.no_chains):
                N_acc = np.sum((self.raw_data.types == 0)*(self.raw_data.stages == i)*(self.raw_data.chains == j))
                N_pr = np.sum((self.raw_data.types == 1)*(self.raw_data.stages == i)*(self.raw_data.chains == j))
                N_r = np.sum((self.raw_data.types == 2)*(self.raw_data.stages == i)*(self.raw_data.chains == j))
                counts[i,j,:] = [N_acc, N_pr, N_r]
        return counts

    def compute_L2_norms(self, observations):
        diff2 = np.square(self.raw_data.observations - observations)
        G_norm = np.sqrt(np.sum(diff2, axis=1))
        return G_norm

    def compute_likelihood_norms(self, observations, noise_cov):
        # transform samples to N(0,1)
        trans_mean = np.array([p["options"]["mu"] for p in self.config["transformations"]])
        trans_std = np.array([p["options"]["sigma"] for p in self.config["transformations"]])
        import surrDAMH.modules.transformations as trs
        trans_params = trs.lognormal_to_normal(self.raw_data.parameters, trans_mean, trans_std)
        # get prior covariance matrix
        prior_mean = np.array(self.config["problem_parameters"]["prior_mean"])
        prior_std = np.array(self.config["problem_parameters"]["prior_std"])
        prior_cov = prior_std ** 2 * np.eye(np.shape(prior_std)[0])

        Pdiff = trans_params - prior_mean
        Gdiff = np.array(self.raw_data.observations - observations)

        lhnorm = np.zeros((self.raw_data.len(),1))
        for i in range(self.raw_data.len()):
            # observations log likelihood
            invCv = np.linalg.solve(noise_cov, Gdiff[i].transpose())
            G_norm = -0.5 * np.dot(Gdiff[i], invCv)
            # prior log likelihood
            invCv = np.linalg.solve(prior_cov, Pdiff[i].transpose())
            p_norm = -0.5 * np.dot(Pdiff[i], invCv)
            # final posterior log likelihood
            lhnorm[i] = G_norm + p_norm

        lhnorm = lhnorm.flatten()
        return lhnorm

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
