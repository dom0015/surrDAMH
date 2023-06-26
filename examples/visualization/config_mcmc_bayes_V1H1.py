#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import os
import sys

import matplotlib.colors
import ruamel.yaml as yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from surrDAMH.modules import analysis as ape
import matplotlib.pyplot as plt
import numpy as np


def load_simulaton_config(simulation_dir):
    conf_path = os.path.join(simulation_dir, "common_files", "config_mcmc_bayes.yaml")
    basename = os.path.basename(conf_path)
    problem_name, fext = os.path.splitext(basename)
    output_dir = os.path.join(simulation_dir, 'saved_samples', problem_name)
    # visualization_dir = os.path.join(output_dir, 'img_Bayes')

    with open(conf_path) as f:
        conf = yaml.safe_load(f)
    return conf, output_dir


def create_analysis(conf, output_dir, no_p, no_b):
    raw_data = ape.RawData()
    raw_data.load(output_dir, no_p, no_b)
    # type: 0-accepted, 1-prerejected, 2-rejected
    raw_data_ar = raw_data.filter(types=[0, 2], stages=[0])
    raw_data_a = raw_data.filter(types=[0], stages=[0])

    visual_ar = ape.Visualization(conf, raw_data_ar)
    visual_a = ape.Visualization(conf, raw_data_a)
    return visual_ar, visual_a



sim1_dir = sys.argv[1] # V1
sim2_dir = sys.argv[2] # H1

conf1, output_dir1 = load_simulaton_config(sim1_dir)
conf2, output_dir2 = load_simulaton_config(sim2_dir)

### PREPARATION:
observations = np.array(conf1["problem_parameters"]["observations"])
par_names = [p["name"] for p in conf1["transformations"]]
print("parameters:", par_names)

visualization_dir = os.path.join(output_dir1, 'img_Bayes')
if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)

no_parameters = conf1["no_parameters"]

# SIMULATION V1
vis_ar_V1, vis_a_V1 = create_analysis(conf1, output_dir1, no_parameters, len(observations))
fits, norms = vis_ar_V1.analysis.find_n_best_fits(observations, count=20, norm="L2")
bestfit_L2_V1 = fits[0]
estimated_distributions_V1 = vis_a_V1.analysis.estimate_distributions(
                output_file=os.path.join(visualization_dir, "parameters2_V1.csv"))
print(estimated_distributions_V1)

# SIMULATION H1
vis_ar_H1, vis_a_H1 = create_analysis(conf2, output_dir2, no_parameters, len(observations))
fits, norms = vis_ar_H1.analysis.find_n_best_fits(observations, count=20, norm="L2")
bestfit_L2_H1 = fits[0]
estimated_distributions_H1 = vis_a_H1.analysis.estimate_distributions(
                output_file=os.path.join(visualization_dir, "parameters2_H1.csv"))
print(estimated_distributions_H1)

fig, axes = vis_a_V1.create_plot_grid()
# V1
vis_a_V1.plot_hist_grid(fig=fig, axes=axes, bins1d=15, bins2d=20, c_1d="Crimson", cmap_2d=plt.cm.Reds)
# vis_a_V1.plot_hist_grid_add_sample(fig=fig, axes=axes, sample=bestfit_L2_V1, color="Red")
# H1
cmap_blues = plt.cm.Blues  # original colormap
fading_blue = cmap_blues(np.arange(cmap_blues.N)) # extract colors
fading_blue[:, -1] = np.linspace(0, 1, cmap_blues.N) # modify alpha
fading_blue = matplotlib.colors.ListedColormap(fading_blue) # convert to colormap
vis_a_H1.plot_hist_grid(fig=fig, axes=axes, bins1d=15, bins2d=20, c_1d="DodgerBlue", cmap_2d=fading_blue)
# vis_a_H1.plot_hist_grid_add_sample(fig=fig, axes=axes, sample=bestfit_L2_H1, color="Blue")
fig.savefig(visualization_dir + "/histograms_s" + str(0) + ".pdf", bbox_inches="tight")
