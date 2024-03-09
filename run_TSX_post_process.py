#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
python3 run_TSX_postprocess.py
"""

import numpy as np
import surrDAMH
# import surrDAMH.post_processing as post
import os
from surrDAMH.priors.independent_components import Uniform, Beta, Lognormal, Normal
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.modules.Gaussian_process import assemble_covariance_matrix
from surrDAMH.stages import Stage
from surrDAMH.solver_specification import SolverSpec
import matplotlib.pyplot as plt

# basic configuration
conf = surrDAMH.Configuration(output_dir="output_TSX", no_parameters=8, no_observations=104, no_solvers=3,
                              use_collector=True, initial_sample_type="prior", transform_before_surrogate=True, save_raw_data=True,
                              no_snapshots_initial=9, no_snapshots_to_update=1)

# prior distribution
prior_mean = np.array([-16, 26, 17, 16, -48, -41, -14, -16], dtype=np.float32)
prior_sd = np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)
list_of_components = [Lognormal(prior_mean[i], prior_sd[i]) for i in range(conf.no_parameters)]
prior = surrDAMH.priors.PriorIndependentComponents(list_of_components)

# observations and likelihood
observations = np.array([275., 426.15387335, 519.56162791, 476.91835845,
                         434.28541983, 395.94912971, 362.91870775, 334.89754005,
                         311.18588382, 291.04532441, 273.82352886, 268.90294123,
                         241.65168662, 221.11548303, 205.17735169, 192.21875682,
                         181.47665216, 172.42512924, 164.69119354, 158.00347887,
                         152.16008627, 147.00781851, 142.42840025, 138.32909792,
                         134.63621267, 133.75887286, 275., 331.38155396,
                         370.73351276, 370.25372879, 369.63272562, 368.71390459,
                         367.41485277, 365.70624488, 363.59622182, 361.11740736,
                         358.31645218, 357.44852263, 351.0293995, 343.91063015,
                         336.49371946, 329.06614102, 321.78852469, 314.76080504,
                         308.03975175, 301.65272745, 295.60775284, 289.90057293,
                         284.51948343, 279.44855845, 274.66977245, 273.51013087,
                         275., 154.60491207,  71.45236294,  76.70999836,
                         83.21164371,  89.82079217,  96.03341936, 101.64846414,
                         106.61222918, 110.94128174, 114.68346148, 115.75476827,
                         121.62247161, 125.77437628, 128.63350118, 130.62182023,
                         131.95983675, 132.81103707, 133.29647802, 133.50601307,
                         133.50663863, 133.34859993, 133.06982694, 132.69915776,
                         132.25865449, 132.14122826, 275., 221.8150888,
                         185.32522539, 187.4275189, 188.87242565, 189.94580339,
                         190.81759421, 191.58603065, 192.30577855, 193.00529174,
                         193.69746946, 193.90470667, 195.2797671, 196.62391099,
                         197.88852785, 199.04501582, 200.07312514, 200.9629336,
                         201.71241553, 202.32504753, 202.80784368, 203.16987678,
                         203.42122425, 203.57225232, 203.63313486, 203.63785925])
block_spec_list = []
for i in range(4):
    spec = dict()
    spec["time_grid"] = np.array([0., 10., 17., 27., 37., 47., 57., 67., 77., 87., 97.,
                                  100., 120., 140., 160., 180., 200., 220., 240., 260., 280., 300.,
                                  320., 340., 360., 365.])
    spec["corr_length"] = 30
    spec["std"] = 5
    spec["cov_type"] = "squared_exponential"
    block_spec_list.append(spec)
noise_cov = assemble_covariance_matrix(block_spec_list)

plt.imshow(noise_cov)
plt.colorbar()
ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
file_path = os.path.join(conf.output_dir, "post_processing_output", "noise_cov.pdf")
plt.savefig(file_path, bbox_inches="tight")

ref_par = np.array([-17.42599444,  26.59339408,  17.38733495,  14.49584642,
                    -49.24249923, -42.02752125, -13.58804331, -17.45912394])

# save histograms grid
samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)
fig, axes = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[0], scale=["ln"]*8)
surrDAMH.post_processing.add_normal_dist_grid(axes, prior_mean, prior_sd)
surrDAMH.post_processing.add_normal_dist_grid(axes, ref_par, prior_sd, no_sigmas_to_show=0, color="orange")
ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms0.pdf")
fig.savefig(file_path, bbox_inches="tight")

samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)
fig, axes = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[1, 2], scale=["ln"]*8)
surrDAMH.post_processing.add_normal_dist_grid(axes, prior_mean, prior_sd)
surrDAMH.post_processing.add_normal_dist_grid(axes, ref_par, prior_sd, no_sigmas_to_show=0, color="orange")
ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms12.pdf")
fig.savefig(file_path, bbox_inches="tight")

parameters_to_disp = [0, 1, 2, 7]
fig, axes = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[1, 2], scale=["ln"]*8, parameters_to_disp=[0, 1, 2, 7])
surrDAMH.post_processing.add_normal_dist_grid(axes, prior_mean[parameters_to_disp], prior_sd[parameters_to_disp])
surrDAMH.post_processing.add_normal_dist_grid(axes, ref_par[parameters_to_disp], prior_sd[parameters_to_disp], no_sigmas_to_show=0, color="orange")
ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms12_0127.pdf")
fig.savefig(file_path, bbox_inches="tight")
