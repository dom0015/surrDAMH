#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 4 with required number of MPI processes):
mpirun -n 4 python3 -m mpi4py own_solver.py

(Here, one process will be used as collector,
and the remaining processes will be used as samplers.)
"""

import os

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage

observations = [754.64805252, 755.01945387, 655.95978987, 594.39699416,
                530.43745741, 515.8816575, 494.97253552, 475.59152216,
                452.86384229, 433.26618307, 382.97842896, 351.99284788,
                332.17098259, 309.92537921, 300.5676362, 303.92681608,
                287.90724713, 291.92276435, 515.43370038, 587.81872627,
                600.23143393, 604.48910264, 600.39683158, 599.11176231,
                603.23124273, 599.23918313, 599.03415368, 601.54081621,
                584.89274096, 571.77321889, 559.71106412, 551.93560232,
                550.11891225, 547.36679965, 546.21690982, 542.02776456,
                182.31709971, 199.55739055, 212.15676692, 224.15477335,
                229.70971358, 238.39284514, 253.75478535, 262.23129018,
                268.99356638, 276.3197908, 278.0979942, 281.01668497,
                280.27204155, 284.22676206, 286.21291704, 290.6920895,
                294.08053105, 294.73164677,  48.08262448,  42.93917678,
                49.94181195,  61.35265637,  59.6642977,  75.47073607,
                81.53250879,  90.51661055,  90.82438687,  89.70334219,
                79.27229886,  79.20607144,  83.22591843,  77.02257697,
                78.10138551,  84.91265489,  75.86837226,  82.69544488]

observations = np.array(observations)

no_subdomains = 9
no_parameters = no_subdomains*4+2  # 38
no_observations = 18*4  # 72

snapshots_par = np.loadtxt('torch_perceptron_par.csv', delimiter=',')
snapshots_obs = np.loadtxt('torch_perceptron_obs.csv', delimiter=',')
snapshots_par_trans = np.loadtxt('torch_perceptron_par_trans.csv', delimiter=',')
# snapshots_par_trans = snapshots_par.copy()

N = snapshots_par.shape[0]

list_of_components = []
for i in range(no_subdomains):
    list_of_components.append(surrDAMH.distributions.independent_components.Lognormal(-40, 3))
for i in range(no_subdomains):
    list_of_components.append(surrDAMH.distributions.independent_components.Lognormal(-25, 3))
for i in range(no_subdomains):
    list_of_components.append(surrDAMH.distributions.independent_components.Lognormal(26, 2))
for i in range(no_subdomains):
    list_of_components.append(surrDAMH.distributions.independent_components.Uniform(0, 0.5))
for i in range(2):
    list_of_components.append(surrDAMH.distributions.independent_components.Lognormal(16, 2))
prior = surrDAMH.distributions.PriorIndependentComponents(list_of_components)

# likelihood (additive Gaussian noise):
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=10.0)

best_likelihood = -np.inf
print(N)
for i in range(N):
    par = snapshots_par[i, :]
    obs = snapshots_obs[i, :]
    par_trans = snapshots_par_trans[i, :]
    lik = likelihood.logpdf(obs)
    if lik > best_likelihood:
        best_likelihood = lik
        index = i
        print(best_likelihood)
        print(lik)

# np.savetxt('torch_perceptron_par_trans.csv', snapshots_par_trans, delimiter=',')

print(best_likelihood)
print(snapshots_obs[index, :])
print(snapshots_par[index, :])
print([snapshots_par_trans[index, :]])

"""
# create instance of own local solver
solver_instance = SolverTSX()

# configuration (specification of basic settings of the sampling framework),
# since local solver instance was specified, solvers pool cannot be used,
# the solver will be evaluated directly on Samplers:
conf = surrDAMH.Configuration(output_dir="out_tsx", no_parameters=no_parameters, no_observations=no_observations,
                              use_solvers_pool=False, use_collector=True,
                              min_snapshots_to_update=0, min_snapshots_initial=0,
                              state_dependent_approximation=False)

# NN updater:
updater = surrDAMH.surrogates.PyTorchNNOngoingUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations,
                                                      hidden_layer_sizes=(48, 60), solver="adam", activation="tanh", learning_rate=1e-3,
                                                      iterations_batch=100, loss_target=1e-6, device="cpu", verbose=False, seed=15)

# Gaussian prior distribution:




# sampling process stages:
list_of_stages = []
# during MH stage, initial surrogate model is constructed:
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.3, time_limit=60*60))
# during DAMH-SMU stage, surrogate model is further updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.3, time_limit=60*60*6,
                            subchain_max_length=10, surrogate_model_updates=True))
# during DAMH stage, surrogate model is used but not updated:
# list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5, max_evaluations=500, surrogate_model_updates=False))

sam = surrDAMH.SamplingFramework(conf, prior=prior, likelihood=likelihood, surrogate_updater=updater,
                                 list_of_stages=list_of_stages, solver_instance=solver_instance)
sam.run()

# post processing:
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

if rank_world == conf.rank_collector:
    updater.save_snapshots()

no_stages = len(list_of_stages)
parameters_to_disp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
if rank_world == 0:
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.get_summary()

    samples.calculate_CpUS([[i] for i in range(no_stages)])
    summary = samples.get_summary()
    summary_path = os.path.join(conf.output_dir, "post_processing_output", "summary.csv")
    summary.to_csv(summary_path)

    # save plot of chains to file:
    for i in range(no_stages):
        fig, _ = samples.plot_chains(stages_to_disp=[i], parameters_to_disp=parameters_to_disp)
        file_path = os.path.join(conf.output_dir, "post_processing_output", "chains_" + str(i) + ".pdf")
        fig.savefig(file_path, bbox_inches="tight")

        # save plot of chain cummulative averages to file:
        fig, _ = samples.plot_chains(average=True, stages_to_disp=[i], parameters_to_disp=parameters_to_disp)
        file_path = os.path.join(conf.output_dir, "post_processing_output", "averages_" + str(i) + ".pdf")
        fig.savefig(file_path, bbox_inches="tight")

        # save histograms grid to file:
        fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[i], parameters_to_disp=parameters_to_disp)
        # ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
        file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms_" + str(i) + ".pdf")
        fig.savefig(file_path, bbox_inches="tight")

        
        """
