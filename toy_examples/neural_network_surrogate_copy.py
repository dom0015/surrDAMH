#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 4 with required number of MPI processes):
mpiexec -n 4 python3 -m mpi4py neural_network_surrogate.py

(Here, one process will be used as collector,
and the remaining processes will be used as samplers.)
"""

import os

import solver_examples.solver_examples
import solver_examples.solver_spec_examples
from mpi4py import MPI
import numpy as np

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

run_sampling = False

no_parameters = 5
no_observations = 1

# solver example, takes 2 parameters, returns 1 observation:
solver_spec = solver_examples.solver_spec_examples.SolverSpecSinProdGeneric(no_parameters=5, no_observations=1, sleep=0.0)

# solver_instance = solver_examples.solver_examples.SinProdGeneric(no_parameters=5, no_observations=1)

initial_samples_distribution = surrDAMH.distributions.Normal(mean=np.zeros((no_parameters,)), sd=1e-5)

# configuration (specification of basic settings of the sampling framework):
conf = surrDAMH.Configuration(output_dir="out_temp_no_sub", no_parameters=5, no_observations=1, min_snapshots_to_update=0, use_solvers_pool=False,
                              min_snapshots_initial=0, state_dependent_approximation=True, initial_sample_type="user_specified",
                              # use_collector=False,
                              initial_samples_distribution=initial_samples_distribution)

# Gaussian prior distribution:
prior = surrDAMH.distributions.Normal(mean=[0.0]*conf.no_parameters, sd=2.0)

# generate 100 random samples from prior, for each calculate observations:
# if rank_world == 0:
#     solver = solver_examples.solver_examples.SinProdGeneric(no_parameters=no_parameters, no_observations=no_observations)
#     pars = np.empty((0, no_parameters))
#     obss = np.empty((0, no_observations))
#     for i in range(100):
#         par = prior.rvs()  # .reshape((1, no_parameters))
#         print(par)
#         solver.set_parameters(par)
#         obs = solver.get_observations().reshape((1, no_observations))
#         print(pars.shape)
#         print(par.shape)
#         pars = np.vstack((pars, par))
#         obss = np.vstack((obss, obs))
#     np.savetxt('snapshots_par.csv', pars, delimiter=',')
#     np.savetxt('snapshots_obs.csv', obss, delimiter=',')

# likelihood (additive Gaussian noise):
# observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=[-2, 2, -1, 1, -1])
observations = [1.0] * conf.no_observations
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=0.5)

# neural network surrogate model:
# updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
# updater = surrDAMH.surrogates.RBFInterpolationUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
updater = surrDAMH.surrogates.PyTorchNNOngoingUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations,
                                                      hidden_layer_sizes=(4, ), solver="adam", activation="tanh", learning_rate=1e-3,
                                                      iterations_batch=100, loss_target=1e-6, device="cpu", verbose=True, seed=15)

if run_sampling and rank_world == conf.rank_collector:
    snapshots_par = np.loadtxt('snapshots_par.csv', delimiter=',')
    snapshots_obs = np.loadtxt('snapshots_obs.csv', delimiter=',')
    snapshots_obs = snapshots_obs.reshape((-1, no_observations))
    updater.add_data(snapshots_par, snapshots_obs, train_on_added_data=True)
    print("TRAIN")
    for i in range(300):
        updater.train()
    updater.verbose = False

# sampling process stages:
no_stages = 10
sds = [0.5+0.1*i for i in range(no_stages)]

print(sds)
list_of_stages = []
# during MH stage, initial surrogate model is constructed:
# list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=1.0, max_evaluations=10, adaptive=False))
# during DAMH-SMU stage, surrogate model is further updated:
# list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.1, max_evaluations=100, surrogate_model_updates=True,
#                             subchain_max_length=1, adaptive=False))
for i in range(no_stages):
    # list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=sds[i], max_evaluations=5000,
    #                             surrogate_model_updates=False))
    list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=sds[i], max_evaluations=5000,
                                surrogate_model_updates=False,
                                subchain_max_length=50,
                                adaptive=False, is_excluded=True))
# during DAMH stage, surrogate model is used but not updated:
# list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5, max_evaluations=500,
#                       surrogate_model_updates=False, send_snapshots_to_collector=False))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater,
                                 # solver_instance=updater.get_evaluator().as_solver(),
                                 prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec,
                                 initial_snapshots=None)
if run_sampling:
    sam.run()

# post processing:
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.get_summary()

    samples.calculate_CpUS([[i] for i in range(no_stages)])
    summary = samples.get_summary()
    summary_path = os.path.join(conf.output_dir, "post_processing_output", "summary.csv")
    summary.to_csv(summary_path)

    # save plot of chains to file:
    for i in range(no_stages):
        fig, _ = samples.plot_chains(stages_to_disp=[i])
        file_path = os.path.join(conf.output_dir, "post_processing_output", "chains_" + str(i) + ".pdf")
        fig.savefig(file_path, bbox_inches="tight")

        # save plot of chain cummulative averages to file:
        fig, _ = samples.plot_chains(average=True, stages_to_disp=[i])
        file_path = os.path.join(conf.output_dir, "post_processing_output", "averages_" + str(i) + ".pdf")
        fig.savefig(file_path, bbox_inches="tight")

        # save histograms grid to file:
        fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[i])
        # ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
        file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms_" + str(i) + ".pdf")
        print(file_path)
        fig.savefig(file_path, bbox_inches="tight")
