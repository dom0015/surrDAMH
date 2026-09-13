#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np


import os
import numpy as np
import surrDAMH
from surrDAMH.surrogates.parent import Updater


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def evaluate_on_a_grid(Solver, par0_grid, par1_grid, filename):
    # create a grid of surrogate evaluations
    # only for no_parameters==2
    if Solver.no_parameters != 2:
        raise ValueError("evaluate_on_a_grid is only implemented for 2 parameters")

    par0_grid = par0_grid.reshape(-1)
    par1_grid = par1_grid.reshape(-1)
    par_grid = np.array(np.meshgrid(par0_grid, par1_grid)).T.reshape(-1, 2)
    # take just first output dimension for visualization, evaluate in a loop one by one
    obs_grid = np.zeros((len(par_grid), ))
    for i, par in enumerate(par_grid):
        obs_grid[i] = Solver.__call__(par)[0]

    # plot it as an image and save to a file
    import matplotlib.pyplot as plt
    plt.imshow(obs_grid.reshape(len(par0_grid), len(par1_grid)), extent=(par0_grid[0], par0_grid[-1], par1_grid[0], par1_grid[-1]), origin='lower')
    plt.colorbar()
    plt.xlabel('par0')
    plt.ylabel('par1')
    plt.title('Surrogate evaluation')
    plt.savefig(filename)
    plt.close()
    return obs_grid.reshape(len(par0_grid), len(par1_grid))

# obsolete, replaced by surrDAMH.modules.test_data.TestData.generate 
def generate_surrogate_test_data(prior_distribution, solver, n_test: int, seed: int,
                                 transform_before_surrogate: bool, conf: surrDAMH.Configuration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng_state = np.random.get_state()
    np.random.seed(seed)
    test_parameters = np.vstack([prior_distribution.rvs() for _ in range(n_test)])
    np.random.set_state(rng_state)

    if transform_before_surrogate:
        surrogate_test_parameters = np.vstack([
            prior_distribution.transform(parameters.copy())
            for parameters in test_parameters
        ])
    else:
        surrogate_test_parameters = test_parameters.copy()
    
    test_observations = np.zeros((n_test, conf.no_observations))
    for i, parameters in enumerate(test_parameters):
        solver.set_parameters(prior_distribution.transform(parameters.copy()))
        test_observations[i, :] = np.asarray(solver.get_observations()).reshape(-1)
    return test_parameters, surrogate_test_parameters, test_observations

# obsolete, replaced by surrDAMH.modules.test_data.TestData.compute_log_posterior_and_weights
def compute_test_log_posterior(prior_distribution, likelihood_distribution,
                               internal_parameters: np.ndarray, observations: np.ndarray):
    log_posterior = np.zeros((internal_parameters.shape[0], 1), dtype=float)
    for i, (parameters, obs) in enumerate(zip(internal_parameters, observations)):
        log_posterior[i, 0] = prior_distribution.logpdf(parameters) + likelihood_distribution.logpdf(obs)
    return log_posterior

# obsolete, replaced by surrDAMH.modules.test_data.TestData.compute_log_posterior_and_weights
def normalized_weights_from_log_posterior(log_posterior: np.ndarray):
    shifted = log_posterior - np.max(log_posterior)
    weights = np.exp(shifted)
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.full_like(log_posterior, 1.0 / log_posterior.shape[0])
    return weights / weight_sum


def surrogate_restart_state_has_snapshots(
        load_surrogate_state: bool, 
        load_surrogate_data_only: bool, 
        surrogate_state_dir: str,
        surrogate_checkpoint_path: str, 
        surrogate_training_data_path: str,
        rank_world: int,
        ) -> bool:
    if not (load_surrogate_state or load_surrogate_data_only):
        return False
    if load_surrogate_state:
        required_files_exist = (
            os.path.exists(surrogate_checkpoint_path)
            and os.path.exists(surrogate_training_data_path)
        )
    else:
        required_files_exist = os.path.exists(surrogate_training_data_path)
    if not required_files_exist:
        return False
    try:
        with np.load(surrogate_training_data_path) as loaded:
            num_snapshots = int(loaded["num_snapshots"])
    except Exception as exc:
        if rank_world == 0:
            print(
                f"Could not inspect surrogate restart state in {surrogate_state_dir}: {exc}. Starting cold.",
                flush=True,
            )
        return False
    if num_snapshots <= 0:
        if rank_world == 0:
            print(
                f"Ignoring surrogate restart state in {surrogate_state_dir} because it contains 0 snapshots. Starting cold.",
                flush=True,
            )
        return False
    return True


def load_surrogate_restart_state_if_available(
        updater: Updater,
        use_surrogate_restart: bool,
        load_surrogate_state: bool,
        load_surrogate_data_only: bool,
        rank_world: int,
        conf: surrDAMH.Configuration,
        surrogate_state_dir: str,
        surrogate_checkpoint_path: str,
        surrogate_training_data_path: str,
        ) -> list[np.ndarray] | None:
    assert updater is not None
    if not use_surrogate_restart or rank_world != conf.rank_collector:
        return None
    if load_surrogate_state and load_surrogate_data_only:
        raise ValueError("Set only one of load_surrogate_state and load_surrogate_data_only")
    if load_surrogate_state:
        if not (os.path.exists(surrogate_checkpoint_path) and os.path.exists(surrogate_training_data_path)):
            print(
                f"Collector restart requested, but surrogate state files are missing in {surrogate_state_dir}. Starting cold.",
                flush=True,
            )
            return None
        loaded_arrays = updater.load_state(
            checkpoint_path=surrogate_checkpoint_path,
            data_path=surrogate_training_data_path,
            load_optimizer=True,
        )
        if loaded_arrays is None:
            return None
        print(
            f"Collector loaded surrogate restart state from {surrogate_state_dir} "
            f"with {loaded_arrays[0].shape[0]} snapshots.",
            flush=True,
        )
        return list(loaded_arrays)
    if load_surrogate_data_only:
        if not os.path.exists(surrogate_training_data_path):
            print(
                f"Collector data-only restart requested, but surrogate training data is missing in {surrogate_state_dir}. Starting cold.",
                flush=True,
            )
            return None
        loaded_arrays = updater.load_training_data(surrogate_training_data_path)
        print(
            f"Collector loaded surrogate training data only from {surrogate_state_dir} "
            f"with {loaded_arrays[0].shape[0]} snapshots. Starting from a fresh network.",
            flush=True,
        )
        return list(loaded_arrays)
    return None