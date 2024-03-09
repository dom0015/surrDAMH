#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:32:40 2021

@author: simona
"""

from mpi4py import MPI
import sys
import ruamel.yaml as yaml
from surrDAMH.modules import Gaussian_process
from typing import Literal
from dataclasses import dataclass
import numpy as np


@dataclass
class Configuration:
    output_dir: str
    no_parameters: int
    no_observations: int
    use_collector: bool = True
    no_solvers: int = 2
    solver_maxprocs: int = 1
    solver_returns_tag: bool = False
    pickled_observations: bool = True
    save_raw_data: bool = False
    transform_before_saving: bool = True
    transform_before_surrogate: bool = True
    initial_sample_type: Literal["lhs", "prior_mean", "user_specified"] = "prior_mean"  # "prior_mean" and "user_specified" will be perturbed
    initial_sample: np.ndarray | None = None  # only if initial_sample_type == "user_specified"
    no_snapshots_to_update: int = 1  # how many snapshots (at least) have to be added to update the surrogate model
    no_snapshots_initial: int = 1  # minimal number of snapshots for the construction of initial surrogate model
    debug: bool = False
    max_buffer_size: int = 1 << 30
    paths_to_append: list[str] = None

    def __post_init__(self) -> None:
        if self.paths_to_append is None:
            self.paths_to_append = []
        else:
            self._append_path()

        size_world = MPI.COMM_WORLD.Get_size()

        # ranks of samplers, collector, solvers pool:
        if self.use_collector:
            self.no_samplers = size_world - 2
            self.rank_collector = self.no_samplers + 1
        else:
            self.no_samplers = size_world - 1
            self.rank_collector = None
        if self.no_samplers < 1:
            print("Number of MPI processes is too low. Use at least \"mpirun -n 4\".")
        self.sampler_ranks = np.arange(self.no_samplers)  # ranks 0, 1, ..., no_samplers-1
        self.solver_pool_rank = self.no_samplers  # rank no_smplers

    def set_from_dict(self, conf_dict: dict = None, conf_dict_path: str = None) -> None:
        """
        Input: conf_dict or path to yaml/json file have to be specified.
        """
        if conf_dict is None:
            with open(conf_dict_path) as f:
                conf_dict = yaml.safe_load(f)

        for key, value in conf_dict.items():
            setattr(self, key, value)

        self._append_path()

# LIKELIHOOD TODO
        if "noise_model" in conf_dict.keys():
            noise_cov = Gaussian_process.assemble_covariance_matrix(
                conf_dict["noise_model"])
            self.problem_parameters["noise_std"] = noise_cov

    def _append_path(self):
        for path in self.paths_to_append:
            sys.path.append(path)
