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


@dataclass
class Configuration:
    output_dir: str
    no_parameters: int
    no_observations: int
    use_surrogate: bool = True
    no_solvers: int = 2
    solver_maxprocs: int = 1
    solver_returns_tag: bool = False
    pickled_observations: bool = True
    save_raw_data: bool = False
    transform_before_saving: bool = True
    initial_sample_type: Literal["lhs", "prior_mean"] = "lhs"
    debug: bool = False
    max_buffer_size: int = 1 << 30
    paths_to_append: list[str] = None

    def __post_init__(self) -> None:
        if self.paths_to_append is None:
            self.paths_to_append = []
        else:
            self._append_path()

        comm_world = MPI.COMM_WORLD
        size_world = comm_world.Get_size()

        if self.use_surrogate:
            self.no_samplers = size_world - 2
            self.rank_collector = self.no_samplers + 1
        else:
            self.no_samplers = size_world - 1
        if self.no_samplers < 1:
            print("Number of MPI processes is too low. Use at least \"mpirun -n 4\".")
        self.solver_parent_rank = self.no_samplers

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
