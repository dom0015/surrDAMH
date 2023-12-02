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
import numpy as np
from typing import Literal


class Configuration:
    def __init__(self, output_dir: str, no_parameters: int, no_observations: int,
                 use_surrogate: bool = True,
                 no_solvers: int = 2,
                 solver_maxprocs: int = 1,
                 solver_returns_tag: bool = False,
                 pickled_observations: bool = True,
                 save_raw_data: bool = False,
                 transform_before_saving: bool = True,
                 initial_sample_type: Literal["lhs", "prior_mean"] = "lhs",
                 debug: bool = False,
                 max_buffer_size=1 << 30) -> None:
        self.output_dir = output_dir
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.use_surrogate = use_surrogate
        self.no_solvers = no_solvers
        self.solver_maxprocs = solver_maxprocs
        self.solver_returns_tag = solver_returns_tag
        self.pickled_observations = pickled_observations
        self.save_raw_data = save_raw_data
        self.transform_before_saving = transform_before_saving
        self.initial_sample_type = initial_sample_type
        self.debug = debug
        self.max_buffer_size = max_buffer_size

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

        # NECESSARY PARAMETERS (no default value):
        self.no_parameters = conf_dict["no_parameters"]
        self.no_observations = conf_dict["no_observations"]

        # NON-NECESSARY PARAMETERS (default value specified in init):
        if "no_solvers" in conf_dict.keys():
            self.no_solvers = conf_dict["no_solvers"]
        if "solver_maxprocs" in conf_dict.keys():
            self.solver_maxprocs = conf_dict["solver_maxprocs"]
        if "solver_returns_tag" in conf_dict.keys():
            self.solver_returns_tag = conf_dict["solver_returns_tag"]
        if "pickled_observations" in conf_dict.keys():
            self.pickled_observations = conf_dict["pickled_observations"]
        if "save_raw_data" in conf_dict.keys():
            self.save_raw_data = conf_dict["save_raw_data"]
        if "transform_before_saving" in conf_dict.keys():
            self.transform_before_saving = conf_dict["transform_before_saving"]
        if "initial_sample_type" in conf_dict.keys():
            self.initial_sample_type = conf_dict["initial_sample_type"]
        if "debug" in conf_dict.keys():
            self.debug = conf_dict["debug"]
        if "max_buffer_size" in conf_dict.keys():
            self.max_buffer_size = conf_dict["max_buffer_size"]

    # OTHER NON-NECESSARY SETTINGS:
        if "paths_to_append" in conf_dict.keys():
            for path in conf_dict["paths_to_append"]:
                sys.path.append(path)

# LIKELIHOOD
        if "noise_model" in conf_dict.keys():
            noise_cov = Gaussian_process.assemble_covariance_matrix(
                conf_dict["noise_model"])
            self.problem_parameters["noise_std"] = noise_cov

# LIST OF STAGES (STAGES LIST):
        self.list_alg = conf_dict["samplers_list"]
        for a in self.list_alg:
            if "max_samples" not in a.keys():
                a["max_samples"] = sys.maxsize
            if "max_evaluations" not in a.keys():
                a["max_evaluations"] = sys.maxsize
            if "time_limit" not in a.keys():
                a["time_limit"] = np.inf


# SURROGATE MODEL SPECIFICATION:
        # if "surrogate_type" in conf.keys():
        #     self.surrogate_type = conf["surrogate_type"]
        # if "surrogate_module_name" in conf.keys():
        #     self.surrogate_module_name = conf["surrogate_module_name"]
        #     self.surrogate_module_path = conf["surrogate_module_path"]
        # self.surr_updater_parameters = {"no_parameters": self.no_parameters,
        #                                 "no_observations": self.no_observations
        #                                 }
        # if "surr_updater_parameters" in conf.keys():
        #     self.surr_updater_parameters.update(conf["surr_updater_parameters"])
        # if "surrogate_type" in conf.keys():
        #     # self.use_surrogate = True
        #     self.rank_surr_collector = self.no_samplers + 1
        #     if conf["surrogate_type"] == "rbf":  # radial basis functions surrogate model
        #         from surrDAMH.surrogates import radial_basis as surr
        #     elif conf["surrogate_type"] == "poly":  # polynomial surrogate model
        #         from surrDAMH.surrogates import polynomial_sklearn as surr
        #     else:
        #         # spec = iu.spec_from_file_location(
        #         #    conf["surrogate_module_name"], conf["surrogate_module_path"])
        #         spec = iu.spec_from_file_location("surrogates.polynomial_sklearn", "surrDAMH/surrogates/polynomial_sklearn.py")
        #         surr = iu.module_from_spec(spec)
        #         spec.loader.exec_module(surr)
        #     self.surr_updater_init = surr.PolynomialTrainer
        #     self.surr_solver_parameters = {"no_parameters": self.no_parameters,
        #                                    "no_observations": self.no_observations
        #                                    }
        #     if "surr_solver_parameters" in conf.keys():
        #         self.surr_solver_parameters.update(
        #             conf["surr_solver_parameters"])
        #     self.surr_updater_parameters = {"no_parameters": self.no_parameters,
        #                                     "no_observations": self.no_observations
        #                                     }
        #     if "surr_updater_parameters" in conf.keys():
        #         self.surr_updater_parameters.update(
        #             conf["surr_updater_parameters"])
        # # else:
        # #     self.use_surrogate = False

# OTHER SETTINGS:
        # COMMUNICATION TYPE 1 (solvers are spawned):
        # self.solver_parent_init = classes_communication.Solver_MPI_parent
        # tmp = {"no_parameters": self.no_parameters,
        #        "no_observations": self.no_observations,
        #        "problem_path": conf_path,
        #        "no_samplers": self.no_samplers,
        #        "pickled_observations": self.pickled_observations
        #        }
        # if "solver_parent_parameters" in conf.keys():
        #     tmp.update(conf["solver_parent_parameters"])
        # self.solver_parent_parameters = []
        # for idx in range(self.no_solvers):  # deflation test
        #     self.solver_parent_parameters.append(tmp.copy())
        #     self.solver_parent_parameters[idx]["solver_id"] = idx

        # if "save_transformed_data" in conf.keys():
        #     self.save_transformed_data = conf["save_transformed_data"]
        # else:
        #     self.save_transformed_data = True
        # if self.save_transformed_data:
        #     self.transform_before_saving = self.transform2
        # else:
        #     self.transform_before_saving = None
