#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:32:40 2021

@author: simona
"""

from mpi4py import MPI
import os
import sys
# sys.path.append(os.getcwd())
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ruamel.yaml as yaml
# import importlib.util as iu
from surrDAMH.modules import classes_communication
from surrDAMH.modules import Gaussian_process
from surrDAMH.modules import transformations as trans
import numpy as np


class Configuration:
    def __init__(self, output_dir: str, use_surrogate=True):
        self.output_dir = output_dir

        comm_world = MPI.COMM_WORLD
        size_world = comm_world.Get_size()

        self.use_surrogate = use_surrogate
        if use_surrogate:
            self.no_samplers = size_world - 2
            self.rank_collector = self.no_samplers + 1
        else:
            self.no_samplers = size_world - 1
        if self.no_samplers < 1:
            print("Number of MPI processes is too low. Use at least \"mpirun -n 4\".")
        self.solver_parent_rank = self.no_samplers

    def set_from_file(self, conf_path):
        # TODO: input - dict or file with dict

        # MODEL PROBLEM CHOICE:
        basename = os.path.basename(conf_path)
        conf_name, fext = os.path.splitext(basename)
        with open(conf_path) as f:
            conf = yaml.safe_load(f)

# NECESSARY PARAMETERS
        self.no_parameters = conf["no_parameters"]
        # length of the vector of observations, not repeated observations
        self.no_observations = conf["no_observations"]
        self.no_solvers = conf["no_solvers"]

# NON-NECESSARY PARAMETERS (DEFAULT SPECIFIED)
        if "solver_returns_tag" in conf.keys():
            self.solver_returns_tag = conf["solver_returns_tag"]
        else:
            self.solver_returns_tag = False

        if "pickled_observations" in conf.keys():
            self.pickled_observations = conf["pickled_observations"]
        else:
            self.pickled_observations = True

        if "solver_maxprocs" in conf.keys():
            self.solver_maxprocs = conf["solver_maxprocs"]
        else:
            self.solver_maxprocs = 1

# SOLVER SPECIFICATION:
        if "paths_to_append" in conf.keys():
            for path in conf["paths_to_append"]:
                sys.path.append(path)

# LIKELIHOOD
        if "noise_model" in conf.keys():
            noise_cov = Gaussian_process.assemble_covariance_matrix(
                conf["noise_model"])
            self.problem_parameters["noise_std"] = noise_cov

# LIST OF STAGES (STAGES LIST):
        self.list_alg = conf["samplers_list"]
        for a in self.list_alg:
            if "max_samples" not in a.keys():
                a["max_samples"] = sys.maxsize
            if "max_evaluations" not in a.keys():
                a["max_evaluations"] = sys.maxsize
            if "time_limit" not in a.keys():
                a["time_limit"] = np.inf
        if "initial_sample_type" in conf.keys():
            self.initial_sample_type = conf["initial_sample_type"]
        else:
            self.initial_sample_type = "lhs"

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
        self.max_buffer_size = 1 << 30
        
        self.debug = False
        if "debug" in conf.keys():
            self.debug = conf["debug"]
        if "save_raw_data" in conf.keys():
            self.save_raw_data = conf["save_raw_data"]
        else:
            self.save_raw_data = False
        # if "save_transformed_data" in conf.keys():
        #     self.save_transformed_data = conf["save_transformed_data"]
        # else:
        #     self.save_transformed_data = True
        # if self.save_transformed_data:
        #     self.transform_before_saving = self.transform2
        # else:
        #     self.transform_before_saving = None
        self.transform_before_saving = True

        # defaultní hodnoty nastavit v initu, v této funkci kdyžta kpřepsat ze souboru
