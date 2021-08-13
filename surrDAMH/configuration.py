#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:32:40 2021

@author: simona
"""

import os
import sys
sys.path.append(os.getcwd())
import json
import importlib.util as iu
from modules import classes_communication
from modules import Gaussian_process

class Configuration:
    def __init__(self, no_samplers, conf_path):
        
### MODEL PROBLEM CHOICE:
        basename = os.path.basename(conf_path)
        conf_name, fext = os.path.splitext(basename)
        with open(conf_path) as f:
            conf = json.load(f)
    
### PROBLEM PARAMETERS:
        if "problem_name" in conf.keys():
            self.problem_name = conf["problem_name"]
        else:
            self.problem_name = conf_name
        self.no_parameters = conf["no_parameters"]
        self.no_observations = conf["no_observations"] # length of the vector of observations, not repeated observations
        self.problem_parameters = conf["problem_parameters"]
        noise_type = None
        if "noise_type" in conf.keys():
            noise_type = conf["noise_type"]
        if noise_type == "Gaussian_process":
            grid = conf["noise_grid"]
            parameters = conf["noise_parameters"]
            cov_type = None
            if "noise_cov_type" in conf.keys():
                cov_type = conf["noise_cov_type"]
            noise_cov = Gaussian_process.assemble_covariance_matrix(grid, parameters, cov_type)
            self.problem_parameters["noise_std"] = noise_cov

            
### SOLVER SPECIFICATION:
        if "paths_to_append" in conf.keys():
            for path in conf["paths_to_append"]:
                sys.path.append(path)
        spec = iu.spec_from_file_location(conf["solver_module_name"], conf["solver_module_path"])
        module = iu.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.child_solver_init = getattr(module, conf["solver_init_name"]) 
        self.child_solver_parameters = conf["solver_parameters"]
        if "solver_returns_tag" in conf.keys():
            self.solver_returns_tag = conf["solver_returns_tag"]
        else:
            self.solver_returns_tag = False
        if "pickled_observations" in conf.keys():
            self.pickled_observations = conf["pickled_observations"]
        else:
            self.pickled_observations = True
        if "save_raw_data" in conf.keys():
            self.save_raw_data = conf["save_raw_data"]
        else:
            self.save_raw_data = False
                
### SAMPLING PARAMETERS:
        self.no_full_solvers = conf["no_solvers"]
        self.no_samplers = no_samplers
        self.list_alg = conf["samplers_list"]

### SURROGATE MODEL SPECIFICATION:
        if "surrogate_type" in conf.keys():
            self.use_surrogate = True
            self.rank_surr_collector = self.no_samplers + 1
            if conf["surrogate_type"] == "rbf": # radial basis functions surrogate model
                from modules import surrogate_rbf as surr
            else: # polynomial surrogate model
                from modules import surrogate_poly as surr
            self.surr_solver_init = surr.Surrogate_apply
            self.surr_updater_init = surr.Surrogate_update
            self.surr_solver_parameters = {"no_parameters": self.no_parameters,
                                            "no_observations": self.no_observations
                                            }
            if "surr_solver_parameters" in conf.keys():
                self.surr_solver_parameters.update(conf["surr_solver_parameters"])
            self.surr_updater_parameters = {"no_parameters": self.no_parameters,
                                            "no_observations": self.no_observations
                                            }
            if "surr_updater_parameters" in conf.keys():
                self.surr_updater_parameters.update(conf["surr_updater_parameters"])
        else:
            self.use_surrogate = False
        
### OTHER SETTINGS:
        # COMMUNICATION TYPE 1 (solvers are spawned):
        self.solver_parent_init = classes_communication.Solver_MPI_parent
        tmp = {"no_parameters": self.no_parameters,
               "no_observations": self.no_observations,
               "problem_path": conf_path,
               "no_samplers": no_samplers,
               "pickled_observations": self.pickled_observations
               }
        if "solver_parent_parameters" in conf.keys():
            tmp.update(conf["solver_parent_parameters"])
        self.solver_parent_parameters = []
        for idx in range(self.no_full_solvers): # deflation test
            self.solver_parent_parameters.append(tmp.copy())
            self.solver_parent_parameters[idx]["solver_id"] = idx 
        self.max_buffer_size = 1<<30
        self.solver_parent_rank = self.no_samplers
        self.debug = False
        if "debug" in conf.keys():
            self.debug = conf["debug"]