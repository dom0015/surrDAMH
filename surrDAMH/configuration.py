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

class Configuration:
    def __init__(self, no_samplers, conf_name = None):
        
### MODEL PROBLEM CHOICE:
        if conf_name is None:
            conf_name = "simple_MPI"
        conf_path = "examples/" + conf_name + ".json"
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
            
### SOLVER SPECIFICATION:
        if "paths_to_append" in conf.keys():
            for path in conf["paths_to_append"]:
                sys.path.append(path)
        spec = iu.spec_from_file_location(conf["solver_module_name"], conf["solver_module_path"])
        module = iu.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.child_solver_init = getattr(module, conf["solver_init_name"]) 
        self.child_solver_parameters = conf["solver_parameters"]
                
### SAMPLING PARAMETERS:
        self.no_full_solvers = conf["no_solvers"]
        self.no_samplers = no_samplers
        self.list_alg = conf["samplers_list"]

### SURROGATE MODEL SPECIFICATION:
        if "surrogate_type" in conf.keys():
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
        
### OTHER SETTINGS:
        # COMMUNICATION TYPE 1 (solvers are spawned):
        self.solver_parent_init = classes_communication.Solver_MPI_parent
        tmp = {"no_parameters": self.no_parameters,
               "no_observations": self.no_observations,
               "problem_name": conf_name,
               "no_samplers": no_samplers
               }
        if "solver_parent_parameters" in conf.keys():
            tmp.update(conf["solver_parent_parameters"])
        self.solver_parent_parameters = [tmp] * self.no_full_solvers
        
        self.max_buffer_size = 1<<30
        self.solver_parent_rank = self.no_samplers
        self.rank_surr_collector = self.no_samplers + 1
