#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:32:40 2021

@author: simona
"""

from modules import classes_communication
import numpy as np
import json
import importlib.util as iu

class Configuration:
    def __init__(self, no_samplers, conf_name = None):
        
### MODEL PROBLEM CHOICE:
        if conf_name is None:
            conf_name = "simple_MPI"
        conf_path = "conf/" + conf_name + ".json"
        with open(conf_path) as f:
            conf = json.load(f)
            
        # if "modules_to_load" in conf.keys():
        #     import importlib.util as iu
        #     for path, name in conf["modules_to_load"]:
        #         spec = iu.spec_from_file_location(name, path)
        #         mod = importlib.util.module_from_spec(spec)
        #         spec.loader.exec_module(mod)
    
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
            import sys
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
            else:
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
        self.max_buffer_size = 1<<30 #20s
        self.solver_parent_rank = self.no_samplers
        self.rank_surr_collector = self.no_samplers + 1
        
        # TYPE 1 - solvers are spawned:
        self.solver_parent_init = classes_communication.Solver_MPI_parent
        tmp = {"no_parameters": self.no_parameters,
               "no_observations": self.no_observations,
               "problem_name": conf_name,
               "no_samplers": no_samplers
               }
        if "solver_parent_parameters" in conf.keys():
            tmp.update(conf["solver_parent_parameters"])
        self.solver_parent_parameters = [tmp] * self.no_full_solvers
        # for i in range(self.no_full_solvers):
        #     self.solver_parent_parameters.append({'no_parameters':self.no_parameters,
        #                                         'no_observations':self.no_observations,
        #                                         'maxprocs':5})