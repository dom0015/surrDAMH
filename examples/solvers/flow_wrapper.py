# -*- coding: utf-8 -*-



import os
import ruamel.yaml as yaml
import modules.transformations as trans
import numpy as np

from flow123d_simulation import endorse_2Dtest

class Wrapper:
    def __init__(self, solver_id = 0):
        work_dir = "/home/domesova/GIT/Endorse-2Dtest-Bayes/flow123d_sim"
        # Create working directory if necessary
        os.makedirs(work_dir, mode=0o775, exist_ok=True)
    
        # read config file and setup paths
        with open("/home/domesova/GIT/Endorse-2Dtest-Bayes/config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict["work_dir"] = work_dir
        config_dict["script_dir"] = "/home/domesova/GIT/Endorse-2Dtest-Bayes"
        config_dict["_aux_flow_path"] = config_dict["local"]["flow_executable"].copy()
        config_dict["_aux_gmsh_path"] = config_dict["local"]["gmsh_executable"].copy()
        config_dict["common_files_dir"] = "/home/domesova/GIT/Endorse-2Dtest-Bayes"
        self.sim = endorse_2Dtest(config_dict, clean=True)
        self.no_parameters = 2
        
    def set_parameters(self,data_par):
        conductivity = trans.normal_to_lognormal(data_par[0])
        biot = trans.normal_to_beta(data_par[1],alfa=5,beta=5)
        self.sim.set_parameters(np.array([conductivity,biot]))
        
    def get_observations(self):
        res = self.sim.get_observations()
        return res