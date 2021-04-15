# -*- coding: utf-8 -*-



import os
import ruamel.yaml as yaml

from flow123d_simulation import endorse_2Dtest

class Wrapper:
    def __init__(self):
        work_dir = "/home/simona/GIT/Endorse-2Dtest-Bayes/flow123d_sim"
        # Create working directory if necessary
        os.makedirs(work_dir, mode=0o775, exist_ok=True)
    
        # read config file and setup paths
        with open("/home/simona/GIT/Endorse-2Dtest-Bayes/config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict["work_dir"] = work_dir
        config_dict["script_dir"] = "/home/simona/GIT/Endorse-2Dtest-Bayes"
        config_dict["_aux_flow_path"] = config_dict["local"]["flow_executable"].copy()
        config_dict["_aux_gmsh_path"] = config_dict["local"]["gmsh_executable"].copy()
        self.sim = endorse_2Dtest(config_dict, clean=True)
        
    def set_parameters(self,data_par):
        self.sim.set_parameters(data_par)
        
    def get_observations(self):
        res = self.sim.get_observations()
        print(res)
        return res[-4:]