#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import os
import sys
import ruamel.yaml as yaml
#sys.path.append(os.getcwd())
#from surrDAMH.surrDAMH.modules import visualization_and_analysis as va
import matplotlib.pyplot as plt
import numpy as np

import importlib.util as iu
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
path = os.path.dirname(path)
path = os.path.join(path,"surrDAMH/modules/visualization_and_analysis.py")
spec = iu.spec_from_file_location("visualization_and_analysis",path)
va = iu.module_from_spec(spec)
spec.loader.exec_module(va)

no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
conf_path = sys.argv[2]
basename = os.path.basename(conf_path)
problem_name, fext = os.path.splitext(basename)
output_dir = os.path.join(sys.argv[3], 'saved_samples', problem_name)
visualization_dir = os.path.join(output_dir, 'img_Bayes')

with open(conf_path) as f:
    conf = yaml.safe_load(f)

if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)

### PREPARATION:
S = va.Samples()
no_parameters = conf["no_parameters"]
scale = ["linear"]*no_parameters
if "transformations" in conf.keys():
    transformations = conf["transformations"]
    for i in range(no_parameters):
        if transformations[i]["type"] == "normal_to_lognormal":
            scale[i] = "log"
else:
    transformations = [None]*no_parameters
S.load_notes(output_dir,no_samplers)
S.load_MH(output_dir,no_parameters)
S.calculate_properties()
S.load_MH_with_posterior(output_dir,no_parameters)

output_dict = S.get_properties(no_samplers)
title = ",".join([str(i) for i in S.notes[0].columns.values])
len_sampler_list = len(output_dict["samplers_list"])
for idx,d in enumerate(output_dict["samplers_list"]):
    d[title] = S.notes[idx].values.tolist()
    d.update(conf["samplers_list"][idx])

mode = S.find_modus()
output_dict["mode"] = mode[0].tolist()

cov_type = None
if "noise_cov_type" in conf.keys():
    cov_type = conf["noise_cov_type"]

observations = np.array(conf["problem_parameters"]["observations"])

### SAMPLES VISUALIZATION:
no_stages = int(S.no_chains/no_samplers)
for i in range(no_stages):
    chains_disp=range(i*no_samplers,(i+1)*no_samplers)
    S.plot_hist_grid(chains_disp=chains_disp, bins1d=None, bins2d=None, scale=scale)
    S.plot_hist_grid_add(transformations,chains_disp=chains_disp, scale=scale)
    plt.savefig(visualization_dir + "/histograms" +str(i)+ ".pdf",bbox_inches="tight")
    S.plot_segment(chains_disp=chains_disp,scale=scale)
    plt.savefig(visualization_dir + "/chains" +str(i)+ ".pdf",bbox_inches="tight")
    S.plot_average(chains_disp=chains_disp,scale=scale)
    plt.savefig(visualization_dir + "/average" +str(i)+ ".pdf",bbox_inches="tight")

if "save_raw_data" in conf.keys():
    if conf["save_raw_data"]:
        no_observations = conf["no_observations"]
        S.hist_G(output_dir + '/raw_data',no_parameters, observations, np.arange(no_observations), range(no_samplers*len_sampler_list))
        plt.savefig(visualization_dir + "/hist_G.pdf",bbox_inches="tight")

with open(os.path.join(output_dir, "output.yaml"), 'w') as f:
    yaml.dump(output_dict, f, default_flow_style=None)#, allow_unicode=True)