#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import os
import sys
import json
sys.path.append("/home/domesova/GIT/Endorse-2Dtest-Bayes/MCMC-Bayes-python/")
from surrDAMH.modules import visualization_and_analysis as va
import matplotlib.pyplot as plt
import numpy as np

no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
conf_path = sys.argv[2]
basename = os.path.basename(conf_path)
problem_name, fext = os.path.splitext(basename)
output_dir = sys.argv[3] + "/"
visualization_dir = output_dir + 'saved_samples/' + problem_name + '/img_prior_outputs'

with open(conf_path) as f:
    conf = json.load(f)

if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)

### PREPARATION:
S = va.Samples()
S.no_chains = no_samplers
no_parameters = conf["no_parameters"]
observations = np.array(conf["problem_parameters"]["observations"])
grid=np.array(conf["noise_grid"])
titles = ["HGT1-5", "HGT1-4", "HGT2-4", "HGT2-3"]
for i in range(4):
    offset = i*len(grid)
    S.hist_G(output_dir + 'saved_samples/' + problem_name + '/prior_outputs',no_parameters, grid, observations, offset+np.arange(len(grid)))
    plt.title(titles[i])
    plt.savefig(visualization_dir + "/prior_G" + str(i+1) + ".pdf",bbox_inches="tight")
    S.show_extremes(output_dir + 'saved_samples/' + problem_name + '/prior_outputs',no_parameters, offset+np.arange(len(grid)))
    plt.savefig(visualization_dir + "/min" + str(i+1) + ".pdf",bbox_inches="tight")
    
S.show_non_converging(output_dir + 'saved_samples/' + problem_name + '/prior_outputs',no_parameters)
plt.savefig(visualization_dir + "/non_converging.pdf",bbox_inches="tight")
