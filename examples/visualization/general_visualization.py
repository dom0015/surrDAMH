#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import os
import sys
import json
sys.path.append(os.getcwd())
from surrDAMH.modules import visualization_and_analysis as va
import matplotlib.pyplot as plt
import numpy as np

conf_path = sys.argv[2]
basename = os.path.basename(conf_path)
problem_name, fext = os.path.splitext(basename)

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 1

if not os.path.exists('saved_samples/' + problem_name + '/img_Bayes'):
    os.makedirs('saved_samples/' + problem_name + '/img_Bayes')

### PREPARATION:
S = va.Samples()
no_parameters = conf["no_parameters"]
scale = ["linear"]*no_parameters
if "transformations" in conf.keys():
    transformations = conf["transformations"]
    for i in range(no_parameters):
        if transformations[i][0] == "normal_to_lognormal":
            scale[i] = "log"
S.load_notes('saved_samples/' + problem_name,no_samplers)
S.load_MH('saved_samples/' + problem_name,no_parameters)
S.calculate_properties()
S.load_MH_with_posterior('saved_samples/' + problem_name,no_parameters)
print("-----")
print("Generates saved_samples/" + problem_name + "/output.txt.")
sys.stdout = open("saved_samples/" + problem_name + "/output.txt", "w")
print("PHASE 1 (MH):")
print(S.notes[0])
print("PHASE 2 (DAMH-SMU):")
print(S.notes[1])
print("PHASE 3 (DAMH):")
print(S.notes[2])
print("-----")
S.print_properties(no_samplers)
print("-----")
mode = S.find_modus()
print("MODE: ", list(mode[0]))
print("-----")
fit = S.find_best_fit('saved_samples/' + problem_name,no_parameters,conf["problem_parameters"]["observations"])
print("BEST FIT (L2)")
print(" - PARAMETERS:", list(fit[0]))
print(" - OUTPUT:", list(fit[1]))

n=int(conf["no_observations"]/4)
grid=np.array(conf["noise_grid"])
grid_max = max(grid)+35

from surrDAMH.modules import Gaussian_process
cov_type = None
if "noise_cov_type" in conf.keys():
    cov_type = conf["noise_cov_type"]
noise_cov = Gaussian_process.assemble_covariance_matrix(grid, conf["noise_parameters"], cov_type)
fit_likelihood = S.find_max_likelihood('saved_samples/' + problem_name,no_parameters,conf["problem_parameters"]["observations"],noise_cov=noise_cov)
plt.savefig('saved_samples/' + problem_name + "/img_Bayes/log_likelihood.pdf",bbox_inches="tight")
print("-----")
print("BEST FIT (likelihood)")
print(" - PARAMETERS:", list(fit_likelihood[0]))
print(" - OUTPUT:", list(fit_likelihood[1]))

observations = np.array(conf["problem_parameters"]["observations"])
plt.figure()
for i in range(4):
    idx=np.arange(n)+i*n
    plt.plot(grid+i*grid_max,observations[idx],'k')
    plt.plot(grid+i*grid_max,fit[1][idx],'b')
    plt.plot(grid+i*grid_max,fit_likelihood[1][idx],'r')
    if i==0:
        plt.legend(["observations","best fit (L2)","best fit (likelihood)"])
plt.grid()
plt.savefig('saved_samples/' + problem_name + "/img_Bayes/best_fit.pdf",bbox_inches="tight")

### SAMPLES VISUALIZATION:
no_stages = int(S.no_chains/no_samplers)
for i in range(no_stages):
    chains_disp=range(i*no_samplers,(i+1)*no_samplers)
    S.plot_hist_grid(chains_disp=chains_disp, bins1d=30, bins2d=30, scale=scale)
    S.plot_hist_grid_add(transformations,chains_disp=chains_disp, scale=scale)
    plt.savefig('saved_samples/' + problem_name + "/img_Bayes/histograms" +str(i)+ ".pdf",bbox_inches="tight")
    S.plot_segment(chains_disp=chains_disp,scale=scale)
    plt.savefig('saved_samples/' + problem_name + "/img_Bayes/chains" +str(i)+ ".pdf",bbox_inches="tight")
    S.plot_average(chains_disp=chains_disp,scale=scale)
    plt.savefig('saved_samples/' + problem_name + "/img_Bayes/average" +str(i)+ ".pdf",bbox_inches="tight")

plt.figure()
plt.imshow(noise_cov)
plt.colorbar()
plt.savefig('saved_samples/' + problem_name + "/img_Bayes/noise_cov.pdf",bbox_inches="tight")
sys.stdout.close()