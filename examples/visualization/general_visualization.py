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

no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
conf_path = sys.argv[2]
basename = os.path.basename(conf_path)
problem_name, fext = os.path.splitext(basename)
output_dir = sys.argv[3]
visualization_dir = output_dir + 'saved_samples/' + problem_name + '/img_Bayes'

with open(conf_path) as f:
    conf = json.load(f)

if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)

### PREPARATION:
S = va.Samples()
no_parameters = conf["no_parameters"]
scale = ["linear"]*no_parameters
if "transformations" in conf.keys():
    transformations = conf["transformations"]
    for i in range(no_parameters):
        if transformations[i][0] == "normal_to_lognormal":
            scale[i] = "log"
S.load_notes(output_dir + 'saved_samples/' + problem_name,no_samplers)
S.load_MH(output_dir + 'saved_samples/' + problem_name,no_parameters)
S.calculate_properties()
S.load_MH_with_posterior(output_dir + 'saved_samples/' + problem_name,no_parameters)
print("-----")
print("Generates " + output_dir + "saved_samples/" + problem_name + "/output.txt.")
sys.stdout = open(output_dir + "saved_samples/" + problem_name + "/output.txt", "w")
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
# fit = S.find_best_fit(output_dir + 'saved_samples/' + problem_name,no_parameters,conf["problem_parameters"]["observations"])
# print("BEST FIT (L2)")
# print(" - PARAMETERS:", list(fit[0]))
# print(" - PARAMETERS (log10):", list(np.log10(fit[0])))
# print(" - OUTPUT:", list(fit[1]))

n=int(conf["no_observations"]/4)
grid=np.array(conf["noise_grid"])
grid_max = max(grid)+35

from surrDAMH.modules import Gaussian_process
cov_type = None
if "noise_cov_type" in conf.keys():
    cov_type = conf["noise_cov_type"]
noise_cov = Gaussian_process.assemble_covariance_matrix(grid, conf["noise_parameters"], cov_type)
# fit_likelihood = S.find_max_likelihood(output_dir + 'saved_samples/' + problem_name,no_parameters,conf["problem_parameters"]["observations"],noise_cov=noise_cov,scale=scale,disp_parameters=[0,1])
# plt.savefig(output_dir + 'saved_samples/' + problem_name + "/img_Bayes/log_likelihood.pdf",bbox_inches="tight")
# print("-----")
# print("BEST FIT (likelihood)")
# print(" - PARAMETERS:", list(fit_likelihood[0]))
# print(" - PARAMETERS (log10):", list(np.log10(fit_likelihood[0])))
# print(" - OUTPUT:", list(fit_likelihood[1]))

observations = np.array(conf["problem_parameters"]["observations"])
# plt.figure()
# for i in range(4):
#     idx=np.arange(n)+i*n
#     plt.plot(grid+i*grid_max,observations[idx],'k')
#     plt.plot(grid+i*grid_max,fit[1][idx],'b')
#     plt.plot(grid+i*grid_max,fit_likelihood[1][idx],'r')
#     if i==0:
#         plt.legend(["observations","best fit (L2)","best fit (likelihood)"])
# plt.grid()
# plt.savefig(visualization_dir + "/best_fit.pdf",bbox_inches="tight")

# fig, axes = plt.subplots(4, 1, figsize=(8, 8))
# for i in range(4):
#     idx=np.arange(n)+i*n
#     axes[i].plot(grid,observations[idx],'k')
#     axes[i].plot(grid,fit_likelihood[1][idx],'r')
#     #axes[i].plot(grid,fit[1][idx],'b')
#     if i==0:
#         axes[0].plot(grid,[300.,528.57076623,712.45437966,714.20995391,713.79667764,711.05357012,706.01356353,698.86811795,689.91018224,679.47744857,667.91006061,665.55351444,640.1023436,613.29924228,586.7780633,561.30572678,537.28346584,514.88299781,494.13460616,474.98884731,457.35429182,441.12044079,426.171219,412.39261503,399.67675066,396.62259926],'g')
#         #axes[i].legend(["observations","output with highest likelihood","output with highest L2 norm","parameters according to Rutqvist"])
#         axes[i].legend(["observations","output with highest likelihood","parameters according to Rutqvist"])
#     axes[i].grid()
# axes[1].plot(grid,[300.        ,  435.19561839,
#         544.01965537,  545.31661875,  545.89270459,  546.09562866,
#         546.03306675,  545.71367351,  545.11018328,  544.18837689,
#         542.92055589,  542.65272154,  539.04407856,  533.83826774,
#         527.38026636,  519.91290944,  511.69074519,  502.95170257,
#         493.90018856,  484.70382465,  475.49494912,  466.37474826,
#         457.41825381,  448.67926664,  440.19476769,  438.10756843],'g')
# axes[2].plot(grid,[300.        ,   72.83352837, -103.67171304,  -92.52875936,
#         -85.71899517,  -80.56162396,  -75.72925698,  -70.60796324,
#         -64.96106105,  -58.74771513,  -52.02411855,  -50.65924429,
#         -36.03012715,  -20.55372369,   -5.11611855,    9.80298782,
#          23.9151355 ,   37.06817734,   49.20461607,   60.32672805,
#          70.47281996,   79.7012137 ,   88.07979256,   95.67939361,
#         102.56978957,  104.20904701],'g')
# axes[3].plot(grid,[300.        ,  199.64235674,
#         119.72130569,  120.60409153,  121.23456015,  121.63613268,
#         121.87482482,  122.01525917,  122.11108623,  122.20475735,
#         122.32884308,  122.35526692,  122.77209053,  123.56265715,
#         124.73206831,  126.26874304,  128.13624646,  130.28486068,
#         132.66153939,  135.2144577 ,  137.89582861,  140.66324426,
#         143.48010051,  146.31548259,  149.14375766,  149.84821152],'g')
# plt.savefig(visualization_dir + "/best_fit2.pdf",bbox_inches="tight")


# plt.figure()
# for i in range(4):
#     idx=np.arange(n)+i*n
#     plt.plot(grid,observations[idx])
# plt.grid()
# plt.savefig(visualization_dir + "/observations.pdf",bbox_inches="tight")

### SAMPLES VISUALIZATION:
no_stages = int(S.no_chains/no_samplers)
for i in range(no_stages):
    chains_disp=range(i*no_samplers,(i+1)*no_samplers)
    S.plot_hist_grid(chains_disp=chains_disp, bins1d=9, bins2d=20, scale=scale)
    S.plot_hist_grid_add(transformations,chains_disp=chains_disp, scale=scale)
    plt.savefig(visualization_dir + "/histograms" +str(i)+ ".pdf",bbox_inches="tight")
    S.plot_segment(chains_disp=chains_disp,scale=scale)
    plt.savefig(visualization_dir + "/chains" +str(i)+ ".pdf",bbox_inches="tight")
    S.plot_average(chains_disp=chains_disp,scale=scale)
    plt.savefig(visualization_dir + "/average" +str(i)+ ".pdf",bbox_inches="tight")
    # S.plot_dots(chains_disp=chains_disp, scale=scale)
    # plt.savefig(visualization_dir + "/dots" +str(i)+ ".pdf",bbox_inches="tight")

# chains_disp=range(20,60)
# S.plot_hist_grid(chains_disp=chains_disp, bins1d=10, bins2d=20, scale=scale)
# S.plot_hist_grid_add(transformations,chains_disp=chains_disp, scale=scale)
# plt.savefig(visualization_dir + "/histograms2_3.pdf",bbox_inches="tight")

# plt.figure()
# plt.imshow(noise_cov)
# plt.colorbar()
# plt.savefig(visualization_dir + "/noise_cov.pdf",bbox_inches="tight")
sys.stdout.close()

# titles = ["HGT1-5", "HGT1-4", "HGT2-4", "HGT2-3"]
# for i in range(4):
#     offset = i*len(grid)
#     S.hist_G(output_dir + 'saved_samples/' + problem_name + '/raw_data',no_parameters, grid, observations, offset+np.arange(len(grid)), range(30,90))
#     plt.title(titles[i])
#     plt.savefig(visualization_dir + "/hist_G" + str(i+1) + ".pdf",bbox_inches="tight")
