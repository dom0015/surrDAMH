#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import os
import sys
import ruamel.yaml as yaml

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from surrDAMH.modules import visualization_and_analysis as va
from surrDAMH.modules import analysis as ape
import matplotlib.pyplot as plt
import numpy as np

no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
conf_path = sys.argv[2]
basename = os.path.basename(conf_path)
problem_name, fext = os.path.splitext(basename)
output_dir = os.path.join(sys.argv[3], 'saved_samples', problem_name)
raw_data_dir = os.path.join(output_dir, 'raw_data')
visualization_dir = os.path.join(output_dir, 'img_Bayes')

with open(conf_path) as f:
    conf = yaml.safe_load(f)

no_stages = len(conf["samplers_list"])
# parameter chains_disp selects samples from 2. and 3. stage
chains_disp = range(no_samplers, no_samplers * no_stages)
print("chains_disp", chains_disp)

observations = np.array(conf["problem_parameters"]["observations"])

par_names = [p["name"] for p in conf["transformations"]]
print("parameters:", par_names)

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
S.load_notes(output_dir,no_samplers)
S.load_MH(output_dir,no_parameters)
S.calculate_properties()
S.load_MH_with_posterior(output_dir,no_parameters)

# output_file = os.path.join(output_dir, "output.txt")
# sys.stdout = open(output_file, "w")

output_dict = S.get_properties(no_samplers)
#samplers_list = conf["samplers_list"]
title = ",".join([str(i) for i in S.notes[0].columns.values])
for idx,d in enumerate(output_dict["samplers_list"]):
    count_list = np.array(S.notes[idx].values.tolist())
    count_list = count_list.sum(axis=0)

    d[title] = S.notes[idx].values.tolist()
    d["acceptance ratio [a/r, a/all]"] = np.array([count_list[0]/count_list[1], count_list[0]/count_list[3]]).tolist()
    d["N samples [a, r, pr, all]"] = np.array(
        [count_list[0], count_list[1], count_list[2], count_list[3]]).tolist()
    d.update(conf["samplers_list"][idx])

mode = S.find_modus()
output_dict["mode"] = mode[0].tolist()


raw_data = ape.RawData()
raw_data.load(output_dir, no_parameters, len(observations))
# type: 0-accepted, 1-prerejected, 2-rejected
raw_data_filtered = raw_data.filter(types=[0,2], stages=range(no_stages+1))

analysis_pe = ape.Analysis(config=conf, raw_data=raw_data_filtered)
bestfit_L2, bestfit_L2_norm = analysis_pe.find_best_fit(observations, norm="L2")
# fit_L2_x, fit_L2_G, fit_L2_norm = S.find_best_fit(raw_data_dir, no_parameters, observations)
output_dict["best_fit_L2"] = {"type": bestfit_L2.type_name(),
                              "parameters": bestfit_L2.parameters().tolist(),
                              "parameters_log": np.log(bestfit_L2.parameters()).tolist(),
                              "parameters_log10": np.log10(bestfit_L2.parameters()).tolist()}
print("-----")
print('BEST FIT (L2)')
print(" - NORM:", bestfit_L2_norm)
print(" - PARAMETERS:", bestfit_L2.parameters())
print(" - OBSERVATIONS:", bestfit_L2.observations())

# fits, norms = raw_data_filtered.find_n_best_fits(observations, count=10, norm="L2")
# print(norms)

from surrDAMH.modules import Gaussian_process
noise_cov = Gaussian_process.assemble_covariance_matrix(conf["noise_model"])

bestfit_LH, bestfit_LH_norm = analysis_pe.find_best_fit(observations, norm="likelihood", noise_cov=noise_cov)
output_dict["best_fit_LH"] = {"type": bestfit_LH.type_name(),
                              "parameters": bestfit_LH.parameters().tolist(),
                              "parameters_log": np.log(bestfit_LH.parameters()).tolist(),
                              "parameters_log10": np.log10(bestfit_LH.parameters()).tolist()}
print("-----")
print('BEST FIT (LikelyHood)')
print(" - NORM:", bestfit_LH_norm)
print(" - PARAMETERS:", bestfit_LH.parameters())
print(" - OBSERVATIONS:", bestfit_LH.observations())

# fits, norms = raw_data_filtered.find_n_best_fits(observations, count=10, norm="likelihood", noise_cov=noise_cov)
# print(norms)

visual = ape.Visualization(conf, raw_data_filtered)
# fig, axes = plt.subplots(1,1)
# visual.plot_likelihood_ij(axes, 0, 1)
# # fig.colorbar(extend="min")
# fig.savefig(os.path.join(visualization_dir, "likelihood_" + str(0) + "_" + str(1) + ".pdf"))
fig, axes = plt.subplots(no_parameters, no_parameters, figsize=(25,25))
plt.subplots_adjust(wspace=0.5, hspace=0.3)
visual.plot_likelihood(fig, axes)
fig.savefig(os.path.join(visualization_dir, "likelihood_all" + ".jpg"))

# grid=np.array(conf["noise_grid"])
# grid_max = max(grid)+35
# fit_likelihood = S.find_max_likelihood(output_dir, no_parameters, conf["problem_parameters"]["observations"],
#                                        noise_cov=noise_cov, scale=scale, disp_parameters=[0,1])
# plt.savefig(visualization_dir + "/log_likelihood.pdf", bbox_inches="tight")
# print("-----")
# print("BEST FIT (likelihood)")
# print(" - PARAMETERS:", list(fit_likelihood[0]))
# print(" - PARAMETERS (log10):", list(np.log10(fit_likelihood[0])))
# print(" - OUTPUT:", list(fit_likelihood[1]))

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

raw_data_accepted = raw_data.filter(types=[0], stages=range(1,no_stages+1))
analysis_pe_accepted = ape.Analysis(config=conf, raw_data=raw_data_accepted)
estimated_distributions = analysis_pe_accepted.estimate_distributions(transformations,
                output_file=os.path.join(visualization_dir, "parameters2.csv"))
print(estimated_distributions)

# collecting samples in the same way as SB does in hist_G_TSX
# estimated_distributions = S.estimate_distributions(raw_data_dir, transformations,
#                                                    chains_disp=chains_disp,
#                                                    output_file=os.path.join(visualization_dir, "parameters.csv"))
# print(estimated_distributions)
### SAMPLES VISUALIZATION:
for i in range(no_stages):
    chains_disp=range(i*no_samplers,(i+1)*no_samplers)

    fig, axes = S.plot_hist_grid(par_names=par_names, chains_disp=chains_disp, bins1d=15, bins2d=20, scale=scale)
    S.plot_hist_grid_add(axes, transformations, estimated_distributions, chains_disp=chains_disp, scale=scale)
    fig.savefig(visualization_dir + "/histograms" +str(i)+ ".pdf",bbox_inches="tight")

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

pressure_boreholes = conf["observe_points"]
print("observe points:", pressure_boreholes)
N = len(pressure_boreholes)
noise_model_list = conf["noise_model"]
for i in range(N):
    # offset = i*len(grid)
    noise_model = noise_model_list[i]
    time_grid = np.array(noise_model["time_grid"])
    offsets = noise_model["range"]
    S.hist_G_TSX(raw_data_dir, no_parameters, time_grid, observations, np.arange(*offsets), chains_disp)
    plt.title(pressure_boreholes[i])
    plt.savefig(visualization_dir + "/hist_G" + str(i+1) + ".pdf",bbox_inches="tight")

S.show_non_converging(raw_data_dir, no_parameters)
plt.savefig(visualization_dir + "/non_converging.png",bbox_inches="tight")

output_dict["estimated_distributions"] = estimated_distributions
with open(os.path.join(output_dir, "output.yaml"), 'w') as f:
    yaml.dump(output_dict, f, default_flow_style=None)