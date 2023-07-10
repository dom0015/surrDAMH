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

### SAMPLES VISUALIZATION:
no_stages = int(S.no_chains/no_samplers)
tau_all = [None] * no_stages
tau2_all = [None] * no_stages
for i in range(no_stages):
    chains_disp=range(i*no_samplers,(i+1)*no_samplers)
    print(chains_disp)
    S.calculate_autocorr_function(chains_range=chains_disp)
    print([len(S.autocorr_function[j]) for j in chains_disp])
    S.calculate_autocorr_function_mean(chains_range=chains_disp)
    [tau,tau2] = S.calculate_autocorr_time_mean(chains_range=chains_disp)
    tau_all[i] = tau
    tau2_all[i] = tau2
    
    # plot samples
    # l=len(S.x[chains_disp[0]])
    # m=min(l,10000)
    # plt.figure()
    # plt.scatter(S.x[chains_disp[0]][:m,0],S.x[chains_disp[0]][:m,1],0.2)
    # plt.grid()
    # plt.xlabel("$u_1$")
    # plt.ylabel("$u_2$")
    # ax = plt.gca()
    # ax.set(xlim=(-6, 6), ylim=(-6, 6))
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig(visualization_dir + "/samples" +str(i)+ ".pdf",bbox_inches="tight")

print(S.notes[0].columns)
data = np.empty((0,len(S.notes[0].columns)-1))
for i in range(no_stages):    
    tmp = np.mean(S.notes[i])
    data = np.vstack((data,tmp[:-1]))
print(data)
plt.figure()
#0 1 10 11 12 2 3 4 5 6 7 8 9
#idx_DAMH = [2,4,6,8,10,12]#[1,3,5,7,9]
idx_DAMH = list(range(1,no_stages))
print(idx_DAMH)
xx = np.arange(0.4,5.2,0.2)
plt.plot(data[idx_DAMH,0]/data[idx_DAMH,3])
acc=(data[idx_DAMH,0]+data[idx_DAMH,1])/data[idx_DAMH,3]
plt.plot(acc)
plt.title("acceptance rate")
plt.figure()
plt.plot(data[idx_DAMH,1]/data[idx_DAMH,3])
plt.title("rejected/all")
plt.figure()

print(tau_all)
tau_all =  np.array(tau_all)
tau = tau_all
tau2_all = np.array(tau2_all)
plt.plot(xx,tau_all[idx_DAMH,:])
plt.plot(xx,tau2_all[idx_DAMH,:])
plt.title("tau")

tau_all = np.mean(np.array(tau_all),axis=1)
tau2_all = np.mean(np.array(tau2_all),axis=1)
plt.figure()
plt.plot(acc*tau_all[idx_DAMH])
plt.plot(acc*tau2_all[idx_DAMH])
plt.title("acc*tau")
plt.show()


print("xx=",xx.tolist())
print("accepted=", data[idx_DAMH,0].tolist())
print("rejected=", data[idx_DAMH,1].tolist())
print("prerejected=", data[idx_DAMH,2].tolist())
print("sum_all=", data[idx_DAMH,3].tolist())
print("tau1=", tau[idx_DAMH,0].tolist())
print("tau2=", tau[idx_DAMH,1].tolist())