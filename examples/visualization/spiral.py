#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import importlib.util as iu
import os
import sys
import json
import numpy as np
import scipy
import matplotlib.pyplot as plt

wdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 
# wdir = os.getcwd() 
sys.path.append(wdir)
from surrDAMH.modules import visualization_and_analysis as va

saved_samples_name = "spiral"
conf_name = "spiral"
conf_path = wdir + "/examples/" + conf_name + ".json"

with open(conf_path) as f:
    conf = json.load(f)

if len(sys.argv)>1:
    no_samplers = int(sys.argv[1]) # number of MH/DAMH chains
else:
    no_samplers = 30

### OBSERVATION OPERATOR
spec = iu.spec_from_file_location(conf["solver_module_name"], wdir + "/" + conf["solver_module_path"])
solver_module = iu.module_from_spec(spec)
spec.loader.exec_module(solver_module)
solver_init = getattr(solver_module, conf["solver_init_name"])     
solver_instance = solver_init(**conf["solver_parameters"])
def G(u1,u2):
    solver_instance.set_parameters(np.array([u1,u2]))
    yy = solver_instance.get_observations()
    return yy
# def contour(u1,y):
#     u2 = np.log(1/(-80*y-3/np.exp(u1)))
#     return u2

prior_mean = conf["problem_parameters"]["prior_mean"]
prior_cov = conf["problem_parameters"]["prior_std"]
offset = 5*np.pi
N = 200
res = np.zeros((N,N))
likelihood = np.zeros((N,N))
prior = np.zeros((N,N))
y = conf["problem_parameters"]["observations"][0]
f_eta = scipy.stats.norm(loc = 0, scale = conf["problem_parameters"]["noise_std"])
f_prior = scipy.stats.multivariate_normal(mean = prior_mean, cov = prior_cov)
x1 = np.linspace(prior_mean[0]-offset,prior_mean[0]+offset,N)
x2 = np.linspace(prior_mean[1]-offset,prior_mean[1]+offset,N)
for idi,i in enumerate(x1):
    for idj,j in enumerate(x2):
        Gij = G(i,j)
        res[idj,idi] = Gij
        likelihood[idj,idi] = f_eta.pdf(y-Gij)
        prior[idj,idi] = f_prior.pdf([i,j])
posterior = prior*likelihood/292.70
fontsize = 14
plt.rcParams['font.size'] = '11'
plt.figure()
plt.imshow(res, origin="lower", extent = [prior_mean[0]-offset,prior_mean[0]+offset,prior_mean[1]-offset,prior_mean[1]+offset],cmap="inferno")
x1 = np.concatenate((x1,[3.625]))
x1.sort()
contour_u2 = np.zeros((N+1,))
# for idi,i in enumerate(x1):
#     contour_u2[idi] = contour(i,y)
plt.plot(x1,contour_u2,label="contour line $G(u)=y$")
plt.ylim([prior_mean[1]-offset,prior_mean[1]+offset])
plt.colorbar()
plt.xlabel("$u_1$", fontsize=fontsize)
plt.ylabel("$u_2$", fontsize=fontsize)
plt.legend()
plt.grid()
plt.show()
for matrix in [likelihood, prior, posterior]:
    plt.figure()
    plt.imshow(matrix, origin="lower", extent = [prior_mean[0]-offset,prior_mean[0]+offset,prior_mean[1]-offset,prior_mean[1]+offset],cmap="Greys")
    plt.colorbar()
    plt.xlabel("$u_1$", fontsize=fontsize)
    plt.ylabel("$u_2$", fontsize=fontsize)
    plt.grid()
    plt.show()

### SAMPLES VISUALIZATION:
folder_samples = wdir + "/saved_samples/" + saved_samples_name
S = va.Samples()
no_parameters = conf["no_parameters"]
S.load_notes(folder_samples,no_samplers)
S.load_MH(folder_samples,no_parameters)

# Which part of the sampling process is analyzed? 0/1/2 = MH/DAMH-SMU/DAMH
setnumber = 0;
S.calculate_properties()
S.print_properties()
chains_disp = range(setnumber*no_samplers,(setnumber+1)*no_samplers)
S.plot_hist_grid(chains_disp = chains_disp, bins1d=30, bins2d=30)
S.plot_average(chains_disp = chains_disp, show_legend = True)

# chain illustration 
plt.figure()
S.plot_raw_data(folder_samples, no_parameters, chains_range=[0])