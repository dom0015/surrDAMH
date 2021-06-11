#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:23:44 2021

@author: domesova
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:02:19 2021

@author: simona
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import importlib.util as iu
# import os
# import sys
# import petsc4py
# import csv

def operation(x):
    #return x
    #minimum = np.min(x,axis=0)
    up = np.quantile(x,0.95,axis=0)
    m = np.mean(x,axis=0)
    down = np.quantile(x,0.05,axis=0)
    #maximum = np.max(x,axis=0)
    return np.array([up,m,down])


plt.rcParams['font.size'] = '16'
fontsize = 20
markersize = 12
linewidth = 3

PROC = 60
problem_name = "deflation_interfaces"
problem_name = "deflation_interfaces_sigma0_5"
problem_name = "deflation_interfaces_sigma1_0"
#problem_name = "deflation_interfaces_sigma2_0"
problem_name = "deflation_grf_sigma0_5deflation_grf"
#problem_name = "deflation_grf_sigma1_0"
#problem_name = "deflation_grf_sigma2_0"

for problem_name in ["deflation_interfaces_sigma0_5",
                     "deflation_interfaces_sigma1_0",
                     "deflation_interfaces_sigma2_0",
                     "deflation_grf_sigma0_5deflation_grf",
                     "deflation_grf_sigma1_0",
                     "deflation_grf_sigma2_0"]:

    # W0 = []
    # iterations0 = []
    # # times0 = []
    # # errors0 = []
    # for seed in range(PROC):
    #     filename = "saved_tests/" + problem_name + "/data_without" + str(seed) + ".csv"
    #     data = pd.read_csv(filename)#, header=None)
    #     W0.append(np.array(data["W"]))
    #     tmp_iter = np.array(data["iter"])
    #     iterations0.append(tmp_iter)
    #     # times0.append(np.array(data["time"])/tmp_iter)
    #     # errors0.append(np.array(data["error"]))
    #     #tmp = np.array(df_samples.iloc[:,1:1+no_parameters])
    
    W = []
    iterations = []
    # times = []
    # errors = []
    for seed in range(PROC):
        filename = "saved_tests/" + problem_name + "/data_with" + str(seed) + ".csv"
        data = pd.read_csv(filename)#, header=None)
        tmp_W = np.array(data["W"])
        W.append(tmp_W)
        tmp_iter = np.array(data["iter"])
        iterations.append(tmp_iter)
        # times.append(np.array(data["time"])/tmp_iter)
        # errors.append(np.array(data["error"]))
        #tmp = np.array(df_samples.iloc[:,1:1+no_parameters])
    
    # plt.figure(); plt.title("W")
    # plt.plot(operation(np.array(W0)).transpose(),'b')
    # plt.plot(operation(np.array(W)).transpose(),'r')
    
    # plt.figure(); plt.title("iter")
    # plt.plot(operation(np.array(iterations0)).transpose(),'b')
    # plt.plot(operation(np.array(iterations)).transpose(),'r')
    
    # plt.figure(); plt.title("time")
    # plt.plot(operation(np.array(times0)).transpose(),'b')
    # plt.plot(operation(np.array(times)).transpose(),'r')
    
    # plt.figure(); plt.title("error")
    # plt.plot(operation(np.array(errors0)).transpose(),'b')
    # plt.plot(operation(np.array(errors)).transpose(),'r')
    
    ## FINAL IMAGES:
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
    # plt.figure()
    # plt.plot(W[0],color=colors[0])
    # plt.plot(iterations[0],color=colors[1])
    # plt.legend(["deflation basis size","number of iterations"])
    # plt.xlabel("number of available solution vectors")
    # plt.tight_layout()
    # plt.xlim([0,150])
    # plt.ylim([0,150])
    # plt.show()
    # plt.grid()
    # plt.savefig('examples/visualization/img/' + problem_name + '_1.pdf')  
    
    plt.figure()
    plt.plot(np.mean(W,axis=0),color=colors[0])
    plt.plot(np.mean(iterations,axis=0),color=colors[1])
    plt.legend(["deflation basis size","number of iterations"])
    plt.plot(np.quantile(W,0.95,axis=0),'--',color=colors[0])
    plt.plot(np.quantile(iterations,0.95,axis=0),'--',color=colors[1])
    plt.plot(np.quantile(W,0.05,axis=0),'--',color=colors[0])
    plt.plot(np.quantile(iterations,0.05,axis=0),'--',color=colors[1],)
    plt.xlabel("number of available solution vectors")
    plt.tight_layout()
    plt.xlim([0,150])
    plt.ylim([0,150])
    plt.show()
    plt.grid()
    plt.savefig('examples/visualization/img/' + problem_name + '_60.pdf')  
    
    # def show_data(data):
    #     print("residual_norm:")
    #     print(data[:,3])
    #     print("comp_time")
    #     print(data[:,2])
    #     print("W size:")
    #     print(data[:,0])
    #     print("iterations:")
    #     print(data[:,1])
    #     plt.figure()
    #     plt.plot(data[:,0])
    #     plt.plot(data[:,1])
    #     plt.plot(data[:,2]*1e3)
    #     plt.plot(data[:,3]*1e7)
    #     plt.legend(["W","iter","time*1e3","error*1e7"])
    #     plt.show()
    
    
    # filename = "saved_tests/deflation_interfaces/data_without" + str(seed) + ".csv"
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    # labels = ["W","iter","time","error"]
    # file = open(filename, 'w')
    # writer = csv.writer(file)
    # writer.writerow(labels)
    # writer.writerows(data_without)#.tolist())
    # file.close()
    
colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
PROC = 30
max_len = 300
for problem_name in ["interfaces_test1_deflation_1e-4",
                     "interfaces_test1_deflation_1e-6",
                     "grf_test1_deflation_1e-4"]:

    W = []
    iterations = []
    errors = []
    plt.figure()
    for seed in range(PROC):
        filename = "saved_test/" + problem_name + "/deflation" + str(seed) + ".csv"
        data = pd.read_csv(filename)#, header=None)
        tmp_W = np.array(data["W"])
        W.append(tmp_W[:max_len])
        tmp_iter = np.array(data["iter"])
        iterations.append(tmp_iter[:max_len])
        tmp_error = np.array(data["error"])
        errors.append(tmp_error[:max_len])
        plt.plot(tmp_W,color=colors[0])
        plt.plot(tmp_iter,color=colors[1])

    plt.figure()
    plt.plot(np.mean(W,axis=0),color=colors[0])
    plt.plot(np.mean(iterations,axis=0),color=colors[1])
    plt.legend(["deflation basis size","number of iterations"])
    plt.plot(np.quantile(W,0.90,axis=0),'--',color=colors[0])
    plt.plot(np.quantile(iterations,0.90,axis=0),'--',color=colors[1])
    plt.plot(np.quantile(W,0.10,axis=0),'--',color=colors[0])
    plt.plot(np.quantile(iterations,0.10,axis=0),'--',color=colors[1],)
    plt.xlabel("number of available solution vectors")
    plt.tight_layout()
    plt.xlim([0,max_len])
    plt.ylim([0,110])
    plt.show()
    plt.grid()
    #plt.savefig('examples/visualization/img/' + problem_name + '_60.pdf')  
