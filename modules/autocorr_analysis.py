#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:01:04 2020

@author: simona
"""

# Autocorrelation analysis using emcee, Foreman-Mackey,
# adapted for the needs of the DAMH-SMU framework.
# Considers "N = no_chains" chains of different lengths (l_1, ..., l_N), 
# each of the chains has "n = no_parameters" components.
# The samples in one chain form a numpy array of shape (l_i, n).
# All samples form a python list of length N.

import numpy as np
import emcee
#import time
import matplotlib.pyplot as plt
#from scipy.optimize import minimize
import pandas as pd
from os import listdir
from os.path import isfile, join

class Samples:
    def __init__(self, samples = None):
        # samples ... list of numpy arrays of shape (l_i, n)
        self.x = samples
        self.known_autocorr_time = False
        
    def calculate_properties(self):
        x = self.x
        self.no_chains = len(x)
        all_chains = range(self.no_chains)
        self.no_parameters = x[0].shape[1]
        self.length = list(x[i].shape[0] for i in all_chains)
        self.var = list(np.var(x[i],axis=0) for i in all_chains)
        self.std = list(np.std(x[i],axis=0) for i in all_chains)
        self.mean = list(np.mean(x[i],axis=0) for i in all_chains)
        self.xp = list(x[i] - self.mean[i] for i in all_chains)
#        mean_all = np.zeros(self.no_parameters)
#        for i in all_chains:
#            mean_all += self.mean[i] * self.length[i]
#        self.mean_all = mean_all / sum(self.length)
            
    def print_properties(self):
        print('known autocorr. time:', self.known_autocorr_time)
        if self.known_autocorr_time:
            print('true autocorr. time:', self.autocorr_time_true)
        print('number of chains:', self.no_chains)
        print('number of parameters:',self.no_parameters)
        print('length:',self.length)
        print('mean:')
        print(self.mean)
        print('std:')
        print(self.std)
        
    def plot_segment(self, begin_disp, end_disp, parameters_disp = None, chains_disp = None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if begin_disp == None:
            begin_disp = [0] * len(parameters_disp)
        if end_disp == None:
            end_disp = [max(self.length)] * len(parameters_disp)
        fig, axes = plt.subplots(1, len(parameters_disp), figsize=(12, 3), sharey=True)
        for idj,j in enumerate(parameters_disp):
            begin_disp[idj] = min(max(self.length),begin_disp[idj])
            end_disp[idj] = min(max(self.length),end_disp[idj])
            for idi,i in enumerate(chains_disp):
                xx = np.arange(begin_disp[idj],min(end_disp[idj],self.length[i]))
                yy = self.x[i][xx,j]
                axes[idj].plot(xx, yy, label=i)
            axes[idj].set_xlim(begin_disp[idj], end_disp[idj]-1)
            axes[idj].legend()
            axes[idj].set_xlabel("$par. {0}$".format(j))
            if self.known_autocorr_time:
                axes[idj].set_title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]));
        axes[0].set_ylabel("samples")
        plt.show()
        
    def plot_hist(self, burn_in, parameters_disp = None, chains_disp = None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        fig, axes = plt.subplots(1, len(parameters_disp), figsize=(12, 3), sharey=True)
        for idj,j in enumerate(parameters_disp):
            for idi,i in enumerate(chains_disp):
                yy = self.x[i][burn_in[idi]:,j]
                axes[idj].hist(yy, label=i)
            axes[idj].legend()
            axes[idj].set_xlabel("$parameter:  {0}$".format(j))
            if self.known_autocorr_time:
                axes[idj].set_title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]));
        axes[0].set_ylabel("samples")
        plt.show()
        
    def plot_average(self, burn_in, begin_disp = None, end_disp = None, parameters_disp = None, chains_disp = None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if begin_disp == None:
            begin_disp = [0] * len(parameters_disp)
        if end_disp == None:
            end_disp = [max(self.length)] * len(parameters_disp)
        fig, axes = plt.subplots(1, len(parameters_disp), figsize=(12, 3), sharey=True)
        for idj,j in enumerate(parameters_disp):
            for idi,i in enumerate(chains_disp):
                xx = np.arange(burn_in[idi],min(end_disp[idj],self.length[i]))
                yy = self.x[i][xx,j]
                axes[idj].plot(xx, np.cumsum(yy)/(1+np.arange(len(yy))), label=i)
            axes[idj].legend()
            axes[idj].set_xlabel("$parameter:  {0}$".format(j))
            if self.known_autocorr_time:
                axes[idj].set_title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]));
        axes[0].set_ylabel("samples")
        plt.show()
        
    def load_MH(self, folder_samples):
        file_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
        file_samples.sort()
        N = len(file_samples)
        self.x = [None] * N
        for i in range(N):
            path_samples = folder_samples + "/" + file_samples[i]
            print(path_samples)
            df_samples = pd.read_csv(path_samples, header=None)
            weights = np.array(df_samples[0])
            tmp = np.array(df_samples.iloc[:,1:])
            self.x[i] = decompress(tmp, weights)
        
    def generate_samples_rand(self,no_parameters,length):
        # no_parameters ... scalar
        # length ... list of scalars l_1, ..., l_N
        no_chains = len(length)
        all_chains = range(no_chains)
        self.x = list(np.random.rand(length[i],no_parameters) for i in all_chains)
        
    def generate_samples_celerite(self, length=[1000], log_c1=[-6.0], log_c2=[-2.0]):
        # length ... list of scalars l_1, ..., l_N
        # log_c1, log_c2 ... list of scalars (len = no_parameters)
        no_chains = len(length)
        no_parameters = len(log_c1)
        # from https://dfm.io/posts/autocorr/ Foreman-Mackey
        # Build the celerite model:
        import celerite
        from celerite import terms
        true_tau = [None] * no_parameters
        gp_samples = [None] * no_parameters
        t = np.arange(max(length))
        for i in range(no_parameters):
            kernel = terms.RealTerm(log_a=0.0, log_c=log_c1[i])
            kernel += terms.RealTerm(log_a=0.0, log_c=log_c2[i])
            # The true autocorrelation time can be calculated analytically:
            true_tau[i] = sum(2*np.exp(t.log_a-t.log_c) for t in kernel.terms)
            true_tau[i] /= sum(np.exp(t.log_a) for t in kernel.terms)
            # Simulate a set of chains:
            gp = celerite.GP(kernel)
            gp.compute(t)
            gp_samples[i] = gp.sample(size=no_chains)
        self.known_autocorr_time = True
        self.autocorr_time_true = true_tau
        x = [None] * no_chains
        for i in range(no_chains):
            x[i] = np.zeros((length[i],no_parameters))
            for j in range(no_parameters):
                x[i][:,j] = gp_samples[j][i,:length[i]]
        self.x = x

    def calculate_autocorr_function(self):
        self.autocorr_function = [None] * self.no_chains
        for i in range(self.no_chains):
            tmp = np.zeros((self.length[i],self.no_parameters))
            for j in range(self.no_parameters):
                tmp[:,j] = emcee.autocorr.function_1d(self.x[i][:,j])
            self.autocorr_function[i] = tmp
            
    def calculate_autocorr_function_mean(self):
        max_length = max(self.length)
        self.autocorr_function_mean = np.zeros((max_length, self.no_parameters))
        for j in range(self.no_parameters):
            tmp = np.zeros(max_length)
            count = np.zeros(max_length)
            for i in range(self.no_chains):
                tmp[:self.length[i]] += self.autocorr_function[i][:,j]
                count[:self.length[i]] += 1
            self.autocorr_function_mean[:,j] = tmp/count
            
    def plot_autocorr_function(self, length_disp, plot_mean=False):
        fig, axes = plt.subplots(1, self.no_parameters, figsize=(12, 3), sharey=True)
        for j in range(self.no_parameters):
            length_disp[j] = min(max(self.length),length_disp[j])
        for j in range(self.no_parameters):
            for i in range(self.no_chains):
                axes[j].plot(self.autocorr_function[i][:length_disp[j],j], label=i)
            if plot_mean:
                axes[j].plot(self.autocorr_function_mean[:,j],label="mean")
            axes[j].set_xlim(0, length_disp[j]-1)
            axes[j].legend()
            axes[j].set_xlabel("$parameter:  {0}$".format(j))
            if self.known_autocorr_time:
                axes[j].set_title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]));
        axes[0].set_ylabel("autocorr. function")
        plt.show()
    
    def calculate_autocorr_time(self, c=5, tol=50, quiet=True):
        self.autocorr_time = [None] * self.no_chains
        for i in range(self.no_chains):
            tmp = np.zeros((self.no_parameters))
            for j in range(self.no_parameters):
                tmp[j] = emcee.autocorr.integrated_time(self.x[i][:,j], c=c, tol=tol, quiet=quiet)
            self.autocorr_time[i] = tmp
            
    def calculate_autocorr_time_mean(self, c=5):
        self.autocorr_time_mean = [None] * self.no_parameters
        self.autocorr_time_mean_beta = [None] * self.no_parameters
        min_length = min(self.length)
        for j in range(self.no_parameters):
            f = self.autocorr_function_mean[:min_length,j]
            self.autocorr_time_mean[j] = autocorr_new(f, c)
            f = self.autocorr_function_mean[:,j]
            self.autocorr_time_mean_beta[j] = autocorr_new(f, c)
            
# Automated windowing procedure following Sokal (1989)
# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def autocorr_new(f, c=5.0):
    # first calculates all autocorr. functions, than averages them
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

def decompress(x, w):
    # x ... "compressed" samples from DAMH-SMU
    # w ... weights
    sum_w = np.sum(w)
    cumsum_w = np.append(0,np.cumsum(w))
    no_unique, no_parameters = x.shape
    xd = np.zeros((sum_w,no_parameters))
    for i in range(no_unique):
        xd[cumsum_w[i]:cumsum_w[i+1],:] = x[i,:]
    return xd