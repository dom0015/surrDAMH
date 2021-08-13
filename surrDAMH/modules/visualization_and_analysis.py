#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:01:04 2020

@author: simona
"""

import numpy as np
#import emcee
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join, getsize
import surrDAMH.modules.grf_eigenfunctions as grf_eigenfunctions

class Samples:
    def __init__(self, samples = None):
        # samples ... list of numpy arrays of shape (l_i, n)
        self.x = samples
        self.known_autocorr_time = False
        
    def load_MH(self, folder_samples, no_parameters):
        folder_samples = folder_samples + '/data'
        file_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
        file_samples.sort()
        N = len(file_samples)
        self.x = [None] * N
        for i in range(N):
            path_samples = folder_samples + "/" + file_samples[i]
            df_samples = pd.read_csv(path_samples, header=None)
            weights = np.array(df_samples[0])
            tmp = np.array(df_samples.iloc[:,1:1+no_parameters])
            self.x[i] = decompress(tmp, weights)
        
    def load_MH_with_posterior(self, folder_samples, no_parameters, surrogate_posterior = False):
        folder_samples = folder_samples + '/data'
        file_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
        file_samples.sort()
        N = len(file_samples)
        self.x_compress = [None] * N
        self.posteriors = [None] * N
        if surrogate_posterior:
            self.surrogate_posteriors = [None] * N
        for i in range(N):
            path_samples = folder_samples + "/" + file_samples[i]
            df_samples = pd.read_csv(path_samples, header=None)
            self.x_compress[i] = np.array(df_samples.iloc[:,1:1+no_parameters])
            self.posteriors[i] = np.array(df_samples.iloc[:,1+no_parameters])
            if surrogate_posterior:
                self.surrogate_posteriors[i] = np.array(df_samples.iloc[:,2+no_parameters])
            
    def load_notes(self, folder_samples, no_samplers):
        folder = folder_samples + '/notes'
        file_samples = [f for f in listdir(folder) if isfile(join(folder, f))]
        file_samples.sort()
        N = int(len(file_samples)/no_samplers)
        self.notes = [pd.DataFrame()] * N
        for n in range(N):
            for i in range(no_samplers):
                path_samples = folder + "/" + file_samples[i+n*no_samplers]
                data = pd.read_csv(path_samples)
                self.notes[n] = self.notes[n].append(data)
            
    def remove_burn_in(self, burn_in):
        self.x = list(self.x[i][burn_in[i]:,:] for i in range(self.no_chains))

    def extract_chains(self, chains_to_keep):
        self.x = [self.x[i] for i in chains_to_keep]
        self.no_chains = len(chains_to_keep)
        
    def calculate_properties(self, burn_in = None):
        self.no_chains = len(self.x)
        all_chains = range(self.no_chains)
        if burn_in == None:
            burn_in = [0] * self.no_chains
        x = list(self.x[i][burn_in[i]:,:] for i in all_chains)
        self.no_parameters = x[0].shape[1]
        self.length = list(x[i].shape[0] for i in all_chains)
        self.length = np.array(self.length)
        self.var = list(np.var(x[i],axis=0) for i in all_chains)
        self.std = list(np.std(x[i],axis=0) for i in all_chains)
        self.mean = list(np.mean(x[i],axis=0) for i in all_chains)
        self.xp = list(x[i] - self.mean[i] for i in all_chains)

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
        
### BASIC VISUALIZATION OF GENERATED CHAINS:
        
    def plot_segment(self, begin_disp = None, end_disp = None, parameters_disp = None, chains_disp = None, show_legend = False):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if begin_disp == None:
            begin_disp = [0] * len(parameters_disp)
        if end_disp == None:
            end_disp = [max([self.length[i] for i in chains_disp])] * len(parameters_disp)
        fig, axes = plt.subplots(1, len(parameters_disp), figsize=(12, 3), sharey=True)
        for idj,j in enumerate(parameters_disp):
            begin_disp[idj] = min(max(self.length),begin_disp[idj])
            end_disp[idj] = min(max(self.length),end_disp[idj])
            for idi,i in enumerate(chains_disp):
                xx = np.arange(begin_disp[idj],min(end_disp[idj],self.length[i]))
                yy = self.x[i][xx,j]
                axes[idj].plot(xx, yy, label=i)
            axes[idj].set_xlim(begin_disp[idj], end_disp[idj]-1)
            if show_legend:
                axes[idj].legend(loc=1)
            axes[idj].set_xlabel("$par. {0}$".format(j))
            axes[idj].grid(True)
            if self.known_autocorr_time:
                axes[idj].set_title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]));
        axes[0].set_ylabel("samples")
        plt.show()
        
    def plot_average(self, burn_in = None, begin_disp = None, end_disp = None, parameters_disp = None, chains_disp = None, show_legend = False):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if burn_in == None:
            burn_in = [0] * len(chains_disp)
        if begin_disp == None:
            begin_disp = [0] * len(parameters_disp)
        if end_disp == None:
            end_disp = [max([self.length[i] for i in chains_disp])] * len(parameters_disp)
        fig, axes = plt.subplots(1, len(parameters_disp), figsize=(12, 3), sharey=True)
        for idj,j in enumerate(parameters_disp):
            for idi,i in enumerate(chains_disp):
                xx = np.arange(burn_in[idi],min(end_disp[idj],self.length[i]))
                yy = self.x[i][xx,j]
                yy = np.cumsum(yy)/(1+np.arange(len(yy)))
                axes[idj].plot(xx[begin_disp[idj]:], yy[begin_disp[idj]:], label=i)
            if show_legend:
                axes[idj].legend(loc=1)
            axes[idj].set_xlabel("$par.  {0}$".format(j))
            axes[idj].grid(True)
            if self.known_autocorr_time:
                axes[idj].set_title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]));
        axes[0].set_ylabel("convergence of averages")
        plt.show()
        
    def plot_average_reverse(self, burn_in = None, begin_disp = None, end_disp = None, parameters_disp = None, chains_disp = None, show_legend = False):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if burn_in == None:
            burn_in = [0] * len(chains_disp)
        if begin_disp == None:
            begin_disp = [0] * len(parameters_disp)
        if end_disp == None:
            end_disp = [max([self.length[i] for i in chains_disp])] * len(parameters_disp)
        fig, axes = plt.subplots(1, len(parameters_disp), figsize=(12, 3), sharey=True)
        for idj,j in enumerate(parameters_disp):
            for idi,i in enumerate(chains_disp):
                xx = np.arange(burn_in[idi],min(end_disp[idj],self.length[i]))
                yy = np.flip(self.x[i][xx,j])
                yy = np.cumsum(yy)/(1+np.arange(len(yy)))
                axes[idj].plot(xx[begin_disp[idj]:], yy[begin_disp[idj]:], label=i)
            if show_legend:
                axes[idj].legend(loc=1)
            axes[idj].set_xlabel("$par.  {0}$".format(j))
            axes[idj].grid(True)
            if self.known_autocorr_time:
                axes[idj].set_title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]));
        axes[0].set_ylabel("convergence of averages")
        plt.show()
    
### HISTOGRAMS:
        
    def plot_hist(self, burn_in = None, parameters_disp = None, chains_disp = None):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if burn_in == None:
            burn_in = [0] * len(chains_disp)
        fig, axes = plt.subplots(1, len(parameters_disp), figsize=(12, 3), sharey=True)
        for idj,j in enumerate(parameters_disp):
            for idi,i in enumerate(chains_disp):
                yy = self.x[i][burn_in[idi]:,j]
                axes[idj].hist(yy, label=i)
            axes[idj].legend(loc=1)
            axes[idj].set_xlabel("$parameter:  {0}$".format(j))
            axes[idj].grid(True)
            if self.known_autocorr_time:
                axes[idj].set_title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]));
        axes[0].set_ylabel("samples")
        plt.show()

    def plot_hist_1d(self, burn_in = None, dimension = 0, chains_disp = None, bins = 20, show = True):
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if burn_in == None:
            burn_in = [0] * len(chains_disp)
        XX = np.zeros((0,))
        for i, chain in enumerate(chains_disp):
            xx = self.x[chain][burn_in[i]:,dimension]
            XX = np.concatenate((XX,xx))
        plt.hist(XX, bins = bins, density = True)
        plt.grid(True)
        if show:
            plt.show()

    def plot_hist_2d(self, burn_in = None, dimensions = [0,1], chains_disp = None, bins = 20, show = True, colorbar = False):
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if burn_in == None:
            burn_in = [0] * len(chains_disp)
        XX = np.zeros((0,))
        YY = np.zeros((0,))
        for i, chain in enumerate(chains_disp):
            xx = self.x[chain][burn_in[i]:,dimensions[0]]
            yy = self.x[chain][burn_in[i]:,dimensions[1]]
            XX = np.concatenate((XX,xx))
            YY = np.concatenate((YY,yy))
        plt.hist2d(XX, YY, bins = bins, cmap = "binary", density = True)
        plt.grid(True)
        if colorbar:
            plt.colorbar()
        if show:
            plt.show()
    
    def plot_hist_grid(self, burn_in = None, parameters_disp = None, chains_disp = None, bins1d = 20, bins2d = 20):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if burn_in == None:
            burn_in = [0] * len(chains_disp)
        n = len(parameters_disp)
        idx = 1
        fig, axes = plt.subplots(n, n, sharex=True, sharey=True) # figsize=(12,12)
        for idi,i in enumerate(parameters_disp):
            for idj,j in enumerate(parameters_disp):
                plt.subplot(n, n, idx)
                if idi==idj:
                    self.plot_hist_1d(dimension = i, burn_in = burn_in, chains_disp=chains_disp, bins=bins1d, show = False)
                else:
                    self.plot_hist_2d(dimensions = [j,i],  burn_in = burn_in, chains_disp=chains_disp, bins=bins2d, show = False)
                if idx<=n:
                    plt.title("$par. {0}$".format(j))
                idx = idx + 1
        plt.show()

### DAMH ANALYSIS:
    
    def load_accepted(self, folder_samples):
        folder = folder_samples + '/DAMH_accepted'
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        files.sort()
        N = len(files)
        self.accepted = [None] * N
        for i in range(N):
            path = folder + "/" + files[i]
            if getsize(path) > 0:
                data = pd.read_csv(path, header=None)
                self.accepted[i] = np.array(data)

    def load_rejected(self, folder_samples):
        folder = folder_samples + '/DAMH_rejected'
        files = [f for f in listdir(folder) if isfile(join(folder, f))]
        files.sort()
        # print(files)
        N = len(files)
        self.rejected = [None] * N
        for i in range(N):
            path = folder + "/" + files[i]
            if getsize(path) > 0:
                data = pd.read_csv(path, header=None)
                self.rejected[i] = np.array(data)
    
    def merge_evaluated(self):
        N = len(self.accepted)
        self.evaluated = [None] * N
        for i in range(N):
            if self.rejected[i] is None:
                tmp = self.accepted[i]
            else:
                tmp = np.concatenate((self.accepted[i],self.rejected[i]))
            order = np.argsort(tmp[:,0])
            self.evaluated[i] = tmp[order]
            
    def plot_evaluated(self, no_samplers, L=None, title=None):
        if L==None:
            L = self.rejected
        N = len(L)
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        r_max = [0] * no_samplers
        x_max = [0] * no_samplers
        for i in range(no_samplers):
            for j in range(int(N/no_samplers)):
                r = L[i+j*no_samplers][:,0]
                axes.plot(r_max[i]+r,range(x_max[i],x_max[i]+len(r)))
                r_max[i] = int(r[-1])
                x_max[i] = len(r)
        plt.title(title)
        plt.show()
    
    def plot_evaluated_sliding(self, chains_range, no_sw, window_length = None,  L = None):
        if L==None:
            L = self.rejected
        # length = min(list(np.max(L[i][:,0]) for i in chains_range))
        length = min(self.length[list(chains_range)])
        if window_length is None:
            sw_step = np.floor(length/(no_sw+1))
            window_length = sw_step*2
        else:
            sw_step = np.floor((length-window_length)/no_sw)
        print("window length:",window_length)
        no_chains = len(chains_range)
        result = np.zeros((no_sw,no_chains))
        for idx,i in enumerate(chains_range):
            for j in range(no_sw):
                bound_lower = sw_step*j
                bound_upper = sw_step*j + window_length
                r = L[i][:,0]
                first = np.where(r>=bound_lower)
                first_idx = first[0][0]
                last = np.where(r<=bound_upper)
                last_idx = last[0][-1]
                tmp = last_idx-first_idx
                result[j,idx] = result[j,idx] + tmp
        plt.plot(result)
        plt.title(str(chains_range))
        plt.show()
        result_mean = np.mean(result,axis=1)
        plt.plot(result_mean)
        plt.title("evaluations in sliding window - mean")
        plt.show()
        return result
    
    def get_raw_data(self, folder_samples, no_parameters, no_observations, chains_range = None):
        folder_samples = folder_samples + '/raw_data'
        file_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
        file_samples.sort()
        if chains_range == None:
            chains_range = range(self.no_chains)
        N = len(chains_range)
        parameters = [None] * N
        observations = [None] * N
        tag = [None] * N
        sample_type = [None] * N
        for idx,i in enumerate(chains_range):
            path_samples = folder_samples + "/" + file_samples[i]
            df_samples = pd.read_csv(path_samples, header=None, names=list(range(2+no_parameters+no_observations)))
            # print(df_samples.iloc[:6,:6])
            sample_type[idx] = np.array(df_samples.iloc[:,0])
            parameters[idx] = np.array(df_samples.iloc[:,1:1+no_parameters])
            tag[idx] = np.array(df_samples.iloc[:,1+no_parameters])
            # if np.sum(df_samples.iloc[:,1+no_parameters]) == np.nan:
            #     print(df_samples)
            observations[idx] = np.array(df_samples.iloc[:,2+no_parameters:])
        return sample_type, parameters, tag, observations

    def analyze_raw_data(self, folder_samples, no_parameters, no_observations, par0 = 0, par1 = 1, chains_range = None, begin_disp = 0, end_disp = None):
        folder_samples = folder_samples + '/raw_data'
        file_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
        file_samples.sort()
        if chains_range == None:
            chains_range = range(self.no_chains)
        N = len(chains_range)
        raw_data = [None] * N
        sample_type = [None] * N
        # plt.figure()
        for idx,i in enumerate(chains_range):
            path_samples = folder_samples + "/" + file_samples[i]
            #print("PATH:", path_samples)
            df_samples = pd.read_csv(path_samples, header=None, names=list(range(2+no_parameters+no_observations)))
            raw_data[idx] = np.array(df_samples.iloc[:,1:1+no_parameters])
            sample_type[idx] = np.array(df_samples.iloc[:,0])
            sample0 = self.x[i][begin_disp,:]
            if end_disp == None:
                end_disp_max = raw_data[idx].shape[0]
            for j in range(begin_disp, end_disp_max):
                #print(raw_data[idx].shape)
                sample1 = raw_data[idx][j,:]
                if sample_type[idx][j] == "accepted":
                    sample0=sample1
                elif sample_type[idx][j] == "prerejected":
                    plt.plot([sample0[par0],sample1[par0]],[sample0[par1],sample1[par1]], linewidth=1, color="silver")
                    #plt.plot(sample1[par0],sample1[par1],'.', color="silver")
            sample0 = self.x[i][begin_disp,:]
            if end_disp == None:
                end_disp_max = raw_data[idx].shape[0]
            for j in range(begin_disp, end_disp_max):
                sample1 = raw_data[idx][j,:]
                if sample_type[idx][j] == "accepted":
                    sample0=sample1
                elif sample_type[idx][j] == "rejected":
                    plt.plot([sample0[par0],sample1[par0]],[sample0[par1],sample1[par1]], linewidth=1, color="tab:orange")
                    #plt.plot(sample1[par0],sample1[par1],'.', color="tab:orange")
            sample0 = self.x[i][begin_disp,:]
            if end_disp == None:
                end_disp_max = raw_data[idx].shape[0]
            for j in range(begin_disp, end_disp_max):
                sample1 = raw_data[idx][j,:]
                if sample_type[idx][j] == "accepted":
                    plt.plot([sample0[par0],sample1[par0]],[sample0[par1],sample1[par1]], color="tab:blue", linewidth=1)
                    #plt.plot(sample1[par0],sample1[par1],'.', color="tab:blue")
                    sample0=sample1
            sample0 = self.x[i][begin_disp,:]
            plt.plot(sample0[par0], sample0[par1], '.', color="tab:red", markersize=10)
        plt.xlabel("$u_1$")
        plt.ylabel("$u_2$")
        plt.grid()
        self.raw_data = raw_data

### AUTOCORRELATION:
# Autocorrelation analysis using emcee, Foreman-Mackey,
# adapted for the needs of the DAMH-SMU framework.
# Considers "N = no_chains" chains of different lengths (l_1, ..., l_N), 
# each of the chains has "n = no_parameters" components.
# The samples in one chain form a numpy array of shape (l_i, n).
# All samples form a python list of length N.

    # def calculate_autocorr_function(self,begin=None,end=None,chains_range=None):
    #     if chains_range==None:
    #         chains_range = range(self.no_chains)
    #     no_chains = len(chains_range)
    #     self.autocorr_function = [None] * no_chains
    #     if end == None:
    #         end = self.length
    #     else:
    #         end = [int(end)] * no_chains
    #     if begin == None:
    #         begin = [0] * no_chains
    #     else:
    #         begin = [int(begin)] * no_chains
    #     for idx,i in enumerate(chains_range):
    #         tmp = np.zeros((end[idx]-begin[idx],self.no_parameters))
    #         for j in range(self.no_parameters):
    #             tmp[:,j] = emcee.autocorr.function_1d(self.x[i][begin[idx]:end[idx],j])
    #         self.autocorr_function[idx] = tmp
            
    def calculate_autocorr_function_mean(self,length=None,chains_range=None):
        if chains_range==None:
            chains_range = range(len(self.autocorr_function))
        no_chains = len(chains_range)
        if length==None:
            length = self.length
            max_length = max(self.length[chains_range])
        else:
            max_length = int(length)
            length = [int(length)] * no_chains
        self.autocorr_function_mean = np.zeros((max_length, self.no_parameters))
        for j in range(self.no_parameters):
            tmp = np.zeros(max_length)
            count = np.zeros(max_length)
            for i in chains_range:
                tmp[:length[i]] += self.autocorr_function[i][:,j]
                count[:length[i]] += 1
            self.autocorr_function_mean[:,j] = tmp/count

    # def calculate_autocorr_time(self, c=5, tol=50, quiet=True):
    #     self.autocorr_time = [None] * self.no_chains
    #     for i in range(self.no_chains):
    #         tmp = np.zeros((self.no_parameters))
    #         for j in range(self.no_parameters):
    #             tmp[j] = emcee.autocorr.integrated_time(self.x[i][:,j], c=c, tol=tol, quiet=quiet)
    #         self.autocorr_time[i] = tmp
        
    def calculate_autocorr_time_mean(self, c=5, length=None):
        self.autocorr_time_mean = [None] * self.no_parameters
        self.autocorr_time_mean_beta = [None] * self.no_parameters
        if length==None:
            length = min(self.length)
        else:
            length = int(length)
        for j in range(self.no_parameters):
            f = self.autocorr_function_mean[:length,j]
            self.autocorr_time_mean[j] = autocorr_FM(f, c)
            f = self.autocorr_function_mean[:,j]
            self.autocorr_time_mean_beta[j] = autocorr_FM(f, c)

    # def calculate_autocorr_time_sliding(self,no_sw=4,window_length=None,chains_range=None):
    #     if chains_range==None:
    #         chains_range = range(self.no_chains)
    #     min_length = min(self.length[list(chains_range)])
    #     if window_length == None:
    #         sw_step = np.floor(min_length/(no_sw+1))
    #         window_length = sw_step*2
    #     else:
    #         sw_step = np.floor((min_length-window_length)/no_sw)
    #     print("window length:",window_length)
    #     self.tau_sliding_mean = [None] * no_sw
    #     self.tau_sliding_max = [None] * no_sw
    #     self.tau_sliding_min = [None] * no_sw
    #     for i in range(no_sw):
    #         begin=i*sw_step
    #         end=i*sw_step+window_length
    #         self.calculate_autocorr_function(begin, end, chains_range=chains_range)
    #         self.calculate_autocorr_function_mean(length=end-begin)
    #         self.calculate_autocorr_time_mean(length=end-begin)
    #         self.tau_sliding_mean[i] = np.mean(self.autocorr_time_mean)
    #         self.tau_sliding_max[i] = np.max(self.autocorr_time_mean)
    #         self.tau_sliding_min[i] = np.min(self.autocorr_time_mean)
    #         print("i, mean, min, max:", i, self.tau_sliding_mean[i], self.tau_sliding_min[i], self.tau_sliding_max[i])

    def plot_autocorr_function(self, length_disp, plot_mean=False, parameters_disp = None, chains_disp = None, show_legend = False):
        if parameters_disp == None:
            parameters_disp = range(self.no_parameters)
        no_parameters_disp = len(parameters_disp)
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        fig, axes = plt.subplots(1, no_parameters_disp, figsize=(12, 3), sharey=True)
        for idj,j in enumerate(parameters_disp):
            length_disp[idj] = min(max(self.length),length_disp[idj])
        for idj,j in enumerate(parameters_disp):
            for idi,i in enumerate(chains_disp):
                axes[idj].plot(self.autocorr_function[i][:length_disp[idj],j], label=i)
            if plot_mean:
                axes[idj].plot(self.autocorr_function_mean[:,j],label="mean")
            axes[idj].set_xlim(0, length_disp[idj]-1)
            if show_legend:
                axes[idj].legend(loc=1)
            axes[idj].set_xlabel("$par. {0}$".format(j))
            axes[idj].grid(True)
            if self.known_autocorr_time:
                axes[idj].set_title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true[j]));
        axes[0].set_ylabel("autocorr. function")
        plt.show()
        
### GAUSSIAN RANDOM FIELDS:
        
    def plot_grf(self, eta, grf_path = None):
        if grf_path == None:
            grf_path = 'modules/unit50.pckl'
        grf_instance = grf_eigenfunctions.GRF(grf_path, truncate=self.no_parameters)
        z = grf_instance.realization_grid_new(eta,np.linspace(0,1,50),np.linspace(0,1,50))
        fig, axes = plt.subplots(1, 1, figsize=(12, 3), sharey=True)
        axes.imshow(z)
        plt.show()

    def plot_grf_minmax(self,chains_disp = None, grf_path = None):
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if grf_path == None:
            grf_path = 'modules/unit50.pckl'
        grf_instance = grf_eigenfunctions.GRF(grf_path, truncate=self.no_parameters)
        for i in chains_disp:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
            idx = np.argmin(self.posteriors[i])
            eta_min = self.x_compress[i][idx]
            z = grf_instance.realization_grid_new(eta_min,np.linspace(0,1,50),np.linspace(0,1,50))
            axes[0].imshow(z.transpose())
            axes[0].invert_yaxis()
            axes[0].set_title("$i={0}: {1}$".format(idx,self.posteriors[i][idx]))
            idx = np.argmax(self.posteriors[i])
            eta_max = self.x_compress[i][idx]
            z = grf_instance.realization_grid_new(eta_max,np.linspace(0,1,50),np.linspace(0,1,50))
            axes[1].imshow(z.transpose())
            axes[1].invert_yaxis()
            axes[1].set_title("$i={0}: {1}$".format(idx,self.posteriors[i][idx]))
            plt.show()
            
    def plot_mean_as_grf(self,chains_disp = None, grf_path = None):
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        if grf_path == None:
            grf_path = 'modules/unit50.pckl'
        grf_instance = grf_eigenfunctions.GRF(grf_path, truncate=self.no_parameters)
        for i in chains_disp:
            eta = self.mean[i]
            z = grf_instance.realization_grid_new(eta,np.linspace(0,1,50),np.linspace(0,1,50))
            fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
            axes[0].imshow(z)
            plt.show()
            
    def plot_mean_and_std_grf(self, burn_in = None, chains_disp = None, grf_path = None, grid_x = 50, grid_y = 50):
        if chains_disp == None:
            chains_disp = range(self.no_chains)
        no_chains_disp = len(chains_disp)
        if grf_path == None:
            grf_path = 'modules/unit50.pckl'
        grf_instance = grf_eigenfunctions.GRF(grf_path, truncate=self.no_parameters)
        if burn_in == None:
            burn_in = [0] * no_chains_disp
        for idi,i in enumerate(chains_disp):
            samples = self.x[i][burn_in[idi]:,:]
            samples_mean, samples_std = grf_instance.samples_mean_and_std(samples)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
            m0 = axes[0].imshow(samples_mean, origin="lower", extent = [0,1,0,1])
            fig.colorbar(m0, ax=axes[0])
            axes[0].set_title('mean')
            m1 = axes[1].imshow(samples_std, origin="lower", extent = [0,1,0,1])
            fig.colorbar(m1, ax=axes[1])
            axes[1].set_title('std')
            plt.show()
            
    def plot_mean_and_std_grf_merged(self, burn_in = None, chains_disp = None, grf_path = None, grid_x = 50, grid_y = 50):
        if grf_path == None:
            grf_path = 'modules/unit50.pckl'
        grf_instance = grf_eigenfunctions.GRF(grf_path, truncate=self.no_parameters)
        if burn_in == None:
            burn_in = [0] * self.no_chains
        samples = np.concatenate((self.x))
        samples_mean, samples_std = grf_instance.samples_mean_and_std(samples)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        m0 = axes[0].imshow(samples_mean, origin="lower", extent = [0,1,0,1])
        fig.colorbar(m0, ax=axes[0])
        axes[0].set_title('mean')
        m1 = axes[1].imshow(samples_std, origin="lower", extent = [0,1,0,1])
        fig.colorbar(m1, ax=axes[1])
        axes[1].set_title('std')
        plt.show()
      
    def generate_samples_rand(self,no_parameters,length):
        # no_parameters ... scalar
        # length ... list of scalars l_1, ..., l_N
        no_chains = len(length)
        all_chains = range(no_chains)
        self.x = list(np.random.rand(length[i],no_parameters) for i in all_chains)
        
    # def generate_samples_celerite(self, length=[1000], log_c1=[-6.0], log_c2=[-2.0]):
    #     # length ... list of scalars l_1, ..., l_N
    #     # log_c1, log_c2 ... list of scalars (len = no_parameters)
    #     no_chains = len(length)
    #     no_parameters = len(log_c1)
    #     # from https://dfm.io/posts/autocorr/ Foreman-Mackey
    #     # Build the celerite model:
    #     import celerite
    #     from celerite import terms
    #     true_tau = [None] * no_parameters
    #     gp_samples = [None] * no_parameters
    #     t = np.arange(max(length))
    #     for i in range(no_parameters):
    #         kernel = terms.RealTerm(log_a=0.0, log_c=log_c1[i])
    #         kernel += terms.RealTerm(log_a=0.0, log_c=log_c2[i])
    #         # The true autocorrelation time can be calculated analytically:
    #         true_tau[i] = sum(2*np.exp(t.log_a-t.log_c) for t in kernel.terms)
    #         true_tau[i] /= sum(np.exp(t.log_a) for t in kernel.terms)
    #         # Simulate a set of chains:
    #         gp = celerite.GP(kernel)
    #         gp.compute(t)
    #         gp_samples[i] = gp.sample(size=no_chains)
    #     self.known_autocorr_time = True
    #     self.autocorr_time_true = true_tau
    #     x = [None] * no_chains
    #     for i in range(no_chains):
    #         x[i] = np.zeros((length[i],no_parameters))
    #         for j in range(no_parameters):
    #             x[i][:,j] = gp_samples[j][i,:length[i]]
    #     self.x = x

# Automated windowing procedure following Sokal (1989)
# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def autocorr_FM(f, c=5.0):
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