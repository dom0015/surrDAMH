#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:11:54 2019

@author: simona
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

from configuration import Configuration
C = Configuration()

class Chain:
    def __init__(self, folder_samples, folder_notes, file_samples, file_notes):
        self.file_samples = file_samples
        self.file_notes = file_notes
        self.path_samples = folder_samples + "/" + file_samples
        self.path_notes = folder_notes + "/" + file_notes
        self.df_samples = pd.read_csv(self.path_samples, header=None)
        self.df_notes = pd.read_csv(self.path_notes)
        self.df_notes['file'] = file_notes
        self.weights = np.array(self.df_samples[0])
        self.values = self.df_samples.iloc[:,1:]
        self.no_parameters = self.values.shape[1]
        self.no_samples = sum(self.weights)
        self.no_samples_unique = len(self.weights)
        self.mean = self.values.mean()
        
    def plot_samples(self):
        xx = np.cumsum(self.weights)
        xx = np.append(0,xx)
        for j in range(self.no_parameters):
            yy = np.array(self.values.iloc[:,j])
            yy = np.append(yy[0],yy)
            plt.step(xx, yy, label=str(j))
        plt.legend(title='Parameter')
        plt.show()

    def plot_histogram(self,dimension = 0, show = True):
        yy = self.values.iloc[:,dimension]
        plt.hist(yy, weights=self.weights, label=str(dimension))
        if show:
            plt.show()

    def plot_histograms(self,one_figure = True):
        for j in range(self.no_parameters):
            self.plot_histogram(dimension = j, show = False)
            if not one_figure:
                plt.title('Parameter' + str(j))
                plt.show()
        if one_figure:
            plt.legend(title='Parameter')
            plt.show()
            
    def plot_correlation(self, show = True):
        # does not take weights into account
        pd.plotting.scatter_matrix(self.values, alpha = 0.2)
        if show:
            plt.show()
        
    def plot_histogram_2d(self, dimensions = [0,1], bins = 20, show = True):
        xx = np.array(self.values.iloc[:,dimensions[0]])
        yy = np.array(self.values.iloc[:,dimensions[1]])
        plt.hist2d(xx, yy, weights = self.weights, bins = bins)
        if show:
            plt.show()
    
    def plot_histograms_grid(self):
        n = self.no_parameters
        idx = 1
        for i in range(n):
            for j in range(n):
                plt.subplot(n, n, idx)
                if i==j:
                    self.plot_histogram(dimension = i, show = False)
                else:
                    self.plot_histogram_2d(dimensions = [j,i], show = False)
                idx = idx + 1
        plt.show()
        
    def function_identity(self, values, dimension = 1):
        return np.array(values[:,dimension])
    
    def plot_empirical_average(self, function):
        data = function(np.array(self.values))
        data_cumsum = np.cumsum(data)
        xx = np.cumsum(self.weights)
        yy = data_cumsum/xx # TO DO
        xx = np.append(0,xx)
        yy = np.append(yy[0],yy)
        plt.step(xx, yy)
        
folder_samples = C.problem_name
folder_notes = folder_samples + "/notes"

files_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
files_samples.sort()
files_notes = [f for f in listdir(folder_notes) if isfile(join(folder_notes, f))]
files_notes.sort()

N = len(files_notes)
chains = []
for i in range(N):
    ch = Chain(folder_samples, folder_notes, files_samples[i], files_notes[i])
    chains.append(ch)

df = pd.DataFrame()
for i in range(N):
    df = df.append(chains[i].df_notes)

print(df)

for i in range(N):
#    chains[i].plot_samples()
#    chains[i].plot_histograms()
#    chains[i].plot_correlation()
#    chains[i].plot_histogram_2d(bins = 30)
#    chains[i].plot_histograms_grid()
    chains[i].plot_empirical_average(function = chains[i].function_identity)
