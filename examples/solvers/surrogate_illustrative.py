#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:00:23 2020

@author: simona
"""

import numpy as np

class Surrogate_apply: # initiated by all SAMPLERs
    def __init__(self, no_parameters, no_observations):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        
    def apply(self, EMPTY, datapoints):
        datapoints = datapoints.reshape(-1,self.no_parameters)
        no_datapoints = datapoints.shape[0]
        x = datapoints[:,0]
        y = datapoints[:,1]
        GS_datapoints = (x**2-y)*(np.log((x-y)**2+1))
        GS_datapoints = GS_datapoints.reshape((no_datapoints,self.no_observations))
        return GS_datapoints
        
class Surrogate_update: # initiated by COLLECTOR
    def __init__(self, no_parameters, no_observations):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        # all processed data (used for surrogate construction):
        self.processed_par = np.empty((0,self.no_parameters))
        self.processed_obs = np.empty((0,self.no_observations))
        self.processed_wei = np.empty((0,1))
        self.no_processed = self.processed_par.shape[0]
        # all data not yet used for surrogate construction:
        self.non_processed_par = np.empty((0,self.no_parameters))
        self.non_processed_obs = np.empty((0,self.no_observations))
        self.non_processed_wei = np.empty((0,1))

    def add_data(self,snapshots):
        # add new data to a matrix of non-processed data
        L = len(snapshots)
        new_par = np.empty((L,self.no_parameters))
        new_obs = np.empty((L,self.no_observations))
        new_wei = np.empty((L,1))
        for i in range(L):
            new_par[i,:] = snapshots[i].sample
            new_obs[i,:] = snapshots[i].G_sample
            new_wei[i,:] = 1
        self.non_processed_par = np.vstack((self.non_processed_par, new_par))
        self.non_processed_obs = np.vstack((self.non_processed_obs, new_obs))
        self.non_processed_wei = np.vstack((self.non_processed_wei, new_wei))

    def update(self):
        no_non_processed = self.non_processed_par.shape[0]
        no_snapshots = self.no_processed + no_non_processed # both processed and non-processed
        
        self.processed_par = np.vstack((self.processed_par, self.non_processed_par))
        self.processed_obs = np.vstack((self.processed_obs, self.non_processed_obs))
        self.processed_wei = np.vstack((self.processed_wei, self.non_processed_wei))
        self.no_processed += no_non_processed
        
        self.non_processed_par = np.empty((0,self.no_parameters))
        self.non_processed_obs = np.empty((0,self.no_observations))
        self.non_processed_wei = np.empty((0,1))
        
        SOL = np.empty((0,))
        return SOL, no_snapshots