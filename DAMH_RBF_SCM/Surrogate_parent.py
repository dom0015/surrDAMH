#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:07:02 2018

@author: dom0015
"""

import numpy as np

class Surrogate_parent:
    def __init__(self):
        print("Surrogate parent")
    
    def add_data(self,alldata_par,alldata_obs,alldata_wei,shared_queue_surrogate,queue_size,Model):
        newdata_par = np.zeros([0,Model.no_parameters])
        newdata_obs = np.zeros([0,Model.no_observations])
        newdata_wei = np.zeros([0,1])
        for i in range(queue_size):
            data_par, data_obs, data_wei = shared_queue_surrogate.get()
            newdata_par = np.vstack((newdata_par, data_par))
            newdata_obs = np.vstack((newdata_obs, data_obs))
            newdata_wei = np.vstack((newdata_wei, data_wei))
        alldata_par = np.vstack((alldata_par, newdata_par))
        alldata_obs = np.vstack((alldata_obs, newdata_obs))
        alldata_wei = np.vstack((alldata_wei, newdata_wei))
        return alldata_par, alldata_obs, alldata_wei, newdata_par, newdata_obs, newdata_wei