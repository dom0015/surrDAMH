#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:30:46 2018

@author: dom0015
"""

import numpy as np

class ModelGauss:

    def __init__(self, no_parameters, no_observations, priorMean, priorStd, noiseStd, observation):
        self.no_parameters = no_parameters    # instance variable unique to each instance
        self.no_observations = no_observations    # instance variable unique to each instance
        self.priorMean = priorMean    # instance variable unique to each instance
        self.priorStd = priorStd    # instance variable unique to each instance
        self.noiseStd = noiseStd    # instance variable unique to each instance
        self.observation = observation    # instance variable unique to each instance
        
    def SetNoisyObservation(self, artificial_observation_without_noise):
        r = np.random.RandomState(1)
        temp=r.normal(0.0,1.0,self.no_observations);
        self.observation = np.multiply(temp,self.noiseStd) + artificial_observation_without_noise
        print('Noisy observation:', self.observation, '=', artificial_observation_without_noise, '+', temp, '*', self.noiseStd)