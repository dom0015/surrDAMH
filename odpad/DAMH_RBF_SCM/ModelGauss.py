#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:30:46 2018

@author: dom0015
"""

import numpy as np

class ModelGauss:
    # shared class, cannot be modified by processes
    # contains only information about Bayesian inverse problem,
    # not about the sampling process

    def __init__(self, no_parameters, no_observations, priorMean, priorStd, noiseStd, observation):
        self.no_parameters = no_parameters    # instance variable unique to each instance
        self.no_observations = no_observations    # instance variable unique to each instance
        self.priorMean = priorMean    # instance variable unique to each instance
        self.priorStd = priorStd    # instance variable unique to each instance
        self.noiseStd = noiseStd    # instance variable unique to each instance
        self.observation = observation    # instance variable unique to each instance
        self.noiseCov = np.diag(noiseStd*noiseStd)
        self.priorStd = np.diag(priorStd*priorStd)
        self.noiseStd = self.noiseCov
        
    def SetNoisyObservation(self, artificial_observation_without_noise):
        r = np.random.RandomState(1)
        temp=r.normal(0.0,1.0,self.no_observations);
        self.observation = np.multiply(temp,self.noiseStd) + artificial_observation_without_noise
        print('Noisy observation:', self.observation, '=', artificial_observation_without_noise, '+', temp, '*', self.noiseStd)