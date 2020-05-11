#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:40:23 2019

@author: simona
"""

from SurrMCMC import SurrMCMC

SM = SurrMCMC() # idx_groups=None, no_chains=1, seed=0, processes_per_solver=1
SM.setForwardModel() # no_parameters=None, no_observations=None, solver=None
SM.setInverseProblem() # type_Bayes='Gauss', priorMean=10, priorStd=1.5, noiseStd=2*0.0001, noisy_observation=None, artificial_real_parameters=8
SM.addStage() # type_sampling='MH', limit_time=10, limit_samples=100, name_stage='default_name', type_proposal='Gauss', proposalStd=0.5
SM.addStage(type_sampling='DAMH',limit_time=20) # type_sampling='MH', limit_time=10, limit_samples=100, name_stage='default_name', type_proposal='Gauss', proposalStd=0.5
# is limit time cumulative?
SM.setSurrogate(type_surrogate='RBF') # type_surrogate='RBF', min_snapshots=5, max_snapshots=100, max_degree=2, initial_iteration=None, no_keep=0, expensive=0, type_kernel=0, type_solver=0
SM.run()