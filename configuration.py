#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:15:15 2019

@author: simona
"""

from modules import classes_communication
from modules import surrogate_solver_rbf as surr
#from modules import full_solver_examples as fse
#from modules import surrogate_solver_examples as sse

class Configuration:
    def __init__(self):
        self.problem_name = "GRF_24to40_02_unlimited"
        self.no_samplers = 8
        self.no_full_solvers = 8
        self.no_parameters = 40
        self.no_observations = 40
        self.rank_full_solver = self.no_samplers
        self.rank_surr_collector = self.no_samplers + 1
        algMH = {'type': 'MH', 
                 'max_samples': 100000, 
                 'time_limit': 60*10,
                 'proposal_std': 0.2,
                 'surrogate_is_updated': True}
        algDAMHSMU = {'type': 'DAMH', 
                   'max_samples': 10000000, 
                   'time_limit': 60*180,
                   'proposal_std': 0.2,
                    'surrogate_is_updated': True}
        # algDAMH = {'type': 'DAMH', 
        #             'max_samples': 10000000, 
        #             'time_limit': 60*60,
        #             'proposal_std': 0.6,
        #             'surrogate_is_updated': False}
        self.list_alg = [algMH, algDAMHSMU]#, algDAMH]
        self.max_buffer_size = 1<<30 #20
        self.surrogate_is_updated = True # TO DO: unused, included to sampler parameters -> remove
        # noise for 1+9 observations:
        noise_std = [0.10049876, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102]
        self.problem_parameters = {# 'noise_std': [0.10049876, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.10049876, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102],
                                   # 'noise_std': [0.10049876, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102],
                                   'noise_std': noise_std * 4,
                                   'prior_mean': 0.0,
                                   'prior_std': 1.0,
                                   # 'observations': [8.93017455, -1.1039663, -0.91462251, -0.50497023, -0.96350122, -1.42191467, -0.48836362, -1.04759721, -1.40030285, -1.0849348,  8.20754808, -0.80696088, -1.04132685, -1.32593244, -0.72397094, -1.3060694,  -1.40514629, -0.55419419, -0.39476416, -0.64918285, 8.930174417531433, -0.6279650110075949, -1.2125797149077837, -0.5982527366895216, -0.7403462536157467, -0.6904980257545852, -1.8159939556070257, -1.8788753194181465, -0.6445604321053241, -0.7211019359525195], # 2500 par, ones, 3*10 obs
                                   # 'observations': [ 8.93017455, -1.1039663, -0.91462251, -0.50497023, -0.96350122, -1.42191467, -0.48836362, -1.04759721, -1.40030285, -1.0849348,  8.20754808, -0.80696088, -1.04132685, -1.32593244, -0.72397094, -1.3060694,  -1.40514629, -0.55419419, -0.39476416, -0.64918285], # 2500 par, ones, 2*10 obs
                                   'observations': [8.93017454511605, -1.1039663032584617, -0.9146225063677776, -0.5049702283627462, -0.963501221872019, -1.4219146682759043, -0.48836361529524097, -1.047597205600013, -1.400302853016347, -1.0849348043739866, 8.207548081705145, -0.8069608775268073, -1.0413268458200349, -1.3259324368769065, -0.7239709404429496, -1.3060693977357836, -1.4051462946035171, -0.5541941920732057, -0.39476416257591956, -0.6491828462637359, 8.930174417531433, -0.6279650110075949, -1.2125797149077837, -0.5982527366895216, -0.7403462536157467, -0.6904980257545852, -1.8159939556070257, -1.8788753194181465, -0.6445604321053241, -0.7211019359525195, 8.207548357304347, -0.9371120457768605, -0.6719444605467054, -1.943175696581782, -1.4402956181931734, -0.5964189953863263, -0.7798851907825796, -0.6190993033255605, -0.8276528090758665, -0.3919639585558014],
                                   #'observations': [ 8.93017455, -1.1039663, -0.91462251, -0.50497023, -0.96350122, -1.42191467, -0.48836362, -1.04759721, -1.40030285, -1.0849348 ], # 2500 par, ones
                                   # 'observations': [ 8.66573862, -1.03993979, -1.14229746, -0.94976422, -0.78099463, -0.66584307, -0.5827063,  -0.78787596, -1.32720872, -1.3891087 ], # 10 par, ones
                                   # 'observations': [10.38232625, -0.52869917, -0.78189012, -1.42572536, -2.47236151, -2.64524228, -1.29127951, -0.56259429, -0.36707952, -0.30745544], # 20 par, seed 2
                                   # 'observations': [10.36728482, -0.96484045, -1.43592265, -1.65478891, -1.94643102, -1.99862472, -1.19716276, -0.58869322, -0.33193985, -0.24888124], # 10 par, seed 2
                                   # 'observations': [10.64239524 -1.40163629 -2.36298113 -5.18486635 -1.03188994 -0.66102128], # 2500 par, seed 2
                                   # 'observations': [ 7.4837606, -1.68701975, -1.81545499, -1.65487159, -1.28726334, -1.03915092], # 10 par
                                   }
### local solver?        
#        self.full_solver_init = fse.Solver_local_ntom
#        self.full_solver_parameters = []
#        for i in range(self.no_full_solvers):
#            self.full_solver_parameters.append({'no_parameters':self.no_parameters, 
#                                                'no_observations':self.no_observations})

### SOLVER TYPE 1 - solvers are spawned
        # TO DO: test if other options are also possible
        from modules import FEM_wrapper4 as FEM_wrapper
        self.child_solver_init = FEM_wrapper.FEM
        self.child_solver_parameters = {'no_parameters': self.no_parameters,
                                        'no_observations': self.no_observations, 
                                        'n': 50,
                                        'quiet': True,
                                        'tolerance': 1e-8,
                                        'PC': "icc",
                                        'use_deflation': False,
                                        'deflation_imp': 1e-2}
        self.full_solver_init = classes_communication.Solver_MPI_parent
        self.full_solver_parameters = []
        for i in range(self.no_full_solvers):
            self.full_solver_parameters.append({'no_parameters':self.no_parameters,
                                                'no_observations':self.no_observations})

### TYPE 2 - solvers are in the same COMM_WORLD (TO DO)
#        self.full_solver_init = classes_communication.Solver_MPI_collector_MPI
#        self.full_solver_parameters = []
#        for i in range(self.no_full_solvers):
#            self.full_solver_parameters.append({'no_parameters':2, 
#                                                'no_observations':2, 
#                                                'rank_solver':i+5})

### SURROGATE
        # TO DO: use 1 iteration of deflation as surrogate
        self.surr_solver_init = surr.Surrogate_apply
        self.surr_solver_parameters = {'no_parameters':self.no_parameters,
                                       'no_observations':self.no_observations}
                                       # 'kernel_type':1}
        self.surr_updater_init = surr.Surrogate_update
        self.surr_updater_parameters = {'no_parameters':self.no_parameters,
                                        'no_observations':self.no_observations}
                                        # 'kernel_type':1}