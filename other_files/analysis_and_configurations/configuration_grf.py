#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:15:15 2019

@author: simona
"""

from modules import classes_communication
import numpy as np

class Configuration:
    def __init__(self):
    
### PROBLEM PARAMETERS
        self.problem_name = "GRF_rbftest"
        self.no_parameters = 40
        no_configurations = 1
        no_windows = 10 # do not change - corresponds to observations_all
        self.no_observations = no_configurations * no_windows
        #noise_std = list(np.sqrt([0.10049876, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102, 0.03480102]))
        noise_std = [0.1] * no_windows
        noise_std[0] = 0.1 * (no_windows-1)
        observations_all = [8.93017454511605, -1.1039663032584617, -0.9146225063677776, -0.5049702283627462, -0.963501221872019, -1.4219146682759043, -0.48836361529524097, -1.047597205600013, -1.400302853016347, -1.0849348043739866, 8.207548081705145, -0.8069608775268073, -1.0413268458200349, -1.3259324368769065, -0.7239709404429496, -1.3060693977357836, -1.4051462946035171, -0.5541941920732057, -0.39476416257591956, -0.6491828462637359, 8.930174417531433, -0.6279650110075949, -1.2125797149077837, -0.5982527366895216, -0.7403462536157467, -0.6904980257545852, -1.8159939556070257, -1.8788753194181465, -0.6445604321053241, -0.7211019359525195, 8.207548357304347, -0.9371120457768605, -0.6719444605467054, -1.943175696581782, -1.4402956181931734, -0.5964189953863263, -0.7798851907825796, -0.6190993033255605, -0.8276528090758665, -0.3919639585558014]
        self.problem_parameters = {'prior_mean': 0.0,
                                   'prior_std': 1.0,
                                   'observations': observations_all[0:self.no_observations],
                                   #'observations': [ 8.77826154, -0.97791952, -0.98783364, -0.6663905, -0.86361077, -1.17667752, -0.66584558, -0.86692055, -1.45951157, -1.1135524, 8.01389101, -0.79169205, -1.07264703, -0.94635772, -0.87144708, -1.35153141, -1.25297801, -0.62401356, -0.50585042, -0.59737366, 8.7782618, -0.77096348, -1.02920786, -0.82651668, -0.50689595, -0.73610558, -1.76177878, -1.78232149, -0.85484636, -0.50962311, 8.01389094, -0.6492758, -1.11468732, -1.66398841, -1.26665593, -0.66067157, -0.57511901, -0.85162245, -0.77442005, -0.45745044],
                                   'noise_std': noise_std*no_configurations,
                                   }

### SOLVER SPECIFICATION
# FEM + grf material, 40 parameters
        from modules import FEM_grf as FEM_wrapper
        self.child_solver_init = FEM_wrapper.FEM
        self.child_solver_parameters = {'no_parameters': self.no_parameters,
                                        'no_observations': self.no_observations, 
                                        'no_configurations': no_configurations,
                                        'n': 50,
                                        'quiet': True,
                                        'tolerance': 1e-8,
                                        'PC': "icc",
                                        'use_deflation': False, # True,
                                        'deflation_imp': 1e-2}
        self.no_full_solvers = 8

### SAMPLING PARAMETERS
        self.no_samplers = 8
        algMH = {'type': 'MH', 
                 'max_samples': 1000, 
                 'time_limit': 1*60, # in seconds
                 'proposal_std': 0.1,
                 'surrogate_is_updated': True,
                 }
        algDAMHSMU = {'type': 'DAMH', 
                   'max_samples': 10000, 
                   'time_limit': 1*60, # in seconds
                   'proposal_std': 0.1,
                   'surrogate_is_updated': True,
                   }
        algDAMH = {'type': 'DAMH', 
                    'max_samples': 100000, 
                    'time_limit': 1*60, # in seconds
                    'proposal_std': 0.1,
                    'surrogate_is_updated': False,
                    }
        self.list_alg = [algMH, algDAMHSMU, algDAMH]

### SURROGATE MODEL SPECIFICATION
        from modules import surrogate_solver_rbf as surr
        self.surr_solver_init = surr.Surrogate_apply
        self.surr_solver_parameters = {'no_parameters':self.no_parameters,
                                       'no_observations':self.no_observations,
                                       'kernel_type':1,
                                       }
        self.surr_updater_init = surr.Surrogate_update
        self.surr_updater_parameters = {'no_parameters':self.no_parameters,
                                        'no_observations':self.no_observations,
                                        'kernel_type':1,
                                        'no_keep':500,
                                        'expensive':False,
                                        }
        
### OTHER SETTINGS
        self.max_buffer_size = 1<<30 #20
        self.rank_full_solver = self.no_samplers
        self.rank_surr_collector = self.no_samplers + 1
        
        # TYPE 1 - solvers are spawned:
        self.full_solver_init = classes_communication.Solver_MPI_parent
        self.full_solver_parameters = []
        for i in range(self.no_full_solvers):
            self.full_solver_parameters.append({'no_parameters':self.no_parameters,
                                                'no_observations':self.no_observations})