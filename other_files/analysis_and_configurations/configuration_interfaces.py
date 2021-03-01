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
        self.problem_name = "interfaces_saving"
        self.no_parameters = 4
        no_configurations = 1
        no_windows = 10 # corresponds to observations_10
        self.no_observations = no_configurations * no_windows
        noise_std = [0.1] * no_windows
        noise_std[0] = 0.1 * (no_windows-1)
        # 2 parameters [2, -1]:
        observations_10 = np.array([14.7170489 , -0.41736385, -2.40350234, -0.66734775, -1.06442428,
                                    -1.65743588, -1.7438058 , -1.52988774, -0.89086761, -4.34240813,
                                    20.22611273, -2.90575579, -1.59780689, -9.31028329, -1.57733198,
                                    -1.341741  , -0.83542656, -0.63571882, -1.27587512, -0.74617407,
                                    14.71704792, -3.11442786, -0.93973316, -0.4564994 , -0.56651853,
                                    -0.40526613, -0.30400738, -2.01407773, -5.22076372, -1.69574645,
                                    20.22611317, -0.26093203, -3.37914667, -3.54942369, -2.9861132 ,
                                    -0.39111649, -5.00583697, -3.5101237 , -0.58067482, -0.56274684])
        # 4 paramteters [2, -1, 1, -2]:
        # observations_10 = np.array([ 8.93406482, -0.48527442, -0.71070188, -0.23678076, -0.5734647 ,
        #                            -1.50087009, -0.55573861, -0.61635841, -2.26819309, -1.9866857 ,
        #                            11.95951321, -1.27701722, -1.51215939, -4.23714353, -2.05795626,
        #                            -0.74296454, -0.35610542, -0.21254544, -1.0033474 , -0.56027266,
        #                             8.93406496, -0.91625996, -0.83284115, -1.31819122, -0.85354592,
        #                            -0.77874648, -0.1090134 , -0.61286766, -2.64062157, -0.87197855,
        #                            11.95951294, -0.90713562, -1.1020357 , -0.93265103, -0.98891822,
        #                            -0.14412132, -4.32066162, -2.16564791, -0.89047374, -0.50786876])
        self.problem_parameters = {'prior_mean': 0.0,
                                   'prior_std': 1.0,
                                   'observations': observations_10[0:self.no_observations],
                                   'noise_std': noise_std*no_configurations,
                                   }

### SOLVER SPECIFICATION
        from modules import FEM_interfaces as FEM_wrapper
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