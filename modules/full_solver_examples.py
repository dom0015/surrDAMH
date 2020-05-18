#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:00:23 2020

@author: simona
"""

import numpy as np

class Solver_local_linela4:
    def __init__(self, f=-0.1, L=1.0, M12=0.25, M23=0.5, M34=0.75):
        self.f = f
        self.L = L
        self.M12 = M12
        self.M23 = M23
        self.M34 = M34
        self.no_parameters = 4
        self.no_observations = 1
        
    def get_solution(self, data_par):
        k1 = data_par[0]
        k2 = data_par[1]
        k3 = data_par[2]
        k4 = data_par[3]
        C4 = (self.f*self.L)/k4
        C3 = C4*k4/k3
        C2 = C3*k3/k2
        C1 = C2*k2/k1
        D1 = 0
        D2 = -self.f/k1*self.M12*self.M12/2 + C1*self.M12 + D1 + self.f/k2*self.M12*self.M12/2 - C2*self.M12
        D3 = -self.f/k2*self.M23*self.M23/2 + C2*self.M23 + D2 + self.f/k3*self.M23*self.M23/2 - C3*self.M23
        D4 = -self.f/k3*self.M34*self.M34/2 + C3*self.M34 + D3 + self.f/k4*self.M34*self.M34/2 - C4*self.M34
        uL = -self.f/k4*self.L*self.L/2 + C4*self.L + D4
        return uL
    
class Solver_local_linela2:
    def __init__(self, f=-0.1, L=1.0, M=0.5):
        self.f = f
        self.L = L
        self.M = M
        self.no_parameters = 2
        self.no_observations = 1
        
    def get_solution(self, data_par):
        k1 = data_par[0]
        k2 = data_par[1]
        D1 = (self.f*self.L)/k2
        C1 = D1*k2/k1
        D2 = -self.f/(2*k1)*(self.M*self.M)+C1*self.M+self.f/(2*k2)*(self.M*self.M)-D1*self.M
        uL = -self.f/(2*k2)*(self.L*self.L)+D1*self.L+D2
        return uL
    
class Solver_local_himmelblau:
    def __init__(self, ):
        self.no_parameters = 2
        self.no_observations = 1
        
    def get_solution(self, data_par):
        x1 = data_par[0]
        x2 = data_par[1]
        y = (x1*x1 + x2 - 11)**2 + (x1 + x2*x2 - 7)**2
        return y
    
class Solver_local_2to2:
    def __init__(self, ):
        self.no_parameters = 2
        self.no_observations = 2
        self.request_solved = True
        self.max_requests = 1
    
    def pass_parameters(self, data_par):
        self.data_par = data_par
        
#    def get_solution(self, ):
#        x1 = self.data_par[0,0]
#        x2 = self.data_par[0,1]
#        y1 = (x1*x1 + x2 - 11)**2 + (x1 + x2*x2 - 7)**2
#        y2 = x1 + x2
#        return np.array([[y1,y2]])
        
    def get_solution(self, ):
        x1 = self.data_par[0]
        x2 = self.data_par[1]
        y1 = (x1*x1 + x2 - 11)**2 + (x1 + x2*x2 - 7)**2
        y2 = x1 + x2
        return np.array([y1,y2])
    
    def terminate(self):
        print("Terminate function is empty.")
        
class Solver_local_ntom:
    def __init__(self, no_parameters = 2, no_observations = 2):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.request_solved = True
        self.max_requests = 1
    
    def pass_parameters(self, data_par):
        self.data_par = data_par
        
    def get_solution(self, ):
        x = self.data_par
        y = np.zeros((self.no_observations,))
        for i in range(self.no_observations):
            y[i] = x[min(i,self.no_parameters-1)]
        return y
    
    def terminate(self):
        print("Terminate function is empty.")