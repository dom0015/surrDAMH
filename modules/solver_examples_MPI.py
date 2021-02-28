#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:00:23 2020

@author: simona
"""

import numpy as np
from mpi4py import MPI

class Solver_linela2exp_MPI:
    def __init__(self, f=-0.1, L=1.0, M=0.5):
        self.f = f
        self.L = L
        self.M = M
        self.no_parameters = 2
        self.no_observations = 1
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if self.rank == 0 or self.rank == self.size-1:
            self.data = np.empty((self.no_observations,))
            
    def set_parameters(self,data_par):
        self.k1 = np.exp(data_par[0])
        self.k2 = np.exp(data_par[1])
        
    def get_observations(self):
        if self.rank == 0 and self.size>1:
            self.comm.Recv(self.data, source=self.size-1)
        elif self.rank == self.size-1:
            D1 = (self.f*self.L)/self.k2
            C1 = D1*self.k2/self.k1
            D2 = -self.f/(2*self.k1)*(self.M*self.M)+C1*self.M+self.f/(2*self.k2)*(self.M*self.M)-D1*self.M
            self.data[0] = -self.f/(2*self.k2)*(self.L*self.L)+D1*self.L+D2
            if self.size>1:
                self.comm.Send(self.data, dest=0)
        # else:
        #     print("I am dwarf", self.rank, "and I'm digging a hole.")
        if self.rank == 0:
            return self.data.copy()
        else:
            return None
