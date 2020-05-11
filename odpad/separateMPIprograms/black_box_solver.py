#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:00:31 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np

class black_box_solver:
    
    def __init__(self):
        self.comm_world = MPI.COMM_WORLD
        self.rank_world = self.comm_world.Get_rank()
        self.size_world = self.comm_world.Get_size()
        if self.rank_world==0:
            universe_size = self.comm_world.Get_attr(MPI.UNIVERSE_SIZE)
            print('universe size is', universe_size)

    def prepare(self, N):
        N = int(N[0])
        self.initial_data = np.zeros(N,)
        if self.rank_world==0:
            for i in 1+np.arange(self.size_world-1):
                temp = np.random.rand(N)
                self.comm_world.Isend(temp, dest=i, tag=15)
        if self.rank_world>0:
            self.comm_world.Recv(self.initial_data, source=0, tag=15)
        self.comm_world.Barrier()
        print('Rank', self.rank_world, '- initial data', self.initial_data)
        
    def solve(self, M):
        if self.rank_world>0:
            temp = np.mean(self.initial_data) + 10*self.rank_world + 1000*M + 100000*len(self.initial_data)
            print('Rank', self.rank_world, '- partial solution', temp)
            self.comm_world.Send(temp, dest = 0, tag=25)
        if self.rank_world==0:
            results = np.zeros(self.size_world,)
            temp = np.zeros(1,)
            for i in 1+np.arange(self.size_world-1):
                self.comm_world.Recv(temp, source=i, tag=25)
                results[i] = temp
            print('Rank 0 received:', results)
            return results