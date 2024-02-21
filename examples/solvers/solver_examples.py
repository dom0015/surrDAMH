#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:00:23 2020

@author: simona
"""

import numpy as np
from mpi4py import MPI
from surrDAMH.solvers import Solver


class Solver_illustrative_local(Solver):
    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters):
        self.x = parameters[0]
        self.y = parameters[1]

    def get_observations(self):
        res = (self.x**2-self.y)*(np.log((self.x-self.y)**2+1))
        return res
        # return convergence_tag, res


class Nonlinear(Solver):
    def __init__(self, coef, solver_id=0, output_dir=None):
        self.no_parameters = 2
        self.no_observations = 1
        self.coef = coef

    def set_parameters(self, parameters):
        self.par = parameters

    def get_observations(self):
        linear_function1 = self.coef[0]*self.par[0] + self.coef[1]*self.par[1] + self.coef[2]
        linear_function2 = self.coef[3]*self.par[0] + self.coef[4]*self.par[1] + self.coef[5]
        res = min(linear_function1, linear_function2)
        return res
        # return convergence_tag, res


class Solver_linela2exp_local(Solver):
    def __init__(self, f=-0.1, length=1.0, m=0.5, solver_id=0, output_dir=None):
        self.f = f
        self.L = length
        self.M = m
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters):
        self.k1 = np.exp(parameters[0])
        self.k2 = np.exp(parameters[1])

    def get_observations(self):
        D1 = (self.f*self.L)/self.k2
        C1 = D1*self.k2/self.k1
        D2 = -self.f/(2*self.k1)*(self.M*self.M)+C1*self.M+self.f/(2*self.k2)*(self.M*self.M)-D1*self.M
        uL = -self.f/(2*self.k2)*(self.L*self.L)+D1*self.L+D2
        return uL


class Solver_linela2exp_MPI:
    def __init__(self, solver_id=0, f=-0.1, L=1.0, M=0.5, output_dir=None):
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

    def set_parameters(self, parameters):
        self.k1 = np.exp(parameters[0])
        self.k2 = np.exp(parameters[1])

    def get_observations(self):
        if self.rank == 0 and self.size > 1:
            self.comm.Recv(self.data, source=self.size-1)
        elif self.rank == self.size-1:
            D1 = (self.f*self.L)/self.k2
            C1 = D1*self.k2/self.k1
            D2 = -self.f/(2*self.k1)*(self.M*self.M)+C1*self.M+self.f/(2*self.k2)*(self.M*self.M)-D1*self.M
            self.data[0] = -self.f/(2*self.k2)*(self.L*self.L)+D1*self.L+D2
            if self.size > 1:
                self.comm.Send(self.data, dest=0)
        # else:
        #     print("Rank", self.rank, "has nothing to do.")
        if self.rank == 0:
            return self.data.copy()
        else:
            return None


class Solver_linela2exp_local_tag:
    def __init__(self, f=-0.1, length=1.0, m=0.5, solver_id=0, output_dir=None):
        self.f = f
        self.L = length
        self.M = m
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters):
        self.k1 = np.exp(parameters[0])
        self.k2 = np.exp(parameters[1])

    def get_observations(self):
        D1 = (self.f*self.L)/self.k2
        C1 = D1*self.k2/self.k1
        D2 = -self.f/(2*self.k1)*(self.M*self.M)+C1*self.M+self.f/(2*self.k2)*(self.M*self.M)-D1*self.M
        uL = -self.f/(2*self.k2)*(self.L*self.L)+D1*self.L+D2
        if int(np.random.rand() < 0.5):
            convergence_tag = 1
        else:
            convergence_tag = -1
        return convergence_tag, uL


class NonlinearGeneric(Solver):
    def __init__(self, solver_id=0, no_parameters=3, no_observations=3,  output_dir=None):
        self.no_parameters = no_parameters
        self.no_observations = no_observations

    def set_parameters(self, parameters):
        self.par = parameters

    def get_observations(self):
        val = np.Inf
        for i in range(self.no_parameters):
            linear_function = np.sum(self.par[:i]) + np.sum(self.par[i+1:]) - self.par[i]
            val = np.min([val, linear_function])
        return val*np.ones((self.no_observations,))
        # return convergence_tag, res


class Generic(Solver):
    def __init__(self, solver_id=0, no_parameters=3, no_observations=3, output_dir=None):
        self.no_parameters = no_parameters
        self.no_observations = no_observations

    def set_parameters(self, parameters):
        self.x = np.mean(parameters)

    def get_observations(self):
        res = self.x*np.ones((self.no_observations,))
        # if int(np.random.rand() < 0.75):
        #     convergence_tag = 1
        # else:
        #     convergence_tag = -1
        # return convergence_tag, res
        return res
