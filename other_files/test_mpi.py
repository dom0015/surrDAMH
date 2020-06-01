#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:59:30 2020

@author: ber0061
"""

from mpi4py import MPI

r = MPI.COMM_WORLD.Get_rank()

print("rank",r)