#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:02:59 2019

@author: simona
"""

from mpi4py import MPI
import psutil

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()
myhost = MPI.Get_processor_name()

comm = MPI.Comm.Get_parent()

print(size_world, rank_world, myhost, psutil.Process().cpu_num())

comm.Disconnect()