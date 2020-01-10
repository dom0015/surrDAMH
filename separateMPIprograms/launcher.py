#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:47:09 2019

@author: simona
"""

from mpi4py import MPI

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

print("LAUNCHER", rank, size, rank_world, size_world)

if rank_world==0:
    Alg = comm.recv(source=0, tag=0)
    Alg.run()
    print(Alg.no_accepted, Alg.no_rejected, Alg.no_accepted/Alg.no_rejected*100, '%')

comm.Disconnect()