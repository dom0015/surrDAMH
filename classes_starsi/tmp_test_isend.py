#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:43:08 2020

@author: simona
"""


from mpi4py import MPI
import numpy as np
import time

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()

if rank_world == 0:
    V = np.random.rand(4)
    time.sleep(1)
    comm_world.isend(V, dest=1, tag=11)
    print("RANK",rank_world,"- sent data",V)

if rank_world == 1:
    req = comm_world.irecv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    for i in range(20):
        time.sleep(0.1)
        r = req.test()
        print("RANK",rank_world,"-",r[0])
        if r[0]:
            data=r[1]
            print(data)