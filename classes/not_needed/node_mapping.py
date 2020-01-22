#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:15:54 2019

@author: simona
"""

#from mpi4py import MPI
#import psutil
#import sys
#
#comm_world = MPI.COMM_WORLD
#size_world = comm_world.Get_size()
#rank_world = comm_world.Get_rank()
#myhost = MPI.Get_processor_name()
#
#print(size_world, rank_world, myhost, psutil.Process().cpu_num())
#
#if rank_world==0:
#    mpi_info = MPI.Info.Create()
#    mpi_info.Set("host", myhost)
##    mpi_info.Set("rank_by", "core")
#    mpi_info.Set("bind_to", "none")
##    mpi_info.Set("npernode", "5")
##    mpi_info.Set("rankfile","rankfilehh")
#    comm = MPI.COMM_SELF.Spawn(sys.executable,
#                               args=['child.py'],
#                               maxprocs=5,
#                               info=mpi_info)
#    comm.Disconnect()