# -*- coding: utf-8 -*-

from mpi4py import MPI
import sys

comms = []

for seed in range(60):
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=['test_deflation_interfaces.py',str(seed)], maxprocs=1)
    comms.append(comm)
    
# for comm in comms:
#     comm.Disconnect()
