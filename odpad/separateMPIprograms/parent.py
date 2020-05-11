#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_SELF.Spawn(sys.executable, args=['child.py'], maxprocs=3)
size = comm.Get_size()
rank = comm.Get_rank()

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

print("PARENT", rank, size, rank_world, size_world)

#N = np.array(10, 'i')
#comm.Bcast([N, MPI.INT], root=MPI.ROOT)
#PI = np.array(0.0, 'd')
#comm.Reduce(None, [PI, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
#print(PI)

sent_data = np.arange(10,dtype='double')
comm.Bcast(sent_data, root=MPI.ROOT)
received_data = np.empty(5,)
comm.Recv(received_data, source=0, tag=11)
print(received_data)

comm.Disconnect()