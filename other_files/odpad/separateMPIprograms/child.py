#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

comm_world = MPI.COMM_WORLD
size_world = comm_world.Get_size()
rank_world = comm_world.Get_rank()

print("CHILD", rank, size, rank_world, size_world)

#N = np.array(0, dtype='i')
#comm.Bcast([N, MPI.INT], root=0)
#h = 1.0 / N; s = 0.0
#for i in range(rank, N, size):
#    x = h * (i + 0.5)
#    s += 4.0 / (1.0 + x**2)
#PI = np.array(s * h, dtype='d')
#comm.Reduce([PI, MPI.DOUBLE], None, op=MPI.SUM, root=0)

received_data = np.zeros(10,)
comm.Bcast(received_data, root=0)
sent_data = np.ones(5,)*np.sum(received_data)
if rank_world==0:
    comm.Send(sent_data, dest=0, tag=11)
    print(sent_data)

comm.Disconnect()