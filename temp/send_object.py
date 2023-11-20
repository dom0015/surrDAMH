# send class instance using mpi4py
# mpirun -n 2 python3 -m mpi4py temp/send_object.py 

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class my_class(object):
    def __init__(self):
        self.x = 10

    def func(self, rank):
        print("Rank=%d is able to calculate the result: %d " % (rank, np.power(self.x, 2)))

obj= None
if rank == 0:
    obj = my_class()
    comm.send(obj, dest=1)

if rank == 1:
    obj = comm.recv(obj, source=0)
    obj.func(rank)