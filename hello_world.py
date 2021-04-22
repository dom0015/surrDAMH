# -*- coding: utf-8 -*-

from mpi4py import MPI

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

print("Hello from rank", rank_world, "of", size_world)