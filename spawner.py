# -*- coding: utf-8 -*-

from mpi4py import MPI
import sys

comms = []

for seed in range(60):
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=['test_deflation_interfaces.py',str(seed)], maxprocs=1)
    comms.append(comm)

# for seed in range(10):
#     print(str(seed))
#     command = "mpirun -n 1 python3 -m mpi4py surrDAMH/process_SAMPLER.py " + str(seed) + " : -n 1 python3 -m mpi4py surrDAMH/process_SOLVER.py interfaces1_updates_rbf500_1proc : -n 1 python3 -m mpi4py surrDAMH/process_COLLECTOR.py"
#     comm = MPI.COMM_SELF.Spawn(sys.executable, args=[command], maxprocs=3)
#     comms.append(comm)

# for comm in comms:
#     comm.Disconnect()


# mpirun -n 1 python3 -m mpi4py surrDAMH/process_SAMPLER.py 3 : -n 1 python3 -m mpi4py surrDAMH/process_SOLVER.py interfaces1_updates_rbf500_1proc : -n 1 python3 -m mpi4py surrDAMH/process_COLLECTOR.py