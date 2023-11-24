#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpirun -n 6 python3 -m mpi4py run_processes.py output_dir examples/simple.json
"""

from mpi4py import MPI
import numpy as np
import sys
from collections import deque
from surrDAMH.configuration import Configuration
import surrDAMH.process_SAMPLER
import surrDAMH.process_SOLVER
import surrDAMH.process_COLLECTOR

output_dir = sys.argv[1]
conf_file_path = sys.argv[2]

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

use_surrogate = True
if use_surrogate:
    no_samplers = size_world - 2
else:
    no_samplers = size_world - 1
if no_samplers < 1:
    print("Number of MPI processes is too low. Use at least \"mpirun -n 4\".")

C = Configuration()
C.set_from_file(no_samplers, output_dir, conf_file_path)

if rank_world == no_samplers:
    surrDAMH.process_SOLVER.run_SOLVER(C)
elif rank_world == no_samplers+1:
    surrDAMH.process_COLLECTOR.run_COLLECTOR(C)
else:
    surrDAMH.process_SAMPLER.run_SAMPLER(C)

comm_world.Barrier()
print("RANK", rank_world, "terminated.")
