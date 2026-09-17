#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:47:09 2019

@author: simona
"""

import sys
import time
import traceback

import numpy as np
from mpi4py import MPI

from surrDAMH.configuration import Configuration
from surrDAMH.modules.communication import ABORT_GRACE_SECONDS
from surrDAMH.solver_specification import SolverSpec
from surrDAMH.solvers import get_solver_from_spec

assert (len(sys.argv) == 3)
solver_id = int(sys.argv[1])
solver_output_dir = sys.argv[2]

parent_comm = MPI.Comm.Get_parent()
rank = parent_comm.Get_rank()

# WS8 fail-loud: the child is spawned as a plain "python process_CHILD.py" (not
# "python -m mpi4py"), so an uncaught exception used to kill only this process while the
# solvers pool kept polling Iprobe() forever and the whole job hung until it was killed
# (measured: a solver raising on its 5th evaluation hangs the job indefinitely). Aborting
# MPI_COMM_WORLD of the spawned group makes the launcher tear the whole job down.
# KeyboardInterrupt / SystemExit are re-raised untouched: they are a requested interruption
# or exit rather than a solver failure, and the launcher forwards signals to every process.
try:
    # new:
    tmp = None
    [conf, solver_spec] = parent_comm.bcast(tmp, root=0)
    conf: Configuration
    solver_spec: SolverSpec

    """ INITIALIZATION OF THE SOLVER """
    solver_instance = get_solver_from_spec(solver_spec, solver_id, solver_output_dir)

    """ SOLVING INCOMING REQUESTS USING LINKED SOLVER """
    # tag is broadcasted by parent
    # parameters are broadcasted by parent
    # methods "set_parameters" and "get_observations" are called by all ranks
    # observations are sent to parent by rank 0
    received_data = np.empty(conf.no_parameters, dtype='d')
    tag = np.array(0, dtype='i')
    solver_is_active = True
    counter = 0
    while solver_is_active:
        parent_comm.Bcast([tag, MPI.INT], root=0)
        if tag == 0:
            parent_comm.Barrier()
            parent_comm.Disconnect()
            solver_is_active = False
        else:
            parent_comm.Bcast([received_data, MPI.DOUBLE], root=0)
            solver_instance.set_parameters(received_data.reshape((conf.no_parameters,)))
            if conf.solver_returns_tag:
                [sent_data, solver_tag] = solver_instance.get_observations()
                if solver_tag < 0:
                    sent_data = np.zeros((conf.no_observations,))
            else:
                sent_data = solver_instance.get_observations()
                solver_tag = 0
            counter += 1
            if rank == 0:
                if conf.pickled_observations:
                    parent_comm.send([sent_data, solver_tag], dest=0, tag=int(tag))
                else:
                    parent_comm.Send(sent_data, dest=0, tag=solver_tag)
    print("Solver at spawned process - evaluations:", counter, flush=True)
except (KeyboardInterrupt, SystemExit):
    raise
except Exception:
    print(f"FATAL: unhandled exception in spawned solver child (solver_id={solver_id},"
          f" rank {rank} of the spawned group); aborting the whole job.", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.stderr.flush()
    sys.stdout.flush()
    time.sleep(ABORT_GRACE_SECONDS)  # let the launcher forward the traceback before it kills the job
    MPI.COMM_WORLD.Abort(1)
    raise  # not reached (Abort does not return), kept so the exception is never swallowed
