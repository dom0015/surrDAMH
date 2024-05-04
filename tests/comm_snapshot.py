import time

import numpy as np
from mpi4py import MPI

from surrDAMH.modules.communication import (CommSnapshot_collector,
                                            CommSnapshot_sampler)

# get mpi rank:
comm_world = MPI.COMM_WORLD
rank = comm_world.Get_rank()
size = comm_world.Get_size()

if rank == 0:  # SAMPLER
    comm = CommSnapshot_sampler(rank_collector=1, max_sampler_isend_requests=10)
    counter = 0
    for i in range(1000):
        # time.sleep(np.random.standard_exponential())
        counter += 1
        snapshot = [counter]
        comm.send_to_collector(snapshot)
    print("SAMPLER: Snapshots sent:", snapshot, flush=True)
    comm.terminate()

else:  # COLLECTOR
    comm = CommSnapshot_collector(rank_sampler=0)
    time.sleep(20.0)
    print("Start receiving", flush=True)

    """ TYPE 1: infinite loop"""
    while True:
        if not comm.all_snapshots_received():
            c = 0
            while comm.snapshot_is_available():
                snapshot = comm.get_snapshot_and_request_new()
                c += 1
            if c > 0:
                print("COLLECTOR: Snapshot received:", c, snapshot, flush=True)
        else:
            print("COLLECTOR: All snapshots received", flush=True)
            break
    #comm.terminate()  # should do nothing here
    """"""

    """ TYPE 2: the loop stops sooner for some reason"""
    # for i in range(10):
    #     time.sleep(np.random.standard_exponential())
    #     if not comm.all_snapshots_received():
    #         if comm.snapshot_is_available():
    #             snapshot = comm.get_snapshot_and_request_new()
    #             print("COLLECTOR: Snapshot received:", snapshot, flush=True)
    #     else:
    #         print("COLLECTOR: All snapshots received", flush=True)
    #         break
    # comm.terminate()  # receive all remaining snapshots
    """"""


comm_world.Barrier()
print("After barrier, rank: ", rank, flush=True)
