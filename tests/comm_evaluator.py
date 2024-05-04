import time

import numpy as np
from mpi4py import MPI

from surrDAMH.modules.communication import (CommEvaluator_collector,
                                            CommEvaluator_sampler)

# get mpi rank:
comm_world = MPI.COMM_WORLD
rank = comm_world.Get_rank()
size = comm_world.Get_size()

if rank == 0:  # SAMPLER
    comm = CommEvaluator_sampler(rank_collector=1, max_buffer_size=int(1 << 30))
    comm.request_evaluator()
    counter = 0
    tt = time.time()
    while time.time() - tt < 10.0:
        # time.sleep(0.01)
        avail = comm.evaluator_is_available()
        # print("SAMPLER: Evaluator available:", avail, flush=True)
        if avail:
            evaluator = comm.get_evaluator()
            comm.request_evaluator()
            counter += 1
            print("SAMPLER: Evaluator received:", evaluator[0], counter, flush=True)

    last_evaluator = comm.get_evaluator_and_terminate()
    print("SAMPLER: Last evaluator received:", last_evaluator[0], flush=True)

else:  # COLLECTOR
    comm = CommEvaluator_collector(rank_sampler=0)
    evaluator = [0]
    update_counter = 0
    last_sent = 0
    while True:
        # print("collector loop running", flush=True)
        # time.sleep(np.random.standard_exponential())
        if True:  # np.random.rand() > 0.5:
            update_counter += 1
            # evaluator = np.full((1_000_000,), update_counter, dtype=int)
            # print("COLLECTOR: Evaluator constructed", update_counter)
        if update_counter > last_sent:
            if comm.sampler_requests_evaluator():
                evaluator = np.full((1_000_000,), update_counter, dtype=int)
                comm.send_evaluator(evaluator)
                last_sent: int = update_counter
                mem = MPI.memory.frombuffer(evaluator).nbytes
                print("COLLECTOR: Evaluator sent:", update_counter, 0.001 * mem, "kB", flush=True)
        if update_counter > 0:  # cannot terminate if no evaluator is constructed
            if comm.sampler_stops():
                if update_counter == last_sent:
                    evaluator = [None]
                comm.terminate(evaluator)
                print("COLLECTOR: termination:", update_counter, flush=True)
                break

comm_world.Barrier()
print("After barrier, rank: ", rank, flush=True)
