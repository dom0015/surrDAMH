"""
Created on Thu Jul 19 10:43:11 2018

@author: dom0015
"""

#from mpi4py import rc
#rc.initialize = False
#rc.finalize = False
import numpy as np
import pandas as pd
import time

def master(comm_world,no_observations,no_parameters,G_data,shared_finisher,no_solvers,no_chains,shared_queue_solver,shared_parents_solver,shared_queue_surrogate,group_leader_ids):
    print('Hello, I am master. I communicate with solver(s).')
    free_solvers = [True] * no_solvers
    from_child = [0] * no_solvers
    parameters = [0] * no_solvers
    data_par = np.zeros(no_parameters)
    no_evaluations = 0
    bound_MH = 100
    while shared_finisher.value < (3+no_chains):
        time.sleep(0.1)
        # if queue is not empty, sends request to free solver (if any)
        if not shared_queue_solver.empty():# == False:
            if True in free_solvers:
                num_child, data_par = shared_queue_solver.get()
                id_solver = free_solvers.index(True)
                from_child[id_solver] = num_child
                parameters[id_solver] = data_par
                comm_world.Isend(data_par, dest=group_leader_ids[id_solver], tag=1)
#                print('Isent data to rank_world', group_leader_ids[id_solver], 'by master:', data_par)
                free_solvers[id_solver] = False
        # checks if data from busy solvers are received
        for id_solver, is_free in enumerate(free_solvers):
            if is_free == False:
                if comm_world.Iprobe(source=group_leader_ids[id_solver],tag=1):
                    data_obs = np.zeros(no_observations)
                    comm_world.Recv(data_obs, source=group_leader_ids[id_solver],tag=1)
#                    print('Received data from rank_world', group_leader_ids[id_solver], 'by master:', data_obs)
                    shared_parents_solver[from_child[id_solver]].send(data_obs)
                    shared_queue_surrogate.put([parameters[id_solver],data_obs])
                    G_data = np.vstack([G_data,np.append(data_obs,parameters[id_solver])])
                    free_solvers[id_solver] = True
                    no_evaluations += 1
        # signal to switch from MH to DAMH-SMU
        if no_evaluations >= bound_MH:
            with shared_finisher.get_lock():
                if shared_finisher.value == 0:
                    shared_finisher.value = 1
    # sends finishing signals to all solvers
    G_DataFrame = pd.DataFrame(G_data)
#    G_DataFrame.to_csv('G_data2_'+str(no_parameters)+'par.csv')
#    G_DataFrame.to_csv('G_data_linela.csv')
    G_DataFrame.to_csv('G_data_linela_MH.csv')
    for idx in group_leader_ids:
        comm_world.Send(data_par, dest=idx, tag=150) # change to None?