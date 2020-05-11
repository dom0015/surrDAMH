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
#from mpi4py import MPI

def master(comm_world,Model,SF,G_data,no_solvers,group_leader_ids):
    print('Master process communicates with solver(s).')
#    free_solvers = [True] * no_solvers
#    from_child = [0] * no_solvers
#    parameters = [0] * no_solvers
    data_par = np.zeros(Model.no_parameters)
    
#    unsolved_waiting = False
#    while SF.shared_finisher.value < (len(SF.stages)+SF.no_chains):
#        if not unsolved_waiting:
#            try:
#                num_child, data_par = SF.shared_queue_solver.get(timeout=0.001)
#                unsolved_waiting = True
#            except:
#                print("Solver queue empty on timeout.")
#        if (True in free_solvers) and unsolved_waiting:
#            id_solver = free_solvers.index(True)
#            from_child[id_solver] = num_child
#            parameters[id_solver] = data_par
#            comm_world.Isend(data_par, dest=group_leader_ids[id_solver], tag=1)
#            free_solvers[id_solver] = False
#            unsolved_waiting = False
#        # check if data from busy solvers are available and receive them
#        for id_solver, is_free in enumerate(free_solvers):
#            if is_free == False:
#                if comm_world.Iprobe(source=group_leader_ids[id_solver],tag=1):
#                    data_obs = np.zeros(Model.no_observations)
#                    comm_world.Recv(data_obs, source=group_leader_ids[id_solver],tag=1)
#                    SF.shared_parents_solver[from_child[id_solver]].send(data_obs)
##                    SF.shared_queue_surrogate.put([parameters[id_solver],data_obs,data_wei])
#                    G_data = np.vstack([G_data,np.append(data_obs,parameters[id_solver])])
#                    free_solvers[id_solver] = True
#    status = MPI.Status()
#    while SF.shared_finisher.value < (len(SF.stages)+SF.no_chains):
#        # waits for solution from any source (solver)
#        print("Master is waiting for MPI solution from any source")
#        comm_world.Probe(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
#        t = status.Get_tag()
#        id_solver_leader = status.Get_source()
#        print("Master found MPI solution from source", id_solver_leader, t)
#
##        for id_solver, is_free in enumerate(SF.shared_free_solvers):
##            if is_free.value == False:
##                if comm_world.Iprobe(source=group_leader_ids[id_solver],tag=1):
#        received_data = np.zeros(Model.no_observations + Model.no_parameters + 1)
#        print("Master is waiting for MPI solution from source",id_solver_leader)
#        comm_world.Recv(received_data, source=id_solver_leader,tag=1)
#        print("Master received MPI solution",received_data, "from", id_solver_leader)
#        num_child = int(received_data[0])
#        data_par = received_data[1:Model.no_parameters+1]
#        data_obs = received_data[1+Model.no_parameters:]
#        SF.shared_parents_solver[num_child].send(data_obs)
##                    SF.shared_queue_surrogate.put([parameters[id_solver],data_obs,data_wei])
#        print("Master sent data to child", num_child)
#        id_solver = group_leader_ids.index(id_solver_leader)
#        print("D", id_solver, id_solver_leader)
#        print(data_obs,data_par)
#        G_data = np.vstack([G_data,np.append(data_obs,data_par)])
#        print("DD")
#        SF.shared_free_solvers[id_solver].value = True
#        print("CC")
        

        # checks if data from busy solvers are received

    while SF.shared_finisher.value < (len(SF.stages)+SF.no_chains):
        time.sleep(0.001)
        # if (full solver) queue is not empty, sends request to free solver (if any)
        if not SF.shared_queue_solver.empty():
            for id_solver, is_free in enumerate(SF.shared_free_solvers):
                if is_free.value == True:
                    num_child, data_par = SF.shared_queue_solver.get()
#                    id_solver = SF.shared_free_solvers.index(True)
#                    from_child[id_solver] = num_child
#                    parameters[id_solver] = data_par
                    sent_data = np.append(num_child,data_par)
                    comm_world.Isend(sent_data, dest=group_leader_ids[id_solver], tag=1)
                    SF.shared_free_solvers[id_solver].value = False
                    break
        # checks if data from busy solvers are received
        for id_solver, is_free in enumerate(SF.shared_free_solvers):
            if is_free.value == False:
                if comm_world.Iprobe(source=group_leader_ids[id_solver],tag=1):
                    received_data = np.zeros(1 + Model.no_parameters + Model.no_observations)
                    comm_world.Recv(received_data, source=group_leader_ids[id_solver],tag=1)
                    num_child = int(received_data[0])
                    data_par = received_data[1:(Model.no_parameters + 1)]
                    data_obs = received_data[(Model.no_parameters + 1):]
                    SF.shared_parents_solver[num_child].send(data_obs)
#                    SF.shared_queue_surrogate.put([parameters[id_solver],data_obs,data_wei])
                    G_data = np.vstack([G_data,np.append(data_obs,data_par)])
                    SF.shared_free_solvers[id_solver].value = True
        
#        print("Process master")
#        time.sleep(0.01)
#        # if (full solver) queue is not empty, sends request to free solver (if any)
#        if not SF.shared_queue_solver.empty():
#            if True in free_solvers:
#                num_child, data_par = SF.shared_queue_solver.get()
#                id_solver = free_solvers.index(True)
#                from_child[id_solver] = num_child
#                parameters[id_solver] = data_par
#                comm_world.Isend(data_par, dest=group_leader_ids[id_solver], tag=1)
#                free_solvers[id_solver] = False
#        # checks if data from busy solvers are received
#        for id_solver, is_free in enumerate(free_solvers):
#            if is_free == False:
#                if comm_world.Iprobe(source=group_leader_ids[id_solver],tag=1):
#                    data_obs = np.zeros(Model.no_observations)
#                    comm_world.Recv(data_obs, source=group_leader_ids[id_solver],tag=1)
#                    SF.shared_parents_solver[from_child[id_solver]].send(data_obs)
##                    SF.shared_queue_surrogate.put([parameters[id_solver],data_obs,data_wei])
#                    G_data = np.vstack([G_data,np.append(data_obs,parameters[id_solver])])
#                    free_solvers[id_solver] = True
                    
    # sends finishing signals to all solvers
    print("Master before wtf")
    G_DataFrame = pd.DataFrame(G_data)
    G_DataFrame.to_csv('G_data_linela_MH.csv')
    for idx in group_leader_ids:
        comm_world.Send(data_par, dest=idx, tag=150) # change to None?
    print("Master process finished")