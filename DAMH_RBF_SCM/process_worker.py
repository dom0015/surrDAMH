#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:06:03 2018

@author: dom0015
"""

import numpy as np
import time

from MHsampler import MHsampler

def worker(num,comm_world,Model,SF,Surrogate,no_solvers,initial_samples,group_leader_ids):
    print("MPI RANK",comm_world.Get_rank())
    if num==0:
        print("Process", num, "is the timer.")
        for i in range(len(SF.stages)):
            time.sleep(SF.stages[i].limit_time)
            with SF.shared_finisher.get_lock():
                if SF.shared_finisher.value < i+1:
                    SF.shared_finisher.value = i+1
                    print("FINISHER VALUE CHANGED TO", SF.shared_finisher.value, "(by timer)")
        time.sleep(60)
                    
    if num==1:
        print("Process", num, "updates the surrogate model.")
        alldata_par = np.zeros([0,Model.no_parameters])
        alldata_obs = np.zeros([0,Model.no_observations])
        alldata_wei = np.zeros([0,1])
        size_all=0
        # snapshots are collected, surrogate model is updated (up to max_snapshots)
        while SF.shared_finisher.value < len(SF.stages):
#            print("Process num",num)
            time.sleep(0.1)
            queue_size=SF.shared_queue_surrogate.qsize()
#            print("DEBUG surr", queue_size)
#            print(queue_size, SF.shared_bool_update.value)
            if queue_size > 0: #and SF.shared_bool_update.value==1:
                # receive new data
                size_all += queue_size
                alldata_par, alldata_obs, alldata_wei, newdata_par, newdata_obs, newdata_wei = Surrogate.add_data(alldata_par,alldata_obs,alldata_wei,SF.shared_queue_surrogate, queue_size, Model)
                no_evaluations = alldata_par.shape[0]
#                print(no_evaluations)
                if no_evaluations >= SF.min_snapshots:# or SF.shared_finisher.value>0:
#                    print("DEBUG:", no_evaluations, SF.shared_finisher.value, queue_size)
                    SOL, no_evaluations, alldata_par, alldata_obs, alldata_wei, TEMP2, RHS = Surrogate.calculate(alldata_par,alldata_obs,alldata_wei)
                    if SF.shared_queue_updater.qsize() > 0: # this surrogate was not received
                        old1, old2, old3 = SF.shared_queue_updater.get()
                    SF.shared_queue_updater.put([0,alldata_par,SOL])
                if no_evaluations >= SF.max_snapshots:
                    print("Number of snapshots", no_evaluations, "exceeded maximum", SF.max_snapshots)
                    return
    
    if num==2:
        print("Process", num, "evaluates the surrogate model.")
        tag, alldata_par, SOL = SF.shared_queue_updater.get()
        print('Surrogate solver received first model.',alldata_par.shape)
        # while at least one chain is being constructed
        while SF.shared_finisher.value < (len(SF.stages)+SF.no_chains):
            try:
                chain_id, data_par, data_par_old = SF.shared_queue_surrogate_solver.get(timeout=1)
                newdata_par = np.zeros([0,Model.no_parameters])
                chain_ids = []
                newdata_par = np.vstack((newdata_par, data_par))
                chain_ids.append(chain_id)
                newdata_par = np.vstack((newdata_par, data_par_old))
                updater_queue_size=SF.shared_queue_updater.qsize()
                if updater_queue_size > 0:
                    for i in range(updater_queue_size):
                        tag, alldata_par, SOL = SF.shared_queue_updater.get()
                    print('Surrogate solver received updated model.',alldata_par.shape, updater_queue_size)            
                queue_size=SF.shared_queue_surrogate_solver.qsize()
                for i in range(queue_size):
                    chain_id, data_par, data_par_old = SF.shared_queue_surrogate_solver.get()
                    newdata_par = np.vstack((newdata_par, data_par))
                    chain_ids.append(chain_id)
                    newdata_par = np.vstack((newdata_par, data_par_old))
                newdata_surrogate = Surrogate.apply(SOL, newdata_par, alldata_par, Model.no_parameters, 0)
                j=0
                for i in chain_ids: # send surrogate results
                    SF.shared_parents_surrogate[i].send([newdata_surrogate[j],newdata_surrogate[j+1],1])
                    j += 2
            except:
                aa=0
#                print("Surrogate queue empty on timeout.")

#    if num == 3:
#        print("Process num", num)
#        while SF.shared_finisher.value < (len(SF.stages)+SF.no_chains):
#            # waits to receive a request and re-sends it to a free solverf
#            no_free_solver = True
#            for id_solver, is_free in enumerate(SF.shared_free_solvers):
#                if is_free.value == True and no_free_solver:
#                    no_free_solver = False
#                    print("Waiting for request")
#                    num_child, data_par = SF.shared_queue_solver.get()
#                    print("nu" , num_child, data_par)
##                    id_solver = SF.shared_free_solvers.index(True)
##                    from_child[id_solver] = num_child
##                    parameters[id_solver] = data_par
#                    sent_data = np.append(num_child, data_par)
#                    comm_world.Isend(sent_data, dest=group_leader_ids[id_solver], tag=1)
#                    print("Worker Isend MPI data",sent_data,"to solver",group_leader_ids[id_solver])
#                    SF.shared_free_solvers[id_solver].value = False
#            if no_free_solver:
#                print("no free solver, sleep")
#                time.sleep(0.1)

    if num>2:
        print("Process", num, "generates a chain.")
        Sampler = MHsampler(num-SF.no_helpers)
        if initial_samples.all() == None:
            Sampler.ChooseInitialSample()
        else:     
            Sampler.initialSample = initial_samples[num-SF.no_helpers,:]
        Sampler.BeforeSampling()
        for i in range(len(SF.stages)):
            Sampler.proposalStd = SF.stages[i].proposalStd
            if SF.stages[i].type_sampling == 'MH':
                with SF.shared_bool_update.get_lock():
                    SF.shared_bool_update.value = 1
                Sampler.MH(SF.stages[i],i+1)
            elif SF.stages[i].type_sampling == 'DAMHSMU':
                with SF.shared_bool_update.get_lock():
                    SF.shared_bool_update.value = 1
                Sampler.DAMHSMU(SF.stages[i],i+1)
            elif SF.stages[i].type_sampling == 'ADAMH':
                with SF.shared_bool_update.get_lock():
                    SF.shared_bool_update.value = 0
                Sampler.ADAMH(SF.stages[i],i+1)
            else:
                with SF.shared_bool_update.get_lock():
                    SF.shared_bool_update.value = 0
                Sampler.DAMHSMU(SF.stages[i],i+1)
            with SF.shared_finisher.get_lock():
                if SF.shared_finisher.value == 0:
                    SF.shared_finisher.value = 1
                    print("FINISHER VALUE CHANGED TO", SF.shared_finisher.value, "by process", num)
                if SF.shared_finisher.value < i+1:
                    SF.shared_finisher.value = i+1
                    print("FINISHER VALUE CHANGED TO", SF.shared_finisher.value, "by process", num)
        Sampler.Finalize()
        with SF.shared_finisher.get_lock():
            SF.shared_finisher.value += 1
            print("FINISHER VALUE CHANGED TO", SF.shared_finisher.value, "by process", num)
    print("Process num.", num, "finished.")
#    return