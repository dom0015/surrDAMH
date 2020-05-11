#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:06:03 2018

@author: dom0015
"""

import numpy as np
import numpy.matlib as npm
import scipy.sparse.linalg as splin

import time
#import process_master as MPI

import MHsampler
import kernel
import rbf

def worker(num,no_observations,no_parameters,no_solvers, no_chains, no_helpers, initial_samples, shared_finisher, shared_queue_solver, shared_children_solver,shared_queue_updater,shared_queue_surrogate_solver,shared_queue_surrogate,shared_parents_surrogate,shared_children_surrogate,group_leader_ids,proposalStd):
    if num==0:
        print("Ahoj, ja jsem timer!")
        time.sleep(60)
        with shared_finisher.get_lock():
            time.sleep(0.1)
            shared_finisher.value = 3
        print("Timer set to 1!")
    if num==1:
        print("Ahoj, ja upgraduji surrogate model!")
        alldata_par = np.zeros([0,no_parameters])
        alldata_obs = np.zeros([0,no_observations])
        size_all=0
        while shared_finisher.value == 0:
            time.sleep(0.1)
            queue_size=shared_queue_surrogate.qsize()
            if queue_size > 0:
                size_all += queue_size
                alldata_par, alldata_obs, newdata_par, newdata_obs = rbf.add_data(alldata_par,alldata_obs,shared_queue_surrogate, queue_size, no_parameters, no_observations)
#            alldata_par = np.vstack((alldata_par, newdata_par))
#            alldata_obs = np.vstack((alldata_obs, newdata_obs)) 

        # calculate first surrogate model and send to surrogate solver
        SOL, no_evaluations, alldata_par, alldata_obs, TEMP2, RHS = rbf.calculate(alldata_par,alldata_obs,no_parameters,None,0,0,0,0)

        if size_all>0 and shared_finisher.value < 3:
            shared_queue_updater.put([0,alldata_par,SOL])
        while shared_finisher.value < 3:
            time.sleep(0.1)
            queue_size=shared_queue_surrogate.qsize()
            if queue_size > 0:
                # receive new data
                size_all += queue_size
                alldata_par, alldata_obs, newdata_par, newdata_obs = rbf.add_data(alldata_par,alldata_obs,shared_queue_surrogate, queue_size, no_parameters, no_observations)
#                alldata_par = np.vstack((alldata_par, newdata_par))
#                alldata_obs = np.vstack((alldata_obs, newdata_obs))
                # calculate updated model and send to surrogate solver
                no_evaluations_old = no_evaluations
                no_evaluations = alldata_par.shape[0]
                initial_iteration=np.append(SOL[:-1],np.zeros([no_evaluations-no_evaluations_old,1]))
                initial_iteration=np.append(initial_iteration,SOL[-1])
                SOL, no_evaluations, alldata_par, alldata_obs, TEMP2, RHS = rbf.calculate(alldata_par,alldata_obs,no_parameters,None,0,0,0,0)
                shared_queue_updater.put([0,alldata_par,SOL])
                if no_evaluations > 5000:
                    print("No evaluations:", no_evaluations)
                    return
    
    if num==2:
        print("Ahoj, ja vyhodnocuji surrogate model!")
        tag, alldata_par, SOL = shared_queue_updater.get()
        print("Tag is",tag)
        if tag == 0:
            with shared_finisher.get_lock():
                if shared_finisher.value == 1:
                    shared_finisher.value = 2
                    version = 1
                    print('Surrogate solver received first model.',alldata_par.shape)
        else:
            return
        chains_version = np.zeros(no_chains)
        while shared_finisher.value < (3+no_chains):
            time.sleep(0.1)
            queue_size=shared_queue_surrogate_solver.qsize()
            if queue_size > 0:
#                print('Surrogate solver received data', queue_size)
                newdata_par = np.zeros([0,no_parameters])
                chain_ids = []
                for i in range(queue_size):
                    chain_id, data_par, data_par_old = shared_queue_surrogate_solver.get()
                    newdata_par = np.vstack((newdata_par, data_par))
                    chain_ids.append(chain_id)
                    if chains_version[chain_id]<version:
                        newdata_par = np.vstack((newdata_par, data_par_old))
                newdata_surrogate = rbf.apply(newdata_par, alldata_par, no_parameters, SOL,0)
                j=0
                for i in chain_ids: # send surrogate results
                    if chains_version[i]<version:
                        shared_parents_surrogate[i].send([newdata_surrogate[j],newdata_surrogate[j+1],1])
                        j += 2
                    else:
                        shared_parents_surrogate[i].send([newdata_surrogate[j],None,0])
                        j += 1
#                print('Surrogate solver returned solution in', time.time()-t)
            queue_size=shared_queue_updater.qsize()
            if queue_size > 0:
                version +=1
                for i in range(queue_size):
                    tag, alldata_par, SOL = shared_queue_updater.get()
#                    print(data_par.shape,alldata_par.shape)
#                    alldata_par = np.vstack((alldata_par, data_par))
                print('Surrogate solver received updated model.',version,alldata_par.shape, queue_size)
        

    if num>2:
        print("Thread", num, "generates a chain.")
        Sampler = MHsampler.MHsampler(proposalStd, num-no_helpers, 0, shared_queue_solver, shared_queue_surrogate_solver, shared_children_solver[num-no_helpers], shared_children_surrogate[num-no_helpers])
        Sampler.shared_finisher = shared_finisher
#        Sampler.ChooseInitialSample()
        Sampler.initialSample = initial_samples[num-no_helpers,:]
        Sampler.MH()
        print('MH sampling finished')
        Sampler.DAMHSMU()
        Sampler.Finalize()
        with shared_finisher.get_lock():
            shared_finisher.value += 1
    print("NUM", num, "FINISHED")
    return