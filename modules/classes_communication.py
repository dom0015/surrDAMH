#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import sys
import time

class Solver_MPI_parent: # initiated by full SOLVER
    def __init__(self, no_parameters, no_observations, maxprocs=1):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_requests = 1
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=['modules/process_CHILD.py'], maxprocs=maxprocs)
        self.tag = 0
        self.received_data = np.zeros(self.no_observations)
    
    def send_parameters(self, data_par):
        self.tag += 1
        self.data_par = data_par.copy()
        self.comm.Send(self.data_par, dest=0, tag=self.tag)
#        print('DEBUG - PARENT Send request FROM', self.comm.Get_rank(), '(', MPI.COMM_WORLD.Get_rank(), ')', 'TO child:', 0, "TAG:", self.tag)
        
    def recv_observations(self):
        self.comm.Recv(self.received_data, source=0, tag=self.tag)
#        print('DEBUG - PARENT Recv solution FROM child', 0, 'TO:', self.comm.Get_rank(), '(', MPI.COMM_WORLD.Get_rank(), ')', "TAG:", self.tag)
        return self.received_data.reshape((1,-1)).copy()
    
    def is_solved(self):
        # check the parent-child communicator if there is an incoming message
        tmp = self.comm.Iprobe(source=0, tag=self.tag)
        if tmp:
            return True
        else:
            return False
    
    def terminate(self):
        self.comm.Send(np.empty((1,self.no_parameters)), dest=0, tag=0)
        self.comm.Barrier()
        self.comm.Disconnect()
        print("Solver spawned by rank", MPI.COMM_WORLD.Get_rank(), "disconnected.")
    
class Solver_MPI_collector_MPI: # initiated by SAMPLERs
    def __init__(self, no_parameters, no_observations, rank_solver, is_updated=False, rank_collector=None):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_requests = 1
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.rank_solver = rank_solver
        self.is_updated = is_updated
        self.rank_collector = rank_collector
        self.tag_solver = 0
#        self.tag_collector = 0
        self.received_data = np.zeros(self.no_observations)
        self.terminated_solver = None
        self.terminated_collector = True
        if not rank_collector is None:
            self.terminated_collector = False
        self.empty_buffer = np.zeros(1)
        
    def send_parameters(self, sent_data):
        self.tag_solver += 1
        self.comm.Send(sent_data, dest=self.rank_solver, tag=self.tag_solver)
#        print('DEBUG - Solver_MPI_collector_MPI (', self.rank, ') Send request FROM', self.rank, 'TO:', self.rank_solver, "TAG:", self.tag_solver)
    
    def recv_observations(self, ):
        self.comm.Recv(self.received_data, source=self.rank_solver, tag=self.tag_solver)
#        print('DEBUG - Solver_MPI_collector_MPI (', self.rank, ') Recv solution FROM', self.rank_solver, 'TO:', self.rank, "TAG:", self.tag_solver)
        return self.received_data.copy()
    
#    def send_to_data_collector(self, sent_snapshot):
#        # needed only if self.is_updated == True
#        self.tag_collector += 1
#        print('DEBUG - Solver_MPI_collector_MPI (', self.rank, ') - sent_snapshot', self.rank_collector, self.comm.Get_size(), self.rank)
#        self.comm.send(sent_snapshot, dest=self.rank_collector, tag=self.tag_collector)
        
#TO DO: implement is_solved method (for non-spawned full solvers)
#    def is_solved(self):
#        # check COMM_WORLD if there is an incoming message from the solver
#        tmp = self.comm.Iprobe(source=self.rank_solver, tag=self.tag_solver)
#        if tmp:
#            return True
#        else:
#            return False
        
    def terminate(self, ):
        # TO DO: assume that terminate may be called multiple times
        if not self.terminated_solver:
#            print('DEBUG - Solver_MPI_collector_MPI terminate',self.rank_solver, self.comm.Get_size(), self.rank)
            self.comm.Send(self.empty_buffer, dest=self.rank_solver, tag=0)
            self.terminated_solver = True
        if not self.terminated_collector:
            self.comm.send([], dest=self.rank_collector, tag=0)
            self.terminated_collector = True

class Solver_local_collector_MPI: # initiated by SAMPLERs
    # local solver (evaluated on SAMPLERs)
    # with external data COLLECTOR (separate MPI process)
    # communicates with: COLLECTOR
    def __init__(self, no_parameters, no_observations, local_solver_instance, is_updated=False, rank_collector=None):
        self.max_requests = 1
        self.comm = MPI.COMM_WORLD
        self.local_solver_instance = local_solver_instance
        self.is_updated = is_updated
        self.rank_collector = rank_collector
        self.solver_data = [None] * 2       # double buffer
        self.solver_data_idx = 0            # idx of current buffer 0/1
        self.solver_data_iterator = 0
        self.tag_terminate = 0
        self.tag_ready_to_receive = 1
        self.tag_sent_data = 2
        self.terminated = None
        self.terminated_data = True
        if not rank_collector is None:
            self.terminated_data = False
        self.computation_in_progress = False # TO DO: remove?
        self.request_recv = self.comm.irecv(source=self.rank_collector, tag=self.tag_sent_data)
        self.request_send = None
        self.empty_buffer = np.zeros((1,))
        self.comm.Isend(self.empty_buffer, dest=self.rank_collector, tag=self.tag_ready_to_receive)
        self.list_snapshots_to_send = []
        self.status = MPI.Status()

    def send_parameters(self, parameters):
        self.parameters = parameters.copy() # TO DO: copy?
        self.computation_in_progress = True

    def recv_observations(self, ):
        # TO DO: check if SOL is not None, otherwise wait for SOL
        computed_observations = self.local_solver_instance.apply(self.solver_data[self.solver_data_idx],self.parameters)
        self.computation_in_progress = False
        return computed_observations

    def send_to_data_collector(self, snapshot_to_send):
        # Adds new snapsahot to a list; if COLLECTOR is ready to receive new
        # snapshots, sends list of snapshots to COLLECTOR and empties the list.
        # (needed only if is_updated == True)
        # TO DO: double buffer for list of snapshots
        self.list_snapshots_to_send.append(snapshot_to_send)
        tmp = self.comm.Iprobe(source=self.rank_collector, tag=self.tag_ready_to_receive)
        if tmp: # if COLLECTOR is ready to receive new snapshots
            tmp = np.zeros((1,)) # TO DO: useless pointer / cancel message
            self.comm.Recv(tmp,source=self.rank_collector, tag=self.tag_ready_to_receive)
#            print('DEBUG - Solver_local_collector_MPI (', self.comm.Get_rank(), ') - sent_snapshots to', self.rank_collector)
            data_to_pickle = self.list_snapshots_to_send.copy()
            if self.request_send is not None:
                self.request_send.wait()
            print("data_to_pickle",len(data_to_pickle))
            self.request_send = self.comm.isend(data_to_pickle, dest=self.rank_collector, tag=self.tag_sent_data)
            self.list_snapshots_to_send = []
#        time.sleep(0.1)
        self.receive_update_if_ready()

    def receive_update_if_ready(self):
        # Receives updated solver data (e.g. for updated surrogate model).
        # TO DO: when to check for updates
        # TO DO: avoid copying of received data
        # check COMM_WORLD if there is an incoming message from COLLECTOR:
#        r = self.request_recv.test()#status=self.status)
#        if r[0]:
        try:
            tmp = self.request_recv.Get_status()
        except:
            print("EX1")    
#        print("RANK", self.comm.Get_rank(), "recv status:", tmp)
        if tmp:
            print("RANK", self.comm.Get_rank(),"will receive update")
            try:
                r = self.request_recv.wait()
                self.solver_data_iterator += 1
                print("RANK", self.comm.Get_rank(), "received update:", self.solver_data_iterator, r[0].shape, r[1].shape)
                self.solver_data[1 - self.solver_data_idx] = r
            except:
                print("EX2")    
            self.solver_data_idx = 1 - self.solver_data_idx
            self.request_recv = self.comm.irecv(source=self.rank_collector, tag=self.tag_sent_data)
            self.comm.Isend(self.empty_buffer, dest=self.rank_collector, tag=self.tag_ready_to_receive)
            
#    def is_solved(self):
#        if self.computation_in_progress:
#            return False
#        else:
#            return True

    def terminate(self, ):
        # TO DO: remove?
#        tmp = self.request_recv.Get_status()
#        if tmp:
#            print("RANK", self.comm.Get_rank(),"will receive update")
#            r = self.request_recv.wait()
#            self.solver_data_iterator += 1
#            print("RANK", self.comm.Get_rank(), "received update:", self.solver_data_iterator, r[0].shape, r[1].shape)
#            self.solver_data[1 - self.solver_data_idx] = r
#            self.solver_data_idx = 1 - self.solver_data_idx
        # TO DO: assume that terminate may be called multiple times
        if not self.terminated:
            self.terminated = True
        if not self.terminated_data:
#            self.comm.send([], dest=self.rank_collector, tag=self.tag_terminate)
            self.comm.Isend(self.empty_buffer, dest=self.rank_collector, tag=self.tag_terminate)
            self.terminated_data = True
