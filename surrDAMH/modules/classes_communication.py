#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

from mpi4py import MPI
import numpy as np
import sys
import os

class Solver_MPI_parent: # initiated by SOLVERS POOL
    def __init__(self, no_parameters, no_observations, no_samplers, problem_path, output_dir, maxprocs=1, solver_id=0, pickled_observations=True):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_requests = 1
        # path hack: find absolute path to process_CHILD.py
        # this makes surrDAMH lib independent of the initial calling path
        rep_dir = os.path.dirname(os.path.abspath(__file__))
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=[rep_dir+'/../process_CHILD.py', str(no_samplers),
                                                              problem_path, str(solver_id), output_dir],
                                        maxprocs=maxprocs)
        self.tag = 0
        self.received_data = np.zeros(self.no_observations)
        self.status = MPI.Status()
        self.pickled_observations = pickled_observations
    
    def send_parameters(self, data_par):
        self.tag += 1
        self.data_par = data_par.copy()
        #self.comm.Barrier()
        self.comm.Bcast([np.array(self.tag,'i'), MPI.INT], root=MPI.ROOT)
        #self.data_par.shape=(-1,)
        self.comm.Bcast([self.data_par, MPI.DOUBLE], root=MPI.ROOT)
        
    def recv_observations(self):
        if self.pickled_observations:
            convergence_tag, self.received_data = self.comm.recv(source=0, tag=self.tag)
        else:
            self.comm.Recv(self.received_data, source=0, tag=MPI.ANY_TAG, status=self.status)
            convergence_tag = self.status.Get_tag()
        return convergence_tag, self.received_data.reshape((1,-1)).copy()
    
    def is_solved(self):
        # check the parent-child communicator if there is an incoming message
        if self.pickled_observations:
            tmp = self.comm.Iprobe(source=0, tag=self.tag)
        else:
            tmp = self.comm.Iprobe(source=0, tag=MPI.ANY_TAG) # tag=self.tag)
        if tmp:
            return True
        else:
            return False
    
    def terminate(self):
        #self.comm.Barrier()
        self.comm.Bcast([np.array(0,'i'), MPI.INT], root=MPI.ROOT)
        self.comm.Barrier()
        self.comm.Disconnect()
        print("Solver spawned by rank", MPI.COMM_WORLD.Get_rank(), "disconnected.")
    
class Solver_MPI_collector_MPI: # initiated by SAMPLERs
    def __init__(self, no_parameters, no_observations, rank_solver, is_updated=False, rank_collector=None, pickled_observations=True):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.max_requests = 1
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.rank_solver = rank_solver
        self.is_updated = is_updated
        self.rank_collector = rank_collector
        self.tag_solver = 0
        self.received_data = np.zeros(self.no_observations)
        self.terminated_solver = None
        self.terminated_collector = True
        if not rank_collector is None:
            self.terminated_collector = False
        self.empty_buffer = np.zeros(1)
        self.status = MPI.Status()
        self.pickled_observations = pickled_observations
        
    def send_parameters(self, sent_data):
        self.tag_solver += 1
        self.comm.Send(sent_data, dest=self.rank_solver, tag=self.tag_solver)

    def recv_observations(self, ):
        if self.pickled_observations:
            [convergence_tag, self.received_data] = self.comm.recv(source=self.rank_solver, tag=self.tag_solver)
        else:
            self.comm.Recv(self.received_data, source=self.rank_solver, tag=MPI.ANY_TAG, status = self.status) # tag=self.tag_solver)
            convergence_tag = self.status.Get_tag()
        return convergence_tag, self.received_data.copy()
        
    def terminate(self, ):
        if not self.terminated_solver:
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
        self.max_buffer_size = 1<<30
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
        self.computation_in_progress = False
        self.request_recv = self.comm.irecv(self.max_buffer_size, source=self.rank_collector, tag=self.tag_sent_data)
        self.request_send = None
        self.empty_buffer = np.zeros((1,))
        self.request_Isend = self.comm.Isend(self.empty_buffer, dest=self.rank_collector, tag=self.tag_ready_to_receive)
        self.list_snapshots_to_send = []
        self.status = MPI.Status()

    def send_parameters(self, parameters):
        self.parameters = parameters.copy() # TO DO: copy?
        self.computation_in_progress = True

    def recv_observations(self, ):
        # if self.solver_data[self.solver_data_idx] == None:
        #     r = self.request_recv.wait()
        #     self.solver_data_iterator += 1
        #     self.solver_data[1 - self.solver_data_idx] = r
        #     self.solver_data_idx = 1 - self.solver_data_idx
        #     self.request_recv = self.comm.irecv(self.max_buffer_size, source=self.rank_collector, tag=self.tag_sent_data)
        #     self.request_Isend.Wait()
        #     self.request_Isend = self.comm.Isend(self.empty_buffer, dest=self.rank_collector, tag=self.tag_ready_to_receive)
        
        computed_observations = self.local_solver_instance.apply(self.solver_data[self.solver_data_idx],self.parameters)
        self.computation_in_progress = False
        return computed_observations

    def send_to_data_collector(self, snapshot_to_send):
        # Adds new snapsahot to a list; if COLLECTOR is ready to receive new
        # snapshots, sends list of snapshots to COLLECTOR and empties the list.
        # (needed only if is_updated == True)
        self.list_snapshots_to_send.append(snapshot_to_send)
        probe = self.comm.Iprobe(source=self.rank_collector, tag=self.tag_ready_to_receive)
        if probe: # if COLLECTOR is ready to receive new snapshots
            tmp = np.zeros((1,))
            self.comm.Recv(tmp,source=self.rank_collector, tag=self.tag_ready_to_receive)
            data_to_pickle = self.list_snapshots_to_send.copy()
            if self.request_send is not None:
                self.request_send.wait()
            self.request_send = self.comm.isend(data_to_pickle, dest=self.rank_collector, tag=self.tag_sent_data)
            self.list_snapshots_to_send = []
        self.receive_update_if_ready()

    def receive_update_if_ready(self):
        # Receives updated solver data (e.g. for updated surrogate model).
        # check COMM_WORLD if there is an incoming message from COLLECTOR:
        try:
            probe = self.request_recv.Get_status()
        except:
            print("EX in Solver_local_collector_MPI")    
        if probe:
            r = self.request_recv.wait()
            self.solver_data_iterator += 1
            self.solver_data[1 - self.solver_data_idx] = r
            self.solver_data_idx = 1 - self.solver_data_idx
            self.request_recv = self.comm.irecv(self.max_buffer_size, source=self.rank_collector, tag=self.tag_sent_data)
            self.request_Isend.Wait()
            self.request_Isend = self.comm.Isend(self.empty_buffer, dest=self.rank_collector, tag=self.tag_ready_to_receive)

    def terminate(self, ):
        if not self.terminated:
            self.terminated = True
        if not self.terminated_data:
            self.request_Isend.Wait()
            self.request_Isend = self.comm.Isend(self.empty_buffer, dest=self.rank_collector, tag=self.tag_terminate)
            self.terminated_data = True
