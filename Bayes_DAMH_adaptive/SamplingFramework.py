#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:30:46 2018

@author: dom0015
"""

import multiprocessing as mp

class SamplingFramework:

    def __init__(self,no_chains, stages, max_snapshots, min_snapshots, no_solvers):
        self.no_chains = no_chains
        self.no_helpers = 3
        self.shared_finisher = mp.Value('i',0)
        self.shared_bool_update = mp.Value('i',0)
        self.shared_free_solvers = []
        for i in range(no_solvers):
            v = mp.Value('b',True)
            self.shared_free_solvers.append(v)
        self.shared_queue_solver = mp.Queue()
        self.shared_queue_surrogate = mp.Queue()
        self.shared_queue_updater = mp.Queue()
        self.shared_queue_surrogate_solver = mp.Queue()
        self.shared_parents_solver = []
        self.shared_children_solver = []
        self.shared_parents_surrogate = []
        self.shared_children_surrogate = []
        self.max_snapshots = max_snapshots
        self.min_snapshots = min_snapshots
        self.stages = stages
        for i in range(self.no_chains):
            new_parent, new_child = mp.Pipe()
            self.shared_parents_solver.append(new_parent)
            self.shared_children_solver.append(new_child)
            new_parent, new_child = mp.Pipe()
            self.shared_parents_surrogate.append(new_parent)
            self.shared_children_surrogate.append(new_child)
            
    def Finalize(self):
        self.shared_queue_solver.cancel_join_thread()
        self.shared_queue_surrogate.cancel_join_thread()
        self.shared_queue_updater.cancel_join_thread()
        self.shared_queue_surrogate_solver.cancel_join_thread()
        print("Queues empty?", self.shared_queue_solver.empty(), self.shared_queue_surrogate.empty(), self.shared_queue_updater.empty(), self.shared_queue_surrogate_solver.empty())