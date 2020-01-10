#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:30:46 2018

@author: dom0015
"""

class Stage:

    def __init__(self, type_sampling, limit_time, limit_samples, name_stage, type_proposal, proposalStd):
        self.limit_time = limit_time
        self.limit_samples = limit_samples
        self.type_sampling = type_sampling
        self.name_stage = name_stage
        self.type_proposal = type_proposal
        self.proposalStd = proposalStd