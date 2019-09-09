#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 12:04:56 2018

@author: dom0015
"""

import numpy as np
import csv
#import time
#import test_solver

class MHsampler:

    Model = None        # class variable shared by all instances
    no_parameters = None
    no_observations = None
    shared_finisher = None

    def __init__(self, proposalStd, chain_id, no_steps, shared_queue_solver, shared_queue_surrogate_solver, shared_children_solver, shared_children_surrogate ):
        self.proposalStd = proposalStd
        self.chain_id = chain_id
        self.no_steps = no_steps
        self.filename = 'data_linela_SMU_1' + str(chain_id) + '.csv'
        self.file = open(self.filename, 'w')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['File opening'])
        self.initialSample = None
        self.currentSample = None
        self.proposedSample = None
        self.lnratio = None
        self.Ap = None
        self.Bp = None
        self.Gp = np.zeros(MHsampler.no_observations)
        self.Ap_surr = None
        self.Gp_surr = np.zeros(MHsampler.no_observations)
        self.Au = None
        self.Bu = None
        self.Gu = np.zeros(MHsampler.no_observations)
        self.Au_surr = None
        self.Gu_surr = np.zeros(MHsampler.no_observations)
        self.shared_queue_solver = shared_queue_solver
        self.shared_queue_surrogate_solver = shared_queue_surrogate_solver
        self.shared_children_solver = shared_children_solver
        self.shared_children_surrogate = shared_children_surrogate
        self.no_accepted = None
        self.no_prerejected = None
        self.no_rejected = None
        
    def ChooseInitialSample(self):
        r = np.random.RandomState(100+self.chain_id)
        temp = r.normal(self.Model.priorMean,self.Model.priorStd)
        self.initialSample = temp
        print('Initial sample:', temp)
        
    def MH(self):
        print('observation from Model:', self.Model.observation)
        no_accepted = 0
        self.currentSample = self.initialSample
        self.shared_queue_solver.put([self.chain_id, self.currentSample])
        self.Gu=self.shared_children_solver.recv()
#        test_solver.solve(self.currentSample,self.Gu)
        self.SetAu()
        self.SetBu()
        gen_uni = np.random.RandomState(200+self.chain_id)
        gen_norm = np.random.RandomState(300+self.chain_id)
        
        #for i in range(1, self.no_steps):
        i=0
        while self.shared_finisher.value < 2: # until surrogate is available
            i+=1
            self.GenerateProposal(gen_norm)
            self.AcceptanceProbabilityMH()
            temp = gen_uni.uniform(0.0,1.0)
            temp = np.log(temp)
            if temp<self.lnratio:
                self.currentSample = self.proposedSample
                self.Au = self.Ap
                self.Bu = self.Bp
                self.Gu = self.Gp
                # KOMPRESE !
                row = [i, 'Accepted']
                no_accepted += 1
            else:
                row = [i, 'Rejected']
            for j in range(self.no_parameters):
                row.append(self.currentSample[j])
            self.writer.writerow(row)

        print('MH accepted samples:', no_accepted, 'of', i)
        self.no_accepted = no_accepted
        self.no_prerejected = 0
        self.no_rejected = i-no_accepted
        
    def DAMHSMU(self):
        print('Start DAMH-SMU sampling')
#        self.shared_queue_surrogate_solver.put([self.chain_id, self.currentSample])
#        self.Gu_surr=self.shared_children_surrogate.recv()
        self.SetAu_surr()
        no_accepted = 0
        no_prerejected = 0
        no_rejected = 0
        gen_uni = np.random.RandomState(400+self.chain_id)
        gen_norm = np.random.RandomState(500+self.chain_id)
        
        #for i in range(1, self.no_steps):
        i=0
        while self.shared_finisher.value < 3: # until surrogate is available
            i+=1
            self.GenerateProposal(gen_norm)
            self.PreAcceptanceProbability()
            temp = gen_uni.uniform(0.0,1.0)
            temp = np.log(temp)
            if temp<self.lnratio_surr: # pre-accepted
                self.AcceptanceProbabilityDAMH()
                print('Preaccepted - comparison:')
                print('Gp exac:', self.Gp)
                print('Gp surr:', self.Gp_surr)
                print('Ap exac:', self.Ap)
                print('Ap surr:', self.Ap_surr)
                temp = gen_uni.uniform(0.0,1.0)
                temp = np.log(temp)
                if temp<self.lnratio: # accepted
                    self.currentSample = self.proposedSample
                    self.Au = self.Ap
                    self.Bu = self.Bp
                    self.Gu = self.Gp
                    self.Au_surr = self.Ap_surr
                    self.Gu_surr = self.Gp_surr
#                    print('Compare (exact, surr):', self.Gu, self.Gu_surr)
                    # KOMPRESE !
                    row = [i, 'Accepted']
                    no_accepted += 1
                else:
                    row = [i, 'Rejected']
                    no_rejected += 1
            else:
                row = [i, 'PreRejected']
                no_prerejected += 1
            for j in range(self.no_parameters):
                row.append(self.currentSample[j])
            self.writer.writerow(row)
        
        print('DAMH accepted samples:', no_accepted, 'of', i, 'rejected:', no_prerejected, no_rejected)
        self.no_accepted = self.no_accepted+no_accepted
        self.no_prerejected = self.no_prerejected+no_prerejected
        self.no_rejected = self.no_rejected+no_rejected
        
    def GenerateProposal(self, gen_norm):
        temp = gen_norm.normal(0.0,1.0,self.no_parameters)
        self.proposedSample = self.currentSample + np.multiply(temp,self.proposalStd)
#        print(self.proposedSample, '=', self.currentSample, '+', temp, '.*', self.proposalStd)
        
    def AcceptanceProbabilityMH(self):
        self.shared_queue_solver.put([self.chain_id, self.proposedSample])
        self.Gp=self.shared_children_solver.recv()
        self.SetAp()
        self.SetBp()
        temp1=self.Au - self.Ap
        temp2=self.Bu - self.Bp
        self.lnratio = temp1+temp2
        
    def AcceptanceProbabilityDAMH(self):
#        t=time.time()
#        print(self.chain_id, ': solver request')
        self.shared_queue_solver.put([self.chain_id, self.proposedSample])
        self.Gp=self.shared_children_solver.recv()
#        print(self.Gp, self.Gp_surr)
#        print(self.chain_id, ': solver received in', time.time()-t)
        self.SetAp()
        temp1=self.Au - self.Ap
        temp2=self.Au_surr - self.Ap_surr
        self.lnratio = temp1-temp2     
        
    def PreAcceptanceProbability(self):
#        t=time.time()
#        print(self.chain_id, ': surrogate request')
        self.shared_queue_surrogate_solver.put([self.chain_id, self.proposedSample, self.currentSample])
        self.Gp_surr, temp0, tag=self.shared_children_surrogate.recv()
#        print(self.chain_id, ': surrogate received in', time.time()-t)
        if tag > 0: # surrogate model was updated
            self.Gu_surr = temp0
            self.SetAu_surr()
        self.SetAp_surr()
        self.SetBp()
        temp1=self.Au_surr - self.Ap_surr
        temp2=self.Bu - self.Bp
#        print(self.Au_surr, self.Ap_surr, self.Bu, self.Bp)
        self.lnratio_surr = temp1+temp2
        
    def SetAu(self):
        a = self.Model.observation - self.Gu
        temp_a = np.multiply(a,a)
        b = self.Model.noiseStd
        temp_b = np.multiply(b,b)
        temp = np.divide(temp_a,temp_b)
        self.Au = np.sum(temp)/2
        
    def SetBu(self):
        a = self.currentSample - self.Model.priorMean
        temp_a = np.multiply(a,a)
        b = self.Model.priorStd
        temp_b = np.multiply(b,b)
        temp = np.divide(temp_a,temp_b)
        self.Bu = np.sum(temp)/2
        
    def SetAu_surr(self):
        a = self.Model.observation - self.Gu_surr
        temp_a = np.multiply(a,a)
        b = self.Model.noiseStd
        temp_b = np.multiply(b,b)
        temp = np.divide(temp_a,temp_b)
        self.Au_surr = np.sum(temp)/2
        
    def SetAp(self):
        a = self.Model.observation - self.Gp
        temp_a = np.multiply(a,a)
        b = self.Model.noiseStd
        temp_b = np.multiply(b,b)
        temp = np.divide(temp_a,temp_b)
        self.Ap = np.sum(temp)/2
        
    def SetBp(self):
        a = self.proposedSample - self.Model.priorMean
        temp_a = np.multiply(a,a)
        b = self.Model.priorStd
        temp_b = np.multiply(b,b)
        temp = np.divide(temp_a,temp_b)
        self.Bp = np.sum(temp)/2
        
    def SetAp_surr(self):
        a = self.Model.observation - self.Gp_surr
        temp_a = np.multiply(a,a)
        b = self.Model.noiseStd
        temp_b = np.multiply(b,b)
        temp = np.divide(temp_a,temp_b)
        self.Ap_surr = np.sum(temp)/2
        
    def SetNoisyObservation(self, artificial_observation_without_noise):
        r = np.random.RandomState(1)
        temp=r.normal(0.0,1.0,self.no_observations);
        self.observation = np.multiply(temp,self.noiseStd) + artificial_observation_without_noise
    
    def Finalize(self):
        self.writer.writerow(['File closing'])
        self.file.close()