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
    SF = None
    seed = None
    
    def __init__(self, proposalStd, chain_id):
        self.proposalStd = proposalStd
        self.chain_id = chain_id
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
        self.Gp = np.zeros(self.Model.no_observations)
        self.Ap_surr = None
        self.Gp_surr = np.zeros(self.Model.no_observations)
        self.Au = None
        self.Bu = None
        self.Gu = np.zeros(self.Model.no_observations)
        self.Au_surr = None
        self.Gu_surr = np.zeros(self.Model.no_observations)
        self.shared_queue_solver = self.SF.shared_queue_solver
        self.shared_queue_surrogate_solver = self.SF.shared_queue_surrogate_solver
        self.shared_children_solver = self.SF.shared_children_solver[chain_id]
        self.shared_children_surrogate = self.SF.shared_children_surrogate[chain_id]
        self.shared_finisher = self.SF.shared_finisher
        self.no_accepted = 0
        self.no_prerejected = 0
        self.no_rejected = 0
        self.no_rejected_current = 0
        self.gen_uni = np.random.RandomState(100+self.chain_id + self.seed)
        self.gen_norm = np.random.RandomState(200+self.chain_id + self.seed)
        self.stage = None
        self.muADAMH = None
        self.covADAMH = None
        self.surrogate_used = False
        
    def WriteSampleToFile(self,no_rejected_current,name_phase):
        # WRITE SAMPLE TO FILE (COMPRESSED FORM)
        row = [name_phase, no_rejected_current + 1]
        for j in range(self.Model.no_parameters):
            row.append(self.currentSample[j])
        self.writer.writerow(row)
        
    def ChooseInitialSample(self):
        temp = self.gen_norm.normal(self.Model.priorMean,self.Model.priorStd)
        self.initialSample = temp
        print('Initial sample:', temp)
        
    def BeforeSampling(self):
        self.currentSample = self.initialSample
        self.shared_queue_solver.put([self.chain_id, self.initialSample])
        print("chain_id",self.chain_id)
        self.Gu=self.shared_children_solver.recv()
        print("Sampler received solution")
        
    def MH(self,stage,value_finisher):
        self.stage = stage
        print('Start MH sampling, observation from Model:', self.Model.observation)
        self.no_accepted = 0
        self.no_rejected = 0
        self.no_rejected_current = 0
        self.SetAu()
        self.SetBu()
        
        i=0
        # tell the sampler what value is he waiting for to finish
        while self.shared_finisher.value < value_finisher and i<stage.limit_samples:
            i+=1
            self.GenerateProposal()
            self.AcceptanceProbabilityMH()
            self.AcceptanceDecision()
        self.WriteSampleToFile(self.no_rejected_current,stage.name_stage)

        print('MH accepted samples:', self.no_accepted, 'of', i)
        
        fields = [self.no_accepted, self.no_prerejected, self.no_rejected, stage.type_sampling, self.seed, self.proposalStd]
        print(fields)
        f = open('notes.csv', 'a')
        writer_notes = csv.writer(f)
        writer_notes.writerow(fields)
        f.close()
        print("Written to file.")
        self.proposalStd *= 4
        
    def DAMHSMU(self,stage,value_finisher):
        self.stage = stage
        print('Start DAMH-SMU sampling')
        self.SetAu_surr()
        self.no_accepted = 0
        self.no_prerejected = 0
        self.no_rejected = 0
        self.no_rejected_current = 0
        i=0
        # tell the sampler what value is he waiting for to finish
        while self.shared_finisher.value < value_finisher and i<stage.limit_samples:
            i+=1
            self.GenerateProposal()
            self.PreAcceptanceProbability()
            temp = self.gen_uni.uniform(0.0,1.0)
            temp = np.log(temp)
            if temp<self.lnratio_surr: # pre-accepted
                self.AcceptanceProbabilityDAMH()
                self.AcceptanceDecision()
            else:
                self.no_prerejected += 1
                self.no_rejected_current += 1
        self.WriteSampleToFile(self.no_rejected_current,stage.name_stage)
        
        print('DAMH-SMU accepted samples:', self.no_accepted, 'of', i, 'rejected:', self.no_prerejected, self.no_rejected)
        
        fields = [self.no_accepted, self.no_prerejected, self.no_rejected, stage.type_sampling, self.seed, self.proposalStd]
        print(fields)
        f = open('notes.csv', 'a')
        writer_notes = csv.writer(f)
        writer_notes.writerow(fields)
        f.close()
        print("Written to file.")
        
    def ADAMH(self,stage,value_finisher):
        self.stage = stage
        print('Start ADAMH sampling',stage.name_stage)
        self.muADAMH = np.zeros((self.Model.no_observations))
        self.covADAMH = np.zeros((self.Model.no_observations,self.Model.no_observations))
        self.SetAu_surrADAMH()
        self.no_accepted = 0
        self.no_prerejected = 0
        self.no_rejected = 0
        self.no_rejected_current = 0
        ALL_ERRORS = np.zeros((0,self.Model.no_observations))
        i=0
        # tell the sampler what value is he waiting for to finish
        while self.shared_finisher.value < value_finisher and i<stage.limit_samples:
            i+=1
            self.GenerateProposal()
            self.PreAcceptanceProbabilityADAMH()
            temp = self.gen_uni.uniform(0.0,1.0)
            temp = np.log(temp)
            if temp<self.lnratio_surr: # pre-accepted
                self.AcceptanceProbabilityDAMH()
                self.AcceptanceDecision()
            else:
                self.no_prerejected += 1
                self.no_rejected_current += 1
            B = np.array(self.Gu - self.Gu_surr).reshape((self.Model.no_observations,1))
            ALL_ERRORS = np.vstack((ALL_ERRORS,B)) # REPLACE BY RECURSION !
#            self.muADAMH = (self.muADAMH*(i-1) + B)/i
            self.muADAMH = np.mean(ALL_ERRORS)
            if i==1:
                self.covADAMH = np.zeros((self.Model.no_observations))
            else:
                #self.covADAMH = ( (i-1)*self.covADAMH + B*B.transpose() - (i)*self.muADAMH*self.muADAMH.transpose() )/i
                #self.covADAMH = (np.matmul((ALL_ERRORS - self.muADAMH).transpose(), ALL_ERRORS - self.muADAMH) )/(i-1)
                self.covADAMH = np.cov(ALL_ERRORS.transpose())
        self.WriteSampleToFile(self.no_rejected_current,stage.name_stage)
        
        print(stage.name_stage, '- ADAMH accepted samples:', self.no_accepted, 'of', i, 'rejected:', self.no_prerejected, self.no_rejected)
        
        fields = [self.no_accepted, self.no_prerejected, self.no_rejected, stage.type_sampling, self.seed, self.proposalStd]
        print(fields)
        f = open('notes.csv', 'a')
        writer_notes = csv.writer(f)
        writer_notes.writerow(fields)
        f.close()
        print("Written to file.")
        
    def AcceptanceDecision(self):
        temp = self.gen_uni.uniform(0.0,1.0)
        temp = np.log(temp)
        if temp<self.lnratio: # accepted
            self.no_accepted += 1
            self.WriteSampleToFile(self.no_rejected_current,self.stage.name_stage)
            self.SF.shared_queue_surrogate.put([self.currentSample,self.Gu,self.no_rejected_current])
            self.no_rejected_current = 0
            self.currentSample = self.proposedSample
            self.Au = self.Ap
            self.Bu = self.Bp
            self.Gu = self.Gp
            self.Au_surr = self.Ap_surr
            self.Gu_surr = self.Gp_surr
#            print("Gu_____",self.Gu)
#            print("Gu_surr",self.Gu_surr)
        else:
            self.no_rejected += 1
            self.no_rejected_current += 1
            self.SF.shared_queue_surrogate.put([self.proposedSample,self.Gp,0])
        
    def GenerateProposal(self):
        temp = self.gen_norm.normal(0.0,1.0,self.Model.no_parameters)
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
        self.shared_queue_solver.put([self.chain_id, self.proposedSample])
        self.Gp=self.shared_children_solver.recv()
#        print("Gp_surr:",self.Gp_surr)
#        print("Gp_____:",self.Gp)
        self.SetAp()
        temp1=self.Au - self.Ap
        temp2=self.Au_surr - self.Ap_surr
        self.lnratio = temp1-temp2     
        
    def PreAcceptanceProbability(self):
        self.shared_queue_surrogate_solver.put([self.chain_id, self.proposedSample, self.currentSample])
        self.Gp_surr, temp0, tag=self.shared_children_surrogate.recv()
        if tag > 0: # surrogate model was updated
            self.Gu_surr = temp0
            self.SetAu_surr()
        self.SetAp_surr()
        self.SetBp()
        temp1=self.Au_surr - self.Ap_surr
        temp2=self.Bu - self.Bp
        self.lnratio_surr = temp1+temp2
        
    def PreAcceptanceProbabilityADAMH(self):
        self.shared_queue_surrogate_solver.put([self.chain_id, self.proposedSample, self.currentSample])
        self.Gp_surr, temp0, tag=self.shared_children_surrogate.recv()
        if not self.surrogate_used:
            print("First surrogate use")
            self.surrogate_used = True
        if tag > 0: # surrogate model was updated - here useless
            self.Gu_surr = temp0
            self.SetAu_surrADAMH()
        self.SetAp_surrADAMH()
        self.SetBp()
        temp1=self.Au_surr - self.Ap_surr
        temp2=self.Bu - self.Bp
        self.lnratio_surr = temp1+temp2
        
    def SetAu(self):
        self.Au = self.Gauss_frac(self.Model.observation, self.Gu, self.Model.noiseStd)
        
    def SetBu(self):
        self.Bu = self.Gauss_frac(self.currentSample, self.Model.priorMean, self.Model.priorStd)
        
    def SetAu_surr(self):
        self.Au_surr = self.Gauss_frac(self.Model.observation, self.Gu_surr, self.Model.noiseStd)
        
    def SetAp(self):
        self.Ap = self.Gauss_frac(self.Model.observation, self.Gp, self.Model.noiseStd)
        
    def SetBp(self):
        self.Bp = self.Gauss_frac(self.proposedSample, self.Model.priorMean, self.Model.priorStd)
        
    def SetAp_surr(self):
        self.Ap_surr = self.Gauss_frac(self.Model.observation, self.Gp_surr, self.Model.noiseStd)
        
    def Gauss_frac(self,par1, par2, C):
        a = par1 - par2
        if a.size>1:
            Ca = np.linalg.solve(C,a)
        elif C==0:
            Ca = 0
        else:
            Ca = a/C
        return np.dot(a,Ca)
#        a = par1 - par2
#        temp_a = np.multiply(a,a)
#        temp_b = np.multiply(noiseStd,noiseStd)
#        temp = np.divide(temp_a,temp_b)
#        return np.sum(temp)/2
    
    def SetAu_surrADAMH(self):
        a = self.Gu_surr + self.muADAMH - self.Model.observation
        C = self.Model.noiseCov + self.covADAMH
        if a.size>1:
            Ca = np.linalg.solve(C,a)
        elif C==0:
            Ca = 0
        else:
            Ca = a/C
        self.Au_surr = np.dot(a,Ca)
        
    def SetAp_surrADAMH(self):
        a = self.Gp_surr + self.muADAMH - self.Model.observation
        C = self.Model.noiseCov + self.covADAMH
        if a.size>1:
            Ca = np.linalg.solve(C,a)
        elif C==0:
            Ca = 0
        else:
            Ca = a/C
        self.Ap_surr = np.dot(a,Ca)
        
#    def SetNoisyObservation(self, artificial_observation_without_noise):
#        r = np.random.RandomState(1)
#        temp=r.normal(0.0,1.0,self.Model.no_observations);
#        self.observation = np.multiply(temp,self.noiseStd) + artificial_observation_without_noise
    
    def Finalize(self):
        self.writer.writerow(['File closing'])
        self.file.close()