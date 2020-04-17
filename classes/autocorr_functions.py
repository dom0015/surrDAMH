#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:24:11 2020

@author: simona
"""

# from stack overflow etc.

import numpy as np
import emcee
import time

class Chain:
    def __init__(self, x):
        self.x = x
        self.var = np.var(x)
        self.std = np.std(x)
        self.mean = np.mean(x)
        self.n = len(x)
        self.xp = x - self.mean
        
    def all_autocorr(self, max_lag = None, quiet = False):
        if max_lag == None:
            max_lag = self.n-1
        res = dict()
        t = time.time()
        res['manual'] = self.manual(max_lag)
        if not quiet:
            print(['manual', time.time()-t])
        t = time.time()
        res['corrcoef1'] = self.corrcoef1(max_lag)
        if not quiet:
            print(['corrcoef1', time.time()-t])
        t = time.time()
        res['fft1'] = self.fft1(max_lag)
        if not quiet:
            print(['fft1', time.time()-t])
        t = time.time()
        res['fft2'] = self.fft2(max_lag)
        if not quiet:
            print(['fft2', time.time()-t])
        t = time.time()
        res['correlate1'] = self.correlate1(max_lag)
        if not quiet:
            print(['correlate1', time.time()-t])
        t = time.time()
        res['fft3'] = self.fft3(max_lag)
        if not quiet:
            print(['fft3', time.time()-t])
        t = time.time()
        res['fft4'] = self.fft4(max_lag)
        if not quiet:
            print(['fft4', time.time()-t])
        t = time.time()
        res['corrcoef2'] = self.corrcoef2(max_lag)
        if not quiet:
            print(['corrcoef2', time.time()-t])
        t = time.time()
        res['correlate2'] = self.correlate2(max_lag)
        if not quiet:
            print(['correlate2', time.time()-t])
        t = time.time()
        res['emcee'] = self.autocorr_emcee(max_lag)
        if not quiet:
            print(['emcee', time.time()-t])
        return res
        
    def manual(self,max_lag):
        '''manualy compute, non partial'''
        corr=[1. if l==0 else np.sum(self.xp[l:]*self.xp[:-l])/self.n/self.var for l in range(max_lag)]
        return np.array(corr)
    
    def corrcoef1(self,max_lag):
        '''numpy.corrcoef, partial'''
        corr=[1. if l==0 else np.corrcoef(self.x[l:],self.x[:-l])[0][1] for l in range(max_lag)]
        return np.array(corr)
    
    def fft1(self,max_lag):
        '''fft, pad 0s, non partial'''
        # pad 0s to 2n-1
        ext_size=2*self.n-1
        # nearest power of 2
        fsize=2**np.ceil(np.log2(ext_size)).astype('int')
        # do fft and ifft
        cf=np.fft.fft(self.xp,fsize)
        sf=cf.conjugate()*cf
        corr=np.fft.ifft(sf).real
        corr=corr/self.var/self.n
        return corr[:max_lag]
    
    def fft2(self,max_lag):
        '''fft, don't pad 0s, non partial'''
        cf=np.fft.fft(self.xp)
        sf=cf.conjugate()*cf
        corr=np.fft.ifft(sf).real/self.var/self.n
        return corr[:max_lag]
    
    def correlate1(self,max_lag):
        '''np.correlate, non partial'''
        corr=np.correlate(self.xp,self.xp,'full')[self.n-1:]/self.var/self.n
        return corr[:max_lag]
    
    def fft3 (self,max_lag) :
        """Compute the autocorrelation of the signal, based on the properties
        of the power spectral density of the signal. """
        f = np.fft.fft(self.xp)
        p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
        pi = np.fft.ifft(p)
        corr = np.real(pi)[:self.n]/np.sum(self.xp**2)
        return corr[:max_lag]
    
    def fft4(self,max_lag):
        r2=np.fft.ifft(np.abs(np.fft.fft(self.x))**2).real
        corr=(r2/self.x.shape-self.mean**2)/self.std**2
        return corr[:max_lag]
    
    def corrcoef2(self,max_lag):
        return np.array([1]+[np.corrcoef(self.x[:-i], self.x[i:])[0,1]  \
            for i in range(1, max_lag)])
    
    def correlate2(self,max_lag):
        corr = np.correlate(self.x,self.x,mode='full')
        corr = corr[corr.size//2:]
        corr = corr/corr[0]
        return corr[:max_lag]
    
    def autocorr_emcee(self,max_lag):
        corr = emcee.autocorr.function_1d(self.x)
        return corr
    
    # integrated autocorr time calculation:
    def integrated_time_emcee(self, c=5, tol=50, quiet=False):
        # used automated windowing procedure following Sokal (1989), i.e. c=5
        # tol=50 follows Foreman-Mackey assuming 32 independent chains
        # tol=100% follows Sokal assuming one chain
        # TO DO: in DAMH-SMU with parallel chains - can we assume equal 
        #        autocorr. time for all parallel chains?
        it = emcee.autocorr.integrated_time(self.x, c=c, tol=tol, quiet=quiet)
        return it