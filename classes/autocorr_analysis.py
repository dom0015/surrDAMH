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

class Samples:
    def __init__(self, samples = None):
        self.x = samples
        self.known_autocorr_time = False
        
    def calculate_properties(self):
        x = self.x
        if isinstance(x,np.ndarray):
            a = x.ndim - 1
            self.var = np.var(x, axis = a)
            self.std = np.std(x, axis = a)
            self.mean = np.mean(x, axis = a)
            self.length = x.shape[a]
            if a==0:
                self.no_chains = 1
                self.xp = x - self.mean
            else:
                self.no_chains = x.shape[0]
                self.xp = x - self.mean.reshape((self.no_chains,1)).repeat(self.length,axis=1)
        else: #assumes list of 1d numpy arrays
            self.no_chains = len(x)
            self.var = list(np.var(x[i]) for i in range(self.no_chains))
            self.std = list(np.std(x[i]) for i in range(self.no_chains))
            self.mean = list(np.mean(x[i]) for i in range(self.no_chains))
            self.length = list(x[i].shape[0] for i in range(self.no_chains))
            self.xp = list(x[i] - self.mean[i] for i in range(self.no_chains))
            
    def print_properties(self):
        print('type:', type(self.x))
        print('known autocorr. time:', self.known_autocorr_time)
        if self.known_autocorr_time:
            print('true autocorr. time:', self.autocorr_time_true)
        print('number of chains:', self.no_chains)
        print('length:',self.length)
        print('mean:',self.mean)
        print('std:',self.std)

    def generate_celerite(self, no_chains=4, length=2000000, log_c1=-6.0, log_c2=-2.0):
        # from https://dfm.io/posts/autocorr/ Foreman-Mackey
        # Build the celerite model:
        import celerite
        from celerite import terms
        kernel = terms.RealTerm(log_a=0.0, log_c=log_c1)
        kernel += terms.RealTerm(log_a=0.0, log_c=log_c2)
        # The true autocorrelation time can be calculated analytically:
        true_tau = sum(2*np.exp(t.log_a-t.log_c) for t in kernel.terms)
        true_tau /= sum(np.exp(t.log_a) for t in kernel.terms)
        self.known_autocorr_time = True
        self.autocorr_time_true = true_tau
        # Simulate a set of chains:
        gp = celerite.GP(kernel)
        t = np.arange(length)
        gp.compute(t)
        self.x = gp.sample(size=no_chains)

    def plot_segment(self, no_chains_disp=1, length_disp=1000):
        import matplotlib.pyplot as plt
        if isinstance(self.x,np.ndarray):
            if self.x.ndim == 1:
                plt.plot(self.x[:length_disp])
            else:
                plt.plot(self.x[:no_chains_disp, :length_disp].T)
        else: #assumes list of 1d numpy arrays
            for i in range(no_chains_disp):
                plt.plot(self.x[i][:length_disp])
        plt.xlim(0, length_disp)
        plt.xlabel("sample number")
        plt.ylabel("$sample$")
        if self.known_autocorr_time:
            plt.title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(self.autocorr_time_true), fontsize=14);
        plt.show()
        
    def autocorr_all(self, max_lag = None, quiet = False):
        if max_lag == None:
            max_lag = self.length-1
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
        
    def ac_manual(self,xp,max_lag):
        '''manualy compute, non partial'''
        '''takes xp'''
        corr=[1. if l==0 else np.sum(self.xp[l:]*self.xp[:-l])/self.length/self.var for l in range(max_lag)]
        return np.array(corr)
    
    def ac_corrcoef1(self,x,max_lag):
        '''numpy.corrcoef, partial'''
        '''takes x'''
        corr=[1. if l==0 else np.corrcoef(self.x[l:],self.x[:-l])[0][1] for l in range(max_lag)]
        return np.array(corr)
    
    def ac_fft1(self,xp,max_lag,n=None,var=None):
        '''fft, pad 0s, non partial'''
        '''takes xp'''
        if n==None:
            n = self.length
        # pad 0s to 2n-1
        ext_size=2*n-1
        # nearest power of 2
        fsize=2**np.ceil(np.log2(ext_size)).astype('int')
        # do fft and ifft
        cf=np.fft.fft(xp,fsize)
        sf=cf.conjugate()*cf
        corr=np.fft.ifft(sf).real
        corr=corr/var/n
        return corr[:max_lag]
    
    def ac_fft2(self,xp,max_lag):
        '''fft, don't pad 0s, non partial'''
        '''takes xp'''
        cf=np.fft.fft(self.xp)
        sf=cf.conjugate()*cf
        corr=np.fft.ifft(sf).real/self.var/self.n
        return corr[:max_lag]
    
    def ac_correlate1(self,xp,max_lag):
        '''np.correlate, non partial'''
        '''takes xp'''
        corr=np.correlate(self.xp,self.xp,'full')[self.n-1:]/self.var/self.n
        return corr[:max_lag]
    
    def ac_fft3 (self,xp,max_lag):
        """Compute the autocorrelation of the signal, based on the properties
        of the power spectral density of the signal. """
        '''takes xp'''
        f = np.fft.fft(self.xp)
        p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
        pi = np.fft.ifft(p)
        corr = np.real(pi)[:self.n]/np.sum(self.xp**2)
        return corr[:max_lag]
    
    def ac_fft4(self,x,max_lag):
        '''takes x'''
        r2=np.fft.ifft(np.abs(np.fft.fft(self.x))**2).real
        corr=(r2/self.x.shape-self.mean**2)/self.std**2
        return corr[:max_lag]
    
    def ac_corrcoef2(self,x,max_lag):
        '''takes x'''
        return np.array([1]+[np.corrcoef(self.x[:-i], self.x[i:])[0,1]  \
            for i in range(1, max_lag)])
    
    def ac_correlate2(self,x,max_lag):
        '''takes x'''
        corr = np.correlate(self.x,self.x,mode='full')
        corr = corr[corr.size//2:]
        corr = corr/corr[0]
        return corr[:max_lag]
    
    def ac_autocorr_emcee(self,x,max_lag):
        '''takes x'''
        corr = emcee.autocorr.function_1d(x)
        return corr
    
    def autocorr_function(self,max_lag):
        if isinstance(self.x,np.ndarray):
            if self.x.ndim==1:
                self.autocorr_f = self.ac_fft1(self.x,max_lag)
            else:
                self.autocorr_f = np.zeros((self.no_chains,max_lag))
                for i in range(self.no_chains):
                    self.autocorr_f[i,:] = self.ac_fft1(self.x[i,:],max_lag,var=self.var[i])
        else: #assumes list of 1d numpy arrays
            self.autocorr_f = np.zeros((self.no_chains,max_lag))
            for i in range(self.no_chains):
                for i in range(self.no_chains):
                    self.autocorr_f[i,:] = self.ac_fft1(self.x[i],max_lag)
    
    # integrated autocorr time calculation:
    def integrated_time_emcee(self, c=5, tol=50, quiet=False):
        # used automated windowing procedure following Sokal (1989), i.e. c=5
        # tol=50 follows Foreman-Mackey assuming 32 independent chains
        # tol=100% follows Sokal assuming one chain
        # TO DO: in DAMH-SMU with parallel chains - can we assume equal 
        #        autocorr. time for all parallel chains?
        it = emcee.autocorr.integrated_time(self.x, c=c, tol=tol, quiet=quiet)
        return it