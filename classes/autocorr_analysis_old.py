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
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def _next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    n = _next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n
    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

# Automated windowing procedure following Sokal (1989)
# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def autocorr_gw2010(y, c=5.0):
    # first averages all chains, than calculates autocorr. func.
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def autocorr_new(y, c=5.0):
    # first calculates all autocorr. functions, than averages them
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

### different autocorrelation time estimation (suitable for shorter chains)
# from https://dfm.io/posts/autocorr/ Foreman-Mackey
def autocorr_ml(y, thin=1, c=5.0):
    # Compute the initial estimate of tau using the standard method
    init = autocorr_new(y, c=c)
    z = y[:, ::thin]
    N = z.shape[1]
    
    # Build the GP model
    tau = max(1.0, init/thin)
    import celerite
    from celerite import terms
    kernel = terms.RealTerm(np.log(0.9*np.var(z)), -np.log(tau),
                        bounds=[(-5.0, 5.0), (-np.log(N), 0.0)])
    kernel += terms.RealTerm(np.log(0.1*np.var(z)), -np.log(0.5*tau),
                            bounds=[(-5.0, 5.0), (-np.log(N), 0.0)])
    gp = celerite.GP(kernel, mean=np.mean(z))
    gp.compute(np.arange(z.shape[1]))

    # Define the objective
    def nll(p):
        # Update the GP model
        gp.set_parameter_vector(p)
        
        # Loop over the chains and compute likelihoods
        v, g = zip(*(
            gp.grad_log_likelihood(z0, quiet=True)
            for z0 in z
        ))
        
        # Combine the datasets
        return -np.sum(v), -np.sum(g, axis=0)

    # Optimize the model
    p0 = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    soln = minimize(nll, p0, jac=True, bounds=bounds)
#    soln = minimize(nll, p0, jac=False, bounds=bounds)
    gp.set_parameter_vector(soln.x)
    
    # Compute the maximum likelihood tau
    a, c = kernel.coefficients[:2]
    tau = thin * 2*np.sum(a / c) / np.sum(a)
    return tau


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

    def autocorr_time_single(self, indices=None, c=5, tol=50, quiet=False):
        if indices==None:
            indices = range(self.no_chains)
        self.autocorr_time_est = [None] * self.no_chains
        if isinstance(self.x,np.ndarray):
            if self.x.ndim == 1:
                self.autocorr_time_est[0] = emcee.autocorr.integrated_time(self.x, c=c, tol=tol, quiet=quiet)
            else:
                for i in indices:
                    self.autocorr_time_est[i] = emcee.autocorr.integrated_time(self.x[i,:], c=c, tol=tol, quiet=quiet)
        else: #assumes list of 1d numpy arrays
           for i in indices:
                self.autocorr_time_est[i] = emcee.autocorr.integrated_time(self.x[i], c=c, tol=tol, quiet=quiet)
        return self.autocorr_time_est

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
    
    def autocorr_function(self,max_lag):
        if isinstance(self.x,np.ndarray):
            if self.x.ndim==1:
                self.autocorr_f = self._ac_fft1(self.x,max_lag)
            else:
                self.autocorr_f = np.zeros((self.no_chains,max_lag))
                for i in range(self.no_chains):
                    self.autocorr_f[i,:] = self._ac_fft1(self.x[i,:],max_lag,var=self.var[i])
        else: #assumes list of 1d numpy arrays
            self.autocorr_f = np.zeros((self.no_chains,max_lag))
            for i in range(self.no_chains):
                for i in range(self.no_chains):
                    self.autocorr_f[i,:] = self._ac_fft1(self.x[i],max_lag)
                    
    def plot_autocorr_function(self, l_disp):
        plt.plot(self.autocorr_f[:,:l_disp].transpose())
        plt.legend(np.arange(self.no_chains))
        plt.show()

    # integrated autocorr time calculation:
    def integrated_time_emcee(self, c=5, tol=50, quiet=False):
        # used automated windowing procedure following Sokal (1989), i.e. c=5
        # tol=50 follows Foreman-Mackey assuming 32 independent chains
        # tol=1000 follows Sokal assuming one chain
        # TO DO: in DAMH-SMU with parallel chains - can we assume equal 
        #        autocorr. time for all parallel chains?
        it = emcee.autocorr.integrated_time(self.x, c=c, tol=tol, quiet=quiet)
        return it
    
    def _ac_manual(self,xp,max_lag):
        '''manualy compute, non partial'''
        '''takes xp'''
        corr=[1. if l==0 else np.sum(self.xp[l:]*self.xp[:-l])/self.length/self.var for l in range(max_lag)]
        return np.array(corr)
    
    def _ac_corrcoef1(self,x,max_lag):
        '''numpy.corrcoef, partial'''
        '''takes x'''
        corr=[1. if l==0 else np.corrcoef(self.x[l:],self.x[:-l])[0][1] for l in range(max_lag)]
        return np.array(corr)
    
    def _ac_fft1(self,xp,max_lag,n=None,var=None):
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
    
    def _ac_fft2(self,xp,max_lag):
        '''fft, don't pad 0s, non partial'''
        '''takes xp'''
        cf=np.fft.fft(self.xp)
        sf=cf.conjugate()*cf
        corr=np.fft.ifft(sf).real/self.var/self.n
        return corr[:max_lag]
    
    def _ac_correlate1(self,xp,max_lag):
        '''np.correlate, non partial'''
        '''takes xp'''
        corr=np.correlate(self.xp,self.xp,'full')[self.n-1:]/self.var/self.n
        return corr[:max_lag]
    
    def _ac_fft3 (self,xp,max_lag):
        """Compute the autocorrelation of the signal, based on the properties
        of the power spectral density of the signal. """
        '''takes xp'''
        f = np.fft.fft(self.xp)
        p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
        pi = np.fft.ifft(p)
        corr = np.real(pi)[:self.n]/np.sum(self.xp**2)
        return corr[:max_lag]
    
    def _ac_fft4(self,x,max_lag):
        '''takes x'''
        r2=np.fft.ifft(np.abs(np.fft.fft(self.x))**2).real
        corr=(r2/self.x.shape-self.mean**2)/self.std**2
        return corr[:max_lag]
    
    def _ac_corrcoef2(self,x,max_lag):
        '''takes x'''
        return np.array([1]+[np.corrcoef(self.x[:-i], self.x[i:])[0,1]  \
            for i in range(1, max_lag)])
    
    def _ac_correlate2(self,x,max_lag):
        '''takes x'''
        corr = np.correlate(self.x,self.x,mode='full')
        corr = corr[corr.size//2:]
        corr = corr/corr[0]
        return corr[:max_lag]
    
    def _ac_autocorr_emcee(self,x,max_lag):
        '''takes x'''
        corr = emcee.autocorr.function_1d(x)
        return corr