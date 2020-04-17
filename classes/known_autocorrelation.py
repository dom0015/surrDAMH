#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:06:45 2020

@author: simona
"""

# from https://dfm.io/posts/autocorr/ Foreman-Mackey

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123456)

# Build the celerite model:
import celerite
from celerite import terms
kernel = terms.RealTerm(log_a=0.0, log_c=-2.0)
kernel += terms.RealTerm(log_a=0.0, log_c=-1.0)
# The true autocorrelation time can be calculated analytically:
true_tau = sum(2*np.exp(t.log_a-t.log_c) for t in kernel.terms)
true_tau /= sum(np.exp(t.log_a) for t in kernel.terms)
print('true tau:',true_tau)

# Simulate a set of chains:
N = 4
l = 2000000
gp = celerite.GP(kernel)
t = np.arange(l)
gp.compute(t)
y = gp.sample(size=N)

# Let's plot a little segment with a few samples:
N_disp = 3
l_disp = int(2*true_tau)
#plt.plot(y[:N_disp, :l_disp].T)
#plt.xlim(0, l_disp)
#plt.xlabel("step number")
#plt.ylabel("$f$")
#plt.title("$\\tau_\mathrm{{true}} = {0:.0f}$".format(true_tau), fontsize=14);
#plt.show()

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n
    
    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

### plot autocorrelation functions and calculate i. t. using emcee
import autocorr_functions
for i in range(N):
    l_cut = int(true_tau * 4000)
    samples = y[i,:l_cut].reshape((l_cut))
#    acf = autocorr_func_1d(samples)
#    plt.plot(acf[:l_disp],label=i)
    c = autocorr_functions.Chain(samples)
    it = c.integrated_time_emcee(c=5, tol=50)
    print(it)
#plt.legend()
#plt.show()
    
### test emcee more chains
import emcee
samples = y[:,:l_cut]
it = emcee.autocorr.integrated_time(samples.transpose())
print('all:', it)

### Make plots of ACF estimate for a few different chain lengths
#window = int(2*true_tau)
#tau = np.arange(window+1)
#f0 = kernel.get_value(tau) / kernel.get_value(0.0)
#
## Loop over chain lengths:
#fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
#for n, ax in zip([10, 100, 1000], axes):
#    nn = int(true_tau * n)
#    ax.plot(tau / true_tau, f0, "k", label="true")
#    ax.plot(tau / true_tau, autocorr_func_1d(y[0, :nn])[:window+1], label="estimate")
#    ax.set_title(r"$N = {0}\,\tau_\mathrm{{true}}$".format(n), fontsize=14)
#    ax.set_xlabel(r"$\tau / \tau_\mathrm{true}$")
#
#axes[0].set_ylabel(r"$\rho_f(\tau)$")
#axes[-1].set_xlim(0, window / true_tau)
#axes[-1].set_ylim(-0.05, 1.05)
#axes[-1].legend(fontsize=14);
#
## averagrage over all autocorrelation functions
#fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
#for n, ax in zip([10, 100, 1000], axes):
#    nn = int(true_tau * n)
#    ax.plot(tau / true_tau, f0, "k", label="true")
#    f = np.mean([autocorr_func_1d(y[i, :nn], norm=False)[:window+1]
#                 for i in range(len(y))], axis=0)
#    f /= f[0]
#    ax.plot(tau / true_tau, f, label="estimate")
#    ax.set_title(r"$N = {0}\,\tau_\mathrm{{true}}$".format(n), fontsize=14)
#    ax.set_xlabel(r"$\tau / \tau_\mathrm{true}$")
#
#axes[0].set_ylabel(r"$\rho_f(\tau)$")
#axes[-1].set_xlim(0, window / true_tau)
#axes[-1].set_ylim(-0.05, 1.05)
#axes[-1].legend(fontsize=14);
#plt.show()

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    # first averages all chains, than calculates autocorr. func.
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    # first calculates all autocorr. functions, than averages them
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

# Compute the estimators for a few different chain lengths
N = np.exp(np.linspace(np.log(100), np.log(y.shape[1]), 10)).astype(int)
gw2010 = np.empty(len(N))
new = np.empty(len(N))
for i, n in enumerate(N):
    gw2010[i] = autocorr_gw2010(y[:, :n])
    new[i] = autocorr_new(y[:, :n])

# Plot the comparisons
plt.loglog(N, gw2010, "o-", label="G\&W 2010")
plt.loglog(N, new, "o-", label="DFM 2017")
ylim = plt.gca().get_ylim()
plt.plot(N, N / 1000.0, "--k", label=r"$\tau = N/1000$")
plt.axhline(true_tau, color="k", label="truth", zorder=-100)
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14);
#plt.show()


### another example /realistic/
#import emcee
#
#def log_prob(p):
#    return np.logaddexp(-0.5*np.sum(p**2), -0.5*np.sum((p-4.0)**2))
#
#sampler = emcee.EnsembleSampler(32, 3, log_prob)
#sampler.run_mcmc(np.concatenate((np.random.randn(16, 3),
#                                 4.0+np.random.randn(16, 3)), axis=0),
#                 500000, progress=True);
#
#chain = sampler.get_chain()[:, :, 0].T
#
#plt.hist(chain.flatten(), 100)
#plt.gca().set_yticks([])
#plt.xlabel(r"$\theta$")
#plt.ylabel(r"$p(\theta)$");
#plt.show()
#
## Compute the estimators for a few different chain lengths
#N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
#gw2010 = np.empty(len(N))
#new = np.empty(len(N))
#for i, n in enumerate(N):
#    gw2010[i] = autocorr_gw2010(chain[:, :n])
#    new[i] = autocorr_new(chain[:, :n])
#
## Plot the comparisons
#plt.loglog(N, gw2010, "o-", label="G\&W 2010")
#plt.loglog(N, new, "o-", label="DFM 2017")
#ylim = plt.gca().get_ylim()
#plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
#plt.ylim(ylim)
#plt.xlabel("number of samples, $N$")
#plt.ylabel(r"$\tau$ estimates")
#plt.legend(fontsize=14);
#plt.show()

### different autocorrelation time estimation (suitable for shorter chains)
from scipy.optimize import minimize

def autocorr_ml(y, thin=1, c=5.0):
    # Compute the initial estimate of tau using the standard method
    init = autocorr_new(y, c=c)
    z = y[:, ::thin]
    N = z.shape[1]
    
    # Build the GP model
    tau = max(1.0, init/thin)
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

# Calculate the estimate for a set of different chain lengths
ml = np.empty(len(N))
ml[:] = np.nan
for j, n in enumerate(N[1:8]):
    i = j+1
    thin = max(1, int(0.05*new[i]))
#    ml[i] = autocorr_ml(chain[:, :n], thin=thin)
    ml[i] = autocorr_ml(y[:, :n], thin=thin)
# Plot the comparisons
#plt.loglog(N, gw2010, "o-", label="G\&W 2010")
#plt.loglog(N, new, "o-", label="DFM 2017")
plt.loglog(N, ml, "o-", label="DFM 2017: ML")
ylim = plt.gca().get_ylim()
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14);
plt.show()