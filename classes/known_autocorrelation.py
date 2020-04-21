#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:06:45 2020

@author: simona
"""

# from https://dfm.io/posts/autocorr/ Foreman-Mackey

import numpy as np
import matplotlib.pyplot as plt
import autocorr_analysis as aa

np.random.seed(123456)
N = 4
S = aa.Samples()
S.generate_celerite(no_chains=N, length=2000000, log_c1=-4.0, log_c2=-2.0)
S.calculate_properties()
S.print_properties()
y = S.x
true_tau = S.autocorr_time_true

### plot autocorrelation functions and calculate i. t. using emcee
for i in range(N):
    l_cut = int(true_tau * 4000)
    samples = y[i,:l_cut].reshape((l_cut))
    acf = aa.autocorr_func_1d(samples)
    plt.plot(acf[:1000],label=i)
    c = aa.Samples(samples)
    it = c.integrated_time_emcee(c=5, tol=50)
    print(it)
plt.legend()
plt.show()
    
### test emcee more chains
import emcee
samples = y[:,:l_cut]
it = emcee.autocorr.integrated_time(samples.transpose())
print('all:', it)

# Compute the estimators for a few different chain lengths
N = np.exp(np.linspace(np.log(100), np.log(y.shape[1]), 10)).astype(int)
gw2010 = np.empty(len(N))
new = np.empty(len(N))
for i, n in enumerate(N):
    gw2010[i] = aa.autocorr_gw2010(y[:, :n])
    new[i] = aa.autocorr_new(y[:, :n])

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

# Calculate the estimate for a set of different chain lengths
ml = np.empty(len(N))
ml[:] = np.nan
for j, n in enumerate(N[1:8]):
    i = j+1
    thin = max(1, int(0.05*new[i]))
#    ml[i] = autocorr_ml(chain[:, :n], thin=thin)
    ml[i] = aa.autocorr_ml(y[:, :n], thin=thin)
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