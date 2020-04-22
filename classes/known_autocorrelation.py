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
l = int(1e6)
S = aa.Samples()
S.generate_celerite(no_chains=N, length=l, log_c1=-5.0, log_c2=-2.0)
S.calculate_properties()
S.print_properties()
y = S.x
true_tau = S.autocorr_time_true

S.autocorr_time_single()
print(S.autocorr_time_est)

S.autocorr_function(l)
S.plot_autocorr_function(1000)

### test emcee more chains
import emcee
it = emcee.autocorr.integrated_time(y.transpose())
print('all:', it)

# Compute the estimators for a few different chain lengths
N = np.exp(np.linspace(np.log(100), np.log(y.shape[1]), 10)).astype(int)
gw2010 = np.empty(len(N))
new = np.empty(len(N))
for i, n in enumerate(N):
    gw2010[i] = aa.autocorr_gw2010(y[:, :n])
    new[i] = aa.autocorr_new(y[:, :n])
ml = np.empty(len(N))
ml[:] = np.nan
# ML
for j, n in enumerate(N[1:8]):
    i = j+1
    thin = max(1, int(0.05*new[i]))
    ml[i] = aa.autocorr_ml(y[:, :n], thin=thin)
# Plot the comparisons
plt.plot(N, gw2010, "o-", label="G\&W 2010")
plt.plot(N, new, "o-", label="DFM 2017")
plt.plot(N, ml, "o-", label="DFM 2017: ML")
ylim = plt.gca().get_ylim()
plt.plot(N, N / 1000.0, "--k", label=r"$\tau = N/1000$")
plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
plt.axhline(true_tau, color="k", label="truth", zorder=-100)
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14);
plt.xscale('log')
plt.show()

print(gw2010)
print(new)
print(ml)

# autocorr_new for several subchains
no_sub = 3 # no. "sliding" subchains of given length
no_len = 10 # no. different subchain lengths
l_sub = np.exp(np.linspace(np.log(l/100),np.log(l),no_len)).astype(int)
sub_it = np.empty((no_sub,no_len))
sub_it_ml = np.empty((no_sub,no_len))
for j_idx,j in enumerate(l_sub[:-1]):
    step = np.floor((l-j)/(no_sub-1)).astype(int)
    for i in range(no_sub):
        subchain = y[:, i*step:i*step+l_sub[j_idx]]
        sub_it[i,j_idx] = aa.autocorr_new(subchain)
        thin = max(1, int(0.05*sub_it[i,j_idx]))
        sub_it_ml[i,j_idx] = aa.autocorr_ml(subchain, thin=thin)
        print(i,j,sub_it[i,j_idx],thin,sub_it_ml[i,j_idx])
sub_it[:,-1] = aa.autocorr_new(y)
thin = max(1, int(0.05*sub_it[0,-1]))
sub_it_ml[:,-1] = aa.autocorr_ml(y, thin=thin)
print(0,l,sub_it[0,-1],thin,sub_it_ml[0,-1])
xx = l_sub.reshape((1,no_len)).repeat(no_sub,axis=0).transpose()
yy = sub_it.transpose()
zz = sub_it_ml.transpose()
plt.scatter(xx,yy)
plt.scatter(xx,zz)
plt.axhline(true_tau, color="k", label="truth", zorder=-100)
plt.xscale('log')
plt.grid()
plt.show()