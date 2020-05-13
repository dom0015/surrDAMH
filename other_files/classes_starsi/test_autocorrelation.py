#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:55:21 2020

@author: simona
"""

import numpy as np
import matplotlib.pyplot as plt

def acf(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

N=10000 # length of chains
X=[]
for j in range(1):
    X0 = np.zeros(N)
    for i in range(N-1):
        X0[i+1] = X0[i] + np.random.rand()-0.5
    X.append(X0)
X1 = np.random.rand(N)-0.5
#X.append(X1)
tmp = X0[:200]
tmp = np.arange(200)
X3 = np.zeros(0)
for i in range(5):
    X3 = np.append(X3,tmp)
X.append(X3)

#M=len(X)
#for i in range(M):
#    plt.plot(X[i])
#plt.show()

import autocorr_functions
x = X[1]
c = autocorr_functions.Chain(x)
res = c.all_autocorr()

fig,ax=plt.subplots()
for name, corr in res.items():
    print(name)
    ax.plot(corr,label=name)
ax.set_xlabel('lag')
ax.set_ylabel('correlation coefficient')
ax.legend()
plt.show()

for name in ['manual', 'fft1', 'correlate1', 'emcee']:
    plt.plot(res[name],label=name)
plt.legend()
plt.show()

it = c.integrated_time_emcee()
print(it)