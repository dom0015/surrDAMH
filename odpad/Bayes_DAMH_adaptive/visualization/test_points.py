#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:49:42 2018

@author: dom0015
"""

import numpy as np
import matplotlib.pyplot as plt
import time

N = 2000
P = np.random.rand(N,2)
plt.figure()
plt.scatter(P[:,0],P[:,1])

P0 = np.reshape(P[:,0],(N,1))
P1 = np.reshape(P[:,1],(N,1))

M0=P0-np.transpose(P0)
M1=P1-np.transpose(P1)

M=np.sqrt(np.multiply(M0,M0)+np.multiply(M1,M1))
maximum = np.max(M)
M=M + maximum*2*np.eye(N)

to_keep = np.ones(N,dtype=bool)
t=time.time()
for i in range(N - 200):
#    S=sum(M)
    argmin = np.argmin(M)
    xx = argmin // N
    yy = argmin % N
    M[xx,yy]=2*maximum
    M[yy,xx]=2*maximum
#    if S[yy]<S[xx]:
#        yy=xx
    M[xx,:]=2*maximum
    M[:,xx]=2*maximum
    to_keep[xx]=False
print(time.time()-t)

plt.figure()
plt.scatter(P[to_keep,0],P[to_keep,1])

