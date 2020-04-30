#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:21:37 2020

@author: simona
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.integrate import quad

def c(r, lam):
    return np.exp(-lam*r)

#def c_approx(r, g, f):
#    tmp = -f*np.power(r,2)
#    return np.sum(g*np.exp(tmp),axis=0)

def c_approx_u(r, g, u):
    M = len(u)
    g = g.reshape((M,1))
    tm = np.cumsum(u).reshape((M,1))
    tm = np.exp(tm)
    f = tm
    tmp = -f*np.power(r,2)
    return np.sum(g*np.exp(tmp),axis=0)

def integral_arg(g, u, lam):
    return lambda r : np.power(c(r,lam)-c_approx_u(r,g,u),2)

def L2norm(param,lam):
    M = int(len(param)/2)
    g = param[:M]
    u = param[M:]
    fun = integral_arg(g, u, lam)
#    print(fun(2))
    I = quad(fun, 0, 1)
    return np.sqrt(I[0])

xx = np.linspace(0,10,100)
lam = 1
cxx = c(xx,lam)
M = 6
g = 1e-2*np.ones((M,))
u = 1e-2*np.ones((M,))
cxx_approx = c_approx_u(xx, g, u)
#fun_lambda = integral_arg(g, u, lam)
#funcxx = fun_lambda(xx)
plt.plot(xx,cxx,label='real')
plt.plot(xx,cxx_approx)
#plt.plot(xx,funcxx)
#plt.show()

param = np.append(g,u)
L2 = L2norm(param,lam)

a = minimize(L2norm, param, args=(lam,), method = 'BFGS')
g = a.x[:M].reshape(M,1)
u = a.x[M:].reshape(M,1)
cxx_approx = c_approx_u(xx, g, u)
plt.plot(xx,cxx_approx,label='opt')

a = basinhopping(L2norm, param, niter=1, minimizer_kwargs={'args':(lam,)})
g = a.x[:M].reshape(M,1)
u = a.x[M:].reshape(M,1)
cxx_approx = c_approx_u(xx, g, u)
plt.plot(xx,cxx_approx,label='stoch')

plt.legend()
plt.show()