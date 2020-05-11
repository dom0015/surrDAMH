#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:07:05 2019

@author: simona
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scm, rbf

def model(k1,k2,k3,k4):
    f = -0.1
    L = 1.0
    M12 = 0.25
    M23 = 0.5
    M34 = 0.75
    C4 = (f*L)/k4
    C3 = C4*k4/k3
    C2 = C3*k3/k2
    C1 = C2*k2/k1
    D1 = 0
    D2 = -f/k1*M12*M12/2 + C1*M12 + D1 + f/k2*M12*M12/2 - C2*M12
    D3 = -f/k2*M23*M23/2 + C2*M23 + D2 + f/k3*M23*M23/2 - C3*M23
    D4 = -f/k3*M34*M34/2 + C3*M34 + D3 + f/k4*M34*M34/2 - C4*M34
    uL = -f/k4*L*L/2 + C4*L + D4
    return uL

def likelihood(y,Gu):
    Sinv=1/(2*0.0001*2*0.0001)
    x=np.exp(-1/2*(y-Gu)*Sinv*(y-Gu))
    return x

def prior(u):
    mu=10
    Sinv=1/(1.5*1.5)
    p=np.exp(-1/2*(u-mu)*Sinv*(u-mu))
    return p

file=pd.read_csv('G_data_linela_MH.csv',skiprows=1,header=None)

Gu=np.array(file[1])
u1=np.array(file[2])
u2=np.array(file[3])
u3=np.array(file[4])
u4=np.array(file[5])
u=file.as_matrix(columns=[2,3,4,5])

#Surrogate = scm.scm()
#surrogate_parameters = [no_parameters,7,None,None,None]

Surrogate = rbf.rbf()
surrogate_parameters = [None,0,0,0,0]

N=150
w=np.ones([N,1])

SOL, no_evaluations, alldata_par, alldata_obs, alldata_wei, TEMP2, RHS = Surrogate.calculate(u[0:N,:],Gu[0:N].reshape([N,1]),w,2,surrogate_parameters)
no_parameters = 4
#newdata_par = u[N:2*N,:]
#newdata_surrogate = Surrogate.apply(SOL, newdata_par, alldata_par, no_parameters, 0)

# evaluate surrogate model on a grid
M=10
bounds=[7,11]
K1 = np.linspace(bounds[0],bounds[1],M)
K2 = np.linspace(bounds[0],bounds[1],M)
K3 = np.linspace(bounds[0],bounds[1],M)
K4 = np.linspace(bounds[0],bounds[1],M)
X1,X2,X3,X4 = np.meshgrid(K1,K2,K3,K4)
K = np.column_stack((X1.flatten(),X2.flatten(),X3.flatten(),X4.flatten()))
Su = Surrogate.apply(SOL,K,alldata_par,no_parameters,0)
Su = Su.flatten()
#Z=Su.reshape((M,M))
#fig = plt.figure()
#plt.imshow(Z,extent=[bounds[0],bounds[1],bounds[0],bounds[1]],origin='lower')
#plt.colorbar()

# evaluate original model on a grid (linela)
f = -0.1
L = 1.0
cM = 0.5
k1=X1.flatten()
k2=X2.flatten()
k3=X3.flatten()
k4=X4.flatten()
Gu=model(k1,k2,k3,k4)
#uL = Gu.reshape((M,M))
#fig = plt.figure()
#plt.imshow(uL,extent=[bounds[0],bounds[1],bounds[0],bounds[1]],origin='lower')
#plt.colorbar()
# evaluate original posterior on a grid
y=model(8,8,8,8)
pi0=prior(k1)*prior(k2)

# absolute difference
#D=np.abs(Z-uL)
#fig = plt.figure()
#plt.imshow(D,extent=[bounds[0],bounds[1],bounds[0],bounds[1]])
#plt.colorbar()

#fig = plt.figure()
#plt.imshow(pi0.reshape((M,M)),extent=[bounds[0],bounds[1],bounds[0],bounds[1]],origin='lower')
#fig = plt.figure()
posterior=pi0*likelihood(y,Gu)
#plt.imshow(posterior.reshape((M,M)),extent=[bounds[0],bounds[1],bounds[0],bounds[1]],origin='lower')
#fig = plt.figure()
posterior_approx=pi0*likelihood(y,Su)
#plt.imshow(abs(posterior-posterior_approx).reshape((M,M)),extent=[bounds[0],bounds[1],bounds[0],bounds[1]],origin='lower')
#plt.imshow(uL,extent=[bounds[0],bounds[1],bounds[0],bounds[1]],origin='lower')
#plt.colorbar()
print(sum(abs(posterior-posterior_approx))/sum(abs(posterior)))
