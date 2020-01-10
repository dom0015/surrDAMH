#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:34:58 2018

@author: dom0015
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import scipy.stats as ss
import time
import csv

#file = pd.read_csv('notes.csv',header=None)
#file = pd.read_csv('notes_equalproposal.csv',header=None)
file = pd.read_csv('notes_4proposal_4par.csv',header=None)
#file = pd.read_csv('notes_4proposal_2par.csv',header=None)
#file = pd.read_csv('notes_doubleproposal.csv',header=None)
MH = file[file[3]=='MH']
DAMH = file[file[3]=='DAMH']
ADAMH = file[file[3]=='ADAMH']
DAMHSMU = file[file[3]=='DAMHSMU']
eff0 = np.append(np.ones(5),np.zeros(10))
eff = np.zeros(0)

repeats=int(MH.shape[0]/15)

for i in range(repeats):
    eff = np.append(eff,eff0)
eff = np.array(eff, dtype=bool)
effp = pd.DataFrame(eff)
MHeff = MH[eff]
MHeff = MHeff.drop(columns=[4,5])
MHsum = MHeff.sum(axis=1)

MHm = MHsum.as_matrix()
MHm = MHm.reshape((repeats,5)).transpose()
MHmu = MHm.mean(axis=0)
print(MHmu)

DAMHm = DAMH.as_matrix(columns=[2])
DAMHm = DAMHm.reshape((repeats,5)).transpose()
DAMHmu = DAMHm.mean(axis=0)
print(DAMHmu)

ADAMHm = ADAMH.as_matrix(columns=[2])
ADAMHm = ADAMHm.reshape((repeats,5)).transpose()
ADAMHmu = ADAMHm.mean(axis=0)
print(ADAMHmu)

DAMHSMUm = DAMHSMU.as_matrix(columns=[2])
DAMHSMUm = DAMHSMUm.reshape((repeats,5)).transpose()
DAMHSMUmu = DAMHSMUm.mean(axis=0)
print(DAMHSMUmu)


idx = MHmu.argsort()
plt.figure(figsize=(5, 4), dpi=400)
plt.plot(MHmu[idx],DAMHmu[idx],'-o')
plt.plot(MHmu[idx],ADAMHmu[idx],'-o')
plt.plot(MHmu[idx],DAMHSMUmu[idx],'-o')
plt.xscale("log")
plt.legend(('DAMH','ADAMH','DAMH-SMU'), fontsize=11)
plt.xlabel("No. of initial snapshots", fontsize=12)
plt.ylabel("No. of rejected samples", fontsize=12)
plt.show()

#
#x_all=np.array(0)
#y_all=np.array(0)
#
#no_chains=5
#all_samples=0;
#for i in range(no_chains):
#    file=pd.read_csv('data_linela_SMU_' + str(i+10) +'.csv',skiprows=1,header=None)
#    file.drop(file.index[len(file)-1])
#    
#    x=np.array(file[2])
#    x = x[~np.isnan(x)]
#    x_all=np.append(x_all,x)
#    y=np.array(file[3])
#    y = y[~np.isnan(y)]
#    y_all=np.append(y_all,y)
#    print(x[0],y[0],x.shape)
#    all_samples = all_samples+x.shape[0]
#plt.figure()  
#plt.hist2d(x_all,y_all,bins=80)
#
##plt.Circle((5.0, 5.0), 1.5, color='white', fill=False, clip_on=False)
#
#plt.figure()
#plt.hist2d(x_all,y_all,bins=50)
#
#print(all_samples)
#
#
#
## Create a figure. Equal aspect so circles look circular
#fig,ax = plt.subplots(1)
##ax.set_aspect('equal')
#
## Show the image
#plt.hist2d(x_all,y_all,bins=50)
#
## Now, loop through coord arrays, and create a circle at each x,y pair
#circ = Circle((5,5),50)
#ax.add_patch(circ)
#
## Show the image
#plt.show()