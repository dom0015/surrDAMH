#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:53:02 2020

@author: domesova
"""

import numpy as np
import matplotlib.pyplot as plt

# DAMH-SMU proposal std:
proposal_std = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# autocorr time calculated from 8 parallel chains:
tau = [901, 597, 542, 468, 469, 698, 1335, 2562, 5682, 16216, 40760]
# no_evaluations * tau / length of the chain:
CPUS0 = [547.287305370548,256.600129648368,179.475809089918,112.477965591436,93.0203978607122,57.6369616278456,47.7951864723707,38.8181086374973,34.3328761217345,38.6972853895235,43.08740030874]
# (no_evaluations/length + 0.002) * tau:
CPUS = [549.089305370548, 257.794129648368, 180.559809089918, 113.413965591436, 93.9583978607122, 59.0329616278456, 50.4651864723707, 43.9421086374973, 45.6968761217345, 71.1292853895235, 124.60740030874]

plt.plot(proposal_std, tau, 'o--')
plt.yscale("log")
# plt.xscale("log")
plt.grid(which="both")

plt.plot(proposal_std, CPUS0, 'o--')
plt.yscale("log")
# plt.xscale("log")
plt.grid(which="both")

plt.plot(proposal_std, CPUS, 'o--')
plt.yscale("log")
# plt.xscale("log")
plt.grid(which="both")
plt.legend(['autocorr. time','eval. per uncorr.','cost per uncorr.'])
plt.xlabel('proposal std')
plt.show()

# MH proposal std:
proposal_std = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
# MH autocorrelation length:
#tau = [662.68703136, 405.22113592, 386.24342313, 300.59732051, 241.54724039, 220.6082899, 242.20767419, 187.92976789, 244.64672502, 309.09970995]
tau = [662.68703136, 455.22113592, 386.24342313, 300.59732051, 241.54724039, 220.6082899, 217.92976789, 232.20767419, 244.64672502, 309.09970995]

plt.plot(proposal_std, tau, 'o--')
plt.yscale("log")
# plt.xscale("log")
plt.grid(which="both")
plt.legend(['autocorr. time','eval. per uncorr.','cost per uncorr.'])
plt.xlabel('proposal std')
plt.show()