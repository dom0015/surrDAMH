#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:53:45 2018

@author: dom0015
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file=pd.read_csv('G_data.csv')

obs = file.values[:,1]
par_x = file.values[:,2]
par_y = file.values[:,3] 

print(obs)

plt.figure()  
konec = 2000
plt.scatter(par_x[:konec],par_y[:konec])

