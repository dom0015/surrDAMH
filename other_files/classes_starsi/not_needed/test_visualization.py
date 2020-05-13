#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:11:54 2019

@author: simona
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = 'my_problem' + "/" + 'my_MH_alg' + str(1) + ".csv"
df = pd.read_csv(filename, header=None)
weights = np.array(df[0])
xx = np.cumsum(weights)
xx = np.append(0,xx)
for i in range(1,df.shape[1]):
    yy = np.array(df[i])
    yy = np.append(yy[0],yy)
    plt.step(xx, yy, label=str(i))
plt.legend(title='Parameter')
plt.show()

for i in range(1,df.shape[1]):
    yy = np.array(df[i])
    print(np.mean(yy))
    plt.hist(yy, weights=weights, label=str(i))
plt.legend(title='Parameter')
plt.show()

dims = [1,2]
xx = np.array(df[dims[0]])
yy = np.array(df[dims[1]])
plt.hist2d(xx,yy,weights=weights,bins=20)
plt.show()

plt.scatter(xx,yy,s=1)
plt.show()