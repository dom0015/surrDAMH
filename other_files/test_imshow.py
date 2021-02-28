# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

x = np.array([0, 1])
y = np.array([2, 3])
M = np.zeros((2,2))
M[0,1] = 10
M[1,0] = 20
M[1,1] = 30
f = scipy.interpolate.interp2d(x,y,M)

xx=np.linspace(0,1,100)
yy=np.linspace(2,3,100)

plt.figure()
plt.imshow(f(xx,yy),origin="lower",extent=[0,1,2,3],cmap="jet")
plt.colorbar()
plt.show()