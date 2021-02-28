
import numpy as np
import matplotlib.pyplot as plt

samples = np.random.multivariate_normal(mean=[5,3], cov=[[4,-2],[-2,4]], size=10000)
#samples = np.random.multivariate_normal(mean=[0,0], cov=[[9,0],[0,9]], size=10000)

plt.figure()
plt.scatter(samples[:,0],samples[:,1],1)
plt.show()