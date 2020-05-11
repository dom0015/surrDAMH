#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:48:24 2018

@author: dom0015
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

x_all=np.array(0)
y_all=np.array(0)

no_chains=4
all_samples=0;
for i in range(no_chains):
    file=pd.read_csv('examplestopped' + str(i) +'.csv',skiprows=1,header=None)
    file.drop(file.index[len(file)-1])
    
    x=np.array(file[2])
    x = x[~np.isnan(x)]
    x_all=np.append(x_all,x)
    y=np.array(file[3])
    y = y[~np.isnan(y)]
    y_all=np.append(y_all,y)
    print(x[0],y[0],x.shape)
    all_samples = all_samples+x.shape[0]
plt.figure()  
plt.hist2d(x_all,y_all,bins=80)

#plt.Circle((5.0, 5.0), 1.5, color='white', fill=False, clip_on=False)

plt.figure()
plt.hist2d(x_all,y_all,bins=50)

print(all_samples)



# Create a figure. Equal aspect so circles look circular
fig,ax = plt.subplots(1)
#ax.set_aspect('equal')

# Show the image
plt.hist2d(x_all,y_all,bins=50)

# Now, loop through coord arrays, and create a circle at each x,y pair
circ = Circle((5,5),50)
ax.add_patch(circ)

# Show the image
plt.show()




## POL5 KERNEL
#LD_LIBRARY_PATH=/home/dom0015/python-workspace/cython_link_mpi/lib mpirun -n 7 --oversubscribe python3 stats.py
#Rank_world: 2 id_group: 2 local rank: 0 local size: 1 group_local_ids: [2]
#Rank_world: 6 id_group: 6 local rank: 0 local size: 1 group_local_ids: [6]
#Rank_world: 1 id_group: 1 local rank: 0 local size: 1 group_local_ids: [1]
#Rank_world: 3 id_group: 3 local rank: 0 local size: 1 group_local_ids: [3]
#Rank_world: 5 id_group: 5 local rank: 0 local size: 1 group_local_ids: [5]
#Hello, I am group leader with local rank 0 and global rank 6
#Rank_world: 4 id_group: 4 local rank: 0 local size: 1 group_local_ids: [4]
#Hello, I am group leader with local rank 0 and global rank 3
#Hello, I am group leader with local rank 0 and global rank 2
#Hello, I am group leader with local rank 0 and global rank 5
#Hello, I am group leader with local rank 0 and global rank 1
#Hello, I am group leader with local rank 0 and global rank 4
#artificial_observation_without_noise: [-0.03333333]
#artificial observation with noise: -0.01
#Ahoj, ja jsem timer!
#Ahoj, ja vyhodnocuji surrogate model!
#Thread 3 generates a chain.
#observation from Model: -0.01
#Thread 4 generates a chain.
#observation from Model: -0.01
#Hello, I am master. I communicate with solver(s).
#Thread 5 generates a chain.
#observation from Model: -0.01
#Ahoj, ja upgraduji surrogate model!
#Thread 6 generates a chain.
#observation from Model: -0.01
#Tag is 0
#Surrogate solver received first model. (97, 2)
#MH accepted samples: 18 of 66
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 2 of 51
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 8 of 61
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 7 of 50
#MH sampling finished
#Start DAMH-SMU sampling
#Surrogate solver received updated model. 2 (229, 2) 1
#Surrogate solver received updated model. 3 (239, 2) 1
#Surrogate solver received updated model. 4 (245, 2) 1
#Surrogate solver received updated model. 5 (247, 2) 1
#Surrogate solver received updated model. 6 (251, 2) 1
#Surrogate solver received updated model. 7 (253, 2) 1
#Surrogate solver received updated model. 8 (259, 2) 1
#Surrogate solver received updated model. 9 (265, 2) 1
#Surrogate solver received updated model. 10 (270, 2) 1
#Surrogate solver received updated model. 11 (277, 2) 1
#Surrogate solver received updated model. 12 (283, 2) 1
#Surrogate solver received updated model. 13 (287, 2) 1
#Surrogate solver received updated model. 14 (291, 2) 1
#Surrogate solver received updated model. 15 (295, 2) 1
#Surrogate solver received updated model. 16 (299, 2) 1
#Surrogate solver received updated model. 17 (304, 2) 1
#Surrogate solver received updated model. 18 (331, 2) 1
#Surrogate solver received updated model. 19 (370, 2) 1
#Surrogate solver received updated model. 20 (394, 2) 1
#Surrogate solver received updated model. 21 (432, 2) 1
#Surrogate solver received updated model. 22 (487, 2) 1
#Surrogate solver received updated model. 23 (532, 2) 1
#Surrogate solver received updated model. 24 (571, 2) 1
#Surrogate solver received updated model. 25 (587, 2) 1
#Surrogate solver received updated model. 26 (605, 2) 1
#Surrogate solver received updated model. 27 (618, 2) 1
#Surrogate solver received updated model. 28 (651, 2) 1
#Surrogate solver received updated model. 29 (669, 2) 1
#Surrogate solver received updated model. 30 (705, 2) 1
#Surrogate solver received updated model. 31 (753, 2) 1
#Surrogate solver received updated model. 32 (810, 2) 1
#Surrogate solver received updated model. 33 (838, 2) 1
#Surrogate solver received updated model. 34 (911, 2) 1
#Surrogate solver received updated model. 35 (1008, 2) 1
#Surrogate solver received updated model. 36 (1095, 2) 1
#Surrogate solver received updated model. 37 (1173, 2) 1
#Surrogate solver received updated model. 38 (1233, 2) 1
#Surrogate solver received updated model. 39 (1311, 2) 1
#Surrogate solver received updated model. 40 (1399, 2) 1
#Surrogate solver received updated model. 41 (1484, 2) 1
#Surrogate solver received updated model. 42 (1595, 2) 1
#Surrogate solver received updated model. 43 (1668, 2) 1
#Surrogate solver received updated model. 44 (1791, 2) 1
#Surrogate solver received updated model. 45 (1900, 2) 1
#Surrogate solver received updated model. 46 (2022, 2) 1
#Surrogate solver received updated model. 47 (2181, 2) 1
#Surrogate solver received updated model. 48 (2315, 2) 1
#Surrogate solver received updated model. 49 (2542, 2) 1
#Surrogate solver received updated model. 50 (2714, 2) 1
#Surrogate solver received updated model. 51 (2883, 2) 1
#Surrogate solver received updated model. 52 (3129, 2) 1
#Surrogate solver received updated model. 53 (3392, 2) 1
#Surrogate solver received updated model. 54 (3712, 2) 1
#Surrogate solver received updated model. 55 (4034, 2) 1
#Surrogate solver received updated model. 56 (4389, 2) 1
#Surrogate solver received updated model. 57 (4783, 2) 1
#Timer set to 1!
#NUM 0 FINISHED
#DAMH accepted samples: 1269 of 14782 rejected: 13512 1
#NUM 4 FINISHED
#DAMH accepted samples: 1260 of 14790 rejected: 13525 5
#NUM 3 FINISHED
#DAMH accepted samples: 1240 of 15037 rejected: 13795 2
#NUM 6 FINISHED
#DAMH accepted samples: 1249 of 14849 rejected: 13599 1
#NUM 5 FINISHED
#NUM 2 FINISHED
#BYE BYE from rank_world 2
#BYE BYE from rank_world 1
#Joining 0
#BYE BYE from rank_world 4
#BYE BYE from rank_world 5
#BYE BYE from rank_world 6
#BYE BYE from rank_world 3


## EXPONENTIAL KERNEL
#Rank_world: 1 id_group: 1 local rank: 0 local size: 1 group_local_ids: [1]
#Rank_world: 2 id_group: 2 local rank: 0 local size: 1 group_local_ids: [2]
#Rank_world: 4 id_group: 4 local rank: 0 local size: 1 group_local_ids: [4]
#Rank_world: 6 id_group: 6 local rank: 0 local size: 1 group_local_ids: [6]
#Rank_world: 3 id_group: 3 local rank: 0 local size: 1 group_local_ids: [3]
#Rank_world: 5 id_group: 5 local rank: 0 local size: 1 group_local_ids: [5]
#Hello, I am group leader with local rank 0 and global rank 1
#Hello, I am group leader with local rank 0 and global rank 6
#Hello, I am group leader with local rank 0 and global rank 5
#Hello, I am group leader with local rank 0 and global rank 3
#Hello, I am group leader with local rank 0 and global rank 4
#Hello, I am group leader with local rank 0 and global rank 2
#artificial_observation_without_noise: [-0.03333333]
#artificial observation with noise: -0.01
#Ahoj, ja upgraduji surrogate model!
#Ahoj, ja jsem timer!
#Ahoj, ja vyhodnocuji surrogate model!
#Thread 3 generates a chain.
#observation from Model: -0.01
#Thread 5 generates a chain.
#observation from Model: -0.01
#Thread 4 generates a chain.
#Hello, I am master. I communicate with solver(s).
#observation from Model: -0.01
#Thread 6 generates a chain.
#observation from Model: -0.01
#Tag is 0
#Surrogate solver received first model. (101, 2)
#MH accepted samples: 8 of 72
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 7 of 71
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 6 of 51
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 12 of 73
#MH sampling finished
#Start DAMH-SMU sampling
#Surrogate solver received updated model. 2 (246, 2) 1
#Surrogate solver received updated model. 3 (305, 2) 1
#Surrogate solver received updated model. 4 (350, 2) 1
#Surrogate solver received updated model. 5 (394, 2) 1
#Surrogate solver received updated model. 6 (430, 2) 1
#Surrogate solver received updated model. 7 (464, 2) 1
#Surrogate solver received updated model. 8 (517, 2) 1
#Surrogate solver received updated model. 9 (571, 2) 1
#Surrogate solver received updated model. 10 (614, 2) 1
#Surrogate solver received updated model. 11 (676, 2) 1
#Surrogate solver received updated model. 12 (744, 2) 1
#Surrogate solver received updated model. 13 (819, 2) 1
#Surrogate solver received updated model. 14 (890, 2) 1
#Surrogate solver received updated model. 15 (938, 2) 1
#Surrogate solver received updated model. 16 (1010, 2) 1
#Surrogate solver received updated model. 17 (1072, 2) 1
#Surrogate solver received updated model. 18 (1140, 2) 1
#Surrogate solver received updated model. 19 (1215, 2) 1
#Surrogate solver received updated model. 20 (1298, 2) 1
#Surrogate solver received updated model. 21 (1379, 2) 1
#Surrogate solver received updated model. 22 (1495, 2) 1
#Surrogate solver received updated model. 23 (1592, 2) 1
#Surrogate solver received updated model. 24 (1712, 2) 1
#Surrogate solver received updated model. 25 (1830, 2) 1
#Surrogate solver received updated model. 26 (1980, 2) 1
#Surrogate solver received updated model. 27 (2139, 2) 1
#Surrogate solver received updated model. 28 (2345, 2) 1
#Surrogate solver received updated model. 29 (2590, 2) 1
#Surrogate solver received updated model. 30 (2853, 2) 1
#Surrogate solver received updated model. 31 (3140, 2) 1
#Surrogate solver received updated model. 32 (3485, 2) 1
#Surrogate solver received updated model. 33 (3908, 2) 1
#Surrogate solver received updated model. 34 (4336, 2) 1
#Surrogate solver received updated model. 35 (4878, 2) 1
#No evaluations: 5525
#Surrogate solver received updated model. 36 (5525, 2) 1
#Timer set to 1!
#NUM 0 FINISHED
#DAMH accepted samples: 1568 of 18278 rejected: 16704 6
#DAMH accepted samples: 1578 of 18011 rejected: 16414 19
#DAMH accepted samples: 1560 of 18051 rejected: 16490 1
#NUM 6 FINISHED
#NUM 4 FINISHED
#NUM 3 FINISHED
#DAMH accepted samples: 1570 of 18139 rejected: 16568 1


# KERNEL LINEAR
#Rank_world: 1 id_group: 1 local rank: 0 local size: 1 group_local_ids: [1]
#Rank_world: 3 id_group: 3 local rank: 0 local size: 1 group_local_ids: [3]
#Rank_world: 5 id_group: 5 local rank: 0 local size: 1 group_local_ids: [5]
#Rank_world: 4 id_group: 4 local rank: 0 local size: 1 group_local_ids: [4]
#Rank_world: 2 id_group: 2 local rank: 0 local size: 1 group_local_ids: [2]
#Rank_world: 6 id_group: 6 local rank: 0 local size: 1 group_local_ids: [6]
#Hello, I am group leader with local rank 0 and global rank 1
#Hello, I am group leader with local rank 0 and global rank 5
#Hello, I am group leader with local rank 0 and global rank 4
#Hello, I am group leader with local rank 0 and global rank 2
#Hello, I am group leader with local rank 0 and global rank 3
#Hello, I am group leader with local rank 0 and global rank 6
#artificial_observation_without_noise: [-0.03333333]
#artificial observation with noise: -0.01
#Ahoj, ja jsem timer!
#Ahoj, ja vyhodnocuji surrogate model!
#Thread 3 generates a chain.
#Ahoj, ja upgraduji surrogate model!
#Hello, I am master. I communicate with solver(s).
#observation from Model: -0.01
#Thread 4 generates a chain.
#Thread 5 generates a chain.
#observation from Model: -0.01
#observation from Model: -0.01
#Thread 6 generates a chain.
#observation from Model: -0.01
#Tag is 0
#Surrogate solver received first model. (96, 2)
#MH accepted samples: 11 of 73
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 8 of 83
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 14 of 92
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 9 of 93
#MH sampling finished
#Start DAMH-SMU sampling
#Surrogate solver received updated model. 2 (341, 2) 1
#Surrogate solver received updated model. 3 (369, 2) 1
#Surrogate solver received updated model. 4 (399, 2) 1
#Surrogate solver received updated model. 5 (432, 2) 1
#Surrogate solver received updated model. 6 (458, 2) 1
#Surrogate solver received updated model. 7 (488, 2) 1
#Surrogate solver received updated model. 8 (519, 2) 1
#Surrogate solver received updated model. 9 (551, 2) 1
#Surrogate solver received updated model. 10 (572, 2) 1
#Surrogate solver received updated model. 11 (605, 2) 1
#Surrogate solver received updated model. 12 (640, 2) 1
#Surrogate solver received updated model. 13 (686, 2) 1
#Surrogate solver received updated model. 14 (721, 2) 1
#Surrogate solver received updated model. 15 (756, 2) 1
#Surrogate solver received updated model. 16 (788, 2) 1
#Surrogate solver received updated model. 17 (835, 2) 1
#Surrogate solver received updated model. 18 (938, 2) 1
#Surrogate solver received updated model. 19 (1011, 2) 1
#Surrogate solver received updated model. 20 (1086, 2) 1
#Surrogate solver received updated model. 21 (1180, 2) 1
#Surrogate solver received updated model. 22 (1277, 2) 1
#Surrogate solver received updated model. 23 (1385, 2) 1
#Surrogate solver received updated model. 24 (1519, 2) 1
#Surrogate solver received updated model. 25 (1653, 2) 1
#Surrogate solver received updated model. 26 (1828, 2) 1
#Surrogate solver received updated model. 27 (1970, 2) 1
#Surrogate solver received updated model. 28 (2168, 2) 1
#Surrogate solver received updated model. 29 (2405, 2) 1
#Surrogate solver received updated model. 30 (2728, 2) 1
#Surrogate solver received updated model. 31 (3065, 2) 1
#Surrogate solver received updated model. 32 (3479, 2) 1
#Surrogate solver received updated model. 33 (3946, 2) 1
#Surrogate solver received updated model. 34 (4625, 2) 1
#No evaluations: 5483
#Surrogate solver received updated model. 35 (5483, 2) 1
#Timer set to 1!
#NUM 0 FINISHED
#DAMH accepted samples: 2148 of 25218 rejected: 23054 16
#NUM 4 FINISHED
#DAMH accepted samples: 2151 of 25290 rejected: 23128 11
#NUM 6 FINISHED
#DAMH accepted samples: 2047 of 25660 rejected: 23596 17
#NUM 5 FINISHED
#DAMH accepted samples: 2105 of 25231 rejected: 23096 30
#NUM 3 FINISHED
#NUM 2 FINISHED
#BYE BYE from rank_world 3
#BYE BYE from rank_world 1
#BYE BYE from rank_world 2
#BYE BYE from rank_world 6
#BYE BYE from rank_world 5
#BYE BYE from rank_world 4
#Joining 0
#Joining 1
#Joining 2
#Joining 3
#Joining 4
#Joining 5
#Joining 6
#Master finished
#Queues empty? True False True True
#Konec 3
#Konec 1
#Konec 2
#Konec 0
#Konec 6
#Konec 4
#Konec 5


## POL5 STOPPED AT 200:
#Rank_world: 2 id_group: 2 local rank: 0 local size: 1 group_local_ids: [2]
#Rank_world: 6 id_group: 6 local rank: 0 local size: 1 group_local_ids: [6]
#Rank_world: 1 id_group: 1 local rank: 0 local size: 1 group_local_ids: [1]
#Rank_world: 3 id_group: 3 local rank: 0 local size: 1 group_local_ids: [3]
#Rank_world: 5 id_group: 5 local rank: 0 local size: 1 group_local_ids: [5]
#Hello, I am group leader with local rank 0 and global rank 6
#Hello, I am group leader with local rank 0 and global rank 1
#Rank_world: 4 id_group: 4 local rank: 0 local size: 1 group_local_ids: [4]
#Hello, I am group leader with local rank 0 and global rank 3
#Hello, I am group leader with local rank 0 and global rank 5
#Hello, I am group leader with local rank 0 and global rank 2
#Hello, I am group leader with local rank 0 and global rank 4
#artificial_observation_without_noise: [-0.03333333]
#artificial observation with noise: -0.01
#Ahoj, ja upgraduji surrogate model!
#Ahoj, ja jsem timer!
#Ahoj, ja vyhodnocuji surrogate model!
#Thread 4 generates a chain.
#Thread 3 generates a chain.
#observation from Model: -0.01
#observation from Model: -0.01
#Hello, I am master. I communicate with solver(s).
#Thread 6 generates a chain.
#Thread 5 generates a chain.
#observation from Model: -0.01
#observation from Model: -0.01
#Tag is 0
#Surrogate solver received first model. (100, 2)
#MH accepted samples: 5 of 44
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 7 of 30
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 1 of 33
#MH sampling finished
#Start DAMH-SMU sampling
#MH accepted samples: 4 of 31
#MH sampling finished
#Start DAMH-SMU sampling
#Surrogate solver received updated model. 2 (138, 2) 1
#Surrogate solver received updated model. 3 (143, 2) 1
#Surrogate solver received updated model. 4 (149, 2) 1
#Surrogate solver received updated model. 5 (151, 2) 1
#Surrogate solver received updated model. 6 (153, 2) 1
#Surrogate solver received updated model. 7 (154, 2) 1
#Surrogate solver received updated model. 8 (156, 2) 1
#Surrogate solver received updated model. 9 (159, 2) 1
#Surrogate solver received updated model. 10 (160, 2) 1
#Surrogate solver received updated model. 11 (164, 2) 1
#Surrogate solver received updated model. 12 (170, 2) 1
#Surrogate solver received updated model. 13 (172, 2) 1
#Surrogate solver received updated model. 14 (174, 2) 1
#Surrogate solver received updated model. 15 (176, 2) 1
#Surrogate solver received updated model. 16 (181, 2) 1
#Surrogate solver received updated model. 17 (185, 2) 1
#Surrogate solver received updated model. 18 (193, 2) 1
#Surrogate solver received updated model. 19 (199, 2) 1
#No evaluations: 201
#Surrogate solver received updated model. 20 (201, 2) 1
#Timer set to 1!
#NUM 0 FINISHED
#DAMH accepted samples: 4312 of 53621 rejected: 49154 155
#NUM 5 FINISHED
#DAMH accepted samples: 4402 of 53879 rejected: 49321 156
#NUM 3 FINISHED
#DAMH accepted samples: 4311 of 54490 rejected: 50048 131
#NUM 4 FINISHED
#DAMH accepted samples: 4265 of 53652 rejected: 48944 443
#NUM 6 FINISHED
#NUM 2 FINISHED
#BYE BYE from rank_world 1
#BYE BYE from rank_world 2
#Joining 0
#BYE BYE from rank_world 5
#BYE BYE from rank_world 4
#BYE BYE from rank_world 6
#BYE BYE from rank_world 3
#Joining 1
#Joining 2
#Joining 3
#Joining 4
#Joining 5
#Joining 6
#Master finished
#Queues empty? True False True True
#Konec 5
#Konec 4
#Konec 1
#Konec 0
#Konec 3
#Konec 2
#Konec 6
