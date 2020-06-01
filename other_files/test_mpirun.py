#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:18:39 2020

@author: ber0061
"""

import os
import sys
syspath = os.path.dirname(os.path.realpath(__file__))

print('Running code')
com = 'mpirun -np 3 ' + str(sys.executable) + " " + syspath + "/test_mpi.py"
print(com)
os.system(com)
print('Done')