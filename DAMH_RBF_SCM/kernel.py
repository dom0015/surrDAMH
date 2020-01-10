#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:18:06 2018

@author: dom0015
"""

import numpy as np

def kernel(arr,kernel_type):
#    arr=arr
    if kernel_type==0:
        return
    if kernel_type==1:
        np.power(arr,3,out=arr)
        return
    if kernel_type==2: # best for 2 parameters
        np.power(arr,5,out=arr)
        return
    if kernel_type==3:
        np.power(arr,7,out=arr)
        return
    if kernel_type==4:
        temp = -np.power(arr,2)
        np.exp(temp,out=arr)
        return
    if kernel_type==5:
        temp = np.power(arr,2)+1
        np.power(temp,-0.5,out=arr)
        return
    if kernel_type==6:
        arr=np.power(arr,5)
        arr=-arr
        return
    if kernel_type==7:
        np.power(arr,9,out=arr)
        return