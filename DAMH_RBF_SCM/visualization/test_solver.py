#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:55:46 2018

@author: dom0015
"""

def solve(data_par,data_obs):
            x = data_par[0]
            y = data_par[1]
            data_obs[0]=pow(pow(x,2)+y-11,2)+pow(x+pow(y,2)-7,2)