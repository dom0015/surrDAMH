#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:11:28 2019

@author: simona
"""

class Examples:

    def linela4(data_par):
        k1 = data_par[0]
        k2 = data_par[1]
        k3 = data_par[2]
        k4 = data_par[3]
        f = -0.1
        L = 1.0
        M12 = 0.25
        M23 = 0.5
        M34 = 0.75
        C4 = (f*L)/k4
        C3 = C4*k4/k3
        C2 = C3*k3/k2
        C1 = C2*k2/k1
        D1 = 0
        D2 = -f/k1*M12*M12/2 + C1*M12 + D1 + f/k2*M12*M12/2 - C2*M12
        D3 = -f/k2*M23*M23/2 + C2*M23 + D2 + f/k3*M23*M23/2 - C3*M23
        D4 = -f/k3*M34*M34/2 + C3*M34 + D3 + f/k4*M34*M34/2 - C4*M34
        uL = -f/k4*L*L/2 + C4*L + D4
        return uL