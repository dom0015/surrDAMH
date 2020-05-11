#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:26:57 2020

@author: simona
"""

import autocorr_analysis as aa
import numpy as np

S = aa.Samples()
m = int(1e5)
length = [10*m, 6*m, 8*m, 10*m, 6*m]

#no_parameters = 3
#S.generate_samples_rand(no_parameters, length)
#length_disp = [100] * no_parameters

#S.generate_samples_celerite(length, [-6.0, -2.0, -2.0], [-6.0, -2.0, -1.0])
##S.generate_samples_celerite(length, [-6.0, -2.0], [-6.0, -2.0])
#length_disp = list(np.multiply(S.autocorr_time_true,5))
#length_disp = list(int(i) for i in length_disp)

S.load_MH('my_problem3')
S.calculate_properties()
length_disp = [1000] * S.no_parameters

S.calculate_properties()
S.print_properties()
S.calculate_autocorr_function()
S.plot_segment(length_disp)
S.plot_hist()
S.calculate_autocorr_time()
S.calculate_autocorr_function_mean()
S.plot_autocorr_function(length_disp,plot_mean=True)
S.calculate_autocorr_time_mean()
print(S.autocorr_time)
print(S.autocorr_time_mean)
print(S.autocorr_time_mean_beta)