#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:35:47 2019

@author: simona
"""

#import main_codes as sa
#
#my_Sol = sa.Solver_local_2to2()
#my_Prob = sa.Problem_Gauss(no_parameters=my_Sol.no_parameters,
#                           noise_std=[2.0, 0.1],
#                           prior_mean=0.0, 
#                           prior_std=1.5,
#                           no_observations=my_Sol.no_observations, 
#                           observations=None,
#                           seed=22,
#                           name='my_problem')
#
#print(my_Prob.sample_prior())
#print(my_Prob.sample_prior())
#print(my_Prob.observations)
##artificial_parameters = np.array([4,2])
##artificial_observations = my_Sol.get_solution(artificial_parameters) + my_Prob.sample_noise(generator=np.random.RandomState(33))
##artificial_observations = my_Sol.get_solution(artificial_parameters) 
#my_Prob.observations = [66.4, 2] #artificial_observations
#print(my_Prob.observations)
#
#my_Prop = sa.Proposal_GaussRandomWalk(no_parameters=my_Sol.no_parameters,
#                                      proposal_std=0.5,
#                                      seed=44)
#my_Alg = sa.Algorithm_MH(my_Prob, my_Prop, my_Sol,
#                         initial_sample=my_Prob.prior_mean,
#                         max_samples=100,
#                         name='my_MH_alg',
#                         seed=55)
#my_Alg.run()
#print(my_Alg.no_accepted, my_Alg.no_rejected, my_Alg.no_accepted/my_Alg.no_rejected*100, '%')