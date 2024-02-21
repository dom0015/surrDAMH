#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:15:50 2020

@author: simona
"""

import numpy as np
import numpy.typing as npt
from surrDAMH.surrogates.parent import Evaluator


class ConstantEvaluator(Evaluator):
    def __init__(self, no_parameters, no_observations) -> None:
        self.no_parameters = no_parameters
        self.no_observations = no_observations

    def __call__(self, datapoints: npt.NDArray):
        # returns 1.0 for each datapoint
        datapoints = datapoints.reshape(-1, self.no_parameters)
        no_datapoints = datapoints.shape[0]
        return np.ones((no_datapoints, self.no_observations))
