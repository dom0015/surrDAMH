#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Literal
import sys
import numpy as np


class Stage:
    def __init__(self,
                 algorithm_type: Literal["MH", "DAMH"] = "MH",
                 proposal_sd: float | None = None,
                 max_samples: int = sys.maxsize,
                 max_evaluations: int = sys.maxsize,
                 time_limit: float = np.inf,
                 surrogate_is_updated: bool = True,
                 is_excluded: bool = False,
                 is_adaptive: bool = False,
                 adaptive_target_rate: float | None = None,  # target acceptance rate of the adaptive algorithm
                 adaptive_corr_limit=None,  # TODO
                 adaptive_sample_limit: int | None = None,
                 use_only_surrogate: bool = False):
        self.algorithm_type = algorithm_type
        self.proposal_sd = proposal_sd  # TODO:  if None previous used?; replace by general Proposal class
        self.max_samples = max_samples
        self.max_evaluations = max_evaluations
        self.time_limit = time_limit
        self.surrogate_is_updated = surrogate_is_updated
        self.is_excluded = is_excluded,
        self.is_adaptive = is_adaptive
        self.adaptive_target_rate = adaptive_target_rate
        self.adaptive_corr_limit = adaptive_corr_limit
        self.adaptive_sample_limit = adaptive_sample_limit
        self.use_only_surrogate = use_only_surrogate

        if self.max_samples == sys.maxsize and self.max_evaluations == sys.maxsize and self.time_limit == np.inf:
            self.max_evaluations = 10
            print(self.algorithm_type, ": No stopping condition specified, max_evaluations set to", self.max_evaluations)
