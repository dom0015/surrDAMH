#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Literal
import sys
import numpy as np
from dataclasses import dataclass


@dataclass
class Stage:
    algorithm_type: Literal["MH", "DAMH"] = "MH"
    proposal_sd: float | None = None
    max_samples: int = sys.maxsize
    max_evaluations: int = sys.maxsize
    time_limit: float = np.inf
    surrogate_is_updated: bool = True
    is_saved: bool = True
    is_excluded: bool = False
    is_adaptive: bool = False
    adaptive_target_rate: float | None = None  # target acceptance rate of the adaptive algorithm
    adaptive_corr_limit = None  # TODO
    adaptive_sample_limit: int | None = None
    use_only_surrogate: bool = False
    name: str | None = None  # will be set later

    def __post_init__(self):
        if self.max_samples == sys.maxsize and self.max_evaluations == sys.maxsize and self.time_limit == np.inf:
            self.max_evaluations = 10
            print(self.algorithm_type, ": No stopping condition specified, max_evaluations set to", self.max_evaluations)
