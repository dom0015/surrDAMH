#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np

from surrDAMH.modules.proposals import Proposal


@dataclass
class Stage:
    algorithm_type: Literal["MH", "DAMH"] = "MH"  # DAMH uses delayed acceptance, MH does not
    proposal: Proposal | None = None
    proposal_sd_or_cov: float | None = None
    adaptive: bool = False
    max_samples: int = sys.maxsize  # termination condition - total number of samples
    max_evaluations: int = sys.maxsize  # termination condition - total number of full model evaluations
    time_limit: float = np.inf  # termination condition - total time
    send_snapshots_to_collector: bool = True  # use snapshots from this stage for surrogate updates
    surrogate_model_updates: bool = True  # only with DAMH, surrogate changes during the stage (DAMH-SMU)
    use_only_surrogate: bool = False  # if True, surrogate is used instead of full model
    save_to_file: bool = True  # samples are saved to file
    is_excluded: bool = False  # if True, the next stage starts from the same sample as this one
    adaptive_target_rate: float | None = None  # target acceptance rate of the adaptive algorithm
    adaptive_corr_limit = None
    adaptive_sample_limit: int | None = None
    name: str | None = None  # will be set later

    def __post_init__(self):
        if self.max_samples == sys.maxsize and self.max_evaluations == sys.maxsize and self.time_limit == np.inf:
            self.max_evaluations = 10
            print(self.algorithm_type, ": No stopping condition specified, max_evaluations set to", self.max_evaluations)
        if self.use_only_surrogate:
            self.send_snapshots_to_collector = False
        if self.algorithm_type == "MH":
            self.surrogate_model_updates = False
