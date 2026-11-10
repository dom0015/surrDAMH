#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note 32, D-32-3 (2026-10-13): ``subchain_stats`` of a DAMH stage whose proposal is a ``Mixture``
gains a LAST column ``component`` = the index of the component drawn for that outer iteration
(``-1`` on an S2 exact-step row). Every other stage's file -- plain proposals and ``Block`` -- is
unchanged (the ``Block`` case is pinned by ``test_fixed_block_runs_are_byte_identical_to_before_c2``).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from surrDAMH.proposals import Block, Mixture, RandomWalk
from surrDAMH.stages import Stage, stage_name

from .test_mixture import _run_local

pytestmark = [pytest.mark.unit]

#: the ``subchain_stats`` header of every non-mixture DAMH stage (S2 ``kernel`` last), unchanged
PLAIN_HEADER = ["iteration", "subchain_max_length", "subchain_accepted", "subchain_acceptance_rate",
                "correction_log_ratio", "outer_proposed_changed", "outer_accepted", "rank_world", "kernel"]

_MIXTURE = Mixture([RandomWalk(scale=0.3), RandomWalk(scale=0.1)], probabilities=[0.7, 0.3])


def _subchain_stats(sampling_output, stages, index):
    path = os.path.join(sampling_output, "subchain_stats", stage_name(stages[index], index), "rank0000.csv")
    with open(path) as f:
        header = f.readline().strip().split(",")
    return header, pd.read_csv(path)


def test_mixture_stage_has_the_component_column_and_plain_stages_do_not(tmp_path):
    stages = [Stage(algorithm="DAMH", subchain_length=2, max_evaluations=300, proposal=RandomWalk(scale=0.3)),
              Stage(algorithm="DAMH", subchain_length=2, max_evaluations=2_200, proposal=_MIXTURE),
              Stage(algorithm="DAMH", subchain_length=2, max_evaluations=200,
                    proposal=Block(groups=[[0], [1, 2]], proposals=[RandomWalk(scale=0.3), RandomWalk(scale=0.3)]))]
    sampling_output = _run_local(tmp_path, stages)
    header, _ = _subchain_stats(sampling_output, stages, 0)
    assert header == PLAIN_HEADER
    header, _ = _subchain_stats(sampling_output, stages, 2)
    assert header == PLAIN_HEADER  # Block: unchanged
    header, frame = _subchain_stats(sampling_output, stages, 1)
    assert header == PLAIN_HEADER + ["component"]
    assert len(frame) >= 2_000
    assert set(frame["component"].unique()) <= {0, 1}
    share = frame["component"].value_counts(normalize=True)
    assert abs(share[0] - 0.7) < 0.05 and abs(share[1] - 0.3) < 0.05, share.to_dict()
    assert (frame["kernel"] == 0).all()
    # the smaller step is accepted more often in the sub-chain: the column attributes the iterations
    rate = frame.groupby("component")["subchain_acceptance_rate"].mean()
    assert rate[1] > rate[0], rate.to_dict()


def test_exact_step_rows_carry_minus_one(tmp_path):
    stages = [Stage(algorithm="DAMH", subchain_length=2, max_evaluations=600, proposal=_MIXTURE,
                    exact_step_probability=0.2)]
    sampling_output = _run_local(tmp_path, stages)
    header, frame = _subchain_stats(sampling_output, stages, 0)
    assert header == PLAIN_HEADER + ["component"]
    exact = frame["kernel"] == 1
    assert exact.sum() > 0 and (~exact).sum() > 0
    assert (frame.loc[exact, "component"] == -1).all()
    assert set(frame.loc[~exact, "component"].unique()) <= {0, 1}
    assert np.isnan(frame.loc[exact, "subchain_acceptance_rate"]).all()  # the S2 row otherwise unchanged
