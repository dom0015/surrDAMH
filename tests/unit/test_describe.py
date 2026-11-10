#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``Configuration.describe()`` / ``Stage.describe()`` (WS5).

The point of these methods is that a setting cannot be silently ignored without showing up
in the start-up log (the ``adaptive_target_rate`` of finding G1 was ignored for a long time
because nothing ever printed it). The first two tests therefore pin the *completeness*
property: every dataclass field must appear in the output, so adding a field without
thinking about it cannot make it invisible.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from surrDAMH.configuration import Configuration
from surrDAMH.configuration import \
    POSTERIOR_AFFECTING_FIELDS as CONFIGURATION_POSTERIOR_FIELDS
from surrDAMH.proposals import RandomWalk
from surrDAMH.stages import POSTERIOR_AFFECTING_FIELDS as STAGE_POSTERIOR_FIELDS
from surrDAMH.stages import Stage


def make_conf(**overrides) -> Configuration:
    kwargs = dict(no_parameters=2, no_observations=2, output_dir="out_describe",
                  use_collector=False, use_solvers_pool=False)
    kwargs.update(overrides)
    return Configuration(**kwargs)


@pytest.mark.unit
def test_configuration_describe_mentions_every_field():
    text = make_conf().describe()
    missing = [field.name for field in fields(Configuration) if field.name not in text]
    assert missing == [], f"Configuration.describe() does not mention: {missing}"


@pytest.mark.unit
def test_stage_describe_mentions_every_field():
    text = Stage(algorithm="DAMH", max_evaluations=10).describe(0)
    missing = [field.name for field in fields(Stage) if field.name not in text]
    assert missing == [], f"Stage.describe() does not mention: {missing}"


@pytest.mark.unit
def test_posterior_affecting_field_names_exist_on_the_dataclasses():
    """Guards against a typo in the two ``POSTERIOR_AFFECTING_FIELDS`` sets."""
    conf_names = {field.name for field in fields(Configuration)}
    stage_names = {field.name for field in fields(Stage)}
    assert CONFIGURATION_POSTERIOR_FIELDS <= conf_names
    assert STAGE_POSTERIOR_FIELDS <= stage_names


@pytest.mark.unit
def test_configuration_describe_marks_posterior_affecting_fields():
    text = make_conf().describe()
    assert "transform_before_surrogate*=" in text  # marked
    assert "debug=" in text and "debug*=" not in text  # not marked


@pytest.mark.unit
def test_configuration_describe_shows_effective_values_and_mpi_layout():
    text = make_conf(min_snapshots_initial=7).describe()
    assert "min_snapshots_initial*=7" in text
    assert "[MPI layout]" in text
    assert "no_samplers=" in text


@pytest.mark.unit
def test_configuration_describe_reports_requested_vs_effective_gradients():
    conf = make_conf(use_surrogate_gradients=False)
    text = conf.describe(use_surrogate_gradients_requested=True)
    assert "requested=True" in text and "in effect=False" in text
    # equal values need no extra line
    assert "requested=" not in conf.describe(use_surrogate_gradients_requested=False)


@pytest.mark.unit
def test_stage_describe_shows_post_init_corrections():
    """An MH stage has ``surrogate_model_updates`` forced off; describe() shows the effective value."""
    stage = Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), surrogate_model_updates=True, max_evaluations=5)
    text = stage.describe(1)
    assert "surrogate_model_updates*=False" in text
    assert "Stage 1 (alg0001_MH)" in text
    assert "[note] MH stage" in text


@pytest.mark.unit
def test_describe_block_stays_short_for_a_three_stage_run():
    """Acceptance criterion of WS5: the whole start-up block fits in ~40 lines."""
    conf = make_conf()
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=10),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=10, subchain_length=5),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=10,
                    surrogate_model_updates=False)]
    block = "\n".join([conf.describe()] + [stage.describe(i) for i, stage in enumerate(stages)])
    assert len(block.splitlines()) <= 40, block
