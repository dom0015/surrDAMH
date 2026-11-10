#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ``surrDAMH.modules.continuation.load_last_samples``.

WS4 (``library_notes/09_improvement_plan.md``): "``load_last_samples`` checks parameter count
and warns on chain-count mismatch." The parameter-count check already raised ``ValueError``
per file; these tests pin that behaviour and the newly added chain-count-mismatch warning
(raised only when there are MORE saved chains than requested -- fewer already raises, unchanged).

Fixture files are written with ``save_last_sample`` itself, so the test exercises the exact
on-disk layout it produces (``<output_dir>/sampling_output/last_sample/<stage>/rank%04d.npz``).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from surrDAMH.modules.continuation import load_last_samples, save_last_sample

pytestmark = pytest.mark.unit


def _conf(output_dir, no_parameters):
    return SimpleNamespace(output_dir=str(output_dir), no_parameters=no_parameters)


def test_load_last_samples_round_trip(tmp_path):
    conf = _conf(tmp_path, no_parameters=2)
    save_last_sample(conf, "alg0000_MH", 0, np.array([0.1, 0.2]))
    save_last_sample(conf, "alg0000_MH", 1, np.array([0.3, 0.4]))

    loaded = load_last_samples(str(tmp_path), no_parameters=2, no_chains=2)
    np.testing.assert_array_equal(loaded, np.array([[0.1, 0.2], [0.3, 0.4]]))


def test_load_last_samples_wrong_parameter_count_raises(tmp_path):
    conf = _conf(tmp_path, no_parameters=3)
    save_last_sample(conf, "alg0000_MH", 0, np.array([0.1, 0.2, 0.3]))

    with pytest.raises(ValueError, match="no_parameters"):
        load_last_samples(str(tmp_path), no_parameters=2, no_chains=1)


def test_load_last_samples_too_few_chains_raises(tmp_path):
    conf = _conf(tmp_path, no_parameters=2)
    save_last_sample(conf, "alg0000_MH", 0, np.array([0.1, 0.2]))

    with pytest.raises(ValueError, match="saved chains"):
        load_last_samples(str(tmp_path), no_parameters=2, no_chains=2)


def test_load_last_samples_more_chains_than_requested_warns_and_truncates(tmp_path):
    """More saved chains than requested: today's behaviour is to silently use the first
    ``no_chains`` files (sorted by filename, i.e. lowest ranks); WS4 asks for a warning that
    documents this, without changing it."""
    conf = _conf(tmp_path, no_parameters=2)
    save_last_sample(conf, "alg0000_MH", 0, np.array([0.1, 0.2]))
    save_last_sample(conf, "alg0000_MH", 1, np.array([0.3, 0.4]))
    save_last_sample(conf, "alg0000_MH", 2, np.array([0.5, 0.6]))

    with pytest.warns(RuntimeWarning, match="saved chains"):
        loaded = load_last_samples(str(tmp_path), no_parameters=2, no_chains=2)
    # unchanged behaviour: first no_chains files (sorted by filename -> rank0000, rank0001)
    np.testing.assert_array_equal(loaded, np.array([[0.1, 0.2], [0.3, 0.4]]))
