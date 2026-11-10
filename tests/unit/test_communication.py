#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the non-MPI helpers of ``surrDAMH.modules.communication``: the ``MPI_TAG_UB``
start-up diagnostic (finding 2.11), the service-loop idle throttle (findings 4.2/M11), and the
single-process/normalization half of the cross-rank configuration check (finding 2.9; its
broadcast needs several ranks and lives in ``tests/mpi/test_mpi_config_consistency.py``).

The message-passing classes of that module are exercised by ``tests/mpi/test_mpi_protocol.py``
(they need two ranks); these two helpers are pure Python.

``check_tag_upper_bound`` is a pure diagnostic: it warns and returns the message, or returns ``None``.
It must never raise and must never change control flow, so the tests cover the "cannot
determine anything" paths as well as the two warning conditions. ``tag_ub`` is passed
explicitly, so no MPI attribute has to be mocked.

Note on the negative cases: the function deliberately swallows every exception (it must not
be able to break a run), which includes a warning turned into an error by
``simplefilter("error")``. They therefore assert on *recorded* warnings, not on raising.
"""

from __future__ import annotations

import sys
import warnings

import pytest

import numpy as np

from surrDAMH.configuration import (POSTERIOR_AFFECTING_FIELDS,
                                    normalize_for_comparison)
from surrDAMH.modules.communication import (IDLE_SPIN_ITERATIONS,
                                            TAG_FIRST_SNAPSHOT,
                                            ServiceLoopThrottle,
                                            check_configuration_consistency,
                                            check_tag_upper_bound)
from surrDAMH.proposals import RandomWalk
from surrDAMH.stages import Stage

pytestmark = pytest.mark.unit


def _stage(**kwargs) -> Stage:
    return Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), **kwargs)


def _call_and_record(stages, tag_ub):
    """Call the check, returning ``(returned_message, [recorded RuntimeWarning messages])``."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        message = check_tag_upper_bound(stages, tag_ub=tag_ub)
    return message, [str(w.message) for w in recorded if issubclass(w.category, RuntimeWarning)]


class TestTagUpperBoundCheck:
    """``check_tag_upper_bound`` warns exactly when a run could plausibly exhaust MPI_TAG_UB."""

    def test_no_warning_when_projection_fits(self):
        message, recorded = _call_and_record([_stage(max_evaluations=1000),
                                              _stage(max_evaluations=2000)], tag_ub=32767)
        assert message is None
        assert recorded == []

    def test_warns_when_projected_tag_count_exceeds_the_limit(self):
        stages = [_stage(max_evaluations=20000), _stage(max_evaluations=20000)]
        with pytest.warns(RuntimeWarning, match=r"MPI_TAG_UB is 32767"):
            message = check_tag_upper_bound(stages, tag_ub=32767)
        assert message is not None
        # the projection is the summed bounds plus the snapshot tag offset
        assert str(40000 + TAG_FIRST_SNAPSHOT) in message

    def test_max_samples_bounds_a_stage_without_max_evaluations(self):
        """A stage bounded only by max_samples is still projectable (it bounds the evaluations)."""
        message, recorded = _call_and_record([_stage(max_samples=100)], tag_ub=32767)
        assert message is None
        assert recorded == []

    def test_warns_for_an_unbounded_stage_only_when_the_limit_is_small(self):
        """A time-limit-only stage cannot be projected; warn only at a genuinely small TAG_UB."""
        unbounded = _stage(time_limit=10.0)
        assert unbounded.max_evaluations == sys.maxsize  # no stopping-condition default kicked in
        with pytest.warns(RuntimeWarning, match=r"no max_evaluations/max_samples bound"):
            assert check_tag_upper_bound([unbounded], tag_ub=32767) is not None
        message, recorded = _call_and_record([unbounded], tag_ub=2**31 - 1)
        assert message is None
        assert recorded == []

    @pytest.mark.parametrize("stages, tag_ub", [
        ([_stage(max_evaluations=10)], None),   # None -> query the (large) real MPI_TAG_UB
        (["not a stage"], "not an int"),        # unusable input: swallowed, not raised
        ([], 10),                               # no stages: projection is the tag offset only
    ])
    def test_never_raises_and_stays_silent_on_unusable_input(self, stages, tag_ub):
        message, recorded = _call_and_record(stages, tag_ub=tag_ub)
        assert message is None
        assert recorded == []


class TestServiceLoopThrottle:
    """
    The solvers-pool/collector idle throttle (findings 4.2/M11).

    ``sleep_seconds=0.0`` throughout: what is asserted is *when* the throttle decides to sleep
    (``step`` returns that decision), not the sleeping itself.
    """

    def test_a_busy_loop_never_sleeps(self):
        throttle = ServiceLoopThrottle(spin_iterations=3, sleep_seconds=0.0)
        assert [throttle.step(True) for _ in range(50)] == [False] * 50

    def test_short_idle_periods_only_spin(self):
        """A sub-spin-budget idle gap (a fast forward model) must not add any sleep latency."""
        throttle = ServiceLoopThrottle(spin_iterations=3, sleep_seconds=0.0)
        assert [throttle.step(False) for _ in range(3)] == [False, False, False]
        assert throttle.step(True) is False  # work arrived within the budget

    def test_sleeps_once_idle_beyond_the_spin_budget(self):
        throttle = ServiceLoopThrottle(spin_iterations=3, sleep_seconds=0.0)
        assert [throttle.step(False) for _ in range(6)] == [False, False, False, True, True, True]

    def test_work_resets_the_spin_budget(self):
        throttle = ServiceLoopThrottle(spin_iterations=2, sleep_seconds=0.0)
        for _ in range(5):
            throttle.step(False)
        assert throttle.consecutive_idle == 5
        throttle.step(True)
        assert throttle.consecutive_idle == 0
        assert [throttle.step(False) for _ in range(3)] == [False, False, True]

    def test_defaults_spin_before_sleeping(self):
        throttle = ServiceLoopThrottle(sleep_seconds=0.0)
        assert throttle.spin_iterations == IDLE_SPIN_ITERATIONS
        assert not any(throttle.step(False) for _ in range(IDLE_SPIN_ITERATIONS))
        assert throttle.step(False) is True


class _DuckDistribution:
    """Stand-in for a user-supplied ``initial_samples_distribution``: no ``__eq__``, so two
    instances are never ``==`` even when they were built identically (as they are on two ranks
    that ran the same script)."""

    def __init__(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)


def _conf(**overrides):
    from surrDAMH.configuration import Configuration
    kwargs = dict(no_parameters=2, no_observations=2, output_dir="out_config_consistency",
                  use_collector=False, use_solvers_pool=False)
    kwargs.update(overrides)
    return Configuration(**kwargs)


class TestConfigurationConsistencyCheck:
    """Finding 2.9: the field snapshot and the value normalization behind
    ``check_configuration_consistency``. The broadcast itself needs several ranks and is covered
    by ``tests/mpi/test_mpi_config_consistency.py``."""

    def test_snapshot_covers_exactly_the_posterior_affecting_fields(self):
        assert set(_conf().requested_posterior_fields()) == set(POSTERIOR_AFFECTING_FIELDS)

    def test_snapshot_keeps_the_requested_value_after_a_local_mutation(self):
        """``core._configure_surrogate_gradients()`` may turn the flag off on one
        rank only; the snapshot (what the script asked for) must not follow."""
        conf = _conf(use_surrogate_gradients=True)
        conf.use_surrogate_gradients = False  # what _configure_surrogate_gradients() does
        assert conf.use_surrogate_gradients_requested is True
        assert conf.requested_posterior_fields_for_comparison()["use_surrogate_gradients"] is True

    def test_equal_configurations_compare_equal_field_by_field(self):
        """Two ranks running the same constructor call, including the awkward field types."""
        left = _conf(lhs_scale=np.array([1.0, 2.0]),
                     initial_samples_distribution=_DuckDistribution([0.3, -0.7]))
        right = _conf(lhs_scale=np.array([1.0, 2.0]),
                      initial_samples_distribution=_DuckDistribution([0.3, -0.7]))
        assert left.requested_posterior_fields_for_comparison() == \
            right.requested_posterior_fields_for_comparison()

    @pytest.mark.parametrize("overrides", [
        {"no_parameters": 3},
        {"lhs_scale": np.array([1.0, 3.0])},
        {"initial_sample_type": "lhs"},
        {"initial_samples_distribution": _DuckDistribution([0.3, 0.7])},
    ])
    def test_differing_configurations_compare_unequal(self, overrides):
        base = _conf(lhs_scale=np.array([1.0, 2.0]),
                     initial_samples_distribution=_DuckDistribution([0.3, -0.7]))
        other = _conf(**{**dict(lhs_scale=np.array([1.0, 2.0]),
                                initial_samples_distribution=_DuckDistribution([0.3, -0.7])),
                         **overrides})
        assert base.requested_posterior_fields_for_comparison() != \
            other.requested_posterior_fields_for_comparison()

    def test_normalized_values_are_plain_data(self):
        """No numpy arrays survive normalization, so the comparison can never raise
        'truth value of an array is ambiguous' and the dict stays cheap to broadcast."""
        normalized = _conf(lhs_scale=np.array([1.0, 2.0]),
                           initial_samples_distribution=_DuckDistribution([0.3, -0.7])
                           ).requested_posterior_fields_for_comparison()
        assert not any(isinstance(value, np.ndarray) for value in normalized.values())

    def test_normalize_survives_an_object_it_cannot_inspect(self):
        class _Opaque:
            __slots__ = ()
        assert normalize_for_comparison(_Opaque()).endswith("_Opaque")

    def test_single_rank_check_is_a_no_op(self):
        """Under plain ``python`` (and for the MPI-free ``run_local``) ``COMM_WORLD`` has size 1:
        there is nothing to compare against and no collective call must be made."""
        assert check_configuration_consistency(_conf()) is None
