"""
Finding 4.1/M10: the collector posts a fixed-size ``irecv`` buffer for the pickled Evaluator.
``process_COLLECTOR._check_evaluator_pickle_size`` fails loudly, right when a fresh evaluator
is built, instead of letting an oversized evaluator truncate an MPI message somewhere else.
"""
import warnings

import numpy as np
import pytest

from surrDAMH.process_COLLECTOR import _check_evaluator_pickle_size


class _TinyEvaluator:
    def __init__(self, payload_size: int) -> None:
        self.payload = np.zeros(payload_size, dtype=np.uint8)


@pytest.mark.unit
def test_evaluator_within_buffer_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_evaluator_pickle_size(_TinyEvaluator(10), max_buffer_size=1 << 20)


@pytest.mark.unit
def test_evaluator_over_half_buffer_warns():
    with pytest.warns(RuntimeWarning, match="over half"):
        _check_evaluator_pickle_size(_TinyEvaluator(1000), max_buffer_size=1500)


@pytest.mark.unit
def test_evaluator_exceeding_buffer_raises():
    with pytest.raises(RuntimeError, match="exceeds Configuration.max_buffer_size"):
        _check_evaluator_pickle_size(_TinyEvaluator(1000), max_buffer_size=100)


# ---------------------------------------------------------------------------
# 2026-09-20: the collector never publishes a surrogate that predicts non-finite
# values on the newest snapshot batch (item C).
# ---------------------------------------------------------------------------

from surrDAMH.process_COLLECTOR import _evaluator_is_finite_on  # noqa: E402


class _ConstantEvaluator:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def __call__(self, datapoints):
        self.calls += 1
        return np.full((np.asarray(datapoints).shape[0], 2), self.value)


@pytest.mark.unit
def test_finite_evaluator_passes_the_probe():
    parameters = np.zeros((3, 2))
    assert _evaluator_is_finite_on(_ConstantEvaluator(1.5), parameters) is True


@pytest.mark.unit
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_evaluator_fails_the_probe(value):
    parameters = np.zeros((3, 2))
    assert _evaluator_is_finite_on(_ConstantEvaluator(value), parameters) is False


@pytest.mark.unit
def test_probe_without_snapshots_cannot_judge_and_accepts():
    """Nothing to probe on: the check must not be turned into a veto (and must not call the
    evaluator with an empty batch either)."""
    evaluator = _ConstantEvaluator(np.nan)
    assert _evaluator_is_finite_on(evaluator, None) is True
    assert _evaluator_is_finite_on(evaluator, np.empty((0, 2))) is True
    assert evaluator.calls == 0


@pytest.mark.unit
def test_probe_ignores_non_finite_probe_points():
    """A healthy surrogate must not be vetoed just because a probe point is not finite."""
    evaluator = _ConstantEvaluator(2.0)
    parameters = np.array([[0.0, 0.0], [np.inf, 1.0]])
    assert _evaluator_is_finite_on(evaluator, parameters) is True
    assert _evaluator_is_finite_on(evaluator, np.array([[np.nan, 1.0]])) is True
    assert evaluator.calls == 1  # only the batch that had a finite row was evaluated
