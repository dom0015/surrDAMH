#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``surrDAMH.modules.surrogate_restart.SurrogateRestart`` (WS5).

In-process round trip: an updater is trained, ``save_state``-ed into a state directory, and
a *fresh* updater is then restored from it through ``SurrogateRestart``. The minibatch MLP
updater is used because it is the only one implementing state persistence today (see
``library_notes/12_evaluator_contract_spec.md`` §1).
"""

from __future__ import annotations

import numpy as np
import pytest

from surrDAMH.modules.surrogate_restart import SurrogateRestart
from surrDAMH.surrogates.torch_perceptron_minibatches import \
    NeuralNetworkUpdater

NO_PARAMETERS = 2
NO_OBSERVATIONS = 2
NO_SNAPSHOTS = 6


def make_updater() -> NeuralNetworkUpdater:
    return NeuralNetworkUpdater(
        no_parameters=NO_PARAMETERS, no_observations=NO_OBSERVATIONS,
        hidden_layer_sizes=(4, 4), solver="adamw", iterations_batch=2, device="cpu",
        verbose=False, output_normalization="identity")


def make_snapshots() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    parameters = rng.normal(size=(NO_SNAPSHOTS, NO_PARAMETERS)).astype(np.float32)
    observations = (parameters @ np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float32)).astype(np.float32)
    multiplicity = np.ones((NO_SNAPSHOTS, 1), dtype=np.float32)
    return parameters, observations, multiplicity


def saved_state_dir(tmp_path) -> str:
    """Train an updater on ``make_snapshots()`` and save its state into ``tmp_path/state``."""
    state_dir = str(tmp_path / "state")
    updater = make_updater()
    updater.add_data(*make_snapshots())
    updater.train()
    SurrogateRestart(state_dir).save(updater)
    return state_dir


@pytest.mark.unit
def test_save_then_restore_state_round_trips_snapshots_and_readiness(tmp_path):
    """``mode="state"``: snapshots, ``training_data_loaded`` and ``pretrained_ready`` all come back."""
    state_dir = saved_state_dir(tmp_path)
    parameters, observations, _ = make_snapshots()

    restored = make_updater()
    snapshots = SurrogateRestart(state_dir, mode="state").apply(restored)

    assert snapshots is not None
    assert len(snapshots) == 3
    assert snapshots[0].shape == (NO_SNAPSHOTS, NO_PARAMETERS)
    np.testing.assert_allclose(snapshots[0], parameters, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(snapshots[1], observations, rtol=1e-5, atol=1e-5)
    assert restored.no_snapshots == NO_SNAPSHOTS
    assert restored.training_data_loaded is True
    # pretrained_ready is what makes run_COLLECTOR hand out an evaluator before the first
    # snapshot arrives, i.e. what allows a DAMH first stage (see process_COLLECTOR)
    assert restored.pretrained_ready is True
    # the updater reports the snapshots itself, so Problem.run_sampling prefers those
    reported = restored.get_initial_snapshots()
    assert reported is not None
    np.testing.assert_allclose(reported[0], snapshots[0])


@pytest.mark.unit
def test_restored_state_reproduces_the_saved_evaluator(tmp_path):
    """The restored network predicts exactly what the saved one did (weights, not just data)."""
    state_dir = str(tmp_path / "state")
    updater = make_updater()
    updater.add_data(*make_snapshots())
    updater.train()
    SurrogateRestart(state_dir).save(updater)
    probe = np.array([[0.3, -0.2], [1.0, 0.4]])
    expected = updater.get_evaluator()(probe)

    restored = make_updater()
    SurrogateRestart(state_dir, mode="state").apply(restored)

    np.testing.assert_allclose(restored.get_evaluator()(probe), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_data_mode_restores_snapshots_but_not_the_network(tmp_path):
    """``mode="data"``: the snapshots are back and retrained on, but nothing is pretrained."""
    state_dir = saved_state_dir(tmp_path)

    restored = make_updater()
    snapshots = SurrogateRestart(state_dir, mode="data").apply(restored)

    assert snapshots is not None
    assert snapshots[0].shape[0] == NO_SNAPSHOTS
    assert restored.no_snapshots == NO_SNAPSHOTS
    assert restored.training_data_loaded is True
    assert restored.pretrained_ready is False


@pytest.mark.unit
def test_none_mode_does_nothing(tmp_path):
    state_dir = saved_state_dir(tmp_path)
    restored = make_updater()
    assert SurrogateRestart(state_dir, mode="none").apply(restored) is None
    assert restored.no_snapshots == 0


@pytest.mark.unit
def test_missing_files_start_cold_instead_of_raising(tmp_path, capsys):
    restored = make_updater()
    assert SurrogateRestart(str(tmp_path / "nothing_here")).apply(restored) is None
    assert restored.no_snapshots == 0
    assert "Starting cold" in capsys.readouterr().out


@pytest.mark.unit
def test_save_without_snapshots_writes_nothing(tmp_path):
    state_dir = str(tmp_path / "state")
    restart = SurrogateRestart(state_dir)
    assert restart.save(make_updater()) is False
    assert not (tmp_path / "state").exists()


@pytest.mark.unit
def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        SurrogateRestart("somewhere", mode="everything")  # type: ignore[arg-type]
