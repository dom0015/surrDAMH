"""
Input clipping of the neural-network surrogate (2026-10-23, paper §6.2): the network sees
``clip_b(x)`` (componentwise clamp to [-b, b]) before its first layer. Pinned here: default
None is the unclipped network; evaluator / jacobian / vjp clip; training and the published
evaluator clip; checkpoint round trip (old checkpoints without the key load as None, mismatch
raises); averageable-state evaluators clip; invalid values raise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surrDAMH.surrogates import NeuralNetworkUpdater
from surrDAMH.surrogates.torch_perceptron_minibatches import PyTorchMLP

pytestmark = pytest.mark.unit

B = 1.5


def _updater(clip, seed=0, **kw):
    return NeuralNetworkUpdater(no_parameters=3, no_observations=2, hidden_layer_sizes=(8, 8), seed=seed,
                                iterations_batch=5, learning_rate=1e-2, input_clip=clip, **kw)


def _trained(clip, seed=0):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 3))
    u = _updater(clip, seed)
    u.add_data(x, np.stack([x.sum(1), (x ** 2).sum(1)], axis=1), train_on_added_data=False)
    u.train()
    return u


def test_default_none_identical_to_huge_clip():
    x = np.random.default_rng(0).normal(size=(10, 3)) * 3
    ev_none = _updater(None).get_evaluator()
    ev_huge = _updater(1e6).get_evaluator()
    assert PyTorchMLP(3, 2, (4,), "silu", 0).input_clip is None
    assert _updater(None).model.input_clip is None
    np.testing.assert_array_equal(ev_none(x), ev_huge(x))


def test_evaluator_clips_and_gradients():
    ev = _trained(B).get_evaluator()
    x = np.array([[3.0, -4.0, 0.5], [0.2, 0.3, -0.4], [-2.0, 1.0, 9.0]])
    np.testing.assert_array_equal(ev(x), ev(np.clip(x, -B, B)))
    inside = np.array([[0.2, 0.3, -0.4]])
    np.testing.assert_array_equal(ev(inside), ev(np.clip(inside, -B, B)))

    p = np.array([3.0, 0.3, -0.4])  # coordinate 0 clipped
    v = np.array([1.0, -0.5])
    g, _ = ev.vjp(p, v)
    g = np.asarray(g).reshape(-1)
    assert g[0] == 0.0 and np.any(g[1:] != 0.0)
    q = np.array([0.1, 0.3, -0.4])
    g_in, _ = ev.vjp(q, v)
    J, _ = ev.jacobian(p)
    J = np.asarray(J).reshape(2, 3)
    assert np.all(J[:, 0] == 0.0) and np.any(J[:, 1:] != 0.0)
    # inside the box the clipped network's derivative equals the plain network's with the same weights
    ev_same = _trained(B).get_evaluator()
    ev_same.model.input_clip = None
    J_in, _ = ev.jacobian(q)
    J_plain, _ = ev_same.jacobian(q)
    np.testing.assert_array_equal(np.asarray(J_in), np.asarray(J_plain))
    g_plain, _ = ev_same.vjp(q, v)
    np.testing.assert_array_equal(np.asarray(g_in).reshape(-1), np.asarray(g_plain).reshape(-1))


def test_training_runs_and_published_evaluator_clips():
    u = _trained(B)
    assert u.model.input_clip == B
    ev = u.get_evaluator()
    assert ev.model.input_clip == B
    x = np.array([[5.0, 5.0, 5.0]])
    np.testing.assert_array_equal(ev(x), ev(np.full((1, 3), B)))
    assert np.all(np.isfinite(ev(x)))


def test_checkpoint_round_trip_old_key_and_mismatch(tmp_path):
    u = _trained(2.0)
    path = str(tmp_path / "ck.pt")
    u.save_checkpoint(path)
    assert torch.load(path, weights_only=True)["updater_hparams"]["input_clip"] == 2.0
    _updater(2.0).load_checkpoint(path)
    with pytest.raises(ValueError, match="input_clip"):
        _updater(3.0).load_checkpoint(path)
    with pytest.raises(ValueError, match="input_clip"):
        _updater(None).load_checkpoint(path)
    # a checkpoint from before the option: no key == None
    u0 = _trained(None)
    p0 = str(tmp_path / "old.pt")
    u0.save_checkpoint(p0)
    ck = torch.load(p0, weights_only=True)
    del ck["updater_hparams"]["input_clip"]
    torch.save(ck, p0)
    fresh = _updater(None)
    fresh.load_checkpoint(p0)
    assert fresh.model.input_clip is None
    with pytest.raises(ValueError, match="input_clip"):
        _updater(2.0).load_checkpoint(p0)


def test_with_averageable_state_still_clips():
    ev = _trained(B).get_evaluator()
    avg = ev.with_averageable_state(ev.averageable_state())
    assert avg.model.input_clip == B
    x = np.array([[4.0, -4.0, 4.0]])
    np.testing.assert_array_equal(avg(x), avg(np.full((1, 3), 1.0) * np.array([B, -B, B])))
    np.testing.assert_array_equal(avg(x), ev(x))


@pytest.mark.parametrize("bad", [0, -1.0, float("inf"), float("nan"), "3"])
def test_invalid_values_raise(bad):
    with pytest.raises(ValueError):
        PyTorchMLP(3, 2, (4,), "silu", 0, input_clip=bad)
    with pytest.raises(ValueError):
        _updater(bad)
