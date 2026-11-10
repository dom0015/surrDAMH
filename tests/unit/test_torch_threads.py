#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ``Configuration.torch_threads`` and ``surrDAMH.modules.torch_threads``
(author decision 2026-09-18, ``library_notes/10_manual_review_notes.md`` §7b "Torch threads").

Uses ``Problem.run_sampling_local`` (MPI-free, single chain) with a tiny ``NeuralNetworkUpdater`` so
the actual torch code path (updater + evaluator construction, training, evaluation) is
exercised, not just the helper function in isolation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import surrDAMH
import surrDAMH.modules.torch_threads as torch_threads_module
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.torch_perceptron_minibatches import NeuralNetworkUpdater

pytestmark = pytest.mark.unit


class ToySolver(Solver):
    """Same model as tests/test_runner_local.py::ToySolver (2 parameters, 1 observation)."""

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters):
        self.x = parameters[0]
        self.y = parameters[1]

    def get_observations(self):
        return np.array([(self.x ** 2 - self.y) * (np.log((self.x - self.y) ** 2 + 1))])


class FixedSample:
    """Degenerate 'distribution' returning a fixed initial sample."""

    def __init__(self, parameters):
        self.parameters = np.array(parameters, dtype=float)

    def rvs(self):
        return self.parameters.copy()


def _make_nn_problem(output_dir, torch_threads):
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(output_dir),
                                  use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedSample([0.3, -0.7]),
                                  torch_threads=torch_threads)
    prior = surrDAMH.distributions.PriorIndependentComponents([
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    ])
    likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=1.0)
    # ~30 evaluations total (2 stages x 15), small net, few optimizer steps: fast (WS4 test brief).
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=15),
             Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=15)]
    updater = NeuralNetworkUpdater(
        no_parameters=2, no_observations=1, hidden_layer_sizes=(4,), solver="adamw",
        iterations_batch=3, seed=0, verbose=False, output_normalization="identity")
    return conf, prior, likelihood, stages, updater


@pytest.fixture
def restore_torch_num_threads():
    """``torch.set_num_threads`` is process-global state; restore it after the test."""
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


@pytest.mark.parametrize("torch_threads", [1, 2, 3, 100])
def test_default_and_positive_values_construct(tmp_path, torch_threads):
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(tmp_path),
                                  use_collector=False, use_solvers_pool=False,
                                  torch_threads=torch_threads)
    assert conf.torch_threads == torch_threads


def test_default_is_one(tmp_path):
    """Author decision 2026-09-18: the default changed from unset/None to a deterministic 1."""
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(tmp_path),
                                  use_collector=False, use_solvers_pool=False)
    assert conf.torch_threads == 1


def test_none_is_accepted(tmp_path):
    """None = leave torch's default alone (opt-out)."""
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(tmp_path),
                                  use_collector=False, use_solvers_pool=False, torch_threads=None)
    assert conf.torch_threads is None


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_non_positive_raises(tmp_path, bad_value):
    with pytest.raises(ValueError, match="torch_threads"):
        surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(tmp_path),
                               use_collector=False, use_solvers_pool=False,
                               torch_threads=bad_value)


@pytest.mark.parametrize("torch_threads", [1, 2])
def test_run_local_with_nn_updater_sets_effective_torch_num_threads(
        tmp_path, torch_threads, restore_torch_num_threads):
    """
    End-to-end: after ``run_sampling_local`` returns, ``torch.get_num_threads()`` equals
    ``conf.torch_threads`` on this (the only) rank, and the manifest records the same value
    as ``environment.torch_num_threads`` (measured post-hoc from ``sys.modules``) alongside
    ``configuration.torch_threads`` (the requested value, via ``dataclasses.fields``).
    """
    # start from a different value so a no-op would be caught:
    torch.set_num_threads(7 if torch_threads != 7 else 6)

    conf, prior, likelihood, stages, updater = _make_nn_problem(tmp_path, torch_threads)
    surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(
        conf, stages, surrogate_updater=updater)

    assert torch.get_num_threads() == torch_threads

    manifest_path = tmp_path / "sampling_output" / "run_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["configuration"]["torch_threads"] == torch_threads
    assert manifest["environment"]["torch_num_threads"] == torch_threads


def test_torch_threads_none_leaves_current_count_unchanged(
        tmp_path, restore_torch_num_threads):
    """``torch_threads=None`` is a no-op: whatever the count was before ``run_sampling_local``, it is the
    same after (this test's own tiny NN construction may still transiently touch torch, but the
    thread count itself is never set by this library when ``torch_threads=None``).

    ``modules.torch_threads._desired_torch_threads`` is reset to ``None`` first: it is
    process-wide state, and another test in this same pytest process may have left a concrete
    value there (a fresh ``mpiexec``/``python`` process -- the launch modes this library
    actually supports -- starts with it already ``None``, see that module's docstring "Known
    limitation" paragraph for what a stale value would otherwise do to a freshly constructed
    updater here).
    """
    torch.set_num_threads(5)
    torch_threads_module._desired_torch_threads = None

    conf, prior, likelihood, stages, updater = _make_nn_problem(tmp_path, None)
    surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(
        conf, stages, surrogate_updater=updater)

    assert torch.get_num_threads() == 5

    manifest_path = tmp_path / "sampling_output" / "run_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["configuration"]["torch_threads"] is None
    assert manifest["environment"]["torch_num_threads"] == 5
