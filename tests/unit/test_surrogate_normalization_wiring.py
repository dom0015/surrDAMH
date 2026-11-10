#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WS6: both runners must feed likelihood-derived output-normalization statistics into a
surrogate updater configured with ``output_normalization="likelihood"`` (the new default
of ``NeuralNetworkUpdater``).

* ``Problem.run_sampling`` does it on every rank that holds an updater,
* ``Problem.run_sampling_local`` (``runner_local.run_local``) does it before the first snapshot is added.

Both are exercised in process with an MPI world of size 1, so no ``mpiexec`` is needed
(marker ``unit``). The manifest's ``surrogate`` block is checked at the same time, because
it is what records which statistics a run actually used.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import surrDAMH
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.torch_perceptron_minibatches import NeuralNetworkUpdater

pytestmark = pytest.mark.unit

OBSERVED_DATA = np.array([2.0, -3.0])
NOISE_SD = np.array([0.25, 4.0])


class LinearToySolver(Solver):
    """2 parameters -> 2 observations, deterministic."""

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = 2
        self.no_observations = 2

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        x, y = self.parameters
        return np.array([x + 0.5 * y, y - 0.25 * x])


class FixedSample:
    def __init__(self, parameters):
        self.parameters = np.array(parameters, dtype=float)

    def rvs(self):
        return self.parameters.copy()


def _make_updater():
    return NeuralNetworkUpdater(
        no_parameters=2, no_observations=2, hidden_layer_sizes=(4,),
        iterations_batch=2, seed=0, verbose=False,
    )


def _make_problem(output_dir, **conf_kwargs):
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=2, output_dir=str(output_dir),
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedSample([0.3, -0.7]),
                                  **conf_kwargs)
    prior = surrDAMH.distributions.PriorIndependentComponents([
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    ])
    likelihood = surrDAMH.distributions.Normal(mean=OBSERVED_DATA, sd=NOISE_SD)
    return conf, prior, likelihood


def _assert_normalized_from_likelihood(updater):
    np.testing.assert_allclose(updater.output_mean, OBSERVED_DATA, atol=1e-6)
    np.testing.assert_allclose(updater.output_scale, NOISE_SD, atol=1e-6)
    assert updater.output_normalization_provenance == "likelihood"


def test_sampling_framework_applies_likelihood_normalization(tmp_path):
    conf, prior, likelihood = _make_problem(tmp_path, use_collector=False, use_solvers_pool=False)
    updater = _make_updater()
    assert updater.output_normalization_provenance == "identity"  # not resolved yet

    surrDAMH.Problem(prior, likelihood, solver=LinearToySolver()).run_sampling(
        conf, [Stage(algorithm="MH", max_evaluations=5)], surrogate_updater=updater)

    _assert_normalized_from_likelihood(updater)


def test_run_local_applies_likelihood_normalization_and_records_it(tmp_path):
    conf, prior, likelihood = _make_problem(tmp_path, use_collector=False, use_solvers_pool=False,
                                            min_snapshots_initial=5, min_snapshots_to_update=5)
    updater = _make_updater()
    stages = [Stage(algorithm="MH", max_evaluations=20, proposal=RandomWalk(scale=0.5))]

    surrDAMH.Problem(prior, likelihood, solver=LinearToySolver()).run_sampling_local(
        conf, stages, surrogate_updater=updater)

    _assert_normalized_from_likelihood(updater)

    manifest_path = os.path.join(str(tmp_path), "sampling_output", "run_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    surrogate_block = manifest["surrogate"]
    assert surrogate_block["weighting"] == "uniform"
    assert surrogate_block["supports_sample_weights"] is True
    assert surrogate_block["output_normalization"] == "likelihood"
    assert surrogate_block["output_normalization_provenance"] == "likelihood"


def test_run_local_warns_and_keeps_identity_for_an_unusable_likelihood(tmp_path):
    class LikelihoodWithoutScale:
        def __init__(self, mean):
            self.mean = np.asarray(mean, dtype=float)

        def logpdf(self, sample):
            return 0.0

        def grad_logpdf(self, sample):
            return np.zeros_like(self.mean)

    conf, prior, _ = _make_problem(tmp_path, use_collector=False, use_solvers_pool=False,
                                   min_snapshots_initial=5, min_snapshots_to_update=5)
    updater = _make_updater()
    stages = [Stage(algorithm="MH", max_evaluations=10, proposal=RandomWalk(scale=0.5))]

    with pytest.warns(RuntimeWarning, match="LikelihoodWithoutScale"):
        surrDAMH.Problem(prior, LikelihoodWithoutScale(OBSERVED_DATA), solver=LinearToySolver()).run_sampling_local(
            conf, stages, surrogate_updater=updater)

    np.testing.assert_array_equal(updater.output_mean, np.zeros(2))
    np.testing.assert_array_equal(updater.output_scale, np.ones(2))
    assert updater.output_normalization_provenance == "identity"
