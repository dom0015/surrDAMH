#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note 32 §1 (2026-10-13): the flattened-likelihood Hamiltonian, ``Hamiltonian(likelihood_weight=tau)``.

* off = byte-identical: ``likelihood_weight=1.0`` given explicitly equals the field left at its
  default (both integrators; the pinned C2/C3 fingerprints are in ``test_mixture.py`` and
  ``test_tempered_hamiltonian.py``);
* map properties at ``tau = 0.3`` on the 2-D quadratic potential of the C3 tests (both
  integrators): involution under the momentum flip, unit Jacobian;
* invariance: an MH stage with the flattened Hamiltonian alone samples the closed-form posterior;
* ``tau = 0`` with the dimension-robust integrator, ``T = pi/2``, ``mass = 1``: the proposal is an
  independent prior draw;
* spec validation; the flattened component inside a ``Mixture`` (carried step, manifest
  resolution, ``describe()``).
"""

from __future__ import annotations

import contextlib
import io
import json
import os

import numpy as np
import pytest

import surrDAMH
from helpers_statistics import assert_matches_closed_form, chain_statistics, read_weighted_samples
from surrDAMH.modules.proposals import (BlockProposal, Hamiltonian as HamiltonianRuntime,
                                        HamiltonianInfinite, HamiltonianInfinite_adaptive)
from surrDAMH.proposals import Hamiltonian, Mixture
from surrDAMH.stages import Stage

from .test_mixture import _build, _run_local
from .test_tempered_hamiltonian import (_A_LIKELIHOOD, _MASS_SD, _ExactLinearGradientEvaluator,
                                        _fingerprint_with_notes, _map, _off_stages)

pytestmark = [pytest.mark.unit,
              pytest.mark.filterwarnings("ignore:.*is written to adaptive_stats:RuntimeWarning")]


# ----------------------------------------------------------------------- off = byte-identical
def _off_stages_explicit(integrator):
    """``_off_stages`` with ``likelihood_weight=1.0`` given; adaptive stages accept it (tau = 1)."""
    return _off_stages(integrator, likelihood_weight=1.0)


@pytest.mark.parametrize("integrator", ["leapfrog", "dimension_robust"])
def test_explicit_unit_weight_equals_the_default(tmp_path, integrator):
    default = _fingerprint_with_notes(_run_local(tmp_path / "a", _off_stages(integrator)))
    explicit = _fingerprint_with_notes(_run_local(tmp_path / "b", _off_stages_explicit(integrator)))
    assert explicit == default
    assert not any("likelihood_weight" in key for entry in explicit["proposal_resolution"] for key in entry)


# ------------------------------------------------------------------------------ map properties
def _flattened(proposal_cls, weight=0.3, num_steps=6, step_size=0.2):
    proposal = proposal_cls(no_parameters=2, step_size=step_size, num_steps=num_steps, sd_or_cov=_MASS_SD, seed=0,
                            likelihood_weight=weight)
    # convention of tests/unit/test_proposals.py: the callables are the gradients of -log density
    proposal.set_gradient_functions(lambda q: _A_LIKELIHOOD @ q, lambda q: np.asarray(q, dtype=float).copy())
    return proposal


@pytest.mark.parametrize("proposal_cls", [HamiltonianRuntime, HamiltonianInfinite])
def test_flattened_map_with_flip_is_an_involution(proposal_cls):
    proposal = _flattened(proposal_cls)
    assert proposal.flattening_on
    q0, p0 = np.array([0.7, -1.1]), np.array([0.4, 0.9])
    q1, p1_end = proposal._leapfrog(q0, p0)
    assert not np.allclose(q1, q0)
    q2, p2_end = proposal._leapfrog(q1, p1_end)
    np.testing.assert_allclose(q2, q0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(p2_end, p0, atol=1e-10, rtol=0)
    # the weight really changes the map
    plain = _flattened(proposal_cls, weight=1.0)
    assert not plain.flattening_on
    assert not np.allclose(plain._leapfrog(q0, p0)[0], q1)


@pytest.mark.parametrize("proposal_cls", [HamiltonianRuntime, HamiltonianInfinite])
def test_flattened_map_has_unit_jacobian(proposal_cls):
    proposal = _flattened(proposal_cls)
    z0 = np.array([0.3, -0.5, 1.2, -0.4])
    h = 1e-6
    jacobian = np.empty((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        jacobian[:, j] = (_map(proposal, z0 + e) - _map(proposal, z0 - e)) / (2 * h)
    assert abs(np.linalg.det(jacobian) - 1.0) < 1e-5, np.linalg.det(jacobian)


def test_likelihood_force_is_the_gradient_itself_at_unit_weight():
    gradient = np.array([1.0, 2.0])
    proposal = _flattened(HamiltonianRuntime, weight=1.0)
    proposal.set_gradient_functions(lambda q: gradient, lambda q: q)
    assert proposal._likelihood_force(np.zeros(2)) is gradient  # no multiplication at all
    proposal.likelihood_weight = 0.25
    np.testing.assert_array_equal(proposal._likelihood_force(np.zeros(2)), 0.25 * gradient)


# ------------------------------------------------------------------------------- invariance
@pytest.mark.parametrize("integrator", ["leapfrog", "dimension_robust"])
def test_flattened_hamiltonian_alone_samples_the_closed_form_posterior(tmp_path, linear_gaussian_problem,
                                                                      integrator):
    problem = linear_gaussian_problem
    spec = Hamiltonian(integrator=integrator, step_size=0.1, integration_time=1.0, integration_time_jitter=0.3,
                       likelihood_weight=0.3, adaptive=False)
    stage = Stage(algorithm="MH", proposal=spec, max_evaluations=8_000)
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="prior", use_surrogate_gradients=True)
    with contextlib.redirect_stdout(io.StringIO()):
        result = surrDAMH.Problem(problem.prior, problem.likelihood, solver=problem.solver).run_sampling_local(
            conf, [stage], surrogate_evaluator=_ExactLinearGradientEvaluator(problem.A))
    stage_result = result.stage_results[0]
    accepted, rejected = stage_result.counter_accepted, stage_result.counter_rejected
    assert accepted / (accepted + rejected) > 0.05, (accepted, rejected)
    samples = read_weighted_samples(tmp_path, stage_result.name, problem.no_parameters)
    statistics = chain_statistics(samples, accepted, rejected)
    print(f"\n[flattened {integrator}] acceptance {accepted / (accepted + rejected):.3f}, max |dev|/SE mean "
          f"{np.max(np.abs(statistics.mean - problem.posterior_mean) / statistics.se_mean):.2f}, cov "
          f"{np.max(np.abs(statistics.cov - problem.posterior_cov) / statistics.se_cov):.2f}")
    assert_matches_closed_form(statistics, problem.posterior_mean, problem.posterior_cov,
                               f"flattened Hamiltonian ({integrator}) MH")
    with open(os.path.join(str(tmp_path), "sampling_output", "run_manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["proposal_resolution"][0]["likelihood_weight"] == 0.3


# ------------------------------------------------------------------------- tau = 0: prior draw
def test_zero_weight_quarter_period_is_an_independent_prior_draw(tmp_path):
    spec = Hamiltonian(integrator="dimension_robust", integration_time=np.pi / 2, step_size=np.pi / 32,
                       mass=1.0, adaptive=False, likelihood_weight=0.0)
    proposal = _build(tmp_path, spec, seed=7)
    assert type(proposal) is HamiltonianInfinite and proposal.num_steps == 16
    # a likelihood gradient that would dominate if it were used
    proposal.set_gradient_functions(lambda q: 50.0 * (np.asarray(q) - 3.0), lambda q: np.asarray(q, dtype=float))
    rng = np.random.default_rng(3)
    starts = rng.normal(size=(2_000, 2)) * 2.0
    proposals = np.array([proposal.propose_sample(start) for start in starts])
    for j in range(2):
        assert abs(np.corrcoef(proposals[:, j], starts[:, j])[0, 1]) < 0.1, j
        assert abs(np.std(proposals[:, j], ddof=1) - 1.0) < 0.1, j
    # exactly the momentum (rotation by pi/2 with mass 1): q' = p
    np.testing.assert_allclose(proposal.propose_sample(starts[0]), proposal.p_current_start, atol=1e-12)


# -------------------------------------------------------------------------- spec validation
def test_spec_validation():
    for weight in (-0.1, 1.01, np.nan):
        with pytest.raises(ValueError, match=r"likelihood_weight must be in \[0, 1\]"):
            Hamiltonian(step_size=0.1, likelihood_weight=weight)
    with pytest.raises(ValueError, match="flattened-likelihood component does not adapt its step"):
        Hamiltonian(likelihood_weight=0.5)  # step_size=None: adaptive by default
    with pytest.raises(ValueError, match="flattened-likelihood component does not adapt its step"):
        Hamiltonian(step_size=0.1, adaptive=True, likelihood_weight=0.5)
    assert Hamiltonian(step_size=0.1, likelihood_weight=0.0).adaptive is False
    assert Hamiltonian(adaptive=False, likelihood_weight=0.2).step_size is None
    assert Hamiltonian(likelihood_weight=1.0).adaptive is True  # off: adapts as before
    # combined with tempering: valid
    both = Hamiltonian(step_size=0.1, likelihood_weight=0.2, tempered_fraction=0.3, tempered_beta=0.1)
    assert both.likelihood_weight == 0.2 and both.tempered_fraction == 0.3
    with pytest.raises(ValueError, match=r"likelihood_weight must be in \[0, 1\]"):
        HamiltonianRuntime(no_parameters=2, likelihood_weight=1.5)


# ----------------------------------------------------------------------------------- mixture
_MIXTURE = Mixture([Hamiltonian(integrator="dimension_robust", integration_time=1.0),
                    Hamiltonian(integrator="dimension_robust", integration_time=3.0, adaptive=False,
                                likelihood_weight=0.2)],
                   probabilities=[0.8, 0.2])


def test_flattened_component_in_a_mixture(tmp_path):
    proposal = _build(tmp_path, _MIXTURE, carried={"step_size": 0.05})
    adaptive, flattened = proposal.list_of_proposals
    assert isinstance(adaptive, HamiltonianInfinite_adaptive) and not adaptive.flattening_on
    assert type(flattened) is HamiltonianInfinite and flattened.flattening_on
    assert flattened.step_size == 0.05 and flattened.num_steps == 60  # the carried step, T = 3
    assert flattened.likelihood_weight == 0.2
    assert proposal.resolution["component1.likelihood_weight"] == 0.2
    assert "component0.likelihood_weight" not in proposal.resolution
    assert proposal.flattening_on and not proposal.tempering_on
    plain = BlockProposal(no_parameters=2, list_of_proposals=[adaptive, HamiltonianInfinite(2)],
                          list_of_groups=[[0, 1]] * 2, is_mixture=True)
    assert not plain.flattening_on


def test_unflattened_resolution_gains_no_key(tmp_path):
    assert "likelihood_weight" not in _build(tmp_path, Hamiltonian(step_size=0.1)).resolution


def test_describe_notes_the_flattening():
    text = Stage(algorithm="DAMH", proposal=_MIXTURE, max_evaluations=10).describe(0)
    assert "component 1: likelihood_weight=0.2: the trajectory's kicks use 0.2 x grad log L" in text
    assert "component 0: likelihood_weight" not in text
    bare = Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1, likelihood_weight=0.0)).describe(0)
    assert "[note] likelihood_weight=0: the trajectory's kicks use 0 x grad log L" in bare
    assert "likelihood_weight=1:" not in Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1)).describe(0)


def test_local_run_with_a_flattened_mixture_component(tmp_path):
    stages = [Stage(algorithm="MH", max_evaluations=60, proposal=Hamiltonian(integrator="dimension_robust",
                                                                              integration_time=0.6)),
              Stage(algorithm="DAMH", subchain_length=2, max_evaluations=60,
                    proposal=Mixture([Hamiltonian(integrator="dimension_robust", integration_time=0.6),
                                      Hamiltonian(integrator="dimension_robust", integration_time=1.8,
                                                  adaptive=False, likelihood_weight=0.2)],
                                     probabilities=[0.7, 0.3]))]
    sampling_output = _run_local(tmp_path, stages)
    with open(os.path.join(sampling_output, "run_manifest.json")) as f:
        resolution = json.load(f)["proposal_resolution"]
    assert "likelihood_weight" not in resolution[0]
    assert resolution[1]["component1.likelihood_weight"] == 0.2
