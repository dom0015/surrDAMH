#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 (2026-10-12, note 30): the tempered-trajectory Hamiltonian component.

* off = byte-identical: pinned fingerprints of fixed/adaptive bare ``Hamiltonian`` runs (both
  integrators) recorded on the code BEFORE C3, and ``tempered_fraction=0.0`` given explicitly
  equals the field left at its default;
* map properties on a 2-D quadratic potential (both integrators): involution under the momentum
  flip, unit Jacobian, the ``(n_t, sqrt_alpha)`` rule and the ``tempered_skipped`` counter;
* invariance: an MH stage with the tempered Hamiltonian alone samples the closed-form posterior;
* spec validation; the tempered component inside a ``Mixture`` (carried step, manifest
  resolution, the summed counter), the conditional ``notes`` column, ``describe()``.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import io
import json
import os

import numpy as np
import pytest

import surrDAMH
from helpers_statistics import assert_matches_closed_form, chain_statistics, read_weighted_samples
from surrDAMH.modules.proposals import (BlockProposal, Hamiltonian as HamiltonianRuntime,
                                        HamiltonianInfinite, HamiltonianInfinite_adaptive)
from surrDAMH.proposals import PCN, Hamiltonian, Mixture
from surrDAMH.stages import Stage, stage_name
from surrDAMH.surrogates.parent import Evaluator

from .test_mixture import _build, _fingerprint, _run_local

pytestmark = [pytest.mark.unit,
              pytest.mark.filterwarnings("ignore:.*is written to adaptive_stats:RuntimeWarning")]


# ----------------------------------------------------------------------- off = byte-identical
def _off_stages(integrator, **extra):
    return [
        Stage(algorithm="MH", max_evaluations=80,
              proposal=Hamiltonian(integrator=integrator, step_size=0.1, num_steps=5, **extra)),
        Stage(algorithm="DAMH", subchain_length=3, max_evaluations=60,
              proposal=Hamiltonian(integrator=integrator, step_size=0.15, integration_time=0.6,
                                   integration_time_jitter=0.3, **extra)),
        Stage(algorithm="DAMH", subchain_length=2, max_evaluations=60,
              proposal=Hamiltonian(integrator=integrator, integration_time=0.5, **extra)),
    ]


def _fingerprint_with_notes(sampling_output):
    digest = _fingerprint(sampling_output)
    for path in sorted(glob.glob(os.path.join(sampling_output, "notes", "*", "rank*.csv"))):
        digest[os.path.relpath(path, sampling_output)] = hashlib.md5(open(path, "rb").read()).hexdigest()
    return digest


def _resolution(name, n_start, n_end, *, dual_averaging=None, integration_time=None, jitter=0.0,
                num_steps_initial, step_size_initial):
    return {"dual_averaging": dual_averaging, "integration_time": integration_time,
            "integration_time_jitter": jitter, "mass_change_r": None, "mass_source": "spec", "name": name,
            "num_steps_initial": num_steps_initial, "posterior_statistics_discounted_from_n": None,
            "posterior_statistics_n_end": n_end, "posterior_statistics_n_start": n_start,
            "step_size_initial": step_size_initial}


_STATISTICS_KEYS = ["carry_over__posterior_statistics", "carry_over__posterior_statistics_n_seen",
                    "summary__posterior_statistics_n"]
_ADAPTIVE_KEYS = ["carry_over__dual_averaging", "carry_over__hamiltonian_mass", "carry_over__posterior_statistics",
                  "carry_over__posterior_statistics_n_seen", "carry_over__step_size",
                  "summary__posterior_statistics_n", "summary__step_size"]


def _reference(hashes, n_end_last):
    """The fingerprint dict of ``_off_stages`` from per-file hashes (the parts both integrators share)."""
    return {
        "adaptive_stats/alg0002_DAMH-SMU/rank0000.csv": hashes["adaptive_stats2"],
        "carry_over/alg0000_MH.npz": _STATISTICS_KEYS + [hashes["carry0"]],
        "carry_over/alg0001_DAMH-SMU.npz": _STATISTICS_KEYS + [hashes["carry1"]],
        "carry_over/alg0002_DAMH-SMU.npz": _ADAPTIVE_KEYS + [hashes["carry2"]],
        "notes/alg0000_MH/rank0000.csv": hashes["notes0"],
        "notes/alg0001_DAMH-SMU/rank0000.csv": hashes["notes1"],
        "notes/alg0002_DAMH-SMU/rank0000.csv": hashes["notes2"],
        "proposal_resolution": [
            _resolution("alg0000_MH", 0, 80, num_steps_initial=5, step_size_initial=0.1),
            _resolution("alg0001_DAMH-SMU", 80, 140, integration_time=0.6, jitter=0.3, num_steps_initial=4,
                        step_size_initial=0.15),
            _resolution("alg0002_DAMH-SMU", 140, n_end_last, dual_averaging="fresh", integration_time=0.5,
                        num_steps_initial=5, step_size_initial=0.1)],
        "raw_data/alg0000_MH/rank0000.csv": hashes["raw0"],
        "raw_data/alg0001_DAMH-SMU/rank0000.csv": hashes["raw1"],
        "raw_data/alg0002_DAMH-SMU/rank0000.csv": hashes["raw2"],
        "samples/alg0000_MH/rank0000.csv": hashes["samples0"],
        "samples/alg0001_DAMH-SMU/rank0000.csv": hashes["samples1"],
        "samples/alg0002_DAMH-SMU/rank0000.csv": hashes["samples2"],
        "subchain_stats/alg0001_DAMH-SMU/rank0000.csv": hashes["subchain1"],
        "subchain_stats/alg0002_DAMH-SMU/rank0000.csv": hashes["subchain2"],
    }


#: recorded 2026-10-12 with the code BEFORE C3 (dev container, /dolfinx-env python), by
#: ``_off_stages`` + ``_fingerprint_with_notes``: a fixed Hamiltonian MH stage (num_steps), a fixed
#: DAMH stage with a jittered integration time and an adaptive DAMH stage, for each integrator
_OFF_REFERENCE = {
    "leapfrog": _reference(dict(
        adaptive_stats2="64666d91877704bedd471221e410cc27",
        carry0="eb526bca8cce21689a0bb4d469bde69c", carry1="267dd7b86fcc97f75ba934e93749c0c5",
        carry2="6b094e5b1c1d58de5c45e3e92899bb41",
        notes0="c57a53cbc2e2e26b9b6ae3b3e4b329d0", notes1="cb1e24453d58f5b8511e9a1495b4cc5a",
        notes2="393985ea2463c1c41f121b22ac5ebc2d",
        raw0="163d04e1ec6ba729a38aac1568cfd355", raw1="16cfac942dbc168f024745cf6c655efc",
        raw2="82ce96232e89732dd043c2db77b4458f",
        samples0="da0cba6161106f62143354ff34ed05df", samples1="de7806b946a3f5e4ddf5971c3cbccf6b",
        samples2="9fec64138877d8bb4182df802c7bb31c",
        subchain1="fe5740ed8eb0a1d7db98edfb4d9f05d8", subchain2="7edc033d69f3707a0d7aa3d50327962c"), 210),
    "dimension_robust": _reference(dict(
        adaptive_stats2="18660a7a044bfd8832eb9aa8541b6ccf",
        carry0="2129c9da0e09b604611fa10a287a7aa3", carry1="656bd108bd6fbc3861b90f3d1a6b1680",
        carry2="edb4b26abd154a4a5f4bbbb0aadf9482",
        notes0="c57a53cbc2e2e26b9b6ae3b3e4b329d0", notes1="cb1e24453d58f5b8511e9a1495b4cc5a",
        notes2="bde35d88fa8ff3a910c08751c586aac2",
        raw0="849bf81d048bbf0d54ce51db5e9674b7", raw1="010dd65a78c70ef3cab2bf9f083ddb80",
        raw2="7e8f180608ab84f1c84b259cae071431",
        samples0="3fdca845856f590abede186eee4e7dab", samples1="b769f7c27ff075e7f26c4de6bbbcb9f4",
        samples2="e5cb0a06152fd917c1dcabf2ace39d13",
        subchain1="231b7439e492b652c2dcca9fd31e3e23", subchain2="3baaac18d9f8522c630a563e340ee38a"), 208),
}


@pytest.mark.parametrize("integrator", ["leapfrog", "dimension_robust"])
def test_untempered_hamiltonian_runs_are_byte_identical_to_before_c3(tmp_path, integrator):
    """Every output file (incl. ``notes``) of fixed and adaptive bare Hamiltonian runs hashes exactly
    as before C3. The hashes are environment-specific (floating-point libraries); if this fails on
    another machine, re-record them there from the pre-C3 code before concluding anything."""
    assert _fingerprint_with_notes(_run_local(tmp_path, _off_stages(integrator))) == _OFF_REFERENCE[integrator]


@pytest.mark.parametrize("integrator", ["leapfrog", "dimension_robust"])
def test_explicit_zero_fraction_equals_the_default(tmp_path, integrator):
    """Machine-independent: ``tempered_fraction=0.0, tempered_beta=1.0`` given = the fields left out."""
    default = _fingerprint_with_notes(_run_local(tmp_path / "a", _off_stages(integrator)))
    explicit = _fingerprint_with_notes(_run_local(tmp_path / "b", _off_stages(
        integrator, tempered_fraction=0.0, tempered_beta=1.0)))
    assert explicit == default


# ------------------------------------------------------------------------------ map properties
_A_LIKELIHOOD = np.array([[2.0, 0.6], [0.6, 1.0]])  # U(q) = q.A.q/2 (+ q.q/2 prior)
_MASS_SD = np.array([1.0, 1.5])


def _tempered(proposal_cls, num_steps=6, fraction=0.34, beta=0.25, step_size=0.2):
    proposal = proposal_cls(no_parameters=2, step_size=step_size, num_steps=num_steps, sd_or_cov=_MASS_SD, seed=0,
                            tempered_fraction=fraction, tempered_beta=beta)
    # convention of tests/unit/test_proposals.py: the callables are the gradients of -log density
    proposal.set_gradient_functions(lambda q: _A_LIKELIHOOD @ q, lambda q: np.asarray(q, dtype=float).copy())
    return proposal


def _map(proposal, z):
    q, p_end = proposal._leapfrog(z[:2], z[2:])
    return np.concatenate([q, p_end])


@pytest.mark.parametrize("proposal_cls", [HamiltonianRuntime, HamiltonianInfinite])
def test_tempered_map_with_flip_is_an_involution(proposal_cls):
    proposal = _tempered(proposal_cls)
    assert proposal._tempering(6) == (2, pytest.approx(np.sqrt(2.0)))  # L=6, f=0.34: n_t=2; beta=1/4: alpha=2
    q0, p0 = np.array([0.7, -1.1]), np.array([0.4, 0.9])
    q1, p1_end = proposal._leapfrog(q0, p0)  # p1_end = -p(L): the flipped final momentum
    assert not np.allclose(q1, q0)
    q2, p2_end = proposal._leapfrog(q1, p1_end)  # map(q', -p') = (q, -p), returned flipped: (q, p)
    np.testing.assert_allclose(q2, q0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(p2_end, p0, atol=1e-10, rtol=0)
    # the tempering really changes the map (same start, untempered)
    plain = _tempered(proposal_cls, fraction=0.0, beta=1.0)
    assert not np.allclose(plain._leapfrog(q0, p0)[0], q1)


@pytest.mark.parametrize("proposal_cls", [HamiltonianRuntime, HamiltonianInfinite])
def test_tempered_map_has_unit_jacobian(proposal_cls):
    proposal = _tempered(proposal_cls)
    z0 = np.array([0.3, -0.5, 1.2, -0.4])
    h = 1e-6
    jacobian = np.empty((4, 4))
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        jacobian[:, j] = (_map(proposal, z0 + e) - _map(proposal, z0 - e)) / (2 * h)
    # the flip has determinant (-1)^2 = 1 in 2-D, so det of the whole map must be 1
    assert abs(np.linalg.det(jacobian) - 1.0) < 1e-5, np.linalg.det(jacobian)
    # ... while one heating scaling alone has det alpha^(d/2) = 2: the cooling really undoes it
    assert proposal._tempering(6)[1] ** 2 == pytest.approx(2.0)


@pytest.mark.parametrize("fraction,beta", [(0.3, 0.1), (0.5, 0.5), (0.1, 0.01), (0.34, 0.25)])
def test_tempering_counts(fraction, beta):
    proposal = _tempered(HamiltonianRuntime, fraction=fraction, beta=beta)
    for num_steps in range(2, 60):
        before = proposal.tempered_skipped
        n_t, sqrt_alpha = proposal._tempering(num_steps)
        assert n_t <= num_steps // 2 and n_t == min(int(np.floor(fraction * num_steps)), num_steps // 2)
        if n_t == 0:
            assert sqrt_alpha == 1.0 and proposal.tempered_skipped == before + 1
        else:
            assert sqrt_alpha ** (2 * n_t) == pytest.approx(1.0 / beta, rel=1e-12)
            assert proposal.tempered_skipped == before
    proposal.tempered_skipped = 0
    assert proposal._tempering(1) == (0, 1.0) and proposal._tempering(0) == (0, 1.0)
    assert proposal.tempered_skipped == 2  # L < 2: never tempered, counted


def test_untempered_counts_nothing():
    proposal = _tempered(HamiltonianRuntime, fraction=0.0, beta=1.0)
    assert not proposal.tempering_on
    assert [proposal._tempering(L) for L in (0, 1, 2, 10)] == [(0, 1.0)] * 4
    assert proposal.tempered_skipped == 0


def test_leapfrog_counts_short_trajectories():
    proposal = _tempered(HamiltonianInfinite, num_steps=1, fraction=0.3, beta=0.1)
    proposal.propose_sample(np.zeros(2))
    proposal.propose_sample(np.zeros(2))
    assert proposal.tempered_skipped == 2
    proposal.num_steps = 3  # floor(0.9) = 0 as well
    proposal.propose_sample(np.zeros(2))
    assert proposal.tempered_skipped == 3
    proposal.num_steps = 4  # floor(1.2) = 1: tempered
    proposal.propose_sample(np.zeros(2))
    assert proposal.tempered_skipped == 3


# ------------------------------------------------------------------------------- invariance
class _ExactLinearGradientEvaluator(Evaluator):
    def __init__(self, A):
        super().__init__(no_parameters=A.shape[1], no_observations=A.shape[0])
        self.A = np.asarray(A, dtype=float)

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.A.T

    def jacobian(self, datapoints):
        return self.A, self.A @ np.asarray(datapoints, dtype=float).reshape(-1)


@pytest.mark.parametrize("integrator", ["leapfrog", "dimension_robust"])
def test_tempered_hamiltonian_alone_samples_the_closed_form_posterior(tmp_path, linear_gaussian_problem,
                                                                     integrator):
    problem = linear_gaussian_problem
    spec = Hamiltonian(integrator=integrator, step_size=0.1, integration_time=1.0, integration_time_jitter=0.3,
                       tempered_fraction=0.3, tempered_beta=0.5, adaptive=False)
    stage = Stage(algorithm="MH", proposal=spec, max_evaluations=8_000)
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="prior", use_surrogate_gradients=True)
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        result = surrDAMH.Problem(problem.prior, problem.likelihood, solver=problem.solver).run_sampling_local(
            conf, [stage], surrogate_evaluator=_ExactLinearGradientEvaluator(problem.A))
    stage_result = result.stage_results[0]
    accepted, rejected = stage_result.counter_accepted, stage_result.counter_rejected
    assert accepted / (accepted + rejected) > 0.05, (accepted, rejected)
    assert "- tempered skipped: 0" in printed.getvalue()  # L = 7..13 here: always tempered
    samples = read_weighted_samples(tmp_path, stage_result.name, problem.no_parameters)
    statistics = chain_statistics(samples, accepted, rejected)
    print(f"\n[tempered {integrator}] acceptance {accepted / (accepted + rejected):.3f}, mean "
          f"{np.array2string(statistics.mean, precision=4)} vs {np.array2string(problem.posterior_mean, precision=4)}"
          f", max |dev|/SE mean {np.max(np.abs(statistics.mean - problem.posterior_mean) / statistics.se_mean):.2f}"
          f", cov {np.max(np.abs(statistics.cov - problem.posterior_cov) / statistics.se_cov):.2f}")
    assert_matches_closed_form(statistics, problem.posterior_mean, problem.posterior_cov,
                               f"tempered Hamiltonian ({integrator}) MH")
    # the conditional notes column
    with open(os.path.join(str(tmp_path), "sampling_output", "notes", stage_result.name, "rank0000.csv")) as f:
        header, row = f.readline().strip().split(","), f.readline().strip().split(",")
    assert header[-1] == "tempered_skipped" and row[-1] == "0"
    with open(os.path.join(str(tmp_path), "sampling_output", "run_manifest.json")) as f:
        resolution = json.load(f)["proposal_resolution"][0]
    assert resolution["tempered_fraction"] == 0.3 and resolution["tempered_beta"] == 0.5


# -------------------------------------------------------------------------- spec validation
def test_spec_validation():
    for kwargs in (dict(tempered_fraction=0.3), dict(tempered_beta=0.5)):
        with pytest.raises(ValueError, match="tempered_fraction > 0 and tempered_beta < 1 go together"):
            Hamiltonian(step_size=0.1, **kwargs)
    for fraction in (-0.1, 0.51, np.nan):
        with pytest.raises(ValueError, match=r"tempered_fraction must be in \[0, 0.5\]"):
            Hamiltonian(step_size=0.1, tempered_fraction=fraction, tempered_beta=0.5)
    for beta in (0.0, -1.0, 1.5, np.nan):
        with pytest.raises(ValueError, match=r"tempered_beta must be in \(0, 1\]"):
            Hamiltonian(step_size=0.1, tempered_fraction=0.3, tempered_beta=beta)
    with pytest.raises(ValueError, match="does not adapt its step"):
        Hamiltonian(tempered_fraction=0.3, tempered_beta=0.1)  # step_size=None: adaptive by default
    with pytest.raises(ValueError, match="does not adapt its step"):
        Hamiltonian(step_size=0.1, adaptive=True, tempered_fraction=0.3, tempered_beta=0.1)
    # valid: pinned step, or adaptive=False with the carried step
    assert Hamiltonian(step_size=0.1, tempered_fraction=0.5, tempered_beta=1e-3).adaptive is False
    assert Hamiltonian(adaptive=False, tempered_fraction=0.3, tempered_beta=0.1).step_size is None
    assert PCN(beta=1.0).beta == 1.0  # the prior-draw component (docstring only)
    with pytest.raises(ValueError):
        HamiltonianRuntime(no_parameters=2, tempered_fraction=0.6, tempered_beta=0.5)


# ----------------------------------------------------------------------------------- mixture
_MIXTURE = Mixture([Hamiltonian(integrator="dimension_robust", integration_time=1.0),
                    Hamiltonian(integrator="dimension_robust", integration_time=3.0, tempered_fraction=0.3,
                                tempered_beta=0.1, adaptive=False)],
                   probabilities=[0.8, 0.2])


def test_tempered_component_in_a_mixture(tmp_path):
    proposal = _build(tmp_path, _MIXTURE, carried={"step_size": 0.05})
    adaptive, tempered = proposal.list_of_proposals
    assert isinstance(adaptive, HamiltonianInfinite_adaptive) and not adaptive.tempering_on
    assert type(tempered) is HamiltonianInfinite and tempered.tempering_on
    assert tempered.step_size == 0.05 and tempered.num_steps == 60  # the carried step, T = 3
    assert proposal.resolution["adaptive_stats_component"] == 0
    assert proposal.resolution["component1.tempered_fraction"] == 0.3
    assert proposal.resolution["component1.tempered_beta"] == 0.1
    assert "component0.tempered_fraction" not in proposal.resolution
    assert proposal.tempering_on and proposal.tempered_skipped == 0
    tempered.tempered_skipped = 3
    assert proposal.tempered_skipped == 3
    # a mixture without tempering has the property but reports it off
    plain = BlockProposal(no_parameters=2, list_of_proposals=[adaptive, HamiltonianInfinite(2)],
                          list_of_groups=[[0, 1]] * 2, is_mixture=True)
    assert not plain.tempering_on and plain.tempered_skipped == 0


def test_untempered_resolution_gains_no_keys(tmp_path):
    proposal = _build(tmp_path, Hamiltonian(step_size=0.1))
    assert not any(key.startswith("tempered") for key in proposal.resolution)


def test_describe_notes_the_tempering():
    stage = Stage(algorithm="DAMH", proposal=_MIXTURE, max_evaluations=10)
    text = stage.describe(0)
    assert "component 1: tempered_fraction=0.3, tempered_beta=0.1: momentum x sqrt(alpha)" in text
    assert "component 0: tempered" not in text
    bare = Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1, tempered_fraction=0.2, tempered_beta=0.5))
    assert "tempered_fraction=0.2, tempered_beta=0.5: momentum x sqrt(alpha)" in bare.describe(0)
    assert "tempered_fraction=" not in Stage(algorithm="MH", proposal=Hamiltonian(step_size=0.1)).describe(0) \
        .split("[note]", 1)[-1]


def test_mixture_run_writes_the_column_only_for_the_tempering_stage(tmp_path):
    stages = [Stage(algorithm="MH", max_evaluations=60, proposal=Hamiltonian(step_size=0.2, num_steps=3)),
              Stage(algorithm="DAMH", subchain_length=2, max_evaluations=60,
                    proposal=Mixture([Hamiltonian(integrator="dimension_robust", step_size=0.2, num_steps=3),
                                      Hamiltonian(integrator="dimension_robust", step_size=0.2, num_steps=1,
                                                  tempered_fraction=0.3, tempered_beta=0.1)]))]
    sampling_output = _run_local(tmp_path, stages)
    names = [stage_name(stage, i) for i, stage in enumerate(stages)]
    rows = {}
    for name in names:
        with open(os.path.join(sampling_output, "notes", name, "rank0000.csv")) as f:
            rows[name] = [line.strip().split(",") for line in f]
    assert "tempered_skipped" not in rows[names[0]][0]
    header, row = rows[names[1]]
    assert header[-1] == "tempered_skipped"
    assert int(row[-1]) > 0  # num_steps=1: every trajectory of the tempered component is skipped
