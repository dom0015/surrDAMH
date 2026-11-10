#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests of the automatic mode (``surrDAMH.auto.plan_auto``, ``Problem.run_sampling_local_auto``;
API_SPEC_auto §5, 2026-10-08).

Planner rules are checked on a stand-in problem (sizes + ``solver_instance``) and a real
``Configuration`` whose MPI layout fields are overridden after construction (a single process
cannot build ``use_collector=True`` with several samplers). The runs use the linear-Gaussian
problem of ``tests/conftest.py`` (closed-form posterior).
"""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

import surrDAMH
from surrDAMH.auto import AutoPlan, AutoTestDataRequest, plan_auto
from surrDAMH.proposals import Hamiltonian, RandomWalk
from surrDAMH.surrogates import PolynomialSklearnUpdater

from helpers_statistics import assert_matches_closed_form, chain_statistics, read_weighted_samples

pytestmark = pytest.mark.unit


def _problem(d=2, m=2, solver_instance=None):
    return SimpleNamespace(no_parameters=d, no_observations=m, solver_instance=solver_instance)


def _conf(tmp_path, no_samplers=1, use_collector=True, **kwargs):
    """A real Configuration with the MPI layout of ``no_samplers`` sampler ranks (+ collector)."""
    conf = surrDAMH.Configuration(output_dir=str(tmp_path / "out"), use_collector=False, use_solvers_pool=False,
                                  **kwargs)
    conf.use_collector = use_collector
    conf.no_samplers = no_samplers
    return conf


def _poly(d=2, m=2):
    return PolynomialSklearnUpdater(no_parameters=d, no_observations=m, max_degree=1)


def _plan(tmp_path, d=2, budget=None, time_limit=None, mode="robust", no_samplers=1, local=False,
          solver_instance=None, updater="poly", conf=None, **kwargs):
    if conf is None:
        conf = _conf(tmp_path, no_samplers=no_samplers)
    if updater == "poly":
        updater = _poly(d)
    return plan_auto(_problem(d, solver_instance=solver_instance), conf, budget=budget, time_limit=time_limit,
                     mode=mode, local=local, surrogate_updater=updater, **kwargs)


# ---------------------------------------------------------------------------------------------
# layout rule
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("d, B, n0", [
    (2, 10_000, 500),      # 20 d = 40 below the floor 0.05 B
    (20, 20_000, 1_000),   # note 25 worked example: clip(400, 1000, 5000)
    (10, 2_000, 200),      # 20 d inside [0.05 B, 0.25 B]
    (100, 5_000, 1_250),   # 20 d = 2000 above the cap 0.25 B (B = 50 d exactly: surrogate still used)
])
def test_warm_up_floor_and_cap(tmp_path, d, B, n0):
    plan = _plan(tmp_path, d=d, budget=B)
    assert plan.per_chain_budget == B and plan.test_data_size == 0
    assert plan.warm_up == {"evaluations": n0}
    warm = plan.stages[0]
    # 2026-10-09: a burn-in stage (the chain continues), not is_excluded (which restarts it)
    assert (warm.algorithm, warm.burn_in, warm.is_excluded, warm.max_evaluations) == ("MH", True, False, n0)
    assert isinstance(warm.proposal, RandomWalk) and warm.proposal.adaptive
    assert [s.max_evaluations for s in plan.stages[1:]] == [(B - n0) // 4] * 3 + [(B - n0) // 4 + (B - n0) % 4]


def test_worked_example_of_note_25_four_chains(tmp_path):
    plan = _plan(tmp_path, d=20, budget=80_000, no_samplers=4)
    assert plan.per_chain_budget == 20_000 and plan.warm_up == {"evaluations": 1_000}
    assert [s.max_evaluations for s in plan.stages[1:]] == [4_750] * 4
    assert plan.conf_settings["min_snapshots_initial"] == 200  # min(10 d, n0 C / 2)
    assert plan.conf_settings["min_snapshots_to_update"] == 40  # max(20, 2 d)


def test_chunks_with_remainder(tmp_path):
    plan = _plan(tmp_path, d=2, budget=1_003)
    assert plan.warm_up == {"evaluations": 50}  # clip(40, 50.15, 250.75) -> 50
    chunks = plan.stages[1:]
    assert [s.max_evaluations for s in chunks] == [238, 238, 238, 239]
    assert sum(s.max_evaluations for s in plan.stages) == 1_003
    for s in chunks:
        assert (s.algorithm, s.subchain_length, s.surrogate_model_updates, s.is_excluded, s.burn_in) == \
            ("DAMH", 1, True, False, False)
    assert plan.stage_names() == ["alg0000_MH-adaptive"] + [f"alg000{i}_DAMH-SMU" for i in range(1, 5)]
    assert len({id(s.proposal) for s in chunks}) == 4  # one spec object per stage


def test_small_budget_runs_a_single_mh_stage(tmp_path):
    plan = _plan(tmp_path, d=4, budget=2 * 199, no_samplers=2)  # B = 199 < 50 d = 200
    assert len(plan.stages) == 1 and plan.chunks == 0 and plan.warm_up is None
    stage = plan.stages[0]
    assert (stage.algorithm, stage.max_evaluations, stage.is_excluded, stage.burn_in) == ("MH", 199, False, False)
    assert isinstance(stage.proposal, RandomWalk) and stage.proposal.adaptive
    assert any("does not pay" in note for note in plan.notes)
    assert "min_snapshots_initial" not in plan.conf_settings and plan.surrogate_test_data is None
    # the collector of an MPI run still needs an updater: the given one is kept, with a note
    assert isinstance(plan.surrogate_updater, PolynomialSklearnUpdater)


def test_small_budget_has_no_held_out_set(tmp_path, linear_gaussian_problem):
    # 2*d=4 points would be planned, but without a surrogate the whole budget goes to the chain
    plan = _plan(tmp_path, d=2, budget=99, local=True, solver_instance=linear_gaussian_problem.solver)
    assert plan.test_data_size == 0 and plan.stages[0].max_evaluations == 99


def test_no_collector_runs_a_single_mh_stage(tmp_path):
    conf = surrDAMH.Configuration(output_dir=str(tmp_path / "o"), use_collector=False, use_solvers_pool=False)
    assert conf.no_samplers == 1
    plan = _plan(tmp_path, budget=100_000, conf=conf, solver_instance=object())
    assert len(plan.stages) == 1 and plan.stages[0].algorithm == "MH"
    assert plan.stages[0].max_evaluations == 100_000  # no held-out set without a surrogate
    assert plan.surrogate_updater is None and plan.surrogate_test_data is None and plan.test_data_size == 0
    assert any("use_collector=False" in note for note in plan.notes)
    assert any("ignored" in note for note in plan.notes)  # the given updater is not used
    # with a time budget too
    plan = _plan(tmp_path, time_limit=10.0, conf=conf, updater=None)
    assert len(plan.stages) == 1 and plan.stages[0].time_limit == 10.0


def test_time_budget_shares(tmp_path):
    plan = _plan(tmp_path, d=3, time_limit=100.0, no_samplers=2)
    assert plan.per_chain_budget is None and plan.budget is None
    assert plan.warm_up == {"time_limit": pytest.approx(15.0)}
    assert plan.stages[0].time_limit == pytest.approx(15.0) and plan.stages[0].burn_in
    assert not plan.stages[0].is_excluded
    assert [s.time_limit for s in plan.stages[1:]] == [pytest.approx(21.25)] * 4
    assert all(s.max_evaluations > 10 ** 15 for s in plan.stages)  # no evaluation limit
    assert any("cannot be enforced" in note for note in plan.notes)
    assert plan.conf_settings["min_snapshots_initial"] == 30  # 10 d (no n0 known)


def test_mode_sets_proposal_and_gradients(tmp_path):
    robust = _plan(tmp_path, budget=10_000)
    assert all(isinstance(s.proposal, RandomWalk) for s in robust.stages)
    assert robust.conf_settings["use_surrogate_gradients"] is False

    fast = _plan(tmp_path, budget=10_000, mode="fast", updater=None)
    assert isinstance(fast.stages[0].proposal, RandomWalk)
    for s in fast.stages[1:]:
        p = s.proposal
        assert isinstance(p, Hamiltonian)
        # C1 (2026-10-11): covariance mass, integration time T = AUTO_FAST_INTEGRATION_TIME instead of num_steps=30;
        # C1b (2026-10-11): T = pi/2 (was 3.0) with jitter 0.5
        assert (p.num_steps, p.integration_time, p.integrator, p.mass, p.step_size, p.adaptive) == \
            (None, 3.0, "dimension_robust", "covariance", None, True)
        assert p.integration_time_jitter == 0.5
    assert "chunk mass: inverse of the chains' pooled posterior covariance" in fast.describe()
    assert "mass='covariance'" in fast.manifest_entry()["proposal"]
    assert f"integration_time={3.0!r}" in fast.manifest_entry()["proposal"]
    assert "integration_time_jitter=0.5" in fast.manifest_entry()["proposal"]
    assert fast.conf_settings["use_surrogate_gradients"] is True


def test_default_network(tmp_path):
    from surrDAMH.surrogates.torch_perceptron_minibatches import NeuralNetworkUpdater
    plan = _plan(tmp_path, d=3, budget=10_000, updater=None)
    updater = plan.surrogate_updater
    assert isinstance(updater, NeuralNetworkUpdater)
    assert (updater.no_parameters, updater.no_observations) == (3, 2)
    assert updater.hidden_layer_sizes == (64, 64) and updater.activation_name == "silu"
    assert updater.solver_name == "adamw" and updater.seed == 0
    assert updater.output_normalization == "likelihood"
    assert plan.surrogate == {"class": "NeuralNetworkUpdater", "default": True,
                              "hparams": {"hidden_layer_sizes": [64, 64], "activation": "silu",
                                          "solver": "adamw", "seed": 0}}


def test_explicit_updater_is_kept(tmp_path):
    updater = _poly()
    plan = _plan(tmp_path, budget=10_000, updater=updater)
    assert plan.surrogate_updater is updater
    assert plan.surrogate["class"] == "PolynomialSklearnUpdater" and plan.surrogate["default"] is False
    assert plan.surrogate["hparams"]["max_degree"] == 1


def test_fast_mode_needs_gradients(tmp_path):
    with pytest.raises(ValueError, match="gradients"):
        _plan(tmp_path, budget=10_000, mode="fast")  # polynomial: no gradients
    conf = _conf(tmp_path, use_surrogate_gradients=False)
    with pytest.raises(ValueError, match="use_surrogate_gradients"):
        _plan(tmp_path, budget=10_000, mode="fast", updater=None, conf=conf)
    # without a surrogate in the plan (too small a budget) the fast mode does not need one
    plan = _plan(tmp_path, budget=50, mode="fast")
    assert len(plan.stages) == 1


@pytest.mark.parametrize("given, expected", [("prior", "lhs"), ("lhs", None), ("user_specified", None)])
def test_initial_sample_type_rule(tmp_path, given, expected):
    conf = _conf(tmp_path, initial_sample_type=given)
    plan = _plan(tmp_path, budget=10_000, conf=conf)
    assert plan.conf_settings.get("initial_sample_type") == expected


def test_user_snapshot_settings_are_kept_with_a_note(tmp_path):
    conf = _conf(tmp_path, min_snapshots_initial=7, min_snapshots_to_update=3)
    plan = _plan(tmp_path, budget=10_000, conf=conf)
    assert "min_snapshots_initial" not in plan.conf_settings
    assert "min_snapshots_to_update" not in plan.conf_settings
    assert any("min_snapshots_initial=7 set by the user is kept" in note for note in plan.notes)
    assert any("min_snapshots_to_update=3 set by the user is kept" in note for note in plan.notes)


def test_held_out_set_size_and_budget_subtraction(tmp_path):
    solver = object()
    plan = _plan(tmp_path, d=2, budget=20_000, local=True, solver_instance=solver)
    assert plan.test_data_size == 256  # min(256, max(4, 400))
    assert plan.per_chain_budget == 20_000 - 256
    assert plan.surrogate_test_data == AutoTestDataRequest(size=256, seed=25347)

    plan = _plan(tmp_path, d=2, budget=1_000, no_samplers=3, solver_instance=solver)
    assert plan.test_data_size == 20  # 0.02 * budget
    assert plan.per_chain_budget == (1_000 - 20) // 3

    plan = _plan(tmp_path, d=40, budget=4 * 2_100, no_samplers=4, solver_instance=solver)
    assert plan.test_data_size == 168  # 2 d = 80 < 0.02 * 8400 = 168

    plan = _plan(tmp_path, d=50, time_limit=60.0, solver_instance=solver)
    assert plan.test_data_size == 100  # time budget: min(256, max(2 d, 64))

    # SolverSpec problem (no solver instance on the collector): no held-out set, a note
    plan = _plan(tmp_path, d=2, budget=20_000)
    assert plan.test_data_size == 0 and plan.surrogate_test_data is None
    assert any("SolverSpec" in note for note in plan.notes)

    # a given set is used as given and not subtracted
    given = (np.zeros((5, 2)), np.zeros((5, 2)))
    plan = _plan(tmp_path, d=2, budget=20_000, solver_instance=solver, surrogate_test_data=given)
    assert plan.surrogate_test_data is given and plan.test_data_size == 0 and plan.per_chain_budget == 20_000


def test_argument_errors(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        _plan(tmp_path, budget=1000, time_limit=10.0)
    with pytest.raises(ValueError, match="exactly one"):
        _plan(tmp_path)
    with pytest.raises(ValueError, match="mode"):
        _plan(tmp_path, budget=1000, mode="bogus")
    with pytest.raises(ValueError, match="positive"):
        _plan(tmp_path, budget=0)
    with pytest.raises(ValueError, match="positive"):
        _plan(tmp_path, time_limit=-1.0)
    conf = _conf(tmp_path, stage_index_offset=2)
    with pytest.raises(ValueError, match="not a continuation"):
        _plan(tmp_path, budget=1000, conf=conf)


def test_describe_and_manifest_entry(tmp_path):
    plan = _plan(tmp_path, d=2, budget=8_000, no_samplers=2, solver_instance=object())
    entry = plan.manifest_entry()
    assert set(entry) == {"mode", "budget", "time_limit", "no_samplers", "per_chain_budget", "test_data_size",
                          "warm_up", "chunks", "stage_names", "proposal", "surrogate", "conf_settings", "notes",
                          "exact_step_probability", "audit_prerejected", "polyak_averaging",
                          "surrogate_error_model"}
    assert entry["exact_step_probability"] == 0.05 and entry["audit_prerejected"] == 0.05  # S2 placeholders
    assert entry["polyak_averaging"] is True  # S3
    assert entry["surrogate_error_model"] is True  # S1
    assert entry["mode"] == "robust" and entry["budget"] == 8_000 and entry["time_limit"] is None
    assert entry["no_samplers"] == 2 and entry["test_data_size"] == 160
    assert entry["per_chain_budget"] == (8_000 - 160) // 2 == 3_920
    assert entry["warm_up"] == {"evaluations": 196} and entry["chunks"] == 4
    assert entry["stage_names"] == plan.stage_names()
    assert entry["proposal"] == "RandomWalk(scale=None, adaptive=True, target_rate=None)"
    assert entry["conf_settings"] == {"initial_sample_type": "lhs", "use_surrogate_gradients": False,
                                      "min_snapshots_initial": 20, "min_snapshots_to_update": 20}
    json.dumps(entry)  # JSON-safe

    text = plan.describe()
    for fragment in ("Auto plan (mode='robust', budget=8000", "160 prior draws", "B = 3920",
                     "196 evaluations per chain, burn-in", "4 DAMH-SMU chunks", "931, 931, 931, 931",
                     "PolynomialSklearnUpdater(", "initial_sample_type='lhs'", "alg0004_DAMH-SMU",
                     "exact random-walk step with probability 0.05", "audit of pre-rejected proposals with probability 0.05"):
        assert fragment in text, (fragment, text)


def test_set_by_auto_updates_the_requested_snapshot(tmp_path):
    conf = surrDAMH.Configuration(output_dir=str(tmp_path / "o"), use_collector=False, use_solvers_pool=False)
    conf._set_by_auto(initial_sample_type="lhs", min_snapshots_initial=20, use_surrogate_gradients=False)
    requested = conf.requested_posterior_fields()
    assert requested["initial_sample_type"] == "lhs" and requested["min_snapshots_initial"] == 20
    assert conf.use_surrogate_gradients_requested is False


# ---------------------------------------------------------------------------------------------
# local runs
# ---------------------------------------------------------------------------------------------

def _local_conf(output_dir):
    return surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  transform_before_saving=False)


def _manifest(output_dir):
    with open(os.path.join(str(output_dir), "sampling_output", "run_manifest.json")) as f:
        return json.load(f)


def _real_problem(lgp):
    return surrDAMH.Problem(lgp.prior, lgp.likelihood, solver=lgp.solver)


def test_local_robust_run_matches_the_closed_form(tmp_path, linear_gaussian_problem):
    lgp = linear_gaussian_problem
    out = tmp_path / "robust"
    budget = 20_000
    run = _real_problem(lgp).run_sampling_local_auto(_local_conf(out), budget=budget, mode="robust",
                                                     surrogate_updater=_poly())
    names = ["alg0000_MH-adaptive"] + [f"alg000{i}_DAMH-SMU" for i in range(1, 5)]
    assert [r.name for r in run.stage_results] == names and [s.name for s in run.stages] == names
    manifest = _manifest(out)
    assert manifest["auto"] == run.auto and run.auto["stage_names"] == names
    assert run.auto["test_data_size"] == 256 and run.auto["per_chain_budget"] == budget - 256
    assert manifest["configuration"]["initial_sample_type"] == "lhs"
    assert [s["max_evaluations"] for s in manifest["stages"]] == [987, 4689, 4689, 4689, 4690]
    # evaluations per stage as planned (acc + rej + audited, S2; prerejected proposals cost no evaluation)
    assert [r.counter_accepted + r.counter_rejected + r.counter_audited
            for r in run.stage_results] == [987, 4689, 4689, 4689, 4690]
    # S2: the DAMH chunks ran exact steps and audits; the MH warm-up neither
    assert run.stage_results[0].counter_exact_steps == run.stage_results[0].counter_audited == 0
    assert all(r.counter_exact_steps > 0 and r.counter_audited > 0 for r in run.stage_results[1:])
    test_data = surrDAMH.TestData.reuse(str(out))
    assert test_data.get_size() == 256
    assert surrDAMH.SamplingRun.load(str(out)).auto == run.auto

    chain = np.vstack([read_weighted_samples(out, name, 2) for name in names[1:]])
    statistics = chain_statistics(chain)
    print(f"\n[auto local robust] {statistics.no_samples} samples, mean {statistics.mean} vs {lgp.posterior_mean}, "
          f"cov {statistics.cov.ravel()} vs {lgp.posterior_cov.ravel()}, "
          f"se_mean {statistics.se_mean}")
    assert_matches_closed_form(statistics, lgp.posterior_mean, lgp.posterior_cov, "auto local robust")


def test_local_default_network_fast_smoke(tmp_path, linear_gaussian_problem):
    out = tmp_path / "fast"
    run = _real_problem(linear_gaussian_problem).run_sampling_local_auto(_local_conf(out), budget=500, mode="fast")
    assert run.auto["surrogate"]["class"] == "NeuralNetworkUpdater" and run.auto["surrogate"]["default"]
    assert [s.algorithm for s in run.stages] == ["MH", "DAMH", "DAMH", "DAMH", "DAMH"]
    assert all(isinstance(s.proposal, Hamiltonian) for s in run.stages[1:])
    assert run.conf.use_surrogate_gradients is True
    assert sum(r.counter_accepted for r in run.stage_results[1:]) > 0
    assert _manifest(out)["auto"]["mode"] == "fast"
    # C1: every chunk resolves its mass from the carried posterior statistics (the warm-up has
    # >= 100 states here only if the budget allows; the source is recorded either way)
    resolution = _manifest(out)["proposal_resolution"]
    assert [r["name"] for r in resolution] == [s.name for s in run.stages]
    for entry in resolution[1:]:
        assert entry["mass_source"].startswith(("covariance (n=", "identity (no statistics)")), entry
        assert entry["integration_time"] == 3.0 and entry["integration_time_jitter"] == 0.5  # C1b
    assert resolution[-1]["posterior_statistics_n_start"] > resolution[1]["posterior_statistics_n_start"]


def test_local_time_limit_run(tmp_path, linear_gaussian_problem):
    import time
    out = tmp_path / "timed"
    started = time.monotonic()
    run = _real_problem(linear_gaussian_problem).run_sampling_local_auto(
        _local_conf(out), time_limit=3.0, surrogate_updater=_poly())
    elapsed = time.monotonic() - started
    names = ["alg0000_MH-adaptive"] + [f"alg000{i}_DAMH-SMU" for i in range(1, 5)]
    assert [r.name for r in run.stage_results] == names
    assert run.auto["time_limit"] == 3.0 and run.auto["warm_up"] == {"time_limit": pytest.approx(0.45)}
    assert elapsed < 30.0, elapsed
    assert surrDAMH.TestData.reuse(str(out)).get_size() == 64
