#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2 (2026-10-09): the exact safety kernel and the audit of pre-rejected proposals of a DAMH stage
(``Stage.exact_step_probability``, ``Stage.audit_prerejected``; notes 24 §6 and 22 §4 P1).

Pinned here: both fields at 0 leave every output byte-identical to the code before S2 (up to the
appended ``kernel``/notes columns); the field checks; ``p = 1`` and ``p = 0.5`` sample the
closed-form posterior, with the exact random walk adapting on its own and the stage proposal
not adapting on exact steps; a Hamiltonian DAMH stage with exact steps; the audit leaves the
chain untouched, writes ``audited`` rows, feeds the updater with multiplicity 0, spends
evaluations and counts the hidden-mass statistic; out-of-box / non-finite candidates are never
audited; the two new seeds in the manifest (also shifted in a continuation); the Auto chunks.
The linear-Gaussian problem is the one of ``tests/conftest.py`` (closed-form posterior).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

import surrDAMH
import surrDAMH.post_processing as pp
from helpers_statistics import (FixedInitialSample, assert_matches_closed_form, chain_statistics,
                                read_weighted_samples)
from surrDAMH.auto import AUTO_AUDIT_PREREJECTED, AUTO_EXACT_STEP_PROBABILITY, plan_auto
from surrDAMH.modules.algorithm_interfaces_local import LocalEvaluatorProvider, LocalSolverAdapter
from surrDAMH.modules.algorithms import (AUDIT_HIDDEN_NATS, SOLVER_TAG_OUT_OF_BOUNDS, Algorithm_DAMH,
                                         Sample)
from surrDAMH.modules.proposals import GaussRandomWalk, GaussRandomWalk_adaptive, Proposal
from surrDAMH.modules.seeds import EXACT_PROPOSAL_SEED_OFFSET, EXACT_STEP_SEED_OFFSET, SEED_FORMULA
from surrDAMH.proposals import Hamiltonian, RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates import PolynomialSklearnUpdater
from surrDAMH.surrogates.parent import Evaluator

pytestmark = pytest.mark.unit

A = np.array([[1.0, 0.5], [0.0, 1.0]])
OBSERVED = A @ np.array([0.7, -0.4])
OBS_SD = np.array([0.3, 0.3])
#: a deliberately wrong (but differentiable) linear surrogate, so the DA correction matters
A_SURROGATE = np.array([[1.1, 0.4], [0.05, 0.9]])


class LinearSolver(Solver):
    """``G(x) = A x``; records every call."""

    def __init__(self):
        self.points: list[np.ndarray] = []

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        self.points.append(self.parameters.copy())
        return A @ self.parameters


class LinearGradientEvaluator(Evaluator):
    """Fixed surrogate ``G~(x) = A' x`` with its Jacobian (a degree-1 polynomial with gradients)."""

    def __init__(self, matrix=A_SURROGATE):
        matrix = np.asarray(matrix, dtype=float)
        super().__init__(no_parameters=matrix.shape[1], no_observations=matrix.shape[0])
        self.matrix = matrix

    def __call__(self, datapoints):
        return np.atleast_2d(np.asarray(datapoints, dtype=float)) @ self.matrix.T

    def jacobian(self, datapoints):
        x = np.asarray(datapoints, dtype=float).reshape(-1)
        return self.matrix, self.matrix @ x


def _problem(solver=None, **kwargs):
    prior = surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(2)])
    return surrDAMH.Problem(prior, surrDAMH.distributions.Normal(mean=OBSERVED, sd=OBS_SD),
                            solver=solver if solver is not None else LinearSolver(), **kwargs)


def _conf(output_dir, initial=(0.3, -0.2), **kwargs):
    return surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedInitialSample(initial),
                                  save_snapshots_to_file=True, **kwargs)


def _poly(degree=1):
    return PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=degree)


def _csv(output_dir, kind, stage_name):
    return pd.read_csv(os.path.join(str(output_dir), "sampling_output", kind, stage_name, "rank0000.csv"))


def _posterior():
    """Closed-form posterior of the prior N(0, I) and the likelihood N(A x, diag(OBS_SD^2))."""
    cov = np.linalg.inv(np.eye(2) + A.T @ np.diag(1.0 / OBS_SD ** 2) @ A)
    return cov @ (A.T @ np.diag(1.0 / OBS_SD ** 2) @ OBSERVED), cov


def _check_posterior(output_dir, stage_name, label):
    mean, cov = _posterior()
    statistics = chain_statistics(read_weighted_samples(output_dir, stage_name, 2))
    print(f"\n[{label}] {statistics.no_samples} samples: mean {statistics.mean} vs {mean} "
          f"(dev/SE {(statistics.mean - mean) / statistics.se_mean}), cov {statistics.cov.ravel()} vs {cov.ravel()} "
          f"(dev/SE {((statistics.cov - cov) / statistics.se_cov).ravel()})")
    assert_matches_closed_form(statistics, mean, cov, label)
    return statistics


# ---------------------------------------------------------------------------------------------
# Byte identity: both fields 0 (default or explicit) is the code before S2. Reference md5s made
# with the code BEFORE S2 (2026-10-09, scratchpad s2/s2_ref_run.py, the same four runs as below);
# the columns appended by S2 (subchain_stats "kernel", notes "exact_steps, exact_accepted,
# audited, audit_hidden") are checked to be 0 and stripped before hashing; so is the notes column
# "polyak_updates" appended by S3 (2026-10-09).
# S6 (2026-10-10, adaptive RandomWalk/PCN adapt on the sub-chain acceptance in a DAMH stage)
# changed exactly the 14 "damh_adaptive" keys of its two adaptive DAMH stages (alg0001, alg0002);
# they were re-recorded with the S6 code. Every other key is still the pre-S2 value.
# ---------------------------------------------------------------------------------------------

# C1 (2026-10-11): stages 1 and 2 of "damh_adaptive" are adaptive random walks AFTER an adaptive
# random-walk stage; they now start from the carried posterior statistics (the intended C1 behaviour
# change, CHANGELOG 2026-10-11), so their entries were re-based on the C1 code ("C1 rebase"). Every
# other entry is still the pre-S2 reference, and the C1-only carry_over keys are stripped.
# C1b (2026-10-11): the burn-in discount of the posterior statistics (item B) changes what stages 1 and 2 of
# "damh_adaptive" (after its burn_in=True warm-up) read: their 14 entries were re-based on the C1b code ("C1b
# rebase"); the new carry_over key summary__posterior_statistics_discounted_from_n is stripped like the C1 keys.
# n_seen (author decision 1, 2026-10-11): the shrinkage weight of the statistics after the burn-in discount is
# 2d/n_seen (undiscounted) instead of 2d/n, which changes the shape stages 1 and 2 of "damh_adaptive" install:
# their 14 entries were re-based again ("n_seen rebase"); the new key carry_over__posterior_statistics_n_seen is
# stripped like the C1 keys.
REFERENCE_MD5 = {
    "damh/last_sample/alg0000_MH/rank0000.npz": "f1229e06b00e628437c1a21dfa0caf3b",
    "damh/last_sample/alg0001_DAMH-SMU/rank0000.npz": "61ef32141102f63146b1d99e72983bcc",
    "damh/notes/alg0000_MH/rank0000.csv": "00e4bb998d5639a0dcd028e4062b970e",
    "damh/notes/alg0001_DAMH-SMU/rank0000.csv": "1bcd9d5d13ff2ef14b4e6f38458a27aa",
    "damh/raw_data/alg0000_MH/rank0000.csv": "0be185d92f3c413c7ae3dbb64b67d852",
    "damh/raw_data/alg0001_DAMH-SMU/rank0000.csv": "f1c37b61545dfa1e9334bbc84c598d93",
    "damh/samples/alg0000_MH/rank0000.csv": "65eeb706d0921eb2611191dc84660b68",
    "damh/samples/alg0001_DAMH-SMU/rank0000.csv": "7fe8a3e7bbb9b9f9294a8a6f478eeee3",
    "damh/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "866b02c29dce63dced0208d9abd862d5",
    "damh_adaptive/adaptive_stats/alg0000_MH-adaptive/rank0000.csv": "47106064c2ff0263b863050189ce5493",
    "damh_adaptive/adaptive_stats/alg0001_DAMH-SMU/rank0000.csv": "5fc6ad71bde01f4534688b1e011008f0",  # n_seen rebase (decision 1, was ab5f6c445c4b76c44a17d5248bf96e33)  # C1b rebase (burn-in discount)  # C1 rebase (was e8d35e8e6c18f8a5d400eaebe5e73696)
    "damh_adaptive/adaptive_stats/alg0002_DAMH/rank0000.csv": "9da13e82bd5371a9b8bc1599b233cca9",  # n_seen rebase (decision 1, was 1cfd862b509c5a29bf487b1a61bb3a37)  # C1b rebase (burn-in discount)  # C1 rebase (was 30dad749a81dfe23fe771c774d0e9da0)
    "damh_adaptive/carry_over/alg0000_MH-adaptive.npz": "5e870c5e37665e57b333945b588e9e7e",
    "damh_adaptive/carry_over/alg0001_DAMH-SMU.npz": "d6b3266b1a457c6c98b7706f5fc0e1e1",  # n_seen rebase (decision 1, was d4a6f010871c62f26ef5395567f1a1ec)  # C1b rebase (burn-in discount)  # C1 rebase (was 79f603bf8f522e47cad90fc9c077e476)
    "damh_adaptive/carry_over/alg0002_DAMH.npz": "fd305452ff7c5eed0247552b56434480",  # n_seen rebase (decision 1, was ed7293b0e80ceb38570dc55837206826)  # C1b rebase (burn-in discount)  # C1 rebase (was 7f16091e4e4bb0bbad50afde2533271a)
    "damh_adaptive/last_sample/alg0000_MH-adaptive/rank0000.npz": "e448e0813c0abb6e16b296d5d2946721",
    "damh_adaptive/last_sample/alg0001_DAMH-SMU/rank0000.npz": "60d88710f12b824f0264e07c6e217d1a",  # n_seen rebase (decision 1, was feeacf4221055a811631bf218f8f164c)  # C1b rebase (burn-in discount)  # C1 rebase (was 48bee970e3e0172ad910ce894bc00e9f)
    "damh_adaptive/last_sample/alg0002_DAMH/rank0000.npz": "d51efa447124d11cec5f5121659b0d8e",  # n_seen rebase (decision 1, was 1c8463fec90f6fd7b9a7ed868ba203f5)  # C1b rebase (burn-in discount)  # C1 rebase (was 708c4965199ab4062bb9bdb9d32858a3)
    "damh_adaptive/notes/alg0000_MH-adaptive/rank0000.csv": "61af274dab3bc7f15b5276fdcfc40829",
    "damh_adaptive/notes/alg0001_DAMH-SMU/rank0000.csv": "4a7144b58a0f67e3a1ca597613ce4ed5",  # n_seen rebase (decision 1, was 36ccaf35282642c1fa4b74ad759c8c73)  # C1b rebase (burn-in discount)  # C1 rebase (was 2d86a7d64f769b2675aa534b81288604)
    "damh_adaptive/notes/alg0002_DAMH/rank0000.csv": "4cf3958b345735504ace50a091080463",  # n_seen rebase (decision 1, was e70a63dd329f3113d79bfae738d6a93c)  # C1b rebase (burn-in discount)  # C1 rebase (was 0def45bb1f3f26391ab0ae4ab82a2041)
    "damh_adaptive/raw_data/alg0000_MH-adaptive/rank0000.csv": "19910ea3b25cbfb836d7077191de4a22",
    "damh_adaptive/raw_data/alg0001_DAMH-SMU/rank0000.csv": "e33e2eae0ce10e2665a6c9045bf62525",  # n_seen rebase (decision 1, was d0a4e5788cd7888b3d78874b93791c8e)  # C1b rebase (burn-in discount)  # C1 rebase (was ce6ed2d854ef3efa2f12747741f534cb)
    "damh_adaptive/raw_data/alg0002_DAMH/rank0000.csv": "691f91933f966badc563c3a4004e1676",  # n_seen rebase (decision 1, was 66460b6770ed033f2a7b54d458eccb31)  # C1b rebase (burn-in discount)  # C1 rebase (was 2a13869f6beb9ecaa3d103aeaaae4962)
    "damh_adaptive/samples/alg0000_MH-adaptive/rank0000.csv": "b4ed29b7b7cb2504132bd33cbec6e0e3",
    "damh_adaptive/samples/alg0001_DAMH-SMU/rank0000.csv": "1f92156986edcc3a78d787408b9033d2",  # n_seen rebase (decision 1, was 19d975ef4a87339f71286c158a90311f)  # C1b rebase (burn-in discount)  # C1 rebase (was cc6d94576350e48bc189c7abd3ef6db1)
    "damh_adaptive/samples/alg0002_DAMH/rank0000.csv": "e512aed03573cc328697f3535b3e14cc",  # n_seen rebase (decision 1, was 53be257de2030f1938de00ba96a90a70)  # C1b rebase (burn-in discount)  # C1 rebase (was a744ae84f03dacb716fb34e7accd30a0)
    "damh_adaptive/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "f23fd2efc1f95bcaf038035afa04c4d8",  # n_seen rebase (decision 1, was c2129e493a3c4f81c02841016028c67d)  # C1b rebase (burn-in discount)  # C1 rebase (was ee215fd0070bbe7eec73929f2414e516)
    "damh_adaptive/subchain_stats/alg0002_DAMH/rank0000.csv": "5487bcafc68a43480b8da618013d48cf",  # n_seen rebase (decision 1, was 93176559d1e4886bd3b18a578c353732)  # C1b rebase (burn-in discount)  # C1 rebase (was ead9badfecd28527a39c6c8037b054a3)
    "hmc/last_sample/alg0000_MH/rank0000.npz": "6db859899035d40acc5e07afadb99b94",
    "hmc/last_sample/alg0001_DAMH-SMU/rank0000.npz": "40360614accf84dd31b59e00c14defba",
    "hmc/notes/alg0000_MH/rank0000.csv": "d06d3d821358b86980f679a7839cf90c",
    "hmc/notes/alg0001_DAMH-SMU/rank0000.csv": "62e63dd47bb98f682b062ba97490a9f2",
    "hmc/raw_data/alg0000_MH/rank0000.csv": "07edc84401bde7a24406a6794b73fe05",
    "hmc/raw_data/alg0001_DAMH-SMU/rank0000.csv": "e5ce0686d11fb9e3b6a187489d94f22b",
    "hmc/samples/alg0000_MH/rank0000.csv": "39f8b2776ee8d57196885c92ee76e743",
    "hmc/samples/alg0001_DAMH-SMU/rank0000.csv": "01fdd10546dbcc4fb2f876ba313ab50e",
    "hmc/subchain_stats/alg0001_DAMH-SMU/rank0000.csv": "6703f4c9e73e2b305c55b24085a7ffca",
    "mh/last_sample/alg0000_MH/rank0000.npz": "528a6c4d6e666347e0daa81b14f527a6",
    "mh/notes/alg0000_MH/rank0000.csv": "06f60dfa65f5b739f1a1402d750a07e4",
    "mh/raw_data/alg0000_MH/rank0000.csv": "75b8dc108e5ad30dd5f79d015d2d715c",
    "mh/samples/alg0000_MH/rank0000.csv": "5af1d512c2cd5732b1b3aece573ff8f6",
}

S2_NOTES = [b"exact_steps", b"exact_accepted", b"audited", b"audit_hidden"]


#: carry_over keys added by C1 (2026-10-11), C1b (the burn-in discount record) and the n_seen fix
#: (2026-10-11): stripped before hashing, a file holding nothing else is skipped
C1_CARRY_OVER_KEYS = {"carry_over__posterior_statistics", "summary__posterior_statistics_n",
                      "carry_over__dual_averaging", "carry_over__hamiltonian_mass",
                      "summary__posterior_statistics_discounted_from_n",
                      "carry_over__posterior_statistics_n_seen"}


def _stripped_md5(path, kind):
    if path.endswith(".npz"):
        with np.load(path) as z:
            keys = sorted(k for k in z.files if k not in C1_CARRY_OVER_KEYS)
            if not keys:
                return None
            return hashlib.md5(b"".join(np.ascontiguousarray(z[k]).tobytes() for k in keys)).hexdigest()
    lines = open(path, "rb").read().split(b"\r\n")
    header = lines[0].split(b",")
    if kind == "subchain_stats":
        assert header[-1] == b"kernel" and all(line.rsplit(b",", 1)[1] == b"0" for line in lines[1:] if line)
        lines = [line.rsplit(b",", 1)[0] if line else line for line in lines]
    if kind == "notes" and header[-1] == b"polyak_updates":  # S3
        assert all(line.rsplit(b",", 1)[1] == b"0" for line in lines[1:] if line)
        lines = [line.rsplit(b",", 1)[0] if line else line for line in lines]
        header = header[:-1]
    if kind == "notes":
        assert header[-4:] == S2_NOTES and all(line.split(b",")[-4:] == [b"0"] * 4 for line in lines[1:] if line)
        lines = [b",".join(line.split(b",")[:-4]) if line else line for line in lines]
    return hashlib.md5(b"\r\n".join(lines)).hexdigest()


def _reference_runs(root, s2_kwargs):
    """The four runs of scratchpad s2/s2_ref_run.py (same order, configurations and seeds)."""
    def problem():
        return _problem()

    problem().run_sampling_local(_conf(root / "mh"), [Stage(algorithm="MH", proposal=RandomWalk(scale=0.8),
                                                            max_evaluations=2000)])
    problem().run_sampling_local(_conf(root / "damh"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.8), max_evaluations=200),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=300, subchain_length=5, **s2_kwargs)],
        surrogate_updater=_poly(2))
    problem().run_sampling_local(_conf(root / "damh_adaptive"), [
        Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=300, burn_in=True),
        Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=400, subchain_length=1, **s2_kwargs),
        Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=300, subchain_length=1,
              surrogate_model_updates=False, **s2_kwargs)],
        surrogate_updater=_poly(1))
    problem().run_sampling_local(_conf(root / "hmc"), [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=100),
        Stage(algorithm="DAMH", proposal=Hamiltonian(step_size=0.2, num_steps=5), max_evaluations=300,
              subchain_length=1, **s2_kwargs)],
        surrogate_evaluator=LinearGradientEvaluator())


@pytest.mark.parametrize("s2_kwargs", [{}, {"exact_step_probability": 0.0, "audit_prerejected": 0.0}],
                         ids=["defaults", "explicit_zero"])
def test_fields_at_zero_are_byte_identical_to_the_code_before_s2(tmp_path, s2_kwargs):
    _reference_runs(tmp_path, s2_kwargs)
    produced = {}
    for kind in ("samples", "raw_data", "subchain_stats", "notes", "adaptive_stats", "last_sample"):
        for path in glob.glob(os.path.join(str(tmp_path), "*", "sampling_output", kind, "*", "rank*.*")):
            key = os.path.relpath(path, str(tmp_path)).replace("/sampling_output", "")
            produced[key] = _stripped_md5(path, kind)
    for path in glob.glob(os.path.join(str(tmp_path), "*", "sampling_output", "carry_over", "*.npz")):
        digest = _stripped_md5(path, "carry_over")
        if digest is not None:  # C1: every stage writes a carry_over file; skip the C1-only ones
            produced[os.path.relpath(path, str(tmp_path)).replace("/sampling_output", "")] = digest
    assert set(produced) == set(REFERENCE_MD5)
    changed = sorted(key for key, expected in REFERENCE_MD5.items() if produced[key] != expected)
    assert not changed, f"changed against the code before S2: {changed}"


# ---------------------------------------------------------------------------------------------
# Field checks
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["exact_step_probability", "audit_prerejected"])
@pytest.mark.parametrize("value", [-0.1, 1.5, np.nan, np.inf, True, "0.1", None])
def test_values_outside_the_unit_interval_raise(name, value):
    with pytest.raises(ValueError, match=rf"Stage.{name} must be a number in \[0, 1\]"):
        Stage(algorithm="DAMH", max_evaluations=10, **{name: value})


def test_fields_are_floats_and_marked_posterior_affecting():
    stage = Stage(algorithm="DAMH", max_evaluations=10, exact_step_probability=1, audit_prerejected=np.float32(0.25))
    assert stage.exact_step_probability == 1.0 and isinstance(stage.exact_step_probability, float)
    assert stage.audit_prerejected == 0.25 and isinstance(stage.audit_prerejected, float)
    text = stage.describe()
    assert "exact_step_probability*=1.0" in text and "audit_prerejected*=0.25" in text


def test_mh_stage_resets_with_a_warning(capsys):
    stage = Stage(algorithm="MH", max_evaluations=10, exact_step_probability=0.3, audit_prerejected=0.2)
    assert stage.exact_step_probability == 0.0 and stage.audit_prerejected == 0.0
    out = capsys.readouterr().out
    assert "Warning: exact_step_probability is only used by a DAMH stage" in out
    assert "Warning: audit_prerejected is only used by a DAMH stage" in out
    Stage(algorithm="MH", max_evaluations=10)  # zero values: no warning
    assert "Warning" not in capsys.readouterr().out


def test_frozen_stage_keeps_the_audit_with_a_note():
    stage = Stage(algorithm="DAMH", max_evaluations=10, surrogate_model_updates=False, audit_prerejected=0.5)
    assert stage.audit_prerejected == 0.5
    assert "audit only measures the surrogate here" in stage.describe()
    assert "audit only" not in Stage(algorithm="DAMH", max_evaluations=10, audit_prerejected=0.5).describe()
    quiet = Stage(algorithm="DAMH", max_evaluations=10, audit_prerejected=0.5, send_snapshots_to_collector=False)
    assert "not sent to the collector" in quiet.describe()


def test_damh_without_an_exact_proposal_refuses_to_run(tmp_path):
    problem = _problem()
    conf = _conf(tmp_path)
    conf.resolve_problem_sizes(2, 2)
    stage = Stage(algorithm="DAMH", max_evaluations=5, exact_step_probability=0.5, name="direct")
    algo = Algorithm_DAMH(stage=stage, proposal=GaussRandomWalk(2, seed=1), initial_sample=Sample(np.zeros(2)),
                          rank_world=0, conf=conf, prior=problem.prior, likelihood=problem.likelihood,
                          observation_provider=LocalSolverAdapter(LinearSolver()),
                          evaluator_provider=LocalEvaluatorProvider(LinearGradientEvaluator()), seed=2)
    with pytest.raises(ValueError, match="needs the exact step's proposal"):
        algo.run()


# ---------------------------------------------------------------------------------------------
# The exact step
# ---------------------------------------------------------------------------------------------

def _exact_run(tmp_path, probability, evaluations, name="run"):
    out = tmp_path / name
    problem = _problem()
    run = problem.run_sampling_local(
        _conf(out, min_snapshots_to_update=500),
        [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=500, burn_in=True),
         Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=evaluations,
               exact_step_probability=probability)],
        surrogate_updater=_poly(1))
    return out, run.stage_results[1]


def test_exact_step_probability_one(tmp_path):
    out, result = _exact_run(tmp_path, 1.0, 20_000)
    stats = _csv(out, "subchain_stats", result.name)
    assert len(stats) == result.counter_exact_steps == 20_000
    assert set(stats["kernel"]) == {1} and result.counter_prerejected == 0
    assert (stats["subchain_accepted"] == 0).all() and stats["subchain_acceptance_rate"].isna().all()
    assert (stats["correction_log_ratio"] == 0).all() and (stats["outer_proposed_changed"] == 1).all()
    assert int(stats["outer_accepted"].sum()) == result.counter_exact_accepted == result.counter_accepted
    raw = _csv(out, "raw_data", result.name)
    assert set(raw["state_type"]) == {"accepted", "rejected"} and len(raw) == 20_000
    # BOTH observation blocks are filled on every exact-step row (surrogate scored first)
    assert raw[["obs_0", "obs_1", "obs_approx_0", "obs_approx_1"]].notna().all().all()
    notes = _csv(out, "notes", result.name)
    assert int(notes["exact_steps"].iloc[0]) == 20_000
    assert int(notes["exact_accepted"].iloc[0]) == result.counter_exact_accepted
    # the exact random walk adapts towards 0.234 on its own acceptances
    assert 0.15 < result.counter_exact_accepted / 20_000 < 0.35
    # the stage proposal never adapted: no adaptive_stats file for the stage
    assert not os.path.exists(os.path.join(str(out), "sampling_output", "adaptive_stats", result.name))
    _check_posterior(out, result.name, "S2 exact step p=1")


def test_exact_step_probability_half(tmp_path):
    out, result = _exact_run(tmp_path, 0.5, 20_000)
    stats = _csv(out, "subchain_stats", result.name)
    exact = stats["kernel"] == 1
    assert int(exact.sum()) == result.counter_exact_steps > 0 and int((~exact).sum()) > 0
    assert int(stats.loc[exact, "outer_accepted"].sum()) == result.counter_exact_accepted
    # evaluations: every exact step and every moved DA iteration
    assert result.counter_accepted + result.counter_rejected == 20_000
    assert int(exact.sum() + stats.loc[~exact, "outer_proposed_changed"].sum()) == 20_000
    # 1:1 rows: kernel rows and raw_data rows (no audits here)
    assert len(_csv(out, "raw_data", result.name)) == len(stats)
    # the stage proposal adapted on the DA iterations only: GaussRandomWalk_adaptive writes one
    # adaptive_stats row per 10 adapt() calls
    adaptive = _csv(out, "adaptive_stats", result.name)
    assert len(adaptive) == int((~exact).sum()) // 10
    assert int(adaptive["n"].iloc[-1]) == 10 * len(adaptive)
    _check_posterior(out, result.name, "S2 exact step p=0.5")


class _CountingRandomWalk(GaussRandomWalk):
    """Fixed random walk that counts its adapt() calls (a stand-in stage proposal)."""

    def __init__(self):
        super().__init__(no_parameters=2, seed=11)
        self.set_covariance(0.5)
        self.adapt_calls = 0

    def adapt(self, *args, **kwargs):
        self.adapt_calls += 1


def test_exact_proposal_adapts_and_the_stage_proposal_only_on_da_steps(tmp_path):
    problem = _problem()
    conf = _conf(tmp_path)
    conf.resolve_problem_sizes(2, 2)
    stage_proposal = _CountingRandomWalk()
    exact_proposal = GaussRandomWalk_adaptive(no_parameters=2, sd_or_cov=0.05, seed=5)
    initial_cov = exact_proposal.sd_or_cov.copy()
    stage = Stage(algorithm="DAMH", max_evaluations=3_000, exact_step_probability=0.5, name="direct")
    algo = Algorithm_DAMH(stage=stage, proposal=stage_proposal, initial_sample=Sample(np.array([0.3, -0.2])),
                          rank_world=0, conf=conf, prior=problem.prior, likelihood=problem.likelihood,
                          observation_provider=LocalSolverAdapter(LinearSolver()),
                          evaluator_provider=LocalEvaluatorProvider(LinearGradientEvaluator()), seed=2,
                          exact_step_seed=4, exact_proposal=exact_proposal)
    algo.run()
    stats = _csv(tmp_path, "subchain_stats", "direct")
    assert stage_proposal.adapt_calls == int((stats["kernel"] == 0).sum()) > 0
    assert exact_proposal.n == int((stats["kernel"] == 1).sum()) == algo.counter_exact_steps > 0
    assert not np.allclose(exact_proposal.sd_or_cov, initial_cov)  # its scale moved away from sd 0.05
    np.testing.assert_array_equal(stage_proposal.sd_or_cov, [0.5, 0.5])  # fixed, untouched
    # the chain state carries the surrogate terms of the installed evaluator (note 24 §4.4)
    np.testing.assert_allclose(algo.current.observations_approx,
                               LinearGradientEvaluator()(algo.current.parameters[None, :])[0])


def test_hamiltonian_damh_stage_with_exact_steps(tmp_path):
    def run(name, probability):
        out = tmp_path / name
        stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=300, burn_in=True),
                  Stage(algorithm="DAMH", proposal=Hamiltonian(step_size=0.2, num_steps=5), max_evaluations=15_000,
                        exact_step_probability=probability)]
        result = _problem().run_sampling_local(_conf(out), stages,
                                               surrogate_evaluator=LinearGradientEvaluator()).stage_results[1]
        return out, result

    out, result = run("p03", 0.3)
    stats = _csv(out, "subchain_stats", result.name)
    exact = stats["kernel"] == 1
    assert int(exact.sum()) == result.counter_exact_steps > 0
    _check_posterior(out, result.name, "S2 Hamiltonian DAMH + exact steps p=0.3")
    # smoke: the HMC (kernel 0) iterations behave as in a run without exact steps
    out0, result0 = run("p0", 0.0)
    stats0 = _csv(out0, "subchain_stats", result0.name)
    hmc_rate = stats.loc[~exact, "outer_accepted"].mean()
    hmc_rate0 = stats0["outer_accepted"].mean()
    print(f"\nHMC outer acceptance per DA iteration: {hmc_rate:.3f} (with exact steps) vs {hmc_rate0:.3f} (without)")
    assert abs(hmc_rate - hmc_rate0) < 0.05


# ---------------------------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------------------------

def _audit_stages(audit, **damh_kwargs):
    return [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=300),
            Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), surrogate_model_updates=False,
                  audit_prerejected=audit, **damh_kwargs)]


def test_audit_does_not_change_the_chain_and_reaches_the_updater(tmp_path):
    runs = {}
    for audit in (0.0, 1.0):
        out = tmp_path / f"audit{audit:g}"
        solver = LinearSolver()
        updater = _poly(2)
        run = _problem(solver).run_sampling_local(_conf(out, min_snapshots_to_update=10_000),
                                                  _audit_stages(audit, max_samples=1_500),
                                                  surrogate_updater=updater)
        runs[audit] = (out, run.stage_results[1], updater, solver)
    (out0, r0, updater0, solver0), (out1, r1, updater1, solver1) = runs[0.0], runs[1.0]
    name = r1.name
    assert name == "alg0001_DAMH"
    for kind in ("samples", "subchain_stats"):
        for stage in ("alg0000_MH", name):
            if kind == "subchain_stats" and stage == "alg0000_MH":
                continue
            a = open(os.path.join(str(out0), "sampling_output", kind, stage, "rank0000.csv"), "rb").read()
            b = open(os.path.join(str(out1), "sampling_output", kind, stage, "rank0000.csv"), "rb").read()
            assert a == b, f"{kind}/{stage}: the audit changed the chain"
    assert (r0.counter_accepted, r0.counter_rejected, r0.counter_prerejected) == \
        (r1.counter_accepted, r1.counter_rejected, r1.counter_prerejected)
    assert r0.counter_audited == 0 and r1.counter_audited == r1.counter_prerejected > 0

    raw = _csv(out1, "raw_data", name)
    types = raw["state_type"].tolist()
    # every pre-rejected iteration (K = 1, all inside the box) is followed by its audited row
    for i, state_type in enumerate(types):
        if state_type == "prerejected":
            assert types[i + 1] == "audited", i
    audited = raw[raw["state_type"] == "audited"]
    assert len(audited) == r1.counter_audited
    assert audited[["obs_0", "obs_1", "obs_approx_0", "obs_approx_1"]].notna().all().all()
    np.testing.assert_allclose(audited[["obs_0", "obs_1"]].to_numpy(),
                               audited[["par_0", "par_1"]].to_numpy() @ A.T)
    likelihood = _problem().likelihood
    exact_ll = [likelihood.logpdf(row) for row in audited[["obs_0", "obs_1"]].to_numpy()]
    np.testing.assert_allclose(audited["log_likelihood"].to_numpy(), exact_ll)
    # the audited point is the pre-rejected PROPOSAL, not the chain state the prerejected row repeats
    prerejected = raw[raw["state_type"] == "prerejected"]
    assert not np.allclose(audited[["par_0", "par_1"]].to_numpy(), prerejected[["par_0", "par_1"]].to_numpy())
    # solver calls: one per audit more
    assert len(solver1.points) == len(solver0.points) + r1.counter_audited
    # the updater got every audited point once, with multiplicity 0
    assert updater1.par.shape[0] == updater0.par.shape[0] + r1.counter_audited
    assert int(np.sum(updater1.mul == 0)) == int(np.sum(updater0.mul == 0)) + r1.counter_audited
    # notes and the summary table
    notes = _csv(out1, "notes", name)
    assert int(notes["audited"].iloc[0]) == r1.counter_audited
    summary = pp.Samples(2, str(out1), selection=False).get_summary()
    assert int(summary["audited"].iloc[1]) == r1.counter_audited
    assert int(summary["exact_steps"].iloc[1]) == 0


def test_audits_spend_the_evaluation_budget(tmp_path):
    out = tmp_path / "budget"
    run = _problem().run_sampling_local(_conf(out), _audit_stages(1.0, max_evaluations=400),
                                        surrogate_updater=_poly(2))
    result = run.stage_results[1]
    assert result.counter_audited > 0
    assert result.counter_accepted + result.counter_rejected + result.counter_audited == 400
    assert result.counter_accepted + result.counter_rejected < 400


def test_hidden_counter_matches_raw_data(tmp_path):
    # a pessimistic fixed surrogate: large parts of the posterior look improbable to it
    bad = LinearGradientEvaluator(np.array([[1.6, 0.9], [-0.4, 0.5]]))
    out = tmp_path / "hidden"
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=200),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.8), max_evaluations=2_000, audit_prerejected=1.0)]
    result = _problem().run_sampling_local(_conf(out), stages, surrogate_evaluator=bad).stage_results[1]
    raw = _csv(out, "raw_data", result.name)
    audited = raw[raw["state_type"] == "audited"]
    likelihood = _problem().likelihood
    approx_ll = np.array([likelihood.logpdf(row) for row in audited[["obs_approx_0", "obs_approx_1"]].to_numpy()])
    gap = audited["log_likelihood"].to_numpy() - approx_ll
    hidden = int(np.sum(gap > AUDIT_HIDDEN_NATS))
    print(f"\naudited {len(audited)}, hidden {hidden}, max gap {gap.max():.1f} nats")
    assert hidden == result.counter_audit_hidden > 0
    assert int(_csv(out, "notes", result.name)["audit_hidden"].iloc[0]) == hidden


def test_out_of_box_candidates_are_never_audited(tmp_path):
    solver = LinearSolver()
    out = tmp_path / "box"
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.3), max_evaluations=100),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=3.0), max_evaluations=300, audit_prerejected=1.0)]
    result = _problem(solver, prior_bound=0.8).run_sampling_local(
        _conf(out), stages, surrogate_evaluator=LinearGradientEvaluator()).stage_results[1]
    raw = _csv(out, "raw_data", result.name)
    out_of_box = int((raw["solver_tag"] == SOLVER_TAG_OUT_OF_BOUNDS).sum())
    prerejected_iterations = int(((raw["state_type"] == "prerejected") & (raw["solver_tag"] == 0)).sum())
    assert out_of_box > 0 and result.counter_audited > 0
    # K = 1: an iteration whose only step was outside the box has nothing to audit
    assert result.counter_audited == prerejected_iterations - out_of_box
    audited = raw[raw["state_type"] == "audited"][["par_0", "par_1"]].to_numpy()
    assert np.all(np.abs(audited) <= 0.8)
    assert np.all(np.abs(np.array(solver.points)) <= 0.8)


class _SequenceProposal(Proposal):
    """Proposes a fixed sequence (ignores the current state); plain MH ratio."""

    def __init__(self, samples):
        super().__init__()
        self.no_parameters = 2
        self._generator = np.random.RandomState(0)
        self.samples = [np.asarray(s, dtype=float) for s in samples]
        self.calls = 0

    def propose_sample(self, current_sample):
        sample = self.samples[self.calls % len(self.samples)]
        self.calls += 1
        return sample.copy()

    def get_log_acceptance_probability(self, log_likelihood_proposed, log_likelihood_current,
                                       log_prior_proposed, log_prior_current):
        return log_likelihood_proposed - log_likelihood_current, log_prior_proposed - log_prior_current


def test_non_finite_and_out_of_box_sub_chains_have_nothing_to_audit(tmp_path):
    problem = _problem(prior_bound=1.0)
    conf = _conf(tmp_path)
    conf.resolve_problem_sizes(2, 2)
    solver = LinearSolver()
    # iteration 1: NaN then out of box (nothing auditable); iteration 2: out of box, then a point
    # inside the box the surrogate rejects with probability ~1 - e^-13 (audited)
    proposal = _SequenceProposal([[np.nan, np.nan], [5.0, 5.0], [5.0, -5.0], [0.9, 0.9]])
    stage = Stage(algorithm="DAMH", max_samples=2, subchain_length=2, audit_prerejected=1.0, name="direct")
    algo = Algorithm_DAMH(stage=stage, proposal=proposal, initial_sample=Sample(np.array([0.7, -0.4])),
                          rank_world=0, conf=conf, prior=problem.prior, likelihood=problem.likelihood,
                          observation_provider=LocalSolverAdapter(solver),
                          evaluator_provider=LocalEvaluatorProvider(LinearGradientEvaluator(A)), seed=2)
    algo.run()
    assert algo.counter_prerejected == 2 and algo.counter_audited == 1
    assert algo.counter_nonfinite_proposals == 1 and algo.counter_out_of_bounds == 2
    # one solver call for the initial state and one for the audit of [0.9, 0.9]
    assert len(solver.points) == 2
    np.testing.assert_array_equal(solver.points[1], [0.9, 0.9])


# ---------------------------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------------------------

def _manifest(output_dir):
    with open(os.path.join(str(output_dir), "sampling_output", "run_manifest.json")) as f:
        return json.load(f)


def test_manifest_seeds_include_the_two_new_streams_and_a_continuation_shifts_them(tmp_path):
    assert (EXACT_STEP_SEED_OFFSET, EXACT_PROPOSAL_SEED_OFFSET) == (4, 5)
    assert "exact_step_seed = seed0+4" in SEED_FORMULA and "exact_proposal_seed = seed0+5" in SEED_FORMULA
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=50),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=30, exact_step_probability=0.5,
                    audit_prerejected=0.5)]
    run = _problem().run_sampling_local(_conf(tmp_path / "a"), stages, surrogate_updater=_poly(1))
    seeds = _manifest(tmp_path / "a")["seeds"]
    assert seeds["formula"] == SEED_FORMULA
    for entry in seeds["per_rank"]:
        assert entry["exact_step_seed"] == entry["seed0"] + 4
        assert entry["exact_proposal_seed"] == entry["seed0"] + 5
    assert [e["exact_step_seed"] for e in seeds["per_rank"]] == [4, 14]
    # a continuation (generation 1) shifts them with seed0
    run.continue_sampling_local(surrDAMH.Configuration(str(tmp_path / "b"), use_collector=False,
                                                       use_solvers_pool=False),
                                [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=20,
                                       exact_step_probability=0.5)],
                                surrogate_evaluator=LinearGradientEvaluator())
    continued = _manifest(tmp_path / "b")["seeds"]["per_rank"]
    assert continued[0]["seed0"] >= 1_000_000
    assert continued[0]["exact_step_seed"] == continued[0]["seed0"] + 4
    assert continued[0]["exact_proposal_seed"] == continued[0]["seed0"] + 5


def test_the_runner_passes_the_exact_step_seed(tmp_path, monkeypatch):
    """The algorithm's S2 stream is seeded with seed0 + 4 and the exact proposal with seed0 + 5."""
    import surrDAMH.runner_local as runner_local
    seen = {}
    original_init = Algorithm_DAMH.__init__

    def recording_init(self, *args, **kwargs):
        seen["exact_step_seed"] = kwargs.get("exact_step_seed")
        seen["exact_proposal"] = kwargs.get("exact_proposal")
        seen["seed"] = kwargs.get("seed")
        original_init(self, *args, **kwargs)

    built = {}
    original_builder = runner_local.build_exact_step_proposal

    def recording_builder(*args, **kwargs):
        built["seed"] = kwargs["seed"]
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(Algorithm_DAMH, "__init__", recording_init)
    monkeypatch.setattr(runner_local, "build_exact_step_proposal", recording_builder)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=50),
              Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=20, exact_step_probability=0.5)]
    _problem().run_sampling_local(_conf(tmp_path), stages, surrogate_updater=_poly(1))
    # stage 1 of 2 stages, rank 0: seed0 = 10
    assert seen["seed"] == 12 and seen["exact_step_seed"] == 14 and built["seed"] == 15
    assert isinstance(seen["exact_proposal"], GaussRandomWalk_adaptive)


# ---------------------------------------------------------------------------------------------
# Auto
# ---------------------------------------------------------------------------------------------

def test_auto_plan_sets_both_fields_on_the_chunks(tmp_path):
    from types import SimpleNamespace
    conf = surrDAMH.Configuration(output_dir=str(tmp_path / "o"), use_collector=False, use_solvers_pool=False)
    plan = plan_auto(SimpleNamespace(no_parameters=2, no_observations=2, solver_instance=None), conf,
                     budget=5_000, time_limit=None, mode="robust", local=True, surrogate_updater=_poly(1))
    warm_up, chunks = plan.stages[0], plan.stages[1:]
    assert (warm_up.exact_step_probability, warm_up.audit_prerejected) == (0.0, 0.0)
    assert len(chunks) == 4
    for stage in chunks:
        assert stage.exact_step_probability == AUTO_EXACT_STEP_PROBABILITY == 0.05
        assert stage.audit_prerejected == AUTO_AUDIT_PREREJECTED == 0.05
    entry = plan.manifest_entry()
    assert entry["exact_step_probability"] == 0.05 and entry["audit_prerejected"] == 0.05


def test_auto_local_run_has_kernel_rows_of_both_kinds(tmp_path):
    out = tmp_path / "auto"
    conf = surrDAMH.Configuration(output_dir=str(out), use_collector=False, use_solvers_pool=False,
                                  save_snapshots_to_file=True)
    run = _problem().run_sampling_local_auto(conf, budget=4_000, mode="robust", surrogate_updater=_poly(1))
    kernels = set()
    for result in run.stage_results[1:]:
        kernels |= set(_csv(out, "subchain_stats", result.name)["kernel"])
        assert result.counter_exact_steps > 0
    assert kernels == {0, 1}
    assert sum(r.counter_audited for r in run.stage_results[1:]) > 0
    manifest_stages = _manifest(out)["stages"]
    assert [s["exact_step_probability"] for s in manifest_stages] == [0.0] + [0.05] * 4
