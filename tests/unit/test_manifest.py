#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the run manifest (``surrDAMH.modules.manifest``), WS4
(``library_notes/09_improvement_plan.md`` §0 principle 4 and WS4 section).

Uses ``Problem.run_sampling_local`` (see ``tests/test_runner_local.py`` for the pattern) rather
than MPI, since ``manifest.py`` is MPI-free and the local runner is the fast, deterministic single-chain path.
"""

import json

import numpy as np
import pytest

import surrDAMH
from surrDAMH.modules.manifest import build_run_manifest, write_run_manifest
from surrDAMH.proposals import Block, RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage

pytestmark = pytest.mark.unit


class ToySolver(Solver):
    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters):
        self.x, self.y = parameters

    def get_observations(self):
        return np.array([(self.x ** 2 - self.y) * (np.log((self.x - self.y) ** 2 + 1))])


class FixedSample:
    """Degenerate 'distribution' returning a fixed initial sample (see tests/test_runner_local.py)."""

    def __init__(self, parameters):
        self.parameters = np.array(parameters, dtype=float)

    def rvs(self):
        return self.parameters.copy()


def _make_problem(output_dir):
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(output_dir),
                                  use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedSample([0.3, -0.7]))
    prior = surrDAMH.distributions.PriorIndependentComponents([
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    ])
    likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=1.0)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20)]
    return conf, prior, likelihood, stages


def _manifest_path(output_dir):
    return output_dir / "sampling_output" / "run_manifest.json"


def _read_manifest(output_dir):
    with open(_manifest_path(output_dir)) as f:
        return json.load(f)


def test_manifest_written_by_run_local(tmp_path):
    conf, prior, likelihood, stages = _make_problem(tmp_path)
    surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(conf, stages)

    path = _manifest_path(tmp_path)
    assert path.is_file()
    manifest = _read_manifest(tmp_path)

    for key in ("manifest_version", "format_version", "surrdamh_version", "runner", "created_at",
               "hostname", "python_version", "package_versions", "git", "configuration",
               "stages", "prior", "likelihood", "surrogate", "solver", "seeds", "mpi",
               "environment", "unverified_options", "continued_from", "finished_at"):
        assert key in manifest, f"missing key {key!r}"

    assert manifest["runner"] == "local"
    assert manifest["mpi"] is None
    assert manifest["format_version"] == 2
    assert manifest["finished_at"] is not None

    commit = manifest["git"]["commit"]
    assert commit is None or (isinstance(commit, str) and len(commit) == 40
                              and all(c in "0123456789abcdef" for c in commit))

    assert len(stages) == 1
    per_rank_stage0 = [e for e in manifest["seeds"]["per_rank"] if e["stage_index"] == 0]
    rank0_entries = [e for e in per_rank_stage0 if e["rank"] == 0]
    assert len(rank0_entries) == 1
    entry = rank0_entries[0]
    no_stages = 1
    expected_seed0 = 10 * (no_stages * 0 + 0)
    assert entry["seed0"] == expected_seed0
    assert entry["proposal_seed"] == expected_seed0 + 1
    assert entry["algorithm_seed"] == expected_seed0 + 2
    # G4: per-CHAIN initial-sample stream, derived from stage 0's seed0 with offset +3
    assert entry["initial_sample_seed"] == 10 * no_stages * 0 + 3

    assert manifest["stages"][0]["name"] == stages[0].name == "alg0000_MH"


def test_manifest_records_initial_sample_seed_per_rank(tmp_path):
    """G4: the initial-sample seed is a pure function of the rank (not of the stage), and
    differs between ranks so that two chains cannot start from the same draw."""
    conf, prior, likelihood, stages = _make_problem(tmp_path)
    conf.no_samplers = 3
    stages = stages + [Stage(algorithm="MH", proposal=RandomWalk(scale=0.4), max_evaluations=5)]
    manifest = build_run_manifest(conf, stages, prior, likelihood, runner="local")

    no_stages = len(stages)
    by_rank: dict[int, set[int]] = {}
    for entry in manifest["seeds"]["per_rank"]:
        by_rank.setdefault(entry["rank"], set()).add(entry["initial_sample_seed"])
    assert set(by_rank) == {0, 1, 2}
    for rank, seeds in by_rank.items():
        assert seeds == {10 * no_stages * rank + 3}, f"rank {rank}: {seeds}"
    assert "initial_sample_seed" in manifest["seeds"]["formula"]


def test_manifest_marks_prior_initial_sample_as_reproducible(tmp_path):
    """G4 (2026-09-17) replaces the old "prior is not reproducible" note: the initial sample
    is now drawn from a per-rank seeded generator, so every initial_sample_type is
    reproducible and there is no ``unreproducible_reason``."""
    conf, prior, likelihood, stages = _make_problem(tmp_path)
    conf.initial_sample_type = "prior"  # bypass user_specified for this check only
    manifest = build_run_manifest(conf, stages, prior, likelihood, runner="local")
    assert manifest["seeds"]["initial_sample_reproducible"] is True
    assert "unreproducible_reason" not in manifest["seeds"]


def test_two_local_runs_produce_identical_samples_and_manifests(tmp_path):
    """WS4 acceptance (local part): two runs of the same configuration are bit-identical."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"

    conf_a, prior_a, likelihood_a, stages_a = _make_problem(dir_a)
    surrDAMH.Problem(prior_a, likelihood_a, solver=ToySolver()).run_sampling_local(conf_a, stages_a)

    conf_b, prior_b, likelihood_b, stages_b = _make_problem(dir_b)
    surrDAMH.Problem(prior_b, likelihood_b, solver=ToySolver()).run_sampling_local(conf_b, stages_b)

    samples_a = (dir_a / "sampling_output" / "samples" / "alg0000_MH" / "rank0000.csv").read_bytes()
    samples_b = (dir_b / "sampling_output" / "samples" / "alg0000_MH" / "rank0000.csv").read_bytes()
    assert samples_a == samples_b

    manifest_a = _read_manifest(dir_a)
    manifest_b = _read_manifest(dir_b)

    volatile_keys = {"created_at", "finished_at", "hostname"}

    def strip(d):
        if isinstance(d, dict):
            return {k: strip(v) for k, v in d.items() if k not in volatile_keys}
        if isinstance(d, list):
            return [strip(v) for v in d]
        return d

    # output_dir differs between the two runs (part of the configuration), and is expected to.
    manifest_a["configuration"].pop("output_dir", None)
    manifest_b["configuration"].pop("output_dir", None)

    assert strip(manifest_a) == strip(manifest_b)


def test_build_run_manifest_json_serialisable_with_block_proposal(tmp_path):
    conf, prior, likelihood, _ = _make_problem(tmp_path)
    stage = Stage(algorithm="MH",
                 proposal=Block(groups=[[0], [1]],
                               proposals=[RandomWalk(scale=0.5), RandomWalk(scale=0.5)]),
                 max_evaluations=10)

    manifest = build_run_manifest(conf, [stage], prior, likelihood, runner="local")
    json.dumps(manifest)  # must not raise

    proposal_dict = manifest["stages"][0]["proposal"]
    assert proposal_dict["type"] == "Block"
    assert proposal_dict["groups"] == [[0], [1]]
    assert [p["type"] for p in proposal_dict["proposals"]] == ["RandomWalk", "RandomWalk"]


def test_build_run_manifest_json_serialisable_with_user_specified_distribution(tmp_path):
    conf, prior, likelihood, stages = _make_problem(tmp_path)
    assert conf.initial_samples_distribution is not None

    manifest = build_run_manifest(conf, stages, prior, likelihood, runner="local")
    json.dumps(manifest)  # must not raise
    assert manifest["configuration"]["initial_samples_distribution"] == "FixedSample"


def test_write_run_manifest_returns_path(tmp_path):
    conf, prior, likelihood, stages = _make_problem(tmp_path)
    manifest = build_run_manifest(conf, stages, prior, likelihood, runner="local")
    path = write_run_manifest(str(tmp_path), manifest)
    assert path == str(_manifest_path(tmp_path))
    assert _manifest_path(tmp_path).is_file()
