#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ``surrDAMH.modules.run_data.read_run`` -- the output format v2 reader
(``docs/outputs.md``, ``library_notes/11_output_format_v2_spec.md``, WS9a).

Everything is checked against a REAL ``run_sampling_local`` output (three stages: MH,
MH with ``save_to_file=False``, DAMH-SMU with snapshots), never against a hand-written
fixture, so the writer and the reader are pinned together.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

import surrDAMH
from surrDAMH.modules.continuation import load_carry_over, save_carry_over
from surrDAMH.modules.manifest import FORMAT_VERSION, RunFormatError
from surrDAMH.modules.run_data import read_run
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.nearest_kdtree import KDTreeUpdater

pytestmark = pytest.mark.unit

A = np.array([[1.0, 0.5], [0.0, 2.0]])
NO_PARAMETERS = 2
NO_OBSERVATIONS = 2


class _LinearSolver(Solver):
    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return A @ self.parameters


class _FixedSample:
    def __init__(self, parameters):
        self.parameters = np.array(parameters, dtype=float)

    def rvs(self):
        return self.parameters.copy()


def _run(output_dir):
    """MH -> MH(save_to_file=False) -> DAMH-SMU, snapshots on, one chain."""
    conf = surrDAMH.Configuration(no_parameters=NO_PARAMETERS, no_observations=NO_OBSERVATIONS,
                                  output_dir=str(output_dir),
                                  use_collector=False, use_solvers_pool=False,
                                  save_snapshots_to_file=True,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=_FixedSample([0.3, -0.7]))
    prior = surrDAMH.distributions.PriorIndependentComponents([
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    ])
    likelihood = surrDAMH.distributions.Normal(mean=np.array([1.0, 2.0]), sd=np.array([0.5, 0.5]))
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_samples=15, max_evaluations=15),
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_samples=10, max_evaluations=10,
              save_to_file=False),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_samples=15, max_evaluations=15,
              subchain_length=2),
    ]
    updater = KDTreeUpdater(no_parameters=NO_PARAMETERS, no_observations=NO_OBSERVATIONS,
                            no_nearest_neighbors=2)
    surrDAMH.Problem(prior, likelihood, solver=_LinearSolver()).run_sampling_local(
        conf, stages, surrogate_updater=updater)
    return str(output_dir)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("read_run"))


# --------------------------------------------------------------------------------------
# stage indexing
# --------------------------------------------------------------------------------------
def test_stage_names_are_in_index_order_including_a_stage_without_samples(run_dir):
    run = read_run(run_dir)
    assert run.stage_names == ["alg0000_MH", "alg0001_MH", "alg0002_DAMH-SMU"]
    assert run.no_stages == 3
    assert run.stage_index("alg0002_DAMH-SMU") == 2


def test_stage_order_comes_from_the_alg_prefix_not_from_listdir(run_dir, monkeypatch):
    """The index is parsed from ``alg%04d``; ``os.listdir`` order must not matter."""
    from surrDAMH.modules import run_data as run_data_module

    real_listdir = os.listdir
    monkeypatch.setattr(run_data_module.os, "listdir",
                        lambda path: list(reversed(sorted(real_listdir(path)))))
    run = read_run(run_dir)
    assert run.stage_names == ["alg0000_MH", "alg0001_MH", "alg0002_DAMH-SMU"]


def test_stage_directory_without_an_index_prefix_is_refused(tmp_path):
    _run(tmp_path)
    samples_dir = tmp_path / "sampling_output" / "samples"
    (samples_dir / "alg0000_MH").rename(samples_dir / "renamed_by_hand")
    with pytest.raises(RunFormatError, match="stage-index prefix"):
        read_run(str(tmp_path))


# --------------------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------------------
def test_samples_columns_and_dtypes(run_dir):
    run = read_run(run_dir)
    assert run.samples_columns == ["multiplicity", "par_0", "par_1", "log_posterior"]
    assert run.no_parameters == NO_PARAMETERS
    assert run.no_observations == NO_OBSERVATIONS

    for stage_index in (0, 2):
        chains = run.samples[stage_index]
        assert len(chains) == 1  # one local chain
        rows = chains[0]
        assert rows.dtype == np.float64
        assert rows.ndim == 2 and rows.shape[1] == 2 + NO_PARAMETERS
        assert rows.shape[0] > 0
        # the multiplicity column is integral and non-negative (A30 allows a leading 0)
        assert np.all(rows[:, 0] >= 0)
        np.testing.assert_array_equal(rows[:, 0], np.round(rows[:, 0]))


def test_a_stage_with_save_to_file_false_has_no_samples(run_dir):
    """Spec §3: such a stage is an EMPTY LIST, not a list of empty arrays."""
    run = read_run(run_dir)
    assert run.samples[1] == []
    assert run.no_chains(1) == 0
    # it did write its last sample, which is why the stage is known at all
    assert run.last_sample["alg0001_MH"].shape == (1, NO_PARAMETERS)


# --------------------------------------------------------------------------------------
# raw_data
# --------------------------------------------------------------------------------------
def test_raw_data_columns_and_nan_placement(run_dir):
    run = read_run(run_dir)
    assert run.raw_data_columns == ["state_type", "par_0", "par_1", "solver_tag",
                                    "obs_0", "obs_1", "obs_approx_0", "obs_approx_1",
                                    "log_likelihood", "log_prior"]
    exact = ["obs_0", "obs_1"]
    approx = ["obs_approx_0", "obs_approx_1"]

    mh = run.raw_data[0][0]
    assert isinstance(mh, pd.DataFrame)
    assert list(mh.columns) == run.raw_data_columns
    assert set(mh["state_type"]) <= {"accepted", "rejected"}
    # an MH stage calls the exact model for every proposal and has no surrogate at all
    assert np.all(np.isfinite(mh[exact].to_numpy(dtype=float)))
    assert np.all(np.isnan(mh[approx].to_numpy(dtype=float)))

    damh = run.raw_data[2][0]
    prerejected = damh["state_type"] == "prerejected"
    assert prerejected.any(), "the DAMH stage produced no prerejected proposal to check"
    # prerejected: the exact model was never called -> NaN; the surrogate block is filled
    assert np.all(np.isnan(damh.loc[prerejected, exact].to_numpy(dtype=float)))
    assert np.all(np.isfinite(damh.loc[prerejected, approx].to_numpy(dtype=float)))
    # every other DAMH row went through the exact model AND has a surrogate value
    assert np.all(np.isfinite(damh.loc[~prerejected, exact].to_numpy(dtype=float)))
    assert np.all(np.isfinite(damh.loc[~prerejected, approx].to_numpy(dtype=float)))
    assert np.all(np.isfinite(damh[["log_likelihood", "log_prior"]].to_numpy(dtype=float)))


def test_raw_data_is_present_for_every_stage_that_ran(run_dir):
    run = read_run(run_dir)
    # save_snapshots_to_file is a run-wide switch, so even the save_to_file=False stage
    # wrote snapshots
    assert [None if chains is None else len(chains) for chains in run.raw_data] == [1, 1, 1]


def test_load_raw_data_false_skips_raw_data(run_dir):
    run = read_run(run_dir, load_raw_data=False)
    assert run.raw_data == [None, None, None]
    assert run.raw_data_loaded is False
    assert len(run.samples[0]) == 1  # everything else is still read


# --------------------------------------------------------------------------------------
# notes / subchain_stats / last_sample / surrogate quality
# --------------------------------------------------------------------------------------
def test_notes_and_subchain_stats(run_dir):
    run = read_run(run_dir)
    notes = run.notes[0]
    assert list(notes.columns) == ["accepted", "rejected", "pre-rejected", "sum", "seed", "out_of_bounds",
                                   "exact_steps", "exact_accepted", "audited", "audit_hidden", "polyak_updates"]
    assert len(notes) == 1  # one chain
    assert run.notes[1].empty  # save_to_file=False -> no notes

    assert run.subchain_stats[0] is None  # MH stage
    stats = run.subchain_stats[2]
    assert stats is not None
    assert list(stats.columns) == ["iteration", "subchain_max_length", "subchain_accepted",
                                   "subchain_acceptance_rate", "correction_log_ratio",
                                   "outer_proposed_changed", "outer_accepted", "rank_world", "kernel"]
    assert len(stats) > 0
    assert set(stats["kernel"]) == {0}  # S2: no exact steps in this fixture

    # adaptive_stats (2026-09-20): no stage of this fixture adapts, so every entry is None --
    # which is also what a run produced before that file existed reads back as
    assert run.adaptive_stats == [None] * len(run.stage_names)


def test_last_sample_and_manifest(run_dir):
    run = read_run(run_dir)
    assert sorted(run.last_sample) == ["alg0000_MH", "alg0001_MH", "alg0002_DAMH-SMU"]
    for parameters in run.last_sample.values():
        assert parameters.shape == (1, NO_PARAMETERS)
        assert parameters.dtype == np.float64
    assert run.manifest["format_version"] == FORMAT_VERSION
    assert run.manifest["runner"] == "local"
    # no collector in a local run
    assert run.surrogate_quality is None
    assert run.surrogate_quality_test is None


# --------------------------------------------------------------------------------------
# refusal of anything that is not format v2
# --------------------------------------------------------------------------------------
def test_missing_manifest_is_refused(tmp_path):
    _run(tmp_path)
    os.remove(tmp_path / "sampling_output" / "run_manifest.json")
    with pytest.raises(RunFormatError) as excinfo:
        read_run(str(tmp_path))
    message = str(excinfo.value)
    assert "run_manifest.json" in message
    assert "no converter" in message
    assert str(FORMAT_VERSION) in message


def test_old_format_version_is_refused(tmp_path):
    _run(tmp_path)
    manifest_path = tmp_path / "sampling_output" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RunFormatError) as excinfo:
        read_run(str(tmp_path))
    message = str(excinfo.value)
    assert "format_version 1" in message
    assert f"expected {FORMAT_VERSION}" in message
    assert "no converter" in message


def test_a_samples_file_without_the_v2_header_is_refused(tmp_path):
    """A pre-v2 (header-less) samples file must not be silently misread."""
    _run(tmp_path)
    path = tmp_path / "sampling_output" / "samples" / "alg0000_MH" / "rank0000.csv"
    rows = path.read_text().splitlines()[1:]  # drop the header, i.e. the v1 layout
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(RunFormatError, match="unexpected CSV header"):
        read_run(str(tmp_path))


def test_samples_refuses_a_directory_that_is_not_format_v2(tmp_path):
    """The same refusal reaches users through ``post_processing.Samples``."""
    import surrDAMH.post_processing as pp

    _run(tmp_path)
    os.remove(tmp_path / "sampling_output" / "run_manifest.json")
    with pytest.raises(RunFormatError):
        pp.Samples(no_parameters=NO_PARAMETERS, samples_dir=str(tmp_path))


# --------------------------------------------------------------------------------------
# carry_over (2026-09-21)
# --------------------------------------------------------------------------------------
class _FakeConf:
    """Minimal stand-in for ``Configuration``: ``save_carry_over`` only reads ``output_dir``."""

    def __init__(self, output_dir):
        self.output_dir = str(output_dir)


def test_save_and_load_carry_over_round_trips_matrix_and_scalars(tmp_path):
    conf = _FakeConf(tmp_path)
    cov = np.array([[0.25, 0.05], [0.05, 0.16]])
    carried = {"scale": cov}
    summary = {"base_cov": cov / 4.0, "log_sigma": -0.7, "n_pooled": 137, "mean": np.array([0.1, -0.2]),
               "beta": 1.0}  # an integral-valued float must come back as a float, not an int

    save_carry_over(conf, "alg0000_MH-adaptive", carried, summary)

    loaded = load_carry_over(str(tmp_path), "alg0000_MH-adaptive")
    assert loaded is not None
    loaded_carry, loaded_summary = loaded

    np.testing.assert_allclose(loaded_carry["scale"], cov)
    np.testing.assert_allclose(loaded_summary["base_cov"], cov / 4.0)
    assert isinstance(loaded_summary["log_sigma"], float)
    assert loaded_summary["log_sigma"] == pytest.approx(-0.7)
    assert isinstance(loaded_summary["n_pooled"], int)
    assert loaded_summary["n_pooled"] == 137
    assert isinstance(loaded_summary["beta"], float) and loaded_summary["beta"] == 1.0
    np.testing.assert_allclose(loaded_summary["mean"], np.array([0.1, -0.2]))

    # the file itself: one .npz, no rank suffix
    path = tmp_path / "sampling_output" / "carry_over" / "alg0000_MH-adaptive.npz"
    assert path.is_file()


def test_load_carry_over_returns_none_when_the_file_is_missing(tmp_path):
    assert load_carry_over(str(tmp_path), "alg0000_MH-adaptive") is None


def test_load_carry_over_maps_legacy_key_names_on_read(tmp_path):
    """Files written before the 2026-09-21 proposal-spec rewrite used the old Stage field names
    (``proposal_sd_or_cov``/``pcn_beta``/``hamiltonian_step_size``) as carry_over keys;
    ``load_carry_over`` must remap them to the new ones (``scale``/``beta``/``step_size``) so an
    old run still renders in a report. Only the carry_over dict is remapped, not summary."""
    conf = _FakeConf(tmp_path)
    save_carry_over(conf, "alg0000_MH-adaptive", {"proposal_sd_or_cov": 0.5}, {"proposal_sd_or_cov": 0.5})
    loaded_carry, loaded_summary = load_carry_over(str(tmp_path), "alg0000_MH-adaptive")
    assert loaded_carry == {"scale": 0.5}
    assert loaded_summary == {"proposal_sd_or_cov": 0.5}  # summary keys are not remapped


def test_read_run_carry_over_is_none_for_a_stage_without_the_file_and_the_pair_for_one_with_it(tmp_path):
    """This fixture (MH -> MH(save_to_file=False) -> DAMH-SMU) has no adaptive stage at all, so
    every entry must be ``None`` -- this also pins the "run predates carry_over/" backwards
    compatibility path, since ``_run`` never calls ``save_carry_over``. A separate ``tmp_path``
    (not the module-scoped ``run_dir`` fixture) so the hand-written file below does not leak
    into the other tests sharing that fixture."""
    root = _run(tmp_path)
    # C1 (2026-10-11): every stage now writes a carry_over file (the pooled posterior statistics);
    # remove them to pin the "run predates carry_over/" path
    run = read_run(root)
    assert all(entry is not None and set(entry[0]) == {"posterior_statistics", "posterior_statistics_n_seen"} for entry in run.carry_over)
    shutil.rmtree(os.path.join(root, "sampling_output", "carry_over"))
    run = read_run(root)
    assert run.carry_over == [None] * len(run.stage_names)

    # now write a carry_over file for stage 0 by hand and re-read: only that entry changes
    conf = _FakeConf(root)
    save_carry_over(conf, "alg0000_MH", {"beta": 0.42}, {"beta": 0.42})
    run_with_carry_over = read_run(root)
    assert run_with_carry_over.carry_over[0] == ({"beta": 0.42}, {"beta": 0.42})
    assert run_with_carry_over.carry_over[1] is None
    assert run_with_carry_over.carry_over[2] is None
