#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ``surrDAMH.post_processing`` (Phase 1 items 34-40 of
``library_notes/07_testing_plan.md``).

Synthetic CSV fixtures are written in **output format v2** (``docs/outputs.md``,
``library_notes/11_output_format_v2_spec.md``) through the production writer
``surrDAMH.modules.monitoring.CsvWriter`` and the production column lists
``surrDAMH.modules.manifest.samples_columns``/``raw_data_columns``, so a column rename in
the library propagates into these fixtures instead of being masked by them:

- ``sampling_output/samples/<stage>/<rank>.csv``: header
  ``multiplicity, par_0..par_{p-1}, log_posterior``, then one row per *kept*
  (post-subchain) state (see ``AlgorithmBase._record_current_sample``).
- ``sampling_output/notes/<stage>/<rank>.csv``: header row
  ``["accepted","rejected","pre-rejected","sum","seed"]`` then one data row
  (``AlgorithmBase._finalize_run``).
- ``sampling_output/raw_data/<stage>/<rank>.csv``: header
  ``state_type, par_0.., solver_tag, obs_0.., obs_approx_0.., log_likelihood, log_prior``,
  then one row per proposed state (``AlgorithmBase._record_proposed_snapshot``).
- ``sampling_output/run_manifest.json``: required by the reader; the fixtures write a
  minimal one with ``format_version`` = ``manifest.FORMAT_VERSION``.

These layouts were derived by reading ``surrDAMH/modules/algorithms.py`` and
``surrDAMH/modules/monitoring.py`` directly (not guessed from the plan's prose), and
cross-checked by constructing fixtures and calling the real ``Samples`` API.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

import surrDAMH
import surrDAMH.post_processing as pp
from surrDAMH.modules.continuation import save_carry_over
from surrDAMH.modules.manifest import (FORMAT_VERSION, MANIFEST_VERSION,
                                       raw_data_columns, samples_columns)
from surrDAMH.modules import manifest as manifest_module
from surrDAMH.modules.monitoring import CsvWriter
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.nearest_kdtree import KDTreeUpdater

pytestmark = pytest.mark.unit


def _stage_spec(**kwargs) -> dict:
    """The exact dict shape ``modules.manifest._stage_dict`` writes for a ``Stage`` into
    ``run_manifest.json`` (2026-09-21 proposal-spec format: ``proposal`` is a nested
    ``{"type": ..., ...}`` dict, not top-level ``proposal_type``/``adaptive`` keys), for
    hand-written manifest fixtures below."""
    return manifest_module._stage_dict(Stage(**kwargs))


# --------------------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------------------
def _write_manifest(root, no_parameters, no_observations=0, unverified_options=None, stages=None):
    """Minimal format-v2 ``run_manifest.json`` (all ``read_run`` actually requires).

    ``stages`` (optional): a list of dicts written verbatim under the ``"stages"`` key,
    read back by ``html_report_extended`` when no ``stages=`` argument is passed to it.
    """
    path = os.path.join(root, "sampling_output")
    os.makedirs(path, exist_ok=True)
    manifest = {"manifest_version": MANIFEST_VERSION,
               "format_version": FORMAT_VERSION,
               "runner": "fixture",
               "unverified_options": list(unverified_options or []),
               "configuration": {"no_parameters": no_parameters,
                                 "no_observations": no_observations}}
    if stages is not None:
        manifest["stages"] = stages
    with open(os.path.join(path, "run_manifest.json"), "w") as f:
        json.dump(manifest, f)


def _write_csv(dirname, basename, header, rows):
    """Write one v2 CSV through the production ``CsvWriter`` (header first, then rows)."""
    writer = CsvWriter(dirname=dirname, basename=basename, header=header)
    for row in rows:
        writer.writerow(row)
    writer.close_file()


def _write_samples_stage(root, stage_name, chains_rows, no_parameters=2, no_observations=0):
    """chains_rows: list (one per chain) of list of [multiplicity, *params, log_posterior]."""
    _write_manifest(root, no_parameters, no_observations)
    samples_dir = os.path.join(root, "sampling_output", "samples", stage_name)
    notes_dir = os.path.join(root, "sampling_output", "notes", stage_name)
    for c, rows in enumerate(chains_rows):
        _write_csv(samples_dir, f"rank{c:04d}.csv", samples_columns(no_parameters), rows)
        accepted = sum(1 for row in rows if row[0] >= 1)
        _write_csv(notes_dir, f"rank{c:04d}.csv",
                   ["accepted", "rejected", "pre-rejected", "sum", "seed"],
                   [[accepted, 0, 0, accepted, c]])


def _write_raw_data_stage(root, stage_name, chain_rows_by_index, no_parameters=2,
                          no_observations=2, header=None):
    raw_dir = os.path.join(root, "sampling_output", "raw_data", stage_name)
    if header is None:
        header = raw_data_columns(no_parameters, no_observations)
    for c, rows in enumerate(chain_rows_by_index):
        _write_csv(raw_dir, f"rank{c:04d}.csv", header, rows)


# --------------------------------------------------------------------------------------
# Item 34: decompress -- shape (sum(weights), p), each row repeated weights[i] times.
# --------------------------------------------------------------------------------------
class TestDecompress:
    def test_shape_and_repetition(self):
        compressed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        weights = np.array([2, 1, 3])
        decompressed = pp.decompress(compressed, weights)
        assert decompressed.shape == (int(weights.sum()), compressed.shape[1])
        expected = np.array([
            [1.0, 2.0], [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0], [5.0, 6.0], [5.0, 6.0],
        ])
        np.testing.assert_array_equal(decompressed, expected)


# --------------------------------------------------------------------------------------
# Item 35: weighted mean/cov vs hand-computed; no_unique_samples equals the compressed
# row count (WS0 item C11 -- verified applied: StageSamples.__init__ now sets
# ``self.no_unique_samples[i] = len(self.weights[i])`` right after loading, not 0).
# --------------------------------------------------------------------------------------
class TestMeanCovAndNoUniqueSamples:
    def test_against_hand_computed_values(self, tmp_path):
        root = str(tmp_path)
        # one chain, 3 unique (compressed) rows with distinct weights
        rows = [
            [2, 1.0, 10.0, -1.0],
            [1, 2.0, 20.0, -1.0],
            [3, 3.0, 30.0, -1.0],
        ]
        _write_samples_stage(root, "alg0000_MH", [rows])

        samples = pp.Samples(no_parameters=2, samples_dir=root)
        assert samples.list_of_stages[0].no_unique_samples == [3]  # C11: compressed row count, not 0

        mean, cov = samples.get_mean_and_cov()
        x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        w = np.array([2, 1, 3])
        expected_mean = np.average(x, axis=0, weights=w)
        expected_cov = np.cov(x, rowvar=False, fweights=w)
        np.testing.assert_allclose(mean, expected_mean)
        np.testing.assert_allclose(cov, expected_cov)

    def test_no_unique_samples_multiple_chains(self, tmp_path):
        root = str(tmp_path)
        rows_chain0 = [[1, float(i), float(i), -1.0] for i in range(5)]
        rows_chain1 = [[2, float(i), float(i), -1.0] for i in range(3)]
        _write_samples_stage(root, "alg0000_MH", [rows_chain0, rows_chain1])

        samples = pp.Samples(no_parameters=2, samples_dir=root)
        assert samples.list_of_stages[0].no_unique_samples == [5, 3]

    def test_zero_weight_boundary_row_is_not_counted_and_not_decompressed(self, tmp_path):
        """
        A30 (2026-09-17): the first row of a stage whose initial state was carried over can
        have weight 0. It must contribute nothing to ``length``, to the weighted mean/cov, or
        to the decompressed chain, and must not be counted in ``no_unique_samples``.
        """
        root = str(tmp_path)
        rows = [
            [0, 1.0, 10.0, -1.0],   # carried-over state, already counted by the previous stage
            [2, 2.0, 20.0, -1.0],
            [1, 3.0, 30.0, -1.0],
        ]
        _write_samples_stage(root, "alg0000_MH", [rows])

        samples = pp.Samples(no_parameters=2, samples_dir=root)
        stage = samples.list_of_stages[0]
        assert stage.no_unique_samples == [2]
        assert stage.length[0] == 3
        assert stage.samples[0].shape == (3, 2)
        np.testing.assert_array_equal(stage.samples[0], np.array([[2.0, 20.0], [2.0, 20.0], [3.0, 30.0]]))

        mean, _ = samples.get_mean_and_cov()
        x = np.array([[2.0, 20.0], [3.0, 30.0]])
        np.testing.assert_allclose(mean, np.average(x, axis=0, weights=np.array([2, 1])))


# --------------------------------------------------------------------------------------
# Item 36: find_best_fits vs a naive full sort, for all three ranking modes;
# 'prerejected' rows excluded; missing log columns -- documented unfixed bug (P10).
# --------------------------------------------------------------------------------------
class TestFindBestFits:
    NO_PARAMETERS = 2
    NO_OBSERVATIONS = 2

    def _make_fixture(self, root, n=30, seed=0):
        _write_samples_stage(root, "alg0000_MH", [[[1, 0.0, 0.0, -1.0]]],
                             no_parameters=self.NO_PARAMETERS, no_observations=self.NO_OBSERVATIONS)
        rng = np.random.default_rng(seed)
        pars = rng.uniform(-3, 3, size=(n, self.NO_PARAMETERS))
        rows = []
        nan_block = [np.nan] * self.NO_OBSERVATIONS
        for i, p in enumerate(pars):
            obs = p ** 2
            log_likelihood = -float(np.sum(obs))
            log_prior = -float(np.sum(p ** 2)) / 2
            if i % 7 == 0:
                # v2: a prerejected proposal has no EXACT observations (NaN block); its
                # surrogate block carries the values the sub-chain decision used.
                state = "prerejected"
                row = [state] + list(p) + [0] + nan_block + list(obs) + [log_likelihood, log_prior]
            else:
                state = "accepted" if i % 2 == 0 else "rejected"
                row = [state] + list(p) + [0] + list(obs) + nan_block + [log_likelihood, log_prior]
            rows.append(row)
        _write_raw_data_stage(root, "alg0000_MH", [rows],
                              no_parameters=self.NO_PARAMETERS, no_observations=self.NO_OBSERVATIONS)
        return pars, rows

    def _naive_non_prerejected(self, rows):
        return [r for r in rows if r[0] != "prerejected"]

    def test_l2_ranking_matches_naive_full_sort(self, tmp_path):
        root = str(tmp_path)
        pars, rows = self._make_fixture(root, n=40, seed=1)
        samples = pp.Samples(no_parameters=self.NO_PARAMETERS, samples_dir=root)

        target = np.array([1.0, 1.0])
        n_best = 6
        df, best_par, best_obs, best_scores = samples.find_best_fits(
            self.NO_OBSERVATIONS, target, n_best=n_best, ranking_mode="l2")

        kept = self._naive_non_prerejected(rows)
        obs_all = np.array([r[1 + self.NO_PARAMETERS + 1:1 + self.NO_PARAMETERS + 1 + self.NO_OBSERVATIONS]
                            for r in kept], dtype=float)
        l2 = np.sqrt(np.sum((obs_all - target) ** 2, axis=1))
        naive_best = np.sort(l2)[:n_best]
        np.testing.assert_allclose(np.sort(best_scores), naive_best)
        assert len(df) == n_best

    def test_posterior_and_likelihood_ranking_match_naive_full_sort(self, tmp_path):
        root = str(tmp_path)
        pars, rows = self._make_fixture(root, n=40, seed=2)
        samples = pp.Samples(no_parameters=self.NO_PARAMETERS, samples_dir=root)

        target = np.array([0.5, 0.5])
        n_best = 5
        kept = self._naive_non_prerejected(rows)
        log_likelihood = np.array([r[-2] for r in kept], dtype=float)
        log_prior = np.array([r[-1] for r in kept], dtype=float)

        _, _, _, best_scores_post = samples.find_best_fits(
            self.NO_OBSERVATIONS, target, n_best=n_best, ranking_mode="posterior")
        naive_post = np.sort(log_likelihood + log_prior)[::-1][:n_best]
        np.testing.assert_allclose(np.sort(best_scores_post)[::-1], naive_post)

        _, _, _, best_scores_lik = samples.find_best_fits(
            self.NO_OBSERVATIONS, target, n_best=n_best, ranking_mode="likelihood")
        naive_lik = np.sort(log_likelihood)[::-1][:n_best]
        np.testing.assert_allclose(np.sort(best_scores_lik)[::-1], naive_lik)

    def test_prerejected_rows_excluded(self, tmp_path):
        root = str(tmp_path)
        pars, rows = self._make_fixture(root, n=40, seed=3)
        samples = pp.Samples(no_parameters=self.NO_PARAMETERS, samples_dir=root)

        prerejected_pars = {tuple(np.round(p, 8)) for r, p in zip(rows, pars) if r[0] == "prerejected"}
        target = np.array([0.0, 0.0])
        _, best_par, _, _ = samples.find_best_fits(
            self.NO_OBSERVATIONS, target, n_best=40, ranking_mode="l2")
        returned_pars = {tuple(np.round(row, 8)) for row in best_par}
        assert returned_pars.isdisjoint(prerejected_pars)

    def _write_truncated_raw_data(self, tmp_path):
        """A raw_data file WITHOUT the log_likelihood/log_prior columns."""
        root = str(tmp_path)
        _write_samples_stage(root, "alg0000_MH", [[[1, 0.0, 0.0, -1.0]]],
                             no_parameters=self.NO_PARAMETERS, no_observations=self.NO_OBSERVATIONS)
        truncated_header = ["state_type", "par_0", "par_1", "solver_tag", "obs_0", "obs_1"]
        rows = [["accepted", 0.1, 0.2, 0, 0.01, 0.04]]
        _write_raw_data_stage(root, "alg0000_MH", [rows], header=truncated_header)
        return root

    @pytest.mark.parametrize("ranking_mode", ["posterior", "likelihood"])
    def test_missing_log_columns_raise(self, tmp_path, ranking_mode):
        """
        Finding P10, FIXED in WS9b: a raw_data file without the log-likelihood/log-prior
        columns (an older, truncated or hand-edited file) used to make every score NaN,
        which became ``-inf`` and ranked those rows last -- an arbitrary but complete-looking
        "best fit" table. ``find_best_fits`` now refuses the file instead.
        """
        root = self._write_truncated_raw_data(tmp_path)
        samples = pp.Samples(no_parameters=self.NO_PARAMETERS, samples_dir=root)
        with pytest.raises(pp.RunFormatError, match="log_likelihood"):
            samples.find_best_fits(self.NO_OBSERVATIONS, np.array([0.0, 0.0]), n_best=5,
                                   ranking_mode=ranking_mode)

    def test_missing_log_columns_do_not_affect_l2_ranking(self, tmp_path):
        """The l2 mode needs obs_*, not the log columns, so the same file still works."""
        root = self._write_truncated_raw_data(tmp_path)
        samples = pp.Samples(no_parameters=self.NO_PARAMETERS, samples_dir=root)
        _, best_par, _, best_scores = samples.find_best_fits(
            self.NO_OBSERVATIONS, np.array([0.0, 0.0]), n_best=5, ranking_mode="l2")
        assert len(best_scores) == 1
        np.testing.assert_allclose(best_scores[0], np.sqrt(0.01 ** 2 + 0.04 ** 2))
        np.testing.assert_allclose(best_par[0], [0.1, 0.2])

    def test_missing_observation_columns_raise_for_l2(self, tmp_path):
        """...and conversely, l2 refuses a file without the observation block."""
        root = str(tmp_path)
        _write_samples_stage(root, "alg0000_MH", [[[1, 0.0, 0.0, -1.0]]],
                             no_parameters=self.NO_PARAMETERS, no_observations=self.NO_OBSERVATIONS)
        header = ["state_type", "par_0", "par_1", "solver_tag", "log_likelihood", "log_prior"]
        _write_raw_data_stage(root, "alg0000_MH", [[["accepted", 0.1, 0.2, 0, -1.0, -2.0]]],
                              header=header)
        samples = pp.Samples(no_parameters=self.NO_PARAMETERS, samples_dir=root)
        with pytest.raises(pp.RunFormatError, match="obs_0"):
            samples.find_best_fits(self.NO_OBSERVATIONS, np.array([0.0, 0.0]), n_best=5,
                                   ranking_mode="l2")

    def test_ranking_order_matches_rank_best_fit_candidates(self, tmp_path):
        """
        Finding P3, FIXED in WS9b: ``find_best_fits`` now ranks through
        ``rank_best_fit_candidates`` -- the unit-tested function -- instead of its own
        argpartition merge. Same data, same order.
        """
        root = str(tmp_path)
        pars, rows = self._make_fixture(root, n=40, seed=7)
        samples = pp.Samples(no_parameters=self.NO_PARAMETERS, samples_dir=root)
        kept = self._naive_non_prerejected(rows)
        log_likelihood = np.array([r[-2] for r in kept], dtype=float)
        log_prior = np.array([r[-1] for r in kept], dtype=float)

        _, _, _, best_scores = samples.find_best_fits(
            self.NO_OBSERVATIONS, np.array([0.0, 0.0]), n_best=6, ranking_mode="posterior")
        expected_order = pp.rank_best_fit_candidates(posteriors=log_likelihood + log_prior,
                                                     mode="posterior")[:6]
        np.testing.assert_allclose(best_scores, (log_likelihood + log_prior)[expected_order])

    def test_chains_to_disp_restricts_the_snapshots_searched(self, tmp_path):
        """``chains_to_disp`` was already honoured here; pin it (finding P5)."""
        root = str(tmp_path)
        _write_samples_stage(root, "alg0000_MH", [[[1, 0.0, 0.0, -1.0]], [[1, 0.0, 0.0, -1.0]]],
                             no_parameters=self.NO_PARAMETERS, no_observations=self.NO_OBSERVATIONS)
        nan_block = [np.nan] * self.NO_OBSERVATIONS
        chain0 = [["accepted", 1.0, 1.0, 0, 5.0, 5.0] + nan_block + [-1.0, -1.0]]
        chain1 = [["accepted", 2.0, 2.0, 0, 0.0, 0.0] + nan_block + [-1.0, -1.0]]
        _write_raw_data_stage(root, "alg0000_MH", [chain0, chain1],
                              no_parameters=self.NO_PARAMETERS, no_observations=self.NO_OBSERVATIONS)
        samples = pp.Samples(no_parameters=self.NO_PARAMETERS, samples_dir=root)

        _, best_par, _, _ = samples.find_best_fits(self.NO_OBSERVATIONS, np.array([0.0, 0.0]),
                                                   n_best=5, chains_to_disp=[0])
        np.testing.assert_allclose(best_par, [[1.0, 1.0]])
        _, best_par, _, _ = samples.find_best_fits(self.NO_OBSERVATIONS, np.array([0.0, 0.0]),
                                                   n_best=5, chains_to_disp=[1])
        np.testing.assert_allclose(best_par, [[2.0, 2.0]])


# --------------------------------------------------------------------------------------
# Item 37: calculate_gelman_rubin -- i.i.d. chains give R-hat ~= 1; offset chains give
# R-hat > 1.
# --------------------------------------------------------------------------------------
class TestGelmanRubin:
    def _make_chains(self, root, stage_name, chain_means, n=300, seed=0):
        rng = np.random.default_rng(seed)
        chains_rows = []
        for mu in chain_means:
            data = rng.normal(loc=mu, scale=1.0, size=(n, 2))
            rows = [[1, row[0], row[1], -1.0] for row in data]
            chains_rows.append(rows)
        _write_samples_stage(root, stage_name, chains_rows)

    def test_iid_chains_rhat_near_one(self, tmp_path):
        root = str(tmp_path)
        self._make_chains(root, "alg0000_MH", [[0.0, 0.0]] * 4, n=400, seed=1)
        samples = pp.Samples(no_parameters=2, samples_dir=root)
        result = samples.calculate_gelman_rubin()
        assert result is not None
        assert result["rhat_max"] < 1.05

    def test_offset_chains_rhat_above_one(self, tmp_path):
        root = str(tmp_path)
        self._make_chains(root, "alg0000_MH", [[0.0, 0.0], [10.0, 10.0], [0.0, 0.0], [10.0, 10.0]],
                          n=400, seed=2)
        samples = pp.Samples(no_parameters=2, samples_dir=root)
        result = samples.calculate_gelman_rubin()
        assert result is not None
        assert result["rhat_max"] > 1.5


# --------------------------------------------------------------------------------------
# Item 38: autocorr_FM / auto_window on rho(k) = phi**k -> tau ~= (1+phi)/(1-phi).
# --------------------------------------------------------------------------------------
class TestAutocorrFM:
    def test_ar1_autocorr_time(self):
        phi = 0.7
        k = np.arange(0, 400)
        rho = phi ** k
        tau = pp.autocorr_FM(rho, c=5)
        expected = (1 + phi) / (1 - phi)
        assert tau == pytest.approx(expected, rel=0.05)

    @pytest.mark.parametrize("phi", [0.3, 0.5, 0.9])
    def test_ar1_autocorr_time_various_phi(self, phi):
        k = np.arange(0, 600)
        rho = phi ** k
        tau = pp.autocorr_FM(rho, c=5)
        expected = (1 + phi) / (1 - phi)
        assert tau == pytest.approx(expected, rel=0.1)


# --------------------------------------------------------------------------------------
# Item 39: Samples(..., load_posterior_surrogate=True) -- either works or is removed
# (finding P1/5.1). WS9b REMOVED it: the "samples" CSV written by
# AlgorithmBase._record_current_sample has never had a posterior_surrogate column (only
# multiplicity, params, log_posterior), so the option could only ever raise IndexError.
# --------------------------------------------------------------------------------------
class TestLoadPosteriorSurrogateRemoved:
    def test_constructor_no_longer_accepts_the_option(self, tmp_path):
        root = str(tmp_path)
        _write_samples_stage(root, "alg0000_MH", [[[1, 0.0, 0.0, -1.0]]])
        with pytest.raises(TypeError, match="load_posterior_surrogate"):
            pp.Samples(no_parameters=2, samples_dir=root, load_posterior_surrogate=True)

    def test_signature_has_no_such_parameter(self):
        import inspect
        for owner in (pp.Samples.__init__, pp.StageSamples.__init__):
            assert "load_posterior_surrogate" not in inspect.signature(owner).parameters

    def test_stage_samples_has_no_posterior_surrogate_attribute(self, tmp_path):
        root = str(tmp_path)
        _write_samples_stage(root, "alg0000_MH", [[[1, 0.0, 0.0, -1.0]]])
        stage = pp.Samples(no_parameters=2, samples_dir=root).list_of_stages[0]
        assert not hasattr(stage, "posterior_surrogate")


# --------------------------------------------------------------------------------------
# WS9b items 3-7: chains_to_disp honoured everywhere, explicit exceptions instead of
# `except BaseException`, Autocorrelation chain count from the analysed stages, no leaked
# figures.
# --------------------------------------------------------------------------------------
def _two_chain_samples(tmp_path, n=200, seed=0, stage_name="alg0000_MH"):
    """Two chains with clearly different means, so a chain subset is visible in any statistic."""
    rng = np.random.default_rng(seed)
    chains_rows = []
    for mu in ([0.0, 0.0], [10.0, 10.0]):
        data = rng.normal(loc=mu, scale=1.0, size=(n, 2))
        chains_rows.append([[1, row[0], row[1], -1.0] for row in data])
    _write_samples_stage(str(tmp_path), stage_name, chains_rows)
    return pp.Samples(no_parameters=2, samples_dir=str(tmp_path))


class TestChainsToDisp:
    def test_get_mean_and_cov_honours_chains_to_disp(self, tmp_path):
        samples = _two_chain_samples(tmp_path, seed=11)
        mean_all, _ = samples.get_mean_and_cov()
        mean_0, _ = samples.get_mean_and_cov(chains_to_disp=[0])
        mean_1, _ = samples.get_mean_and_cov(chains_to_disp=[1])
        assert np.all(mean_0 < 1.0) and np.all(mean_1 > 9.0)
        np.testing.assert_allclose(mean_all, (mean_0 + mean_1) / 2, rtol=1e-12)

    def test_collect_samples_matrix_honours_chains_to_disp(self, tmp_path):
        samples = _two_chain_samples(tmp_path, n=50, seed=12)
        assert samples._collect_samples_matrix().shape[0] == 100
        assert samples._collect_samples_matrix(chains_to_disp=[1]).shape[0] == 50

    def test_gelman_rubin_on_one_chain_is_unavailable(self, tmp_path):
        samples = _two_chain_samples(tmp_path, n=50, seed=13)
        assert samples.calculate_gelman_rubin() is not None
        assert samples.calculate_gelman_rubin(chains_to_disp=[0]) is None  # needs >= 2 chains

    def test_effective_sample_size_honours_chains_to_disp(self, tmp_path):
        samples = _two_chain_samples(tmp_path, n=400, seed=14)
        assert samples.calculate_effective_sample_size()["total_samples"] == 800
        assert samples.calculate_effective_sample_size(chains_to_disp=[0])["total_samples"] == 400

    def test_unknown_chain_is_reported(self, tmp_path):
        samples = _two_chain_samples(tmp_path, n=20, seed=15)
        with pytest.raises(IndexError, match="chains_to_disp"):
            samples.get_mean_and_cov(chains_to_disp=[5])

    def test_unknown_stage_is_reported(self, tmp_path):
        samples = _two_chain_samples(tmp_path, n=20, seed=16)
        with pytest.raises(IndexError, match="stages_to_disp"):
            samples.get_mean_and_cov(stages_to_disp=[3])

    def test_plots_accept_chains_to_disp(self, tmp_path):
        """Every sample-derived plot takes the argument (P5: none of them ignores it)."""
        import inspect
        for name in ("plot_chains", "plot_hist_grid", "plot_hist_marginals", "plot_autocorr",
                     "plot_parameter_correlation_heatmap", "hist_observations"):
            assert "chains_to_disp" in inspect.signature(getattr(pp.Samples, name)).parameters
        # ...and the one that cannot honour it does not offer it
        assert "chains_to_disp" not in inspect.signature(pp.Samples.plot_acceptance_rates).parameters


# --------------------------------------------------------------------------------------
# WS9 bullet 5 (remainder): stages_to_disp accepts stage NAMES (surrDAMH.stages.stage_name())
# in addition to the existing positional indices, resolved by Samples._resolve_stages.
# --------------------------------------------------------------------------------------
def _two_named_stage_samples(tmp_path, seed=0):
    """Two stages with clearly different means, so selecting by name is observable."""
    root = str(tmp_path)
    rng = np.random.default_rng(seed)
    stage0 = [[[1, float(x), float(y), -1.0] for x, y in rng.normal(loc=0.0, size=(60, 2))]]
    stage1 = [[[1, float(x), float(y), -1.0] for x, y in rng.normal(loc=10.0, size=(60, 2))]]
    _write_samples_stage(root, "alg0000_MH", stage0)
    _write_samples_stage(root, "alg0001_DAMH-SMU", stage1)
    return pp.Samples(no_parameters=2, samples_dir=root)


class TestStagesToDispByName:
    def test_old_integer_list_calls_are_unaffected(self, tmp_path):
        """(a) plain int lists/None still behave exactly as before."""
        samples = _two_named_stage_samples(tmp_path, seed=40)
        mean_stage0, _ = samples.get_mean_and_cov(stages_to_disp=[0])
        mean_stage1, _ = samples.get_mean_and_cov(stages_to_disp=[1])
        mean_all, _ = samples.get_mean_and_cov()  # None -> all stages, as before
        assert np.all(mean_stage0 < 1.0)
        assert np.all(mean_stage1 > 9.0)
        np.testing.assert_allclose(mean_all, (mean_stage0 + mean_stage1) / 2, rtol=1e-12)

    def test_name_list_resolves_to_the_same_stage_as_the_equivalent_index_list(self, tmp_path):
        """(b) a name resolves to the same stage as its positional index, incl. mixed lists."""
        samples = _two_named_stage_samples(tmp_path, seed=41)
        mean_by_index, cov_by_index = samples.get_mean_and_cov(stages_to_disp=[1])
        mean_by_name, cov_by_name = samples.get_mean_and_cov(stages_to_disp=["alg0001_DAMH-SMU"])
        np.testing.assert_array_equal(mean_by_index, mean_by_name)
        np.testing.assert_array_equal(cov_by_index, cov_by_name)

        mean_mixed, _ = samples.get_mean_and_cov(stages_to_disp=[0, "alg0001_DAMH-SMU"])
        mean_all, _ = samples.get_mean_and_cov()
        np.testing.assert_array_equal(mean_mixed, mean_all)

        assert samples._resolve_stages(["alg0000_MH", "alg0001_DAMH-SMU"]) == [0, 1]

    def test_unknown_name_raises_a_clear_error(self, tmp_path):
        """(c) an unknown name raises, rather than silently doing nothing."""
        samples = _two_named_stage_samples(tmp_path, seed=42)
        with pytest.raises(ValueError, match="stages_to_disp"):
            samples.get_mean_and_cov(stages_to_disp=["not_a_real_stage"])
        with pytest.raises(ValueError, match="not_a_real_stage"):
            samples._resolve_stages(["not_a_real_stage"])


class TestExplicitExceptions:
    def test_decompress_samples_false_says_so(self, tmp_path):
        """
        Finding P6: with ``decompress_samples=False`` these methods used to print
        "CHAIN ... NOT AVAILABLE" per chain from an ``except BaseException``, blaming the
        data for a constructor choice. They now raise and name the choice.
        """
        root = str(tmp_path)
        _write_samples_stage(root, "alg0000_MH", [[[1, 0.0, 0.0, -1.0], [1, 1.0, 1.0, -1.0]]])
        samples = pp.Samples(no_parameters=2, samples_dir=root, decompress_samples=False)
        with pytest.raises(AttributeError, match="decompress_samples=False"):
            samples._collect_samples_matrix()
        with pytest.raises(AttributeError, match="decompress_samples=False"):
            samples.plot_chains()

    def test_compressed_statistics_still_work_without_decompression(self, tmp_path):
        """get_mean_and_cov uses the compressed rows, so it is unaffected."""
        root = str(tmp_path)
        _write_samples_stage(root, "alg0000_MH", [[[2, 1.0, 10.0, -1.0], [1, 3.0, 30.0, -1.0]]])
        samples = pp.Samples(no_parameters=2, samples_dir=root, decompress_samples=False)
        mean, _ = samples.get_mean_and_cov()
        np.testing.assert_allclose(mean, np.average([[1.0, 10.0], [3.0, 30.0]], axis=0,
                                                    weights=[2, 1]))

    def test_malformed_burn_in_is_reported_once(self, tmp_path):
        samples = _two_chain_samples(tmp_path, n=20, seed=17)
        with pytest.raises(ValueError, match="burn_in"):
            samples.get_mean_and_cov(burn_in=[[0]])          # 1 chain given, 2 exist
        with pytest.raises(ValueError, match="burn_in"):
            samples.get_mean_and_cov(stages_to_disp=[0], burn_in=[[0, 0], [0, 0]])  # 2 stages


class TestAutocorrelationChainCount:
    def test_chain_count_comes_from_the_analysed_stage(self, tmp_path):
        """
        Finding P7: ``no_chains`` used to be read off GLOBAL stage 0 even when stage 0 was
        not among the stages analysed. Here stage 0 has one chain and stage 1 has two.
        """
        root = str(tmp_path)
        rng = np.random.default_rng(21)
        one_chain = [[[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(120, 2))]]
        two_chains = [[[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(120, 2))]
                      for _ in range(2)]
        _write_samples_stage(root, "alg0000_MH", one_chain)
        _write_samples_stage(root, "alg0001_DAMH", two_chains)
        samples = pp.Samples(no_parameters=2, samples_dir=root)
        assert [stage.no_chains for stage in samples.list_of_stages] == [1, 2]

        autocorr = pp.Autocorrelation(samples, stages_to_disp=[1])
        assert autocorr.no_chains == 2                    # was 1 before WS9b
        autocorr.calculate_autocorr_function()
        assert len(autocorr.autocorr_function) == 2

    def test_chains_to_disp_selects_the_chains(self, tmp_path):
        samples = _two_chain_samples(tmp_path, n=150, seed=22)
        assert pp.Autocorrelation(samples, stages_to_disp=[0]).no_chains == 2
        assert pp.Autocorrelation(samples, stages_to_disp=[0], chains_to_disp=[1]).no_chains == 1


class TestReportConfigurationSection:
    """WS9 bullet 5: the report states the effective configuration and the unverified options."""

    def _report(self, tmp_path, unverified_options=None, **kwargs):
        root = str(tmp_path)
        rng = np.random.default_rng(41)
        rows = [[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(60, 2))]
        _write_samples_stage(root, "alg0000_MH", [rows, rows])
        _write_manifest(root, 2, 0, unverified_options=unverified_options)
        samples = pp.Samples(no_parameters=2, samples_dir=root)
        output_file = os.path.join(root, "report.html")
        samples.html_report_extended(output_file=output_file, **kwargs)
        with open(output_file) as f:
            return f.read()

    def test_effective_configuration_from_the_manifest_is_a_table(self, tmp_path):
        html = self._report(tmp_path)
        assert "sampling_output/run_manifest.json" in html
        assert "<tr><td>no_parameters</td><td>2</td></tr>" in html
        assert "<tr><td>no_observations</td><td>0</td></tr>" in html

    def test_unverified_options_are_listed(self, tmp_path):
        html = self._report(tmp_path, unverified_options=["use_surrogate_gradients was disabled by SamplingFramework"])
        assert "Unverified options" in html
        assert "use_surrogate_gradients was disabled by SamplingFramework" in html

    def test_no_unverified_options_says_so(self, tmp_path):
        html = self._report(tmp_path)
        assert "Unverified options" in html
        assert "no option flagged as" in html

    def test_explicit_configuration_object_wins(self, tmp_path):
        html = self._report(tmp_path, configuration={"proposal_sd": 0.5})
        assert "the configuration object passed to this report" in html
        assert "<tr><td>proposal_sd</td><td>0.5</td></tr>" in html


class _FakeConf:
    """Minimal stand-in for ``Configuration``: ``save_carry_over`` only reads ``output_dir``."""

    def __init__(self, output_dir):
        self.output_dir = str(output_dir)


class TestCarryOverSection:
    """
    2026-09-21: "5. Proposal Adaptation" gains a "Carried over to the next stage" subsection,
    sourced from ``sampling_output/carry_over/<stage>.npz`` (``RunData.carry_over`` /
    ``Samples.carry_over``, written by ``modules.continuation.save_carry_over``).
    """

    def _report(self, tmp_path, write_carry_over=True, **kwargs):
        root = str(tmp_path)
        rng = np.random.default_rng(7)
        rows0 = [[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(30, 2))]
        rows1 = [[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(20, 2))]
        _write_samples_stage(root, "alg0000_MH-adaptive", [rows0])
        _write_samples_stage(root, "alg0001_MH", [rows1])
        stages = [_stage_spec(name="alg0000_MH-adaptive", algorithm="MH"),
                 _stage_spec(name="alg0001_MH", algorithm="MH", proposal=RandomWalk(scale=0.4))]
        _write_manifest(root, 2, 0, stages=stages)

        if write_carry_over:
            cov = np.array([[0.09, 0.02], [0.02, 0.04]])  # sd = [0.3, 0.2]
            save_carry_over(_FakeConf(root), "alg0000_MH-adaptive",
                            {"scale": cov},
                            {"base_cov": cov, "log_sigma": 0.0, "n_pooled": 500,
                             "mean": np.array([0.1, -0.1])})

        samples = pp.Samples(no_parameters=2, samples_dir=root)
        output_file = os.path.join(root, "report.html")
        samples.html_report_extended(output_file=output_file, **kwargs)
        with open(output_file) as f:
            return f.read()

    def test_carry_over_section_and_sd_table_present(self, tmp_path):
        html = self._report(tmp_path)
        assert "Carried over to the next stage" in html
        assert "consumed by stage alg0001_MH" in html
        assert "Proposal standard deviations" in html
        assert "0.3" in html  # sqrt(0.09), the sd of parameter 0

    def test_last_stage_says_nothing_consumed_it(self, tmp_path):
        """The manifest's second (non-adaptive) stage has no successor."""
        cov = np.array([[0.09, 0.0], [0.0, 0.04]])
        root = str(tmp_path)
        rng = np.random.default_rng(9)
        rows = [[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(20, 2))]
        _write_samples_stage(root, "alg0000_MH-adaptive", [rows])
        _write_manifest(root, 2, 0, stages=[_stage_spec(name="alg0000_MH-adaptive", algorithm="MH")])
        save_carry_over(_FakeConf(root), "alg0000_MH-adaptive", {"scale": cov},
                        {"base_cov": cov, "log_sigma": 0.0, "n_pooled": 200, "mean": np.zeros(2)})
        samples = pp.Samples(no_parameters=2, samples_dir=root)
        output_file = os.path.join(root, "report.html")
        samples.html_report_extended(output_file=output_file)
        with open(output_file) as f:
            html = f.read()
        assert "this was the last stage; nothing consumed it" in html

    def test_missing_carry_over_file_shows_an_orange_note(self, tmp_path):
        html = self._report(tmp_path, write_carry_over=False)
        assert "no carry_over file" in html
        assert "Carried over to the next stage" not in html


class TestPoolModeReportNote:
    """
    Finding 2.7: ``use_solvers_pool=True`` leaves the reporting rank without a live
    Solver, so ``par_names``/posterior field statistics/the best-fit solver
    visualization used to be dropped with no note anywhere. ``html_report_extended``'s
    new ``pool_mode_note`` argument (wired from ``SamplingRun._write_report_rank0``
    in ``core.py``) makes the report say so instead of silently falling back.
    """

    NOTE = "Not available: pool mode (`use_solvers_pool=True`) has no live Solver object on the reporting rank."

    def _report(self, tmp_path, **kwargs):
        samples = _two_chain_samples(tmp_path, n=30, seed=50)
        output_file = os.path.join(str(tmp_path), "report.html")
        samples.html_report_extended(output_file=output_file, **kwargs)
        with open(output_file) as f:
            return f.read()

    def test_par_names_gap_is_explained_when_pool_mode_note_given(self, tmp_path):
        html = self._report(tmp_path, par_names=None, pool_mode_note=self.NOTE)
        assert self.NOTE in html
        assert "generic labels" in html

    def test_par_names_note_absent_without_pool_mode_note(self, tmp_path):
        """Non-pool runs (pool_mode_note=None) keep the plain fallback, unexplained."""
        html = self._report(tmp_path, par_names=None)
        assert self.NOTE not in html

    def test_par_names_note_absent_when_par_names_is_explicitly_given(self, tmp_path):
        """A caller-supplied par_names list means there is no gap to explain."""
        html = self._report(tmp_path, par_names=["a", "b"], pool_mode_note=self.NOTE)
        assert "generic labels" not in html

    def test_field_statistics_section_shows_the_note_instead_of_the_generic_message(self, tmp_path):
        html = self._report(tmp_path, field_statistics=None, pool_mode_note=self.NOTE)
        assert self.NOTE in html
        assert "No posterior field statistics were supplied for this report." not in html

    def test_field_statistics_section_keeps_generic_message_without_pool_mode_note(self, tmp_path):
        html = self._report(tmp_path, field_statistics=None)
        assert "No posterior field statistics were supplied for this report." in html


class TestBestFitVisualizationNote:
    """Finding 2.7: core._insert_best_fit_visualization_note fills the (previously
    nonexistent) best-fit solver visualization section of an already-written report."""

    def _base_html(self, tmp_path):
        path = os.path.join(str(tmp_path), "report.html")
        with open(path, "w") as f:
            f.write("<html><body><p>existing content</p></body></html>")
        return path

    def test_pool_mode_note_is_inserted(self, tmp_path):
        from surrDAMH.core import _insert_best_fit_visualization_note
        path = self._base_html(tmp_path)
        _insert_best_fit_visualization_note(path, pool_mode_note="Not available: pool mode.",
                                            image_filenames=[])
        html = open(path).read()
        assert "Not available: pool mode." in html
        assert "<p>existing content</p>" in html  # additive: nothing existing was removed

    def test_images_are_embedded_when_no_pool_mode_note(self, tmp_path):
        from surrDAMH.core import _insert_best_fit_visualization_note
        path = self._base_html(tmp_path)
        _insert_best_fit_visualization_note(
            path, pool_mode_note=None,
            image_filenames=["best_fit_solver_visualization_1.png"])
        html = open(path).read()
        assert '<img src="best_fit_solver_visualization_1.png"' in html

    def test_neutral_message_when_nothing_available(self, tmp_path):
        from surrDAMH.core import _insert_best_fit_visualization_note
        path = self._base_html(tmp_path)
        _insert_best_fit_visualization_note(path, pool_mode_note=None, image_filenames=[])
        html = open(path).read()
        assert "No best-fit solver visualization was generated" in html


class TestCollapsibleReport:
    """
    2026-09-21 layout: every top-level section and every per-stage block is a
    ``<details>`` element that is collapsed when the report is opened; the only
    ``<details ... open>`` is the table of contents. An "Expand all" / "Collapse all"
    control pair is provided.
    """

    def test_sections_are_collapsed_details_and_toc_is_open(self, tmp_path):
        samples = _two_chain_samples(tmp_path, seed=60)
        output_file = os.path.join(str(tmp_path), "report.html")
        samples.html_report_extended(output_file=output_file)
        with open(output_file) as f:
            html = f.read()

        details_tags = re.findall(r'<details[^>]*>', html)
        section_tags = [t for t in details_tags if 'class="section"' in t]
        assert section_tags, 'no <details class="section"> found'
        assert all(' open' not in t for t in section_tags)

        toc_tags = [t for t in details_tags if 'class="toc"' in t]
        assert len(toc_tags) == 1
        assert ' open' in toc_tags[0]

        assert 'Expand all' in html
        assert html.count('<details') == html.count('</details>')

    def test_per_stage_blocks_are_collapsed_details_with_expensive_sections(self, tmp_path):
        samples = _two_named_stage_samples(tmp_path, seed=61)
        output_file = os.path.join(str(tmp_path), "report.html")
        samples.html_report_extended(output_file=output_file, include_expensive_sections=True)
        with open(output_file) as f:
            html = f.read()

        for stage_name in ("alg0000_MH", "alg0001_DAMH-SMU"):
            for prefix in ("individual_", "autocorr_"):
                block_id = f"{prefix}{stage_name}"
                match = re.search(rf'<details class="stage" id="{re.escape(block_id)}"[^>]*>', html)
                assert match is not None, f"missing block {block_id!r}"
                assert ' open' not in match.group(0)

        assert html.count('<details') == html.count('</details>')


class TestStagesSpecificationSection:
    """
    2026-09-21: the new "Sampling Stages" section renders either the ``stages=`` argument
    or ``manifest["stages"]`` as a table (one column per stage, one row per ``Stage``
    field), and says which source it used.
    """

    def _two_stage_run_with_manifest_stages(self, tmp_path, seed=70):
        root = str(tmp_path)
        rng = np.random.default_rng(seed)
        rows0 = [[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(30, 2))]
        rows1 = [[1, float(x), float(y), -1.0] for x, y in rng.normal(loc=5.0, size=(30, 2))]
        _write_samples_stage(root, "alg0000_MH-adaptive", [rows0])
        _write_samples_stage(root, "alg0001_DAMH", [rows1])
        stage_specs = [
            _stage_spec(name="alg0000_MH-adaptive", algorithm="MH", proposal=RandomWalk(scale=0.5),
                       max_samples=600),
            _stage_spec(name="alg0001_DAMH", algorithm="DAMH", proposal=RandomWalk(scale=0.3),
                       max_samples=400),
        ]
        _write_manifest(root, 2, 0, stages=stage_specs)
        return root

    def test_manifest_stages_render_as_a_table(self, tmp_path):
        root = self._two_stage_run_with_manifest_stages(tmp_path, seed=71)
        samples = pp.Samples(no_parameters=2, samples_dir=root)
        output_file = os.path.join(root, "report.html")
        samples.html_report_extended(output_file=output_file, stages_to_disp=[0])
        with open(output_file) as f:
            html = f.read()

        assert "<td>proposal *</td>" in html
        assert "RandomWalk" in html
        # sys.maxsize (both stages' max_evaluations) renders as "unbounded"
        assert "<tr><td>max_evaluations</td><td>unbounded</td><td>unbounded</td></tr>" in html
        assert "sampling_output/run_manifest.json" in html
        assert "from the stage list passed to this report" not in html
        # stages_to_disp=[0]: stage 0 included, stage 1 not
        assert "<tr><td>included in this report</td><td>yes</td><td>no</td></tr>" in html

    def test_stages_argument_overrides_the_manifest(self, tmp_path):
        root = self._two_stage_run_with_manifest_stages(tmp_path, seed=72)
        samples = pp.Samples(no_parameters=2, samples_dir=root)
        stage_objs = [
            Stage(algorithm="MH", proposal=RandomWalk(), max_samples=10, max_evaluations=10),
            Stage(algorithm="DAMH", proposal=RandomWalk(), max_samples=10, max_evaluations=10),
        ]
        output_file = os.path.join(root, "report.html")
        samples.html_report_extended(output_file=output_file, stages=stage_objs)
        with open(output_file) as f:
            html = f.read()

        assert "from the stage list passed to this report" in html

    def test_no_stages_anywhere_says_so(self, tmp_path):
        samples = _two_chain_samples(tmp_path, seed=73)
        output_file = os.path.join(str(tmp_path), "report.html")
        samples.html_report_extended(output_file=output_file)
        with open(output_file) as f:
            html = f.read()

        assert "No stage specification was recorded" in html


class TestProposalAdaptationSection:
    """
    2026-09-21: the "Proposal Adaptation" section (per displayed adaptive stage) and the
    underlying ``SamplesPlots.plot_adaptation`` it calls.
    """

    RW_HEADER = ["n", "mean_acceptance_probability", "log_sigma", "trace_C_over_d",
                "shrinkage_delta", "rank_world"]
    PCN_HEADER = ["n", "mean_acceptance_probability", "beta", "rank_world"]

    def _write_adaptive_stats(self, root, stage_name, header, chains_rows):
        adaptive_dir = os.path.join(root, "sampling_output", "adaptive_stats", stage_name)
        for c, rows in enumerate(chains_rows):
            _write_csv(adaptive_dir, f"rank{c:04d}.csv", header, rows)

    def _rw_rows(self, rank):
        return [[10, 0.3, 0.7, float('nan'), float('nan'), rank],
                [20, 0.25, 0.72, 0.5, 0.04, rank]]

    def _pcn_rows(self, rank):
        return [[10, 0.2, 0.5, rank], [20, 0.22, 0.48, rank]]

    def _report_fixture(self, tmp_path, seed=80):
        """Stage 0: adaptive RW with a 2-chain trace. Stage 1: declared non-adaptive."""
        root = str(tmp_path)
        rng = np.random.default_rng(seed)
        rows0 = [[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(30, 2))]
        rows1 = [[1, float(x), float(y), -1.0] for x, y in rng.normal(loc=5.0, size=(30, 2))]
        _write_samples_stage(root, "alg0000_MH-adaptive", [rows0, rows0])
        _write_samples_stage(root, "alg0001_DAMH", [rows1])
        self._write_adaptive_stats(root, "alg0000_MH-adaptive", self.RW_HEADER,
                                   [self._rw_rows(0), self._rw_rows(1)])
        stage_specs = [
            _stage_spec(name="alg0000_MH-adaptive", algorithm="MH"),
            _stage_spec(name="alg0001_DAMH", algorithm="DAMH", proposal=RandomWalk(scale=0.3)),
        ]
        _write_manifest(root, 2, 0, stages=stage_specs)
        return root

    def test_adaptive_stage_shows_a_block_and_the_non_adaptive_stage_does_not(self, tmp_path):
        root = self._report_fixture(tmp_path, seed=81)
        samples = pp.Samples(no_parameters=2, samples_dir=root)
        output_file = os.path.join(root, "report.html")
        samples.html_report_extended(output_file=output_file, include_expensive_sections=False)
        with open(output_file) as f:
            html = f.read()

        assert 'id="adaptation_alg0000_MH-adaptive"' in html
        assert 'alt="Proposal adaptation alg0000_MH-adaptive"' in html
        assert "Last logged period" in html
        assert 'id="adaptation_alg0001_' not in html

    def test_no_adaptive_stage_says_so(self, tmp_path):
        samples = _two_chain_samples(tmp_path, seed=82)
        output_file = os.path.join(str(tmp_path), "report.html")
        samples.html_report_extended(output_file=output_file)
        with open(output_file) as f:
            html = f.read()

        assert "No displayed stage used an adaptive proposal." in html

    def test_declared_adaptive_without_a_trace_says_so(self, tmp_path):
        root = str(tmp_path)
        rng = np.random.default_rng(83)
        rows = [[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(20, 2))]
        _write_samples_stage(root, "alg0000_MH-adaptive", [rows])
        _write_manifest(root, 2, 0, stages=[
            _stage_spec(name="alg0000_MH-adaptive", algorithm="MH")])
        samples = pp.Samples(no_parameters=2, samples_dir=root)
        output_file = os.path.join(root, "report.html")
        samples.html_report_extended(output_file=output_file)
        with open(output_file) as f:
            html = f.read()

        assert "no adaptive_stats trace was found" in html

    # -- SamplesPlots.plot_adaptation, called directly ------------------------------------

    def _samples_with_rw_and_pcn_traces(self, tmp_path, seed=90):
        """Stage 0: adaptive RW (2 chains). Stage 1: adaptive pCN (2 chains). Stage 2: non-adaptive."""
        root = str(tmp_path)
        rng = np.random.default_rng(seed)
        rows0 = [[1, float(x), float(y), -1.0] for x, y in rng.normal(size=(20, 2))]
        rows1 = [[1, float(x), float(y), -1.0] for x, y in rng.normal(loc=5.0, size=(20, 2))]
        rows2 = [[1, float(x), float(y), -1.0] for x, y in rng.normal(loc=-5.0, size=(20, 2))]
        _write_samples_stage(root, "alg0000_MH-adaptive", [rows0, rows0])
        _write_samples_stage(root, "alg0001_MH-adaptive-pCN", [rows1, rows1])
        _write_samples_stage(root, "alg0002_MH", [rows2])
        self._write_adaptive_stats(root, "alg0000_MH-adaptive", self.RW_HEADER,
                                   [self._rw_rows(0), self._rw_rows(1)])
        self._write_adaptive_stats(root, "alg0001_MH-adaptive-pCN", self.PCN_HEADER,
                                   [self._pcn_rows(0), self._pcn_rows(1)])
        return pp.Samples(no_parameters=2, samples_dir=root)

    def test_random_walk_trace_has_four_panels(self, tmp_path):
        import matplotlib.pyplot as plt

        samples = self._samples_with_rw_and_pcn_traces(tmp_path, seed=91)
        fig, axes = samples.plot_adaptation(0)
        assert fig is not None
        assert len(axes) == 4
        plt.close(fig)

    def test_pcn_trace_has_two_panels(self, tmp_path):
        import matplotlib.pyplot as plt

        samples = self._samples_with_rw_and_pcn_traces(tmp_path, seed=92)
        fig, axes = samples.plot_adaptation(1)
        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)

    def test_chains_to_disp_selects_the_second_chain(self, tmp_path):
        import matplotlib.pyplot as plt

        samples = self._samples_with_rw_and_pcn_traces(tmp_path, seed=93)
        fig, axes = samples.plot_adaptation(0, chains_to_disp=[1])
        assert fig is not None
        # acceptance-probability axis: one line per displayed chain (no target_rate line here)
        assert len(axes[0].get_lines()) == 1
        plt.close(fig)

    def test_unknown_chain_returns_none(self, tmp_path):
        samples = self._samples_with_rw_and_pcn_traces(tmp_path, seed=94)
        fig, axes = samples.plot_adaptation(0, chains_to_disp=[7])
        assert fig is None
        assert axes is None

    def test_non_adaptive_stage_returns_none(self, tmp_path):
        samples = self._samples_with_rw_and_pcn_traces(tmp_path, seed=95)
        fig, axes = samples.plot_adaptation(2)
        assert fig is None
        assert axes is None


class TestFiguresAreNotLeaked:
    def test_failing_plot_closes_its_figure(self, tmp_path):
        """Finding P8: a plot that raises used to leave its figure open."""
        import matplotlib.pyplot as plt

        samples = _two_chain_samples(tmp_path, n=30, seed=31)
        plt.close("all")
        before = len(plt.get_fignums())
        # parameters_to_disp addresses a parameter the run does not have -> IndexError deep
        # inside the per-parameter loop, after plt.subplots() has already created the figure
        with pytest.raises(IndexError):
            samples.plot_chains(parameters_to_disp=[0, 1, 5])
        assert len(plt.get_fignums()) == before

    def test_extended_report_closes_every_figure_it_makes(self, tmp_path):
        import matplotlib.pyplot as plt

        samples = _two_chain_samples(tmp_path, n=80, seed=33)
        plt.close("all")
        samples.html_report_extended(output_file=os.path.join(str(tmp_path), "report.html"),
                                     include_expensive_sections=True)
        assert plt.get_fignums() == []

    def test_successful_plot_hands_the_figure_to_the_caller(self, tmp_path):
        import matplotlib.pyplot as plt

        samples = _two_chain_samples(tmp_path, n=30, seed=32)
        plt.close("all")
        fig, _ = samples.plot_chains()
        assert len(plt.get_fignums()) == 1
        plt.close(fig)
        assert len(plt.get_fignums()) == 0


# --------------------------------------------------------------------------------------
# Item 40: golden-file test of the column layout of samples/raw_data/notes/subchain_stats.
# --------------------------------------------------------------------------------------
GOLDEN_A = np.array([[1.0, 0.5], [0.0, 2.0]])


class _GoldenSolver(Solver):
    """``G(x) = GOLDEN_A @ x`` -- defined here so the golden values depend only on this
    file and on the library, not on a shared fixture someone may retune."""

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return GOLDEN_A @ self.parameters


class _GoldenInitialSample:
    def __init__(self, parameters):
        self.parameters = np.array(parameters, dtype=float)

    def rvs(self):
        return self.parameters.copy()


def _run_golden_sampling(output_dir):
    """MH -> DAMH-SMU(K=2, KDTree surrogate), fixed initial sample, fixed seeds."""
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=2, output_dir=str(output_dir),
                                  use_collector=False, use_solvers_pool=False,
                                  save_snapshots_to_file=True,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=_GoldenInitialSample([0.3, -0.7]))
    prior = surrDAMH.distributions.PriorIndependentComponents([
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    ])
    likelihood = surrDAMH.distributions.Normal(mean=np.array([1.0, 2.0]), sd=np.array([0.5, 0.5]))
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_samples=20, max_evaluations=20),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_samples=20, max_evaluations=20,
              subchain_length=2, surrogate_model_updates=True),
    ]
    updater = KDTreeUpdater(no_parameters=2, no_observations=2, no_nearest_neighbors=2)
    return surrDAMH.Problem(prior, likelihood, solver=_GoldenSolver()).run_sampling_local(
        conf, stages, surrogate_updater=updater)


# Expected header + first 3 data rows of every per-stage CSV of the run above.
# PROVENANCE: generated once on 2026-09-17 by running ``_run_golden_sampling`` with the
# format-v2 writers and pasting the files' first lines verbatim (the run is deterministic:
# fixed initial sample, seeds from ``modules.seeds``, no MPI). They are NOT hand-derived
# reference values -- their job is to make any future drift of the OUTPUT FORMAT (column
# names, column order, NaN placement, the exact/surrogate observation split) visible.
GOLDEN_FILES = {
    "samples/alg0000_MH/rank0000.csv": [
        "multiplicity,par_0,par_1,log_posterior",
        "10,0.3,-0.7,-25.615",
        "1,0.3211068733577964,-0.40859239314208884,-17.568135685343403",
        "1,-0.2292027152486642,0.16376946177771823,-8.26660637955927",
    ],
    "samples/alg0001_DAMH-SMU/rank0000.csv": [
        "multiplicity,par_0,par_1,log_posterior",
        "1,0.9175496937896609,1.0512164448864132,-1.3872395985407036",
        "2,0.6450926979858279,1.0491020972171126,-0.8352260288160988",
        "1,0.9899338530865385,1.329698185233753,-3.101121707389872",
    ],
    "raw_data/alg0000_MH/rank0000.csv": [
        "state_type,par_0,par_1,solver_tag,obs_0,obs_1,obs_approx_0,obs_approx_1,log_likelihood,log_prior",
        "rejected,1.1121726818316209,-1.0058782068250376,0,0.6092335784191021,-2.011756413650075,nan,nan,-32.493775837395305,-1.1243595205890464",
        "rejected,0.03591412386827214,-1.2364843110780852,0,-0.5823280316707704,-2.4729686221561704,nan,nan,-45.022420589209524,-0.7650916379177363",
        "rejected,0.7327038146623392,-1.8507693484401413,0,-0.1926808595577314,-3.7015386968802826,nan,nan,-67.86006228955736,-1.9811010305731442",
    ],
    "raw_data/alg0001_DAMH-SMU/rank0000.csv": [
        "state_type,par_0,par_1,solver_tag,obs_0,obs_1,obs_approx_0,obs_approx_1,log_likelihood,log_prior",
        "prerejected,0.9175496937896609,1.0512164448864132,0,nan,nan,0.4284109847511373,1.1000596440243484,-2.2732132933334945,-0.973476727286565",
        "accepted,0.6450926979858279,1.0491020972171126,0,1.1696437465943843,2.098204194434225,0.42837884062725884,1.1126394476408834,-0.07684612912610954,-0.7583798996899892",
        "prerejected,0.6450926979858279,1.0491020972171126,0,nan,nan,1.1491229651741213,1.8875576560898486,-0.06976187889266189,-0.7583798996899892",
    ],
    # out_of_bounds appended 2026-10-09 (S0); exact_steps..audit_hidden appended 2026-10-09 (S2);
    # polyak_updates appended 2026-10-09 (S3)
    "notes/alg0000_MH/rank0000.csv": [
        "accepted,rejected,pre-rejected,sum,seed,out_of_bounds,exact_steps,exact_accepted,audited,audit_hidden,polyak_updates",
        "5,15,0,20,2,0,0,0,0,0,0",
    ],
    "notes/alg0001_DAMH-SMU/rank0000.csv": [
        "accepted,rejected,pre-rejected,sum,seed,out_of_bounds,exact_steps,exact_accepted,audited,audit_hidden,polyak_updates",
        "10,8,2,20,12,0,0,0,0,0,0",
    ],
    "subchain_stats/alg0001_DAMH-SMU/rank0000.csv": [  # kernel appended 2026-10-09 (S2)
        "iteration,subchain_max_length,subchain_accepted,subchain_acceptance_rate,correction_log_ratio,outer_proposed_changed,outer_accepted,rank_world,kernel",
        "0,2,0,0.0,0.0,0,0,0,0",
        "1,2,2,1.0,0.044894293882068226,1,1,0,0",
        "2,2,0,0.0,0.0,0,0,0,0",
    ],
}


def _assert_field_equal(expected: str, actual: str, where: str):
    """Exact match for text fields; numeric fields compared at 1e-12 relative tolerance."""
    try:
        expected_value = float(expected)
    except ValueError:
        assert actual == expected, f"{where}: expected {expected!r}, got {actual!r}"
        return
    actual_value = float(actual)
    if np.isnan(expected_value):
        assert np.isnan(actual_value), f"{where}: expected NaN, got {actual!r}"
    else:
        assert actual_value == pytest.approx(expected_value, rel=1e-12, abs=1e-12), \
            f"{where}: expected {expected!r}, got {actual!r}"


def test_golden_file_column_layout(tmp_path):
    """
    Item 40 (was deferred to WS9, implemented in WS9a): pin the on-disk OUTPUT FORMAT of a
    short, deterministic two-stage run -- header row and first three data rows of every
    ``samples``/``raw_data``/``notes``/``subchain_stats`` file.

    Headers are compared verbatim; data fields are compared as values (text exact, numbers
    to 1e-12 relative, NaN vs non-NaN exact), so the test fails on a changed column name,
    a changed column order, a lost/added column, a differently placed NaN or a changed
    exact-vs-surrogate observation split -- the things format v2 fixes -- without being
    brittle about a last-ulp difference in an unrelated library.

    The expected values were GENERATED ONCE with the v2 writers on 2026-09-17 and pasted in
    (see ``GOLDEN_FILES``); to regenerate, run ``_run_golden_sampling`` and copy the first
    four lines of each file.
    """
    result = _run_golden_sampling(tmp_path)
    assert [stage.name for stage in result.stage_results] == ["alg0000_MH", "alg0001_DAMH-SMU"]

    sampling_output = tmp_path / "sampling_output"
    for relative_path, expected_lines in GOLDEN_FILES.items():
        path = sampling_output / relative_path
        assert path.is_file(), f"{relative_path} was not written"
        lines = path.read_text().splitlines()
        assert len(lines) >= len(expected_lines), f"{relative_path}: only {len(lines)} lines"
        assert lines[0] == expected_lines[0], f"{relative_path}: header changed"
        for row_index, expected_line in enumerate(expected_lines[1:], start=1):
            expected_fields = expected_line.split(",")
            actual_fields = lines[row_index].split(",")
            assert len(actual_fields) == len(expected_fields), (
                f"{relative_path} row {row_index}: {len(actual_fields)} fields, "
                f"expected {len(expected_fields)}")
            for field_index, (expected, actual) in enumerate(zip(expected_fields, actual_fields)):
                _assert_field_equal(expected, actual,
                                    f"{relative_path} row {row_index} field {field_index} "
                                    f"({expected_lines[0].split(',')[field_index]})")

    # and the whole thing is readable back through the public reader
    samples = pp.Samples(no_parameters=2, samples_dir=str(tmp_path))
    assert samples.stage_names == ["alg0000_MH", "alg0001_DAMH-SMU"]
    assert not samples.summary.empty
