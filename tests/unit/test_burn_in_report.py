#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``Stage.burn_in``, every chain visible in the report, the posterior over the selection mask
(2026-10-09, ``API_SPEC_burn_in_report`` §1-§4).

* ``burn_in`` continues the chain into the next stage, ``is_excluded`` restarts it; both set the
  default ``include = 0`` of ``selection.json``; both together are refused;
* ``Samples`` loads every chain, ``Samples.posterior_view()`` is the posterior subset;
* the report shows every chain with a badge, aggregates the pooled sections over the included
  chains only, writes ``chains_in_posterior`` to ``summary.csv`` and tells the user how to change
  the selection;
* the automatic mode's warm-up is a ``burn_in`` stage: chunk 1 continues from its last state.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import pytest

import surrDAMH
import surrDAMH.post_processing as pp
from surrDAMH.modules.run_data import read_run
from surrDAMH.post_processing.selection import default_selection, selection_path, stage_flags
from surrDAMH.proposals import RandomWalk
from surrDAMH.stages import Stage

from .test_lineage_report import _html, _selection, _summary_csv, _synthetic_run, _write_selection

pytestmark = pytest.mark.unit


class _ToySolver(surrDAMH.solvers.Solver):
    def __init__(self, solver_id=0, output_dir=None):
        super().__init__(solver_id, output_dir, no_parameters=2, no_observations=1)

    def set_parameters(self, parameters):
        self.x = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return np.array([self.x[0] + 0.5 * self.x[1]])


def _problem():
    prior = surrDAMH.distributions.Normal(mean=0.0, sd=1.0, dim=2)
    likelihood = surrDAMH.distributions.Normal(mean=[0.3], sd=0.5)
    return surrDAMH.Problem(prior, likelihood, solver=_ToySolver())


def _local_run(output_dir, stages):
    conf = surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False)
    return _problem().run_sampling_local(conf, stages)


def _rows(output_dir, stage_name, chain=0):
    path = os.path.join(str(output_dir), "sampling_output", "samples", stage_name, f"rank{chain:04d}.csv")
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def _first_stage(flag):
    return Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=40, **{flag: True})


# ---------------------------------------------------------------------------------------------
# §1: the flag
# ---------------------------------------------------------------------------------------------

def test_burn_in_continues_the_chain_is_excluded_restarts_it(tmp_path):
    second = dict(algorithm="MH", proposal=RandomWalk(scale=0.3), max_evaluations=30)
    _local_run(tmp_path / "burn", [_first_stage("burn_in"), Stage(**second)])
    first_rows, next_rows = _rows(tmp_path / "burn", "alg0000_MH"), _rows(tmp_path / "burn", "alg0001_MH")
    # the chain continues: the second stage starts at the burn-in stage's LAST state
    np.testing.assert_array_equal(next_rows[0, 1:3], first_rows[-1, 1:3])
    assert not np.array_equal(first_rows[0, 1:3], first_rows[-1, 1:3])  # the chain moved, so this is a real check

    _local_run(tmp_path / "excl", [_first_stage("is_excluded"), Stage(**second)])
    first_rows, next_rows = _rows(tmp_path / "excl", "alg0000_MH"), _rows(tmp_path / "excl", "alg0001_MH")
    # is_excluded restarts: the second stage starts at the FIRST state of the excluded stage
    np.testing.assert_array_equal(next_rows[0, 1:3], first_rows[0, 1:3])
    assert not np.array_equal(first_rows[0, 1:3], first_rows[-1, 1:3])


def test_burn_in_and_is_excluded_together_are_refused():
    with pytest.raises(ValueError, match="burn_in=True and is_excluded=True"):
        Stage(max_evaluations=10, burn_in=True, is_excluded=True)


def test_flag_in_describe_and_manifest(tmp_path):
    stage = _first_stage("burn_in")
    assert "burn_in*=True" in stage.describe()
    _local_run(tmp_path / "run", [stage, Stage(proposal=RandomWalk(scale=0.5), max_evaluations=20)])
    with open(tmp_path / "run" / "sampling_output" / "run_manifest.json") as f:
        specs = json.load(f)["stages"]
    assert [(s["burn_in"], s["is_excluded"]) for s in specs] == [(True, False), (False, False)]


def test_default_selection_excludes_burn_in_and_is_excluded_stages(tmp_path):
    _local_run(tmp_path / "run", [_first_stage("burn_in"), _first_stage("is_excluded"),
                                  Stage(proposal=RandomWalk(scale=0.5), max_evaluations=20)])
    run_data = read_run(str(tmp_path / "run"))
    selection = default_selection(run_data)
    assert selection["posterior"] == {"alg0000_MH": [0], "alg0001_MH": [0], "alg0002_MH": [1]}
    flags = stage_flags(run_data)
    assert [(name, flags[name]["burn_in"], flags[name]["is_excluded"]) for name in run_data.stage_names] == [
        ("alg0000_MH", True, False), ("alg0001_MH", False, True), ("alg0002_MH", False, False)]


# ---------------------------------------------------------------------------------------------
# §2/§3: load everything, mask the posterior only; the report
# ---------------------------------------------------------------------------------------------

def _pooled_mean_from_html(html):
    overall = html[html.index('id="overall"'):html.index('id="individual"')]
    block = overall[overall.index("<h4>Posterior Mean:</h4>"):overall.index("<h4>Covariance Matrix:</h4>")]
    return np.array([float(v) for v in re.findall(r"Parameter \d+: (-?\d+\.\d+)", block)])


def test_report_shows_every_chain_and_aggregates_the_included_ones(tmp_path):
    root = _synthetic_run(str(tmp_path / "run"), first_stage_flag="burn_in")
    surrDAMH.SamplingRun.load(root).write_report(include_expensive_sections=False, no_best_fits=0)
    content = _selection(root)
    assert list(content["posterior"].values()) == [[0, 0], [1, 1], [1, 1]]
    flags = stage_flags(read_run(root))
    assert flags["alg0000_MH"] == {"is_excluded": False, "burn_in": True}
    content["posterior"]["alg0002_MH"] = [1, 0]  # chain 1 of the last stage excluded by hand
    _write_selection(root, content)

    samples = surrDAMH.SamplingRun.load(root).write_report(include_expensive_sections=True, no_best_fits=0)
    html = _html(root)

    # every stage and chain loaded and shown, with its badge
    assert samples.chain_indices == [[0, 1], [0, 1], [0, 1]]
    for name in ("alg0000_MH", "alg0001_MH-adaptive", "alg0002_MH"):
        assert f'alt="Stage {name} Chain Traces"' in html
    expected_badges = {("alg0000_MH", 0): "not in posterior (burn-in stage)",
                       ("alg0000_MH", 1): "not in posterior (burn-in stage)",
                       ("alg0001_MH-adaptive", 0): "in posterior", ("alg0001_MH-adaptive", 1): "in posterior",
                       ("alg0002_MH", 0): "in posterior", ("alg0002_MH", 1): "not in posterior (excluded by user)"}
    for (name, chain), status in expected_badges.items():
        assert samples.chain_status(samples.stage_names.index(name), chain, samples.posterior_view()) == status
        block = html[html.index(f'id="individual_{name}"'):]
        block = block[:block.index("</details>")]
        assert f"chain {chain}: {status}</span>" in block, (name, chain)
    assert "not in the posterior as burn-in stages (burn_in=True): alg0000_MH chain 0; alg0000_MH chain 1." in html
    assert "excluded by the user: alg0002_MH chain 1." in html
    assert "Skipped: every chain" not in html and "is excluded by the selection" not in html

    # the pooled mean: included chains only, computed independently from the CSV files
    included = [("alg0001_MH-adaptive", 0), ("alg0001_MH-adaptive", 1), ("alg0002_MH", 0)]
    rows = np.vstack([_rows(root, name, chain) for name, chain in included])
    expected = np.average(rows[:, 1:3], axis=0, weights=rows[:, 0])
    np.testing.assert_allclose(_pooled_mean_from_html(html), expected, atol=1e-6)
    np.testing.assert_allclose(samples.posterior_view().get_mean_and_cov()[0], expected, rtol=1e-12)
    every = np.vstack([_rows(root, name, chain) for name in samples.stage_names for chain in (0, 1)])
    assert not np.allclose(expected, np.average(every[:, 1:3], axis=0, weights=every[:, 0]), atol=0.1)
    assert ("Aggregated over the chains with posterior=1 in selection.json: 2 stages "
            "(alg0001_MH-adaptive, alg0002_MH), 3 chain-stage pairs.") in html

    # summary.csv: every stage, every chain's counters, plus chains_in_posterior (last column)
    summary = _summary_csv(root)
    assert list(summary.index) == samples.stage_names
    assert list(summary.columns)[-1] == "chains_in_posterior"
    assert list(summary["chains_in_posterior"]) == ["0/2", "2/2", "1/2"]
    assert list(summary["accepted"]) == [10 + 20, 11 + 21, 12 + 22]

    # the user note, verbatim, at the top of the overall section and in the selection section
    output_dir = os.path.abspath(root)
    note = (f"Which chains form the posterior is decided by <code>{output_dir}/post_processing_output/selection.json"
            "</code>: set each stage's chain list in <code>posterior</code> to 1 (used) or 0 (not), "
            "then re-run <code>python -c 'import surrDAMH; "
            f"surrDAMH.SamplingRun.load(\"{output_dir}\").write_report()'</code> (pass <code>problem=</code> to "
            "<code>load</code> for the solver-dependent sections).")
    assert html.count(note) == 2
    overall = html.index('id="overall"')
    assert overall < html.index(note, overall) < html.index("<h3>2.1 ")
    assert html.index(note, html.index('id="selection"')) > html.index('id="selection"')
    assert selection_path(root) == f"{output_dir}/post_processing_output/selection.json"


def test_posterior_view_without_exclusions_equals_selection_false(tmp_path):
    run = _local_run(tmp_path / "run", [Stage(proposal=RandomWalk(scale=0.5), max_evaluations=60),
                                        Stage(proposal=RandomWalk(adaptive=True), max_evaluations=60)])
    reference = pp.Samples(2, run.output_dir, selection=False)
    default = run.write_report(include_expensive_sections=False, no_best_fits=0)  # creates selection.json
    assert default.selection_mask is not None
    for view in (default.posterior_view(), reference.posterior_view()):
        assert view.chain_indices == reference.chain_indices
        for got, expected in zip(view.list_of_stages, reference.list_of_stages):
            assert got.no_chains == expected.no_chains and got.length == expected.length
            assert got.no_unique_samples == expected.no_unique_samples
            for name in ("samples_compressed", "weights", "samples"):
                for a, b in zip(getattr(got, name), getattr(expected, name)):
                    np.testing.assert_array_equal(a, b)
        pd.testing.assert_frame_equal(view.summary, reference.summary)
        for a, b in zip(view.notes, reference.notes):
            pd.testing.assert_frame_equal(a, b)
        for a, b in zip(view.get_mean_and_cov(), reference.get_mean_and_cov()):
            np.testing.assert_array_equal(a, b)
        ess, ess_reference = view.calculate_effective_sample_size(), reference.calculate_effective_sample_size()
        assert ess["total_samples"] == ess_reference["total_samples"]
        assert ess["ess_per_param"] == ess_reference["ess_per_param"]
    # the view shares the arrays, nothing is copied
    assert np.shares_memory(default.posterior_view().list_of_stages[0].samples[0], default.list_of_stages[0].samples[0])


def test_posterior_view_burn_in_rows_share_the_arrays(tmp_path):
    root = _synthetic_run(str(tmp_path / "run"), first_stage_flag="burn_in")
    content = default_selection(read_run(root))
    content["drop_first_rows"] = {"alg0002_MH": [7, 0]}
    samples = pp.Samples(2, root, selection=content)
    view = samples.posterior_view()
    full, part = samples.list_of_stages[2], view.list_of_stages[2]
    np.testing.assert_array_equal(part.samples_compressed[0], full.samples_compressed[0][7:])
    offset = int(full.weights[0][:7].sum())
    np.testing.assert_array_equal(part.samples[0], full.samples[0][offset:])
    assert part.length[0] == full.length[0] - offset and np.shares_memory(part.samples[0], full.samples[0])
    assert view.list_of_stages[0].no_chains == 0 and view.chain_indices[0] == []
    assert view.posterior_chains(2) == [0, 1] and samples.chains_in_posterior(view) == ["0/2", "2/2", "2/2"]


# ---------------------------------------------------------------------------------------------
# automatic mode: the warm-up is a burn-in stage, chunk 1 continues from it
# ---------------------------------------------------------------------------------------------

def test_auto_local_chunk_one_continues_from_the_warm_up(tmp_path, linear_gaussian_problem):
    from surrDAMH.surrogates import PolynomialSklearnUpdater
    lgp = linear_gaussian_problem
    conf = surrDAMH.Configuration(output_dir=str(tmp_path / "auto"), use_collector=False, use_solvers_pool=False,
                                  transform_before_saving=False)
    run = surrDAMH.Problem(lgp.prior, lgp.likelihood, solver=lgp.solver).run_sampling_local_auto(
        conf, budget=2_000, mode="robust",
        surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1))
    warm, chunk = run.stages[0], run.stages[1]
    assert warm.burn_in and not warm.is_excluded
    warm_rows, chunk_rows = _rows(conf.output_dir, warm.name), _rows(conf.output_dir, chunk.name)
    np.testing.assert_array_equal(chunk_rows[0, 1:3], warm_rows[-1, 1:3])
    assert not np.array_equal(chunk_rows[0, 1:3], warm_rows[0, 1:3])
    # its default selection leaves the warm-up out of the posterior
    selection = default_selection(read_run(conf.output_dir))
    values = list(selection["posterior"].values())
    assert values[0] == [0] and all(v == [1] for v in values[1:])
