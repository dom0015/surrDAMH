#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lineage report and selection mask (2026-10-08, ``API_SPEC_lineage_report`` B1-B3, cases 1-6 and
8; the MPI case 7 is ``tests/mpi/test_mpi_continuation.py::test_lineage_report_with_selection``).

* B1: ``modules.run_data.read_lineage``, ``Samples(include_previous=...)``, per-stage directories;
* B2: ``post_processing_output/selection.json`` created / validated / applied by
  ``SamplingRun.write_report``, the "Selection and re-run" section (since 2026-10-09 the mask
  selects the posterior only -- ``Samples.posterior_view()`` -- every chain stays loaded and shown;
  see also ``test_burn_in_report.py``);
* ``SamplingRun.load(output_dir)`` without a problem;
* a plain single run gives the same numbers with the default selection as with ``selection=False``.

Lineages come from ``run_sampling_local`` + ``continue_sampling_local`` (one chain); the
multi-chain selection cases use a synthetic format-v2 run written through the production
``CsvWriter`` (as in ``test_post_processing.py``), with chains of deliberately different means.
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
from surrDAMH.modules import manifest as manifest_module
from surrDAMH.modules.manifest import FORMAT_VERSION, MANIFEST_VERSION, RunFormatError, samples_columns
from surrDAMH.modules.monitoring import CsvWriter
from surrDAMH.modules.run_data import read_lineage, read_run
from surrDAMH.post_processing.selection import selection_note, selection_path
from surrDAMH.proposals import RandomWalk
from surrDAMH.stages import Stage

from .test_continue_sampling import OBSERVED, _conf, _fixed_rw_stage, _problem, _stages_a

pytestmark = pytest.mark.unit

REPORT = dict(include_expensive_sections=False, no_best_fits=0)


def _summary_csv(output_dir):
    return pd.read_csv(os.path.join(str(output_dir), "post_processing_output", "summary.csv"), index_col=0)


def _html(output_dir):
    with open(os.path.join(str(output_dir), "post_processing_output", "report_extended.html")) as f:
        return f.read()


def _selection(output_dir):
    with open(selection_path(output_dir)) as f:
        return json.load(f)


def _write_selection(output_dir, content):
    with open(selection_path(output_dir), "w") as f:
        json.dump(content, f)


def _stage_names(output_dir):
    return sorted(os.listdir(os.path.join(str(output_dir), "sampling_output", "samples")))


# ---------------------------------------------------------------------------------------------
# case 1: A -> B, same problem
# ---------------------------------------------------------------------------------------------

@pytest.fixture
def lineage_ab(tmp_path):
    """A (adaptive MH -> MH) continued by B (one MH stage), raw snapshots saved in both."""
    run_a = _problem().run_sampling_local(_conf(tmp_path / "a", save_snapshots_to_file=True), _stages_a())
    run_b = run_a.continue_sampling_local(_conf(tmp_path / "b", save_snapshots_to_file=True), [_fixed_rw_stage()])
    return run_a, run_b


def test_1_lineage_same_problem(lineage_ab, tmp_path):
    run_a, run_b = lineage_ab
    dir_a, dir_b = os.path.abspath(tmp_path / "a"), os.path.abspath(tmp_path / "b")
    alone_a, alone_b = read_run(dir_a), read_run(dir_b)

    data = read_lineage(dir_b)
    assert data.stage_names == alone_a.stage_names + alone_b.stage_names == \
        ["alg0000_MH-adaptive", "alg0001_MH", "alg0002_MH"]
    assert len(set(data.stage_names)) == data.no_stages
    for got, expected in zip(data.samples, alone_a.samples + alone_b.samples):
        assert len(got) == len(expected) == 1
        np.testing.assert_array_equal(got[0], expected[0])
    assert data.output_dirs == [dir_a, dir_b]
    assert data.stage_output_dir == [dir_a, dir_a, dir_b]
    assert data.same_problem is True and data.generation == 1
    assert data.output_dir == dir_b and data.manifest == alone_b.manifest
    assert [m["lineage"]["generation"] for m in data.manifests[1:]] == [1]
    assert set(data.last_sample) == set(data.stage_names)

    # Samples: the default loads the lineage, with each stage's own directory
    samples = pp.Samples(2, dir_b)
    assert samples.stage_names == data.stage_names and samples.lineage_note is None
    assert samples.stage_sampling_dirs == [os.path.join(d, "sampling_output") for d in (dir_a, dir_a, dir_b)]
    assert samples.sampling_output_dir == os.path.join(dir_b, "sampling_output")

    # raw_data streaming across the two directories
    par, obs, _ = samples.load_snapshots(no_observations=2)
    par_a, _, _ = pp.Samples(2, dir_a).load_snapshots(no_observations=2)
    par_b, _, _ = pp.Samples(2, dir_b, include_previous=False).load_snapshots(no_observations=2)
    assert par.shape[0] == par_a.shape[0] + par_b.shape[0] > par_b.shape[0] > 0
    np.testing.assert_array_equal(par, np.vstack((par_a, par_b)))
    assert obs.shape == (par.shape[0], 2) and samples._raw_data_available([2])

    # the report over the lineage: one summary row per lineage stage
    report_samples = run_b.write_report(**REPORT)
    summary = _summary_csv(dir_b)
    assert list(summary.index) == data.stage_names
    html = _html(dir_b)
    assert "<h3>Lineage</h3>" in html and dir_a in html
    assert list(_summary_csv(dir_b)["sum"]) == [float(n["sum"].sum()) for n in data.notes]
    np.testing.assert_allclose(report_samples.get_mean_and_cov()[0], samples.get_mean_and_cov()[0])

    # include_previous=False: B only
    only_b = pp.Samples(2, dir_b, include_previous=False)
    assert only_b.stage_names == ["alg0002_MH"] and only_b.stage_sampling_dirs == [os.path.join(dir_b, "sampling_output")]
    run_b.write_report(include_previous=False, selection=False, **REPORT)
    assert list(_summary_csv(dir_b).index) == ["alg0002_MH"]


# ---------------------------------------------------------------------------------------------
# case 2: A -> B with problem= override
# ---------------------------------------------------------------------------------------------

def test_2_lineage_with_problem_override(tmp_path):
    run_a = _problem().run_sampling_local(_conf(tmp_path / "a"), _stages_a())
    run_b = run_a.continue_sampling_local(_conf(tmp_path / "b"), [_fixed_rw_stage()],
                                          problem=_problem(observed=OBSERVED + 0.2))
    dir_b = str(tmp_path / "b")
    assert read_lineage(dir_b).same_problem is False

    default = pp.Samples(2, dir_b)
    assert default.stage_names == ["alg0002_MH"]
    assert "different Problem" in default.lineage_note and "include_previous=True" in default.lineage_note

    both = pp.Samples(2, dir_b, include_previous=True)
    assert both.stage_names == ["alg0000_MH-adaptive", "alg0001_MH", "alg0002_MH"]
    assert "Problem changed" in both.lineage_note

    run_b.write_report(include_previous=True, **REPORT)
    assert len(_summary_csv(dir_b)) == 3
    assert "The Problem changed within this lineage" in _html(dir_b)
    # the default selection file was created for the loaded (3-stage) lineage
    assert list(_selection(dir_b)["posterior"]) == both.stage_names
    assert _selection(dir_b)["runs"] == [os.path.abspath(tmp_path / "a"), os.path.abspath(tmp_path / "b")]
    # ...and is still valid for B alone (stages that are not loaded are ignored)
    run_b.write_report(**REPORT)
    assert len(_summary_csv(dir_b)) == 1 and len(_selection(dir_b)["posterior"]) == 3
    assert "different Problem" in _html(dir_b)


# ---------------------------------------------------------------------------------------------
# case 3: depth-2 lineage A -> B -> C
# ---------------------------------------------------------------------------------------------

def test_3_depth_two_lineage(tmp_path):
    run_a = _problem().run_sampling_local(_conf(tmp_path / "a"), _stages_a())
    run_b = run_a.continue_sampling_local(_conf(tmp_path / "b"),
                                          [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=100),
                                           _fixed_rw_stage(scale=0.3, evaluations=40)])
    surrDAMH.SamplingRun.load(str(tmp_path / "b"), _problem()).continue_sampling_local(
        _conf(tmp_path / "c"), [_fixed_rw_stage()])
    dir_c = str(tmp_path / "c")
    data = read_lineage(dir_c)
    assert data.output_dirs == [os.path.abspath(tmp_path / n) for n in "abc"]
    assert data.stage_names == ["alg0000_MH-adaptive", "alg0001_MH", "alg0002_MH-adaptive", "alg0003_MH",
                                "alg0004_MH"]
    assert data.generation == 2 and data.same_problem
    samples = pp.Samples(2, dir_c)
    assert samples.no_stages == 5
    assert [os.path.basename(os.path.dirname(d)) for d in samples.stage_sampling_dirs] == list("aabbc")
    surrDAMH.SamplingRun.load(dir_c).write_report(**REPORT)
    assert len(_summary_csv(dir_c)) == 5


# ---------------------------------------------------------------------------------------------
# case 4: old-style manual continuation (stage-name clash)
# ---------------------------------------------------------------------------------------------

def test_4_manual_continuation_name_clash(tmp_path):
    _problem().run_sampling_local(_conf(tmp_path / "a"), _stages_a())
    manual = _conf(tmp_path / "m", initial_sample_type="continued", continued_from_dir=str(tmp_path / "a"))
    _problem().run_sampling_local(manual, _stages_a())
    dir_m = str(tmp_path / "m")
    assert _stage_names(dir_m) == _stage_names(tmp_path / "a")

    with pytest.raises(RunFormatError, match=r"stage name 'alg0000_MH-adaptive'.*include_previous=False"):
        read_lineage(dir_m)
    with pytest.raises(RunFormatError, match=r"include_previous=False"):
        pp.Samples(2, dir_m, include_previous=True)
    only_m = pp.Samples(2, dir_m, include_previous=False)
    assert only_m.stage_names == ["alg0000_MH-adaptive", "alg0001_MH"]
    # the default does not follow a manual link (same problem not recorded)
    default = pp.Samples(2, dir_m)
    assert default.stage_names == only_m.stage_names and "continued by hand" in default.lineage_note


def test_lineage_with_a_moved_earlier_run(lineage_ab, tmp_path):
    """The default falls back to this run when an earlier run cannot be read (moved/deleted)."""
    os.rename(tmp_path / "a", tmp_path / "a_moved")
    default = pp.Samples(2, str(tmp_path / "b"))
    assert default.stage_names == ["alg0002_MH"] and "could not be read" in default.lineage_note
    with pytest.raises(RunFormatError, match=r"include_previous=False"):
        read_lineage(str(tmp_path / "b"))


# ---------------------------------------------------------------------------------------------
# case 5 and 6: the selection mask, SamplingRun.load(output_dir) without a problem
# ---------------------------------------------------------------------------------------------

def _write_csv(dirname, basename, header, rows):
    writer = CsvWriter(dirname=dirname, basename=basename, header=header)
    for row in rows:
        writer.writerow(row)
    writer.close_file()


def _synthetic_run(root, n=60, seed=0, first_stage_flag="is_excluded"):
    """
    Two chains, three stages: ``alg0000_MH`` (an ``is_excluded`` -- or, with
    ``first_stage_flag="burn_in"``, a ``burn_in`` -- stage, far away at 50),
    ``alg0001_MH-adaptive`` and ``alg0002_MH``; chain 0 sits at 0, chain 1 at 10, so excluding a
    chain moves every mean. The adaptive stage has an ``adaptive_stats`` trace per chain.
    """
    rng = np.random.default_rng(seed)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=n, **{first_stage_flag: True}),
              Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=n),
              Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=n)]
    names = ["alg0000_MH", "alg0001_MH-adaptive", "alg0002_MH"]
    specs = []
    for stage, name in zip(stages, names):
        stage.name = name
        specs.append(manifest_module._stage_dict(stage))
    sampling = os.path.join(root, "sampling_output")
    os.makedirs(sampling, exist_ok=True)
    with open(os.path.join(sampling, "run_manifest.json"), "w") as f:
        json.dump({"manifest_version": MANIFEST_VERSION, "format_version": FORMAT_VERSION, "runner": "local",
                   "configuration": {"no_parameters": 2, "no_observations": 0}, "stages": specs}, f)
    for index, name in enumerate(names):
        for chain, centre in enumerate((50.0 if index == 0 else 0.0, 50.0 if index == 0 else 10.0)):
            data = rng.normal(loc=centre, size=(n, 2))
            rows = [[1 + (i % 3), x, y, -1.0] for i, (x, y) in enumerate(data)]
            _write_csv(os.path.join(sampling, "samples", name), f"rank{chain:04d}.csv", samples_columns(2), rows)
            accepted = 10 * (chain + 1) + index
            _write_csv(os.path.join(sampling, "notes", name), f"rank{chain:04d}.csv",
                       ["accepted", "rejected", "pre-rejected", "sum", "seed"], [[accepted, 5, 0, accepted + 5, chain]])
            if index == 1:
                _write_csv(os.path.join(sampling, "adaptive_stats", name), f"rank{chain:04d}.csv",
                           ["n", "mean_acceptance_probability", "log_sigma", "rank_world"],
                           [[k, 0.3, 0.1 * chain + 0.01 * k, chain] for k in range(1, 6)])
    return root


def _chain_mean(root, stage, chain):
    rows = np.loadtxt(os.path.join(root, "sampling_output", "samples", stage, f"rank{chain:04d}.csv"),
                      delimiter=",", skiprows=1)
    return np.average(rows[:, 1:3], axis=0, weights=rows[:, 0])


def test_5_selection_file_created_edited_and_applied(tmp_path):
    root = _synthetic_run(str(tmp_path / "run"))
    loaded = surrDAMH.SamplingRun.load(root)  # case 6: no problem
    assert loaded.problem is None and loaded.conf.no_parameters == 2

    # first report: the file is created with the documented defaults
    first = loaded.write_report(**REPORT)
    content = _selection(root)
    assert content == {
        "posterior": {"alg0000_MH": [0, 0], "alg0001_MH-adaptive": [1, 1], "alg0002_MH": [1, 1]},
        "note": selection_note(root)}
    # 2026-10-09: the excluded stage is loaded and shown (all chains), only left out of the posterior
    summary = _summary_csv(root)
    assert list(summary.index) == ["alg0000_MH", "alg0001_MH-adaptive", "alg0002_MH"]
    assert list(summary["chains_in_posterior"]) == ["0/2", "2/2", "2/2"]
    html = _html(root)
    assert "Skipped: every chain" not in html and "is excluded by the selection" not in html
    assert "not in the posterior as excluded stages (is_excluded=True): alg0000_MH chain 0; alg0000_MH chain 1." in html
    assert "chain 0: not in posterior (excluded stage)" in html
    assert "excluded by the user: none." in html
    assert selection_path(root) in html and "(created with the defaults by this report)" in html
    assert f'import surrDAMH\nsurrDAMH.SamplingRun.load("{os.path.abspath(root)}").write_report()' in html
    assert html.rstrip().endswith("</details>\n</body>\n</html>")
    assert html.index('id="selection"') > html.index('id="best_fit_visualization"')
    assert "Not available: no Problem given" in html  # case 6
    assert first.chain_indices == [[0, 1], [0, 1], [0, 1]]
    mean_default = first.posterior_view().get_mean_and_cov()[0]
    expected = np.mean([_chain_mean(root, s, c) for s in ("alg0001_MH-adaptive", "alg0002_MH") for c in (0, 1)],
                       axis=0)
    np.testing.assert_allclose(mean_default, expected, rtol=0.05)

    # edit: drop chain 1 of alg0002_MH -> the mean moves, the HTML lists the pair
    content["posterior"]["alg0002_MH"] = [1, 0]
    _write_selection(root, content)
    second_all = surrDAMH.SamplingRun.load(root).write_report(**REPORT)
    assert second_all.chain_indices == [[0, 1], [0, 1], [0, 1]]
    second = second_all.posterior_view()
    assert second.chain_indices == [[], [0, 1], [0]]
    assert second.list_of_stages[2].no_chains == 1
    mean_edit = second.get_mean_and_cov(stages_to_disp=[2])[0]
    np.testing.assert_allclose(mean_edit, _chain_mean(root, "alg0002_MH", 0))
    assert not np.allclose(second.get_mean_and_cov()[0], mean_default)
    assert "excluded by the user: alg0002_MH chain 1." in _html(root)
    assert "(created with the defaults" not in _html(root)
    summary = _summary_csv(root)
    assert summary.loc["alg0002_MH", "accepted"] == 12 + 22  # summary.csv: every chain (2026-10-09)
    assert summary.loc["alg0002_MH", "chains_in_posterior"] == "1/2"
    assert second.summary.loc["alg0002_MH", "accepted"] == 12  # the view: chain 0's notes only
    assert summary.loc["alg0001_MH-adaptive", "accepted"] == 11 + 21
    # chains_to_disp keeps the ORIGINAL numbers and is intersected with the mask
    assert second._resolve_chains(2, [1]) == [] and second._resolve_chains(1, [1]) == [1]
    kept_0 = [(second.list_of_stages[s].samples_compressed[0], second.list_of_stages[s].weights[0]) for s in (1, 2)]
    np.testing.assert_allclose(second.get_mean_and_cov(stages_to_disp=[1, 2], chains_to_disp=[0])[0],
                               np.average(np.vstack([x for x, _ in kept_0]), axis=0,
                                          weights=np.concatenate([w for _, w in kept_0])))

    # excluding a chain of the adaptive stage drops its adaptive_stats rows, labels stay original
    content["posterior"]["alg0001_MH-adaptive"] = [0, 1]
    _write_selection(root, content)
    third_all = surrDAMH.SamplingRun.load(root).write_report(**REPORT)
    assert set(third_all.adaptive_stats[1]["rank_world"]) == {0, 1}
    third = third_all.posterior_view()
    assert set(third.adaptive_stats[1]["rank_world"]) == {1}
    fig, axes = third.plot_adaptation(1)
    assert [line.get_label() for line in axes[0].get_lines() if line.get_label().startswith("chain")] == ["chain 1"]
    import matplotlib.pyplot as plt
    plt.close(fig)

    # drop_first_rows from the file: rows drop, the mean moves
    content["posterior"]["alg0001_MH-adaptive"] = [1, 1]
    content["posterior"]["alg0002_MH"] = [1, 1]
    content["drop_first_rows"] = {"alg0002_MH": [40, 0]}
    _write_selection(root, content)
    burnt_all = surrDAMH.SamplingRun.load(root).write_report(**REPORT)
    assert burnt_all.list_of_stages[2].samples_compressed[0].shape[0] == 60
    burnt = burnt_all.posterior_view()
    assert burnt.list_of_stages[2].samples_compressed[0].shape[0] == 60 - 40
    assert burnt.list_of_stages[2].samples_compressed[1].shape[0] == 60
    assert not np.allclose(burnt.get_mean_and_cov(stages_to_disp=[2])[0], mean_default)
    assert "1 (in posterior), drop_first_rows=40" in _html(root)

    # selection=False ignores the file: every chain of every stage, the burn-in stage included
    everything = surrDAMH.SamplingRun.load(root).write_report(selection=False, **REPORT)
    assert everything.posterior_view().chain_indices == [[0, 1], [0, 1], [0, 1]]
    assert len(_summary_csv(root)) == 3
    plain = pp.Samples(2, root)
    np.testing.assert_array_equal(everything.get_mean_and_cov()[0], plain.get_mean_and_cov()[0])
    assert "No selection was applied" in _html(root)

    # an explicit selection path
    other = str(tmp_path / "other.json")
    content["drop_first_rows"] = {"alg0002_MH": [0, 0]}
    content["posterior"]["alg0002_MH"] = [0, 1]
    with open(other, "w") as f:
        json.dump(content, f)
    explicit = surrDAMH.SamplingRun.load(root).write_report(selection=other, **REPORT)
    assert explicit.posterior_view().chain_indices[2] == [1] and explicit.selection_path == other


def test_5_selection_errors(tmp_path):
    root = _synthetic_run(str(tmp_path / "run"))
    surrDAMH.SamplingRun.load(root).write_report(**REPORT)
    content = _selection(root)

    content["posterior"]["alg0001_MH-adaptive"] = [1, 1, 1]
    _write_selection(root, content)
    with pytest.raises(ValueError, match=re.escape(selection_path(root)) + r".*'alg0001_MH-adaptive' has 2 chain"):
        pp.Samples(2, root, selection=selection_path(root))
    with pytest.raises(ValueError, match=r"selection\.json.*alg0001_MH-adaptive"):
        surrDAMH.SamplingRun.load(root).write_report(**REPORT)

    content["posterior"]["alg0001_MH-adaptive"] = [1, 1]
    del content["posterior"]["alg0002_MH"]
    _write_selection(root, content)
    with pytest.raises(ValueError, match=r"selection\.json: stage 'alg0002_MH'.*missing"):
        pp.Samples(2, root, selection=True)

    content["posterior"]["alg0002_MH"] = [1, 1]
    content["drop_first_rows"] = {"alg0002_MH": [0, 61]}
    _write_selection(root, content)
    with pytest.raises(ValueError, match=r"chain 1: drop_first_rows=61"):
        pp.Samples(2, root, selection=True)

    # every stage excluded: a clear error, not an empty report
    content["posterior"] = {name: [0, 0] for name in content["posterior"]}
    content["drop_first_rows"] = {name: [0, 0] for name in content["posterior"]}
    _write_selection(root, content)
    with pytest.raises(ValueError, match=r"every stage of the report is excluded by the selection"):
        surrDAMH.SamplingRun.load(root).write_report(**REPORT)
    with pytest.raises(FileNotFoundError):
        pp.Samples(2, root, selection=str(tmp_path / "missing.json"))

    # the old (pre-2026-10-09) layout is rejected with a clear, actionable message
    old = {"format": 1, "lineage": [root],
          "stages": [{"name": "alg0000_MH", "include": [0, 0], "burn_in": [0, 0]}]}
    _write_selection(root, old)
    with pytest.raises(ValueError, match=r"layout changed on 2026-10-09.*delete"):
        pp.Samples(2, root, selection=selection_path(root))


def test_6_load_without_problem_on_a_real_run(tmp_path):
    run = _problem().run_sampling_local(_conf(tmp_path / "a"), _stages_a())
    loaded = surrDAMH.SamplingRun.load(str(tmp_path / "a"))
    assert loaded.problem is None and loaded.solver_instance is None
    assert (loaded.conf.no_parameters, loaded.conf.no_observations) == (2, 2)
    samples = loaded.write_report()  # default (expensive) sections, best fits requested
    html = _html(tmp_path / "a")
    assert html.count("Not available: no Problem given") >= 2  # field statistics + best-fit visualization
    np.testing.assert_array_equal(samples.get_mean_and_cov()[0], run.write_report(**REPORT).get_mean_and_cov()[0])
    with pytest.raises(ValueError, match=r"loaded without a problem.*problem="):
        loaded.continue_sampling_local(_conf(tmp_path / "b"), [_fixed_rw_stage()])


# ---------------------------------------------------------------------------------------------
# case 8: a plain single run is unchanged by the new code paths
# ---------------------------------------------------------------------------------------------

def _strip_selection_section(html):
    start = html.index('    <details class="section" id="selection">')
    end = html.index("</body>")
    html = html[:start] + html[end:]
    return html.replace('            <li><a href="#selection">Selection and re-run</a></li>\n', "")


def test_8_plain_single_run_numbers_are_unchanged(tmp_path):
    run = _problem().run_sampling_local(_conf(tmp_path / "a"), _stages_a())
    output_dir = str(tmp_path / "a")
    reference = run.write_report(selection=False)
    summary_reference = open(os.path.join(output_dir, "post_processing_output", "summary.csv")).read()
    html_reference = _html(output_dir)

    default = run.write_report()
    assert os.path.isfile(selection_path(output_dir))
    assert open(os.path.join(output_dir, "post_processing_output", "summary.csv")).read() == summary_reference
    for got, expected in zip(default.get_mean_and_cov(), reference.get_mean_and_cov()):
        np.testing.assert_array_equal(got, expected)
    ess, ess_reference = default.calculate_effective_sample_size(), reference.calculate_effective_sample_size()
    assert ess["total_samples"] == ess_reference["total_samples"]
    assert ess["ess_per_param"] == ess_reference["ess_per_param"]
    # the HTML differs only by the "Selection and re-run" section (and its table-of-contents entry)
    assert _strip_selection_section(_html(output_dir)) == _strip_selection_section(html_reference)
    # ...and a Samples built with the old arguments is the selection=False one
    old_style = pp.Samples(2, output_dir)
    assert old_style.selection_mask is None and old_style.chain_indices == [[0], [0]]
    assert default.posterior_view().chain_indices == [[0], [0]]
    for got, expected in zip(old_style.get_mean_and_cov(), reference.get_mean_and_cov()):
        np.testing.assert_array_equal(got, expected)
