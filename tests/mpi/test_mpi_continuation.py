#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``SamplingRun.continue_sampling`` under real MPI (2026-10-08, ``API_SPEC_continuation`` §3,
cases A-D). One driver (``mpiexec -n 3``, run once per module) samples the linear-Gaussian
problem of ``conftest.py`` (closed-form posterior):

* run A: 2 samplers + collector (in-process solver from a ``SolverSpec``), adaptive MH ->
  DAMH-SMU with a degree-1 polynomial updater, held-out test data saved into A; afterwards the
  collector saves the updater state into A (``SurrogateRestart(...).save``);
* B  (case A) ``continue_sampling`` with the defaults and a DAMH-SMU FIRST stage with
  ``RandomWalk()`` (scale carried);
* P / L (case B) ``chains="prior"`` / ``"lhs"``, one short MH stage;
* C  (case C) explicit ``PolynomialSklearnUpdater(max_degree=2)`` + ``SurrogateRestart(mode="data")``;
* D  (case D) explicit fixed ``surrogate_evaluator``, ``use_collector=False`` with a solvers pool
  (2 samplers + pool), frozen DAMH.

The local / 1-sampler cases (E-H) are in ``tests/unit/test_continue_sampling.py``.
"""

from __future__ import annotations

import ast
import json
import os
import re

import numpy as np
import pytest

from helpers_statistics import (BURN_IN_FRACTION, assert_matches_closed_form, pooled_chain_statistics,
                                read_weighted_samples)

from .conftest import _run_mpi, requires_mpiexec, write_driver, write_solver_module

pytestmark = [pytest.mark.mpi, requires_mpiexec]

NO_SAMPLERS = 2

DRIVER_BODY = """
    from mpi4py import MPI
    from surrDAMH.surrogates import PolynomialSklearnUpdater
    from surrDAMH.surrogates.reuse import SurrogateReused

    rank = MPI.COMM_WORLD.Get_rank()
    out = lambda name: os.path.join(OUTPUT_DIR, name)
    problem = surrDAMH.Problem(solver_module.make_prior(), solver_module.make_likelihood(), solver=solver_spec())

    def conf(name, **kwargs):
        kwargs.setdefault("use_collector", True)
        kwargs.setdefault("use_solvers_pool", False)
        return surrDAMH.Configuration(output_dir=out(name), transform_before_saving=False,
                                      min_snapshots_initial=20, min_snapshots_to_update=200, **kwargs)

    # ---- run A ----
    conf_a = conf("A", initial_sample_type="lhs")
    assert conf_a.no_samplers == 2
    updater = PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1)
    test_data = surrDAMH.TestData.generate(problem, size=20)
    if rank == 0:
        test_data.save(out("A"))
    stages_a = [Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=1500),
                Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=3000, subchain_length=5)]
    run_a = problem.run_sampling(conf_a, stages_a, surrogate_updater=updater, surrogate_test_data=test_data)
    if rank == conf_a.rank_collector:
        surrDAMH.SurrogateRestart(state_dir=os.path.join(out("A"), "sampling_output")).save(updater)
    MPI.COMM_WORLD.Barrier()

    # ---- B: defaults, DAMH-SMU first stage, carried scale ----
    print("CASE B START", flush=True)
    run_b = run_a.continue_sampling(conf("B"), [Stage(algorithm="DAMH", proposal=RandomWalk(),
                                                      max_evaluations=6000, subchain_length=5)])
    assert run_b.generation == 1 and run_b.stages[0].name == "alg0002_DAMH-SMU"

    # ---- P / L: new chains ----
    for name, chains in (("P", "prior"), ("L", "lhs")):
        run_a.continue_sampling(conf(name), [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5),
                                                   max_evaluations=30)], chains=chains)

    # ---- C: explicit updater + restart from A's training data ----
    run_a.continue_sampling(conf("C"), [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5),
                                              max_evaluations=200, subchain_length=3)],
                            surrogate_updater=PolynomialSklearnUpdater(no_parameters=2, no_observations=2,
                                                                       max_degree=2),
                            surrogate_restart=surrDAMH.SurrogateRestart(
                                state_dir=os.path.join(out("A"), "sampling_output"), mode="data"))

    # ---- D: frozen evaluator, no collector, solvers pool ----
    conf_d = conf("D", use_collector=False, use_solvers_pool=True, no_solvers=1)
    assert conf_d.no_samplers == 2
    run_a.continue_sampling(conf_d, [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=200,
                                           subchain_length=3, surrogate_model_updates=False)],
                            surrogate_evaluator=SurrogateReused(out("A")).get_evaluator())
    print("DRIVER FINISHED", flush=True)
    """

CARRY_RE = re.compile(r"Sampler at rank (\d+) carry-over from the previous run (\w+): (\[[^\]]*\])")


@pytest.fixture(scope="module")
def continuation_runs(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("continuation")
    write_solver_module(workspace)
    driver = write_driver(workspace, "continuation", DRIVER_BODY)
    result = _run_mpi(driver, n=3, timeout=300)
    assert not result.timed_out and result.returncode == 0, result.describe()
    assert "DRIVER FINISHED" in result.stdout, result.describe()
    return result, workspace / "out_continuation"


def _manifest(output_dir):
    with open(output_dir / "sampling_output" / "run_manifest.json") as f:
        return json.load(f)


def _rows(output_dir, kind, stage, rank):
    return np.loadtxt(output_dir / "sampling_output" / kind / stage / f"rank{rank:04d}.csv",
                      delimiter=",", skiprows=1, ndmin=2)


def _stage_dirs(output_dir):
    return sorted(os.listdir(output_dir / "sampling_output" / "samples"))


def test_a_default_mpi_continuation(continuation_runs, linear_gaussian_problem):
    from surrDAMH.modules.continuation import load_carry_over
    from surrDAMH.modules.seeds import ALGORITHM_SEED_OFFSET, GENERATION_SEED_STRIDE, stage_seed0

    result, root = continuation_runs
    dir_a, dir_b = root / "A", root / "B"
    assert _stage_dirs(dir_a) == ["alg0000_MH-adaptive", "alg0001_DAMH-SMU"]
    assert _stage_dirs(dir_b) == ["alg0002_DAMH-SMU"]

    lineage = _manifest(dir_b)["lineage"]
    assert lineage == {"continued_from": str(dir_a), "generation": 1, "stage_index_offset": 2,
                       "no_stages_lineage": 3, "same_problem": True, "same_problem_lineage": True,
                       "chains": "continue", "initial_sample_is_carried_over": True,
                       "dirs": [str(dir_a), str(dir_b)]}

    # seeds: lineage formula with the generation-1 shift, disjoint from every seed0 of A (any
    # rank, any stage), the ones the chains really used (notes column 4, "seed")
    seeds_a = {s["seed0"] for s in _manifest(dir_a)["seeds"]["per_rank"]}
    for entry in _manifest(dir_b)["seeds"]["per_rank"]:
        rank = entry["rank"]
        assert entry["stage_index"] == 2
        assert entry["seed0"] == stage_seed0(3, rank, 2) + GENERATION_SEED_STRIDE
        assert entry["seed0"] not in seeds_a
        assert _rows(dir_b, "notes", "alg0002_DAMH-SMU", rank)[0, 4] == entry["seed0"] + ALGORITHM_SEED_OFFSET

    for rank in range(NO_SAMPLERS):
        # each chain continues from A's last state, counted once (no +1 in B)
        last = np.load(dir_a / "sampling_output" / "last_sample" / "alg0001_DAMH-SMU" / f"rank{rank:04d}.npz")
        samples = _rows(dir_b, "samples", "alg0002_DAMH-SMU", rank)
        np.testing.assert_array_equal(samples[0, 1:3], last["parameters"])
        accepted, rejected, prerejected = _rows(dir_b, "notes", "alg0002_DAMH-SMU", rank)[0, :3]
        assert samples[:, 0].sum() == accepted + rejected + prerejected

    # B's proposal started from A's pooled carry-over (the later of A's two adaptive stages wins)
    carried_a, _ = load_carry_over(str(dir_a), "alg0001_DAMH-SMU")
    text_b = result.stdout.split("CASE B START", 1)[1]
    printed = {int(m.group(1)): ast.literal_eval(m.group(3))
               for m in CARRY_RE.finditer(text_b) if m.group(2) == "scale"}
    assert sorted(printed) == [0, 1], result.describe()
    for value in printed.values():
        np.testing.assert_array_equal(np.reshape(value, np.shape(carried_a["scale"])), carried_a["scale"])

    # the collector started pretrained from A's saved state, and the test data were reused
    assert "Reused surrogate state (PolynomialSklearnUpdater)" in result.stdout
    assert _manifest(dir_b)["surrogate"]["updater_class"] == "PolynomialSklearnUpdater"
    assert (dir_b / "sampling_output" / "surrogate_quality_test.csv").is_file()
    assert (dir_b / "sampling_output" / "surrogate_test_data.npz").is_file()
    assert (dir_b / "sampling_output" / "surrogate_checkpoint.pt").is_file()  # saved for the next one

    # posterior of A (no stage excluded) + B, chain by chain, against the closed form
    chains = [np.vstack([read_weighted_samples(dir_a, "alg0000_MH-adaptive", 2, rank),
                         read_weighted_samples(dir_a, "alg0001_DAMH-SMU", 2, rank),
                         read_weighted_samples(dir_b, "alg0002_DAMH-SMU", 2, rank)])
              for rank in range(NO_SAMPLERS)]
    statistics = pooled_chain_statistics(chains, burn_in_fraction=BURN_IN_FRACTION)
    print(f"[A+B] chains {[len(c) for c in chains]}, mean {statistics.mean}, cov {statistics.cov.ravel()}")
    assert_matches_closed_form(statistics, linear_gaussian_problem.posterior_mean,
                               linear_gaussian_problem.posterior_cov, "continuation A+B")


def test_b_prior_and_lhs_chains(continuation_runs):
    """New chains: the initial states are neither A's last nor A's first ones but what a fresh run
    with that ``initial_sample_type`` and the generation-1 lineage seeds draws; counted (+1)."""
    import surrDAMH
    from surrDAMH.distributions.parent import rvs_with_generator
    from surrDAMH.modules import lhs_normal
    from surrDAMH.modules.seeds import GENERATION_SEED_STRIDE, initial_sample_seed

    _, root = continuation_runs
    prior = surrDAMH.distributions.PriorIndependentComponents(
        [surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0) for _ in range(2)])
    lhs = lhs_normal.lhs_normal(loc=prior.mean, scale=1.0, n=NO_SAMPLERS, seed=GENERATION_SEED_STRIDE)
    for name, chains in (("P", "prior"), ("L", "lhs")):
        output_dir = root / name
        lineage = _manifest(output_dir)["lineage"]
        assert (lineage["chains"], lineage["initial_sample_is_carried_over"]) == (chains, False)
        assert _stage_dirs(output_dir) == ["alg0002_MH"]
        for rank in range(NO_SAMPLERS):
            last = np.load(root / "A" / "sampling_output" / "last_sample" / "alg0001_DAMH-SMU" / f"rank{rank:04d}.npz")
            samples = _rows(output_dir, "samples", "alg0002_MH", rank)
            assert not np.array_equal(samples[0, 1:3], last["parameters"])
            first_a = _rows(root / "A", "samples", "alg0000_MH-adaptive", rank)[0, 1:3]  # A started from lhs
            assert not np.array_equal(samples[0, 1:3], first_a)
            expected = (rvs_with_generator(prior, np.random.default_rng(initial_sample_seed(3, rank)
                                                                        + GENERATION_SEED_STRIDE))
                        if chains == "prior" else lhs[rank])
            np.testing.assert_array_equal(samples[0, 1:3], expected)
            accepted, rejected, prerejected = _rows(output_dir, "notes", "alg0002_MH", rank)[0, :3]
            assert samples[:, 0].sum() == accepted + rejected + prerejected + 1


def test_c_explicit_updater_with_data_restart(continuation_runs):
    result, root = continuation_runs
    dir_c = root / "C"
    assert _stage_dirs(dir_c) == ["alg0002_DAMH-SMU"]
    surrogate = _manifest(dir_c)["surrogate"]
    assert surrogate["updater_class"] == "PolynomialSklearnUpdater"
    assert surrogate["updater_parameters"]["max_degree"] == 2  # the given one, not A's (1)
    assert "restored surrogate training data only from" in result.stdout


def test_d_frozen_evaluator_without_collector(continuation_runs):
    result, root = continuation_runs
    dir_d = root / "D"
    manifest = _manifest(dir_d)
    assert _stage_dirs(dir_d) == ["alg0002_DAMH"]
    assert manifest["surrogate"]["updater_class"] is None
    assert manifest["surrogate"]["evaluator_class"] == "PolynomialSklearnEvaluator"
    assert manifest["mpi"]["rank_solvers_pool"] == 2 and manifest["mpi"]["rank_collector"] is None
    for rank in range(NO_SAMPLERS):
        last = np.load(root / "A" / "sampling_output" / "last_sample" / "alg0001_DAMH-SMU" / f"rank{rank:04d}.npz")
        np.testing.assert_array_equal(_rows(dir_d, "samples", "alg0002_DAMH", rank)[0, 1:3], last["parameters"])


def test_lineage_report_with_selection(continuation_runs, linear_gaussian_problem):
    """B3 case 7 (2026-10-08, ``API_SPEC_lineage_report``): the report of B over its lineage A -> B
    (2 samplers each), first with the default selection, then with chain 1 of B's stage excluded in
    ``selection.json``. Written in this process (``SamplingRun.load``: runner "mpi", world size 1)."""
    import pandas as pd

    import surrDAMH
    from surrDAMH.post_processing.selection import selection_path

    _, root = continuation_runs
    dir_a, dir_b = root / "A", root / "B"
    lgp = linear_gaussian_problem
    problem = surrDAMH.Problem(lgp.prior, lgp.likelihood, solver=lgp.solver, no_parameters=2, no_observations=2)
    stages = ["alg0000_MH-adaptive", "alg0001_DAMH-SMU", "alg0002_DAMH-SMU"]
    summary_path = dir_b / "post_processing_output" / "summary.csv"

    loaded = surrDAMH.SamplingRun.load(str(dir_b), problem)
    assert loaded.lineage == [str(dir_a), str(dir_b)]
    samples = loaded.write_report(include_expensive_sections=False, no_best_fits=0)
    assert samples.stage_names == stages and samples.chain_indices == [[0, 1]] * 3
    assert list(pd.read_csv(summary_path, index_col=0).index) == stages
    selection = json.loads(open(selection_path(str(dir_b))).read())
    assert selection["runs"] == [str(dir_a), str(dir_b)]
    assert selection["posterior"] == {stages[0]: [1, 1], stages[1]: [1, 1], stages[2]: [1, 1]}

    selection["posterior"][stages[2]] = [1, 0]
    with open(selection_path(str(dir_b)), "w") as f:
        json.dump(selection, f)
    loaded_all = surrDAMH.SamplingRun.load(str(dir_b), problem).write_report(include_expensive_sections=False,
                                                                             no_best_fits=0)
    # 2026-10-09: every chain stays loaded; the selection picks the posterior (posterior_view)
    assert loaded_all.chain_indices == [[0, 1]] * 3
    samples = loaded_all.posterior_view()
    assert samples.chain_indices == [[0, 1], [0, 1], [0]]
    summary = pd.read_csv(summary_path, index_col=0)
    assert list(summary.index) == stages
    assert list(summary["chains_in_posterior"]) == ["2/2", "2/2", "1/2"]
    accepted_rank0 = _rows(dir_b, "notes", stages[2], 0)[0, 0]
    accepted_b = sum(_rows(dir_b, "notes", stages[2], rank)[0, 0] for rank in range(NO_SAMPLERS))
    accepted_a = sum(_rows(dir_a, "notes", stages[1], rank)[0, 0] for rank in range(NO_SAMPLERS))
    assert summary.loc[stages[2], "accepted"] == accepted_b  # summary.csv: every chain
    assert samples.summary.loc[stages[2], "accepted"] == accepted_rank0  # the posterior view
    assert summary.loc[stages[1], "accepted"] == accepted_a
    assert set(samples.subchain_stats[2]["rank_world"]) == {0}
    assert set(samples.subchain_stats[1]["rank_world"]) == {0, 1}
    np.testing.assert_allclose(samples.list_of_stages[2].samples_compressed[0],  # (pandas vs loadtxt parsing)
                               _rows(dir_b, "samples", stages[2], 0)[:, 1:3], rtol=1e-12)
    with open(dir_b / "post_processing_output" / "report_extended.html") as f:
        html = f.read()
    assert f"excluded by the user: {stages[2]} chain 1." in html
    assert "chain 1: not in posterior (excluded by user)" in html
    assert "<th>chain 0</th><th>chain 1</th>" in html and "<h3>Lineage</h3>" in html
