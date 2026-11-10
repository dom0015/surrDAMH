#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ``SamplingRun.continue_sampling`` / ``continue_sampling_local`` (2026-10-08,
``API_SPEC_continuation`` sections 1-3), everything a single process can check:

* F  ``continue_sampling_local`` with the defaults, and the local <-> 1-sampler MPI cross
     continuations (the MPI runner is called in this process with a world of size 1, which is
     exactly chain 0 of an ``mpiexec -n 1`` run);
* B  ``chains="prior"`` / ``"lhs"`` against a fresh run with the same seeds;
* E  ``problem=`` override;
* G  a depth-2 lineage A -> B -> C (offsets, ``SamplingRun.load(...).lineage``, carry-over);
* H  the error cases;
* the default surrogate reuse in process (polynomial updater state, ``LocalSurrogateManager``).

"Builder used the carried value" is checked by equivalence: a continuation whose first stage is
``RandomWalk(adaptive=False)`` (scale ``None`` -> carried) must reproduce, byte for byte, a plain
run of the old manual kind (``initial_sample_type="continued"``) given that scale explicitly and
the same lineage offsets -- except the first row's multiplicity, which the continuation does not
count twice (A30 across runs).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np
import pytest

import surrDAMH
from surrDAMH.distributions import Normal, NormalComponent, PriorIndependentComponents
from surrDAMH.distributions.parent import rvs_with_generator
from surrDAMH.modules import lhs_normal
from surrDAMH.modules.algorithm_interfaces_local import LocalSurrogateManager
from surrDAMH.modules.continuation import load_carry_over
from surrDAMH.modules.manifest import _seeds_dict
from surrDAMH.modules.seeds import (ALGORITHM_SEED_OFFSET, GENERATION_SEED_STRIDE, initial_sample_seed,
                                    stage_seed0)
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates import PolynomialSklearnUpdater
from surrDAMH.surrogates.reuse import SurrogateReused, surrogate_state_paths

pytestmark = pytest.mark.unit

A_MATRIX = np.array([[1.0, 0.5], [0.0, 1.0]])
OBSERVED = A_MATRIX @ np.array([0.7, -0.4])


class LinearSolver(Solver):
    """``G(x) = A @ x`` (the linear-Gaussian toy problem of ``tests/conftest.py``)."""

    def __init__(self, solver_id=0, output_dir=None):
        super().__init__(solver_id, output_dir, no_parameters=2, no_observations=2)

    def get_observations(self):
        return A_MATRIX @ np.asarray(self.parameters, dtype=float)


def _problem(observed=OBSERVED):
    prior = PriorIndependentComponents([NormalComponent(mu=0.0, sigma=1.0) for _ in range(2)])
    return surrDAMH.Problem(prior, Normal(mean=np.asarray(observed, dtype=float), sd=0.3), solver=LinearSolver())


def _conf(output_dir, **kwargs):
    return surrDAMH.Configuration(output_dir=str(output_dir), use_collector=False, use_solvers_pool=False,
                                  transform_before_saving=False, **kwargs)


def _stages_a():
    return [Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=300),
            Stage(algorithm="MH", proposal=RandomWalk(scale=0.4), max_evaluations=60)]


def _fixed_rw_stage(scale=None, evaluations=80):
    return Stage(algorithm="MH", proposal=RandomWalk(scale=scale, adaptive=False), max_evaluations=evaluations)


def _samples(output_dir, stage_name, rank=0):
    path = os.path.join(str(output_dir), "sampling_output", "samples", stage_name, f"rank{rank:04d}.csv")
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def _notes(output_dir, stage_name, rank=0):
    """Values of the notes row: accepted, rejected, pre-rejected, sum, seed (index 4), out_of_bounds."""
    path = os.path.join(str(output_dir), "sampling_output", "notes", stage_name, f"rank{rank:04d}.csv")
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)[0]


def _md5(output_dir, kind, stage_name, rank=0):
    path = os.path.join(str(output_dir), "sampling_output", kind, stage_name, f"rank{rank:04d}.csv")
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _manifest(output_dir):
    with open(os.path.join(str(output_dir), "sampling_output", "run_manifest.json")) as f:
        return json.load(f)


def _last_sample(output_dir, stage_name, rank=0):
    path = os.path.join(str(output_dir), "sampling_output", "last_sample", stage_name, f"rank{rank:04d}.npz")
    return np.load(path)["parameters"]


def _stage_dirs(output_dir):
    return sorted(os.listdir(os.path.join(str(output_dir), "sampling_output", "samples")))


@pytest.fixture
def run_a(tmp_path):
    """Plain local run A: adaptive MH -> fixed MH (two stages)."""
    return _problem().run_sampling_local(_conf(tmp_path / "a"), _stages_a())


# ---------------------------------------------------------------------------------------------
# F: continue_sampling_local with the defaults
# ---------------------------------------------------------------------------------------------

def test_f_local_default_continuation(run_a, tmp_path):
    run_b = run_a.continue_sampling_local(_conf(tmp_path / "b"), [_fixed_rw_stage()])

    # numbering after A's two stages, seeds of the 3-stage lineage
    assert _stage_dirs(tmp_path / "b") == ["alg0002_MH"]
    assert run_b.stage_results[0].name == "alg0002_MH"
    assert run_b.conf.stage_index_offset == 2 and run_b.conf.no_stages_lineage == 3
    lineage = _manifest(tmp_path / "b")["lineage"]
    assert lineage == {"continued_from": os.path.abspath(tmp_path / "a"), "generation": 1,
                       "stage_index_offset": 2, "no_stages_lineage": 3, "same_problem": True,
                       "same_problem_lineage": True, "chains": "continue",
                       "initial_sample_is_carried_over": True,
                       "dirs": [os.path.abspath(tmp_path / "a"), os.path.abspath(tmp_path / "b")]}
    assert run_b.lineage == lineage["dirs"] and run_b.generation == 1
    assert run_a.lineage == [os.path.abspath(tmp_path / "a")] and run_a.generation == 0
    assert "lineage" not in _manifest(tmp_path / "a")  # plain-run manifests are unchanged

    seeds = _manifest(tmp_path / "b")["seeds"]["per_rank"]
    seed0_b = stage_seed0(3, 0, 2) + GENERATION_SEED_STRIDE  # generation 1
    assert [(s["stage_index"], s["seed0"]) for s in seeds] == [(2, seed0_b)]
    assert seeds[0]["initial_sample_seed"] == initial_sample_seed(3, 0) + GENERATION_SEED_STRIDE
    assert _notes(tmp_path / "b", "alg0002_MH")[4] == seed0_b + ALGORITHM_SEED_OFFSET
    seeds_a = {(s["rank"], s["stage_index"]): s["seed0"] for s in _manifest(tmp_path / "a")["seeds"]["per_rank"]}
    assert seeds[0]["seed0"] not in seeds_a.values()

    # the chain continues from A's last state, which A already counted (A30: no +1)
    samples = _samples(tmp_path / "b", "alg0002_MH")
    np.testing.assert_array_equal(samples[0, 1:3], _last_sample(tmp_path / "a", "alg0001_MH"))
    accepted, rejected, prerejected = _notes(tmp_path / "b", "alg0002_MH")[:3]
    assert samples[:, 0].sum() == accepted + rejected + prerejected

    # reproducible: the same continuation again is byte-identical
    run_a.continue_sampling_local(_conf(tmp_path / "b2"), [_fixed_rw_stage()])
    for kind in ("samples", "notes"):
        assert _md5(tmp_path / "b", kind, "alg0002_MH") == _md5(tmp_path / "b2", kind, "alg0002_MH")


def test_f_carried_scale_and_seeds_equal_the_manual_equivalent(run_a, tmp_path):
    """The builder of the continuation's first stage used A's carry-over: a manual run given that
    scale explicitly (old-style ``initial_sample_type="continued"``, same lineage offsets) writes the
    same chain; only the first row's multiplicity differs by the ``+1`` A30 removes."""
    carried, _ = load_carry_over(str(tmp_path / "a"), "alg0000_MH-adaptive")
    run_a.continue_sampling_local(_conf(tmp_path / "b"), [_fixed_rw_stage()])

    manual = _conf(tmp_path / "manual", initial_sample_type="continued", continued_from_dir=str(tmp_path / "a"),
                   stage_index_offset=2, no_stages_lineage=3, lineage_generation=1)
    _problem().run_sampling_local(manual, [_fixed_rw_stage(scale=np.asarray(carried["scale"]))])

    cont, man = _samples(tmp_path / "b", "alg0002_MH"), _samples(tmp_path / "manual", "alg0002_MH")
    np.testing.assert_array_equal(cont[:, 1:], man[:, 1:])
    np.testing.assert_array_equal(cont[1:, 0], man[1:, 0])
    assert man[0, 0] == cont[0, 0] + 1
    assert _md5(tmp_path / "b", "notes", "alg0002_MH") == _md5(tmp_path / "manual", "notes", "alg0002_MH")


def test_f_local_and_one_sampler_mpi_continuations_are_byte_identical(tmp_path):
    """Local A continued locally == local A continued by the MPI runner (world size 1), and the
    same for an MPI A: chain 0 of the MPI continuation is the local continuation, byte for byte."""
    problem = _problem()
    local_a = problem.run_sampling_local(_conf(tmp_path / "la"), _stages_a())
    mpi_a = problem.run_sampling(_conf(tmp_path / "ma"), _stages_a())
    assert mpi_a.runner == "mpi"
    for kind in ("samples", "notes"):
        for name in ("alg0000_MH-adaptive", "alg0001_MH"):
            assert _md5(tmp_path / "la", kind, name) == _md5(tmp_path / "ma", kind, name)

    local_a.continue_sampling_local(_conf(tmp_path / "la_l"), [_fixed_rw_stage()])
    local_a.continue_sampling(_conf(tmp_path / "la_m"), [_fixed_rw_stage()])
    mpi_a.continue_sampling_local(_conf(tmp_path / "ma_l"), [_fixed_rw_stage()])
    run = mpi_a.continue_sampling(_conf(tmp_path / "ma_m"), [_fixed_rw_stage()])
    assert run.runner == "mpi" and run.generation == 1
    for kind in ("samples", "notes"):
        reference = _md5(tmp_path / "la_l", kind, "alg0002_MH")
        for other in ("la_m", "ma_l", "ma_m"):
            assert _md5(tmp_path / other, kind, "alg0002_MH") == reference, (kind, other)
    assert _manifest(tmp_path / "la_m")["lineage"]["continued_from"] == os.path.abspath(tmp_path / "la")


# ---------------------------------------------------------------------------------------------
# seed streams of a lineage are disjoint (generation offset, 2026-10-08)
# ---------------------------------------------------------------------------------------------

def _seed_sets(no_samplers, no_stages, offset=0, generation=0, initial_sample_type="prior"):
    from types import SimpleNamespace
    conf = SimpleNamespace(initial_sample_type=initial_sample_type, no_samplers=no_samplers,
                           stage_index_offset=offset, no_stages_lineage=(offset + no_stages) if generation else None,
                           lineage_generation=generation)
    per_rank = _seeds_dict(conf, [None] * no_stages)["per_rank"]
    seed0 = {e["seed0"] for e in per_rank}
    streams = ({e["proposal_seed"] for e in per_rank} | {e["algorithm_seed"] for e in per_rank}
               | {e["initial_sample_seed"] for e in per_rank})
    assert len(seed0) == no_samplers * no_stages  # no collision inside one run either
    return seed0, streams


@pytest.mark.parametrize("no_samplers", [1, 2, 7])
def test_seed0_of_a_continuation_never_equals_one_of_an_earlier_run(no_samplers):
    """A: 2 stages (generation 0); B: 3 stages (offset 2, generation 1); C: 2 stages (offset 5,
    generation 2). Without the generation offset B rank 0 / alg0002 had A rank 1 / alg0000's seed0."""
    seed0_a, streams_a = _seed_sets(no_samplers, 2)
    seed0_b, streams_b = _seed_sets(no_samplers, 3, offset=2, generation=1)
    seed0_c, streams_c = _seed_sets(no_samplers, 2, offset=5, generation=2)
    assert not seed0_a & seed0_b and not seed0_a & seed0_c and not seed0_b & seed0_c
    assert not streams_a & streams_b and not streams_a & streams_c and not streams_b & streams_c
    # generation 0 is the plain formula
    assert seed0_a == {stage_seed0(2, r, i) for r in range(no_samplers) for i in range(2)}


def test_seeds_of_a_real_depth_two_lineage_are_disjoint(run_a, tmp_path):
    run_b = run_a.continue_sampling_local(_conf(tmp_path / "b"), [_fixed_rw_stage() for _ in range(3)])
    run_b.continue_sampling_local(_conf(tmp_path / "c"), [_fixed_rw_stage() for _ in range(2)])
    seeds = {name: {e["seed0"] for e in _manifest(tmp_path / name)["seeds"]["per_rank"]} for name in "abc"}
    assert not seeds["a"] & seeds["b"] and not seeds["a"] & seeds["c"] and not seeds["b"] & seeds["c"]
    used = {name: {_notes(tmp_path / name, stage)[4] for stage in _stage_dirs(tmp_path / name)} for name in "abc"}
    assert not used["a"] & used["b"] and not used["a"] & used["c"] and not used["b"] & used["c"]
    assert _manifest(tmp_path / "c")["lineage"]["generation"] == 2
    assert all(e["seed0"] >= 2 * GENERATION_SEED_STRIDE for e in _manifest(tmp_path / "c")["seeds"]["per_rank"])


# ---------------------------------------------------------------------------------------------
# B: chains="prior" / "lhs"
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("chains", ["prior", "lhs"])
def test_b_new_chains_equal_a_fresh_run_with_the_same_seeds(run_a, tmp_path, chains):
    run = run_a.continue_sampling_local(_conf(tmp_path / "b"), [_fixed_rw_stage(scale=0.5)], chains=chains)
    assert _manifest(tmp_path / "b")["lineage"]["chains"] == chains
    assert _manifest(tmp_path / "b")["lineage"]["initial_sample_is_carried_over"] is False
    first = _samples(tmp_path / "b", "alg0002_MH")[0, 1:3]
    assert not np.array_equal(first, _last_sample(tmp_path / "a", "alg0001_MH"))
    # generation-shifted seeds: not A's starting point, nor a generation-0 draw of that type
    assert not np.array_equal(first, _samples(tmp_path / "a", "alg0000_MH-adaptive")[0, 1:3])
    if chains == "prior":
        expected = rvs_with_generator(run.problem.prior, np.random.default_rng(
            initial_sample_seed(3, 0) + GENERATION_SEED_STRIDE))
        generation_0 = rvs_with_generator(run.problem.prior, np.random.default_rng(initial_sample_seed(3, 0)))
    else:
        expected = lhs_normal.lhs_normal(loc=run.problem.prior.mean, scale=1.0, n=1, seed=GENERATION_SEED_STRIDE)[0]
        generation_0 = lhs_normal.lhs_normal(loc=run.problem.prior.mean, scale=1.0, n=1, seed=0)[0]
    np.testing.assert_array_equal(first, expected)
    assert not np.array_equal(first, generation_0)

    # a fresh run with the same lineage fields (offset, stage count, generation) is the same run
    fresh = _conf(tmp_path / "fresh", initial_sample_type=chains, stage_index_offset=2, no_stages_lineage=3,
                  lineage_generation=1)
    _problem().run_sampling_local(fresh, [_fixed_rw_stage(scale=0.5)])
    for kind in ("samples", "notes"):  # the new state is counted (+1), exactly like a fresh run
        assert _md5(tmp_path / "b", kind, "alg0002_MH") == _md5(tmp_path / "fresh", kind, "alg0002_MH")


# ---------------------------------------------------------------------------------------------
# E: problem= override
# ---------------------------------------------------------------------------------------------

def test_e_problem_override(run_a, tmp_path):
    new_problem = _problem(observed=OBSERVED + 0.2)
    run = run_a.continue_sampling_local(_conf(tmp_path / "b"), [_fixed_rw_stage()], problem=new_problem)
    assert run.problem is new_problem
    lineage = _manifest(tmp_path / "b")["lineage"]
    assert lineage["same_problem"] is False and lineage["same_problem_lineage"] is False
    # and it stays False down the lineage
    run.continue_sampling_local(_conf(tmp_path / "c"), [_fixed_rw_stage()])
    lineage_c = _manifest(tmp_path / "c")["lineage"]
    assert lineage_c["same_problem"] is True and lineage_c["same_problem_lineage"] is False

    prior3 = PriorIndependentComponents([NormalComponent(mu=0.0, sigma=1.0) for _ in range(3)])
    wrong =surrDAMH.Problem(prior3, Normal(mean=np.zeros(4), sd=0.3), solver=_SumSolver(4))
    with pytest.raises(ValueError, match=r"no_parameters=3, no_observations=4.*no_parameters=2, no_observations=2"):
        run_a.continue_sampling_local(_conf(tmp_path / "d"), [_fixed_rw_stage()], problem=wrong)


class _SumSolver(Solver):
    def __init__(self, n_out):
        self.n_out = n_out

    def get_observations(self):
        return np.full(self.n_out, float(np.sum(self.parameters)))


# ---------------------------------------------------------------------------------------------
# G: depth-2 lineage
# ---------------------------------------------------------------------------------------------

def test_g_depth_two_lineage(run_a, tmp_path):
    stages_b = [Stage(algorithm="MH", proposal=RandomWalk(), max_evaluations=200),  # adaptive, carried start
                Stage(algorithm="MH", proposal=RandomWalk(scale=0.3), max_evaluations=40)]
    run_a.continue_sampling_local(_conf(tmp_path / "b"), stages_b)
    assert _stage_dirs(tmp_path / "b") == ["alg0002_MH-adaptive", "alg0003_MH"]

    # continue a run rebuilt from disk
    loaded_b = surrDAMH.SamplingRun.load(str(tmp_path / "b"), _problem())
    assert loaded_b.generation == 1
    run_c = loaded_b.continue_sampling_local(_conf(tmp_path / "c"), [_fixed_rw_stage()])
    assert _stage_dirs(tmp_path / "c") == ["alg0004_MH"]
    lineage = _manifest(tmp_path / "c")["lineage"]
    assert (lineage["stage_index_offset"], lineage["no_stages_lineage"], lineage["generation"]) == (4, 5, 2)

    expected_dirs = [os.path.abspath(tmp_path / name) for name in ("a", "b", "c")]
    assert run_c.lineage == expected_dirs
    loaded_c = surrDAMH.SamplingRun.load(str(tmp_path / "c"), _problem())
    assert loaded_c.lineage == expected_dirs and loaded_c.generation == 2

    # B's carry-over (later than A's) reaches C: manual equivalent with B's scale
    carried_b, _ = load_carry_over(str(tmp_path / "b"), "alg0002_MH-adaptive")
    carried_a, _ = load_carry_over(str(tmp_path / "a"), "alg0000_MH-adaptive")
    assert not np.array_equal(carried_a["scale"], carried_b["scale"])
    manual = _conf(tmp_path / "manual", initial_sample_type="continued", continued_from_dir=str(tmp_path / "b"),
                   stage_index_offset=4, no_stages_lineage=5, lineage_generation=2)
    _problem().run_sampling_local(manual, [_fixed_rw_stage(scale=np.asarray(carried_b["scale"]))])
    np.testing.assert_array_equal(_samples(tmp_path / "c", "alg0004_MH")[:, 1:],
                                  _samples(tmp_path / "manual", "alg0004_MH")[:, 1:])


# ---------------------------------------------------------------------------------------------
# H: errors
# ---------------------------------------------------------------------------------------------

def test_h_error_cases(run_a, tmp_path):
    stages = [_fixed_rw_stage()]
    with pytest.raises(ValueError, match=r"continue_sampling sets the initial samples; use chains="):
        run_a.continue_sampling_local(_conf(tmp_path / "x", initial_sample_type="continued",
                                            continued_from_dir=str(tmp_path / "a")), stages)
    with pytest.raises(ValueError, match=r"continue_sampling sets the initial samples; use chains="):
        run_a.continue_sampling_local(_conf(tmp_path / "x", initial_sample_type="lhs"), stages)
    with pytest.raises(ValueError, match=r"set by continue_sampling"):
        run_a.continue_sampling_local(_conf(tmp_path / "x", stage_index_offset=5), stages)
    with pytest.raises(ValueError, match=r"set by continue_sampling"):
        run_a.continue_sampling_local(_conf(tmp_path / "x", lineage_generation=1), stages)
    with pytest.raises(ValueError, match=r"needs its own output_dir"):
        run_a.continue_sampling_local(_conf(tmp_path / "a"), stages)
    with pytest.raises(ValueError, match=r"chains must be one of \('continue', 'prior', 'lhs'\), got 'bogus'"):
        run_a.continue_sampling_local(_conf(tmp_path / "x"), stages, chains="bogus")
    with pytest.raises(FileNotFoundError, match=r"no run manifest"):
        surrDAMH.SamplingRun.load(str(tmp_path / "missing"), _problem())

    # a run without last_sample/: copy A without it
    no_last = tmp_path / "no_last"
    shutil.copytree(tmp_path / "a", no_last, ignore=shutil.ignore_patterns("last_sample"))
    previous = surrDAMH.SamplingRun.load(str(no_last), _problem())
    with pytest.raises(FileNotFoundError, match=r"chains='continue' needs the last states"):
        previous.continue_sampling_local(_conf(tmp_path / "x"), stages)
    previous.continue_sampling_local(_conf(tmp_path / "y"), stages, chains="prior")  # fine without them

    # surrogate training data but no checkpoint: the default cannot restore it
    updater = PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1)
    updater.add_data(np.ones((3, 2)), np.ones((3, 2)))
    updater.save_training_data(surrogate_state_paths(str(tmp_path / "a"))[1])
    with pytest.raises(ValueError, match=r"no surrogate checkpoint.*load_training_data"):
        run_a.continue_sampling_local(_conf(tmp_path / "z"), stages)
    with pytest.raises(ValueError, match=r"SurrogateRestart\(state_dir=.*mode=\"data\"\)"):
        run_a.continue_sampling(_conf(tmp_path / "z2"), stages)


# ---------------------------------------------------------------------------------------------
# default surrogate reuse, in process
# ---------------------------------------------------------------------------------------------

def test_polynomial_updater_state_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    parameters = rng.normal(size=(40, 2))
    original = PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=2, weighting="multiplicity")
    original.add_data(parameters, parameters @ A_MATRIX.T, np.ones((40, 1)))
    assert original.supports_state_persistence() and original.get_initial_snapshots() is None
    sampling_output = tmp_path / "sampling_output"
    surrDAMH.SurrogateRestart(state_dir=str(sampling_output)).save(original)

    restored = SurrogateReused(str(tmp_path))
    assert isinstance(restored, PolynomialSklearnUpdater)
    assert (restored.max_degree, restored.weighting, restored.num_snapshots) == (2, "multiplicity", 40)
    assert restored.pretrained_ready and restored.training_data_loaded
    np.testing.assert_array_equal(restored.get_initial_snapshots()[0], parameters)
    probe = rng.normal(size=(5, 2))
    np.testing.assert_allclose(restored.get_evaluator()(probe), original.get_evaluator()(probe))

    # the local surrogate manager counts restored snapshots without adding them twice
    manager = LocalSurrogateManager(SurrogateReused(str(tmp_path)), min_snapshots_initial=1,
                                    initial_snapshots=restored.get_initial_snapshots())
    assert manager.updater.num_snapshots == 40 and manager.no_snapshots_total == 40
    assert manager.evaluator_is_available()


def test_local_continuation_reuses_the_saved_surrogate_and_saves_its_own(tmp_path):
    problem = _problem()
    updater = PolynomialSklearnUpdater(no_parameters=2, no_observations=2, max_degree=1)
    run_a = problem.run_sampling_local(_conf(tmp_path / "a"), [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5),
                                                                     max_evaluations=60)],
                                       surrogate_updater=updater)
    surrDAMH.SurrogateRestart(state_dir=str(tmp_path / "a" / "sampling_output")).save(updater)

    # a DAMH FIRST stage works without any MH warm-up: the surrogate is restored
    damh = [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=40, subchain_length=3)]
    run_b = run_a.continue_sampling_local(_conf(tmp_path / "b"), damh)
    assert run_b.stage_results[0].name == "alg0001_DAMH-SMU"
    assert _manifest(tmp_path / "b")["surrogate"]["updater_class"] == "PolynomialSklearnUpdater"
    # its state (A's snapshots + B's) was saved at the end, so a further continuation reuses it too
    checkpoint, data = surrogate_state_paths(str(tmp_path / "b"))
    assert os.path.isfile(checkpoint) and os.path.isfile(data)
    with np.load(data) as loaded:
        assert loaded["parameters"].shape[0] > updater.num_snapshots
    run_b.continue_sampling_local(_conf(tmp_path / "c"),
                                  [Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=20)])
    assert _stage_dirs(tmp_path / "c") == ["alg0002_DAMH-SMU"]
