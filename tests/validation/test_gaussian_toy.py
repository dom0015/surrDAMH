#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 statistical validation on the linear-Gaussian toy problem
(library_notes/07_testing_plan.md, items V1-V3).

The problem is the ``linear_gaussian_problem`` fixture from ``tests/conftest.py``:
``G(x) = A x``, Gaussian likelihood, standard-normal prior, so the posterior is known
in closed form. Chains are produced by the MPI-free single-chain runner
``Problem.run_sampling_local`` and read back from the CSV files the sampler writes
to ``<output_dir>/sampling_output/samples/<stage>/rank0000.csv`` (column 0 = weight,
columns 1..no_parameters = parameters).

| # | setup                                                                   | expectation          |
|---|-------------------------------------------------------------------------|----------------------|
| V1| one ``MH`` stage                                                         | matches closed form  |
| V2| ``DAMH``, wrong but FIXED surrogate ``A' = 1.3 A``,                       | same posterior as V1 |
|   | ``surrogate_model_updates=False``, ``subchain_length in {1, 5, 20}``  | for every K          |
| V3| as V2 but ``surrogate_model_updates=True`` with a real (retrained)        | same posterior as V1 |
|   | degree-1 ``PolynomialSklearnUpdater``, same K values                     | for every K          |
| V3b| as V3 with a 3-nearest-neighbour ``KDTreeUpdater``: a surrogate that      | same posterior as V1 |
|   | stays wrong while it keeps changing (the variant that stresses 1.2)      | for every K          |
| V3c| as V3 with a small torch MLP (``NeuralNetworkUpdater``)        | same posterior as V1 |
|   | trained on the fly, ``subchain_length in {1, 5, 20}``                 | for every K          |
| V3d| the full three-stage chain with that MLP: MH -> DAMH-SMU(K=5) ->          | frozen stage matches |
|   | DAMH(K=5, frozen surrogate)                                              | the closed form      |

V2 is the decisive test: delayed acceptance with a *wrong* surrogate must leave the
target distribution unchanged, because the sub-chain is only a proposal mechanism and the
outer acceptance step corrects for it (see the derivation comment above
``Algorithm_DAMH.run`` in ``surrDAMH/modules/algorithms.py``).

Tolerances are Monte-Carlo informed rather than arbitrary: the standard error of every
compared quantity (each mean component, each covariance entry) is estimated by
*batch means* on the decompressed chain, which needs no autocorrelation-time estimate
and stays reliable for the strongly autocorrelated DAMH chains. A deviation is accepted
if it is below ``TOLERANCE_IN_SE`` standard errors. Runs are fully deterministic (fixed
sampler/proposal seeds and a fixed initial sample), so these tests do not flake.

The chain I/O, the batch-means machinery and the tolerance itself live in
``tests/helpers_statistics.py``, shared with the multi-chain MPI checks in
``tests/mpi/test_mpi_posterior.py``.

Wall time (one core, 2026-09-16): V1 ~4 s, V2 ~12/16/26 s and V3 ~11/18/25 s for K = 1/5/20;
the whole module ~2 minutes. V3c adds ~5/6/13 s for K = 1/5/20 and V3d ~10 s (2026-09-17).
"""

from __future__ import annotations

import numpy as np
import pytest

# tests/ is on sys.path (pytest imports tests/conftest.py from there), see helpers_statistics
from helpers_statistics import (TOLERANCE_IN_SE, ChainStatistics, FixedInitialSample,
                                FixedLinearEvaluator)
from helpers_statistics import assert_close as _assert_close
from helpers_statistics import assert_matches_closed_form, chain_statistics, max_deviation_in_se
from helpers_statistics import read_weighted_samples as _read_weighted_samples

from surrDAMH.configuration import Configuration
from surrDAMH.proposals import RandomWalk
import surrDAMH
from surrDAMH.stages import Stage
from surrDAMH.surrogates import (KDTreeUpdater, NeuralNetworkUpdater,
                                 PolynomialSklearnUpdater)
from surrDAMH.surrogates.parent import Evaluator

pytestmark = pytest.mark.validation

PROPOSAL_SD = 0.5  # ~1.6 posterior sd; gives an MH acceptance rate of ~34 % on this problem
SURROGATE_ERROR_FACTOR = 1.3  # the deliberately wrong surrogate is A' = 1.3 * A
assert TOLERANCE_IN_SE == 4.0  # the shared tolerance these tests were written against

MH_EVALUATIONS = 200_000
DAMH_EVALUATIONS = {1: 100_000, 5: 100_000, 20: 60_000}
DAMH_SMU_EVALUATIONS = {1: 20_000, 5: 20_000, 20: 10_000}  # smaller: the updater refits on all snapshots
DAMH_SMU_KDTREE_EVALUATIONS = {1: 30_000, 5: 30_000, 20: 15_000}
DAMH_SMU_NN_EVALUATIONS = {1: 20_000, 5: 20_000, 20: 20_000}
SMU_BOOTSTRAP_EVALUATIONS = 200  # short MH stage that produces the first snapshots

# V3c/V3d surrogate: small enough (and trained briefly enough) that it stays clearly short of the
# exact linear model, so the outer acceptance step is really exercised - see the V3c docstring.
NN_HIDDEN_LAYER_SIZES = (16, 16)
NN_ITERATIONS_BATCH = 50  # optimizer steps per train() call
NN_MIN_SNAPSHOTS_INITIAL = 50
NN_MIN_SNAPSHOTS_TO_UPDATE = 200


def _neural_network_updater(problem, updater_class=NeuralNetworkUpdater):
    """The torch MLP updater used by V3c/V3d (``weighting``/``output_normalization`` at default)."""
    return updater_class(no_parameters=problem.no_parameters,
                         no_observations=problem.no_observations,
                         hidden_layer_sizes=NN_HIDDEN_LAYER_SIZES,
                         iterations_batch=NN_ITERATIONS_BATCH, seed=0)


def _chain_statistics(samples, stage_result) -> ChainStatistics:
    """Statistics of one local chain (thin wrapper around the shared helper)."""
    return chain_statistics(samples,
                            counter_accepted=stage_result.counter_accepted,
                            counter_rejected=stage_result.counter_rejected,
                            counter_prerejected=stage_result.counter_prerejected)


def _run_stages(problem, output_dir, stages: list[Stage], min_snapshots_initial: int = 1,
                min_snapshots_to_update: int = 1, **surrogate_kwargs):
    """Run ``stages`` locally and return the ``SamplingRun`` (``stage_results``: all stages).

    ``surrogate_kwargs``: ``updater=`` or ``evaluator=`` (passed on as ``surrogate_updater`` /
    ``surrogate_evaluator``)."""
    conf = Configuration(no_parameters=problem.no_parameters,
                         no_observations=problem.no_observations,
                         output_dir=str(output_dir),
                         use_collector=False,
                         use_solvers_pool=False,
                         # start at the posterior mean: deterministic and free of burn-in bias
                         initial_sample_type="user_specified",
                         initial_samples_distribution=FixedInitialSample(problem.posterior_mean),
                         min_snapshots_initial=min_snapshots_initial,
                         min_snapshots_to_update=min_snapshots_to_update)
    surrogate = {f"surrogate_{name}": value for name, value in surrogate_kwargs.items()}
    return surrDAMH.Problem(problem.prior, problem.likelihood, solver=problem.solver).run_sampling_local(
        conf, stages, **surrogate)


def _stage_statistics(problem, output_dir, stage_result) -> ChainStatistics:
    """Statistics of the chain one finished stage wrote."""
    samples = _read_weighted_samples(str(output_dir), stage_result.name, problem.no_parameters)
    return _chain_statistics(samples, stage_result)


def _run_chain(problem, output_dir, stages: list[Stage], stage_index: int = -1,
               min_snapshots_initial: int = 1, min_snapshots_to_update: int = 1,
               **surrogate_kwargs) -> ChainStatistics:
    """Run ``stages`` locally and summarize the chain produced by ``stages[stage_index]``."""
    result = _run_stages(problem, output_dir, stages, min_snapshots_initial=min_snapshots_initial,
                         min_snapshots_to_update=min_snapshots_to_update, **surrogate_kwargs)
    return _stage_statistics(problem, output_dir, result.stage_results[stage_index])


def _assert_matches_closed_form(statistics: ChainStatistics, problem, label: str) -> None:
    assert_matches_closed_form(statistics, problem.posterior_mean, problem.posterior_cov, label)


def _report(label: str, statistics: ChainStatistics, problem) -> None:
    """One summary line per analysed stage (visible with ``pytest -s``), used by V3c/V3d.

    The surrogate of those two cases is retrained from data the chain itself produces, so unlike
    V1-V3b their numbers are worth reading even on a pass; this keeps them out of a scratch script.
    """
    print(f"\n[{label}] {statistics.no_samples} samples after burn-in, "
          f"acc/rej/prerej {statistics.counter_accepted}/{statistics.counter_rejected}/"
          f"{statistics.counter_prerejected}\n"
          f"    mean {np.array2string(statistics.mean, precision=4)} "
          f"vs closed form {np.array2string(problem.posterior_mean, precision=4)} "
          f"(max |dev|/SE {max_deviation_in_se(statistics.mean, problem.posterior_mean, statistics.se_mean):.2f})\n"
          f"    cov  {np.array2string(statistics.cov.ravel(), precision=4)} "
          f"vs closed form {np.array2string(problem.posterior_cov.ravel(), precision=4)} "
          f"(max |dev|/SE {max_deviation_in_se(statistics.cov, problem.posterior_cov, statistics.se_cov):.2f})",
          flush=True)


def _assert_matches_reference_chain(statistics: ChainStatistics, reference: ChainStatistics,
                                    label: str) -> None:
    """Compare two independent chains; the standard errors of both are combined in quadrature."""
    _assert_close(f"{label} posterior mean vs V1 (MH)", statistics.mean, reference.mean,
                  np.hypot(statistics.se_mean, reference.se_mean))
    _assert_close(f"{label} posterior covariance vs V1 (MH)", statistics.cov, reference.cov,
                  np.hypot(statistics.se_cov, reference.se_cov))


# The V1 reference chain is deterministic and is reused by V2/V3, so it is computed once per
# session. ``linear_gaussian_problem`` is function-scoped, hence the explicit module-level cache
# instead of a session-scoped fixture.
_MH_REFERENCE_CACHE: dict[str, ChainStatistics] = {}


@pytest.fixture
def mh_reference(linear_gaussian_problem, tmp_path_factory) -> ChainStatistics:
    """V1: plain Metropolis-Hastings chain, used as the reference posterior for V2 and V3."""
    if "statistics" not in _MH_REFERENCE_CACHE:
        stage = Stage(algorithm="MH", proposal=RandomWalk(scale=PROPOSAL_SD),
                      max_evaluations=MH_EVALUATIONS)
        _MH_REFERENCE_CACHE["statistics"] = _run_chain(
            linear_gaussian_problem, tmp_path_factory.mktemp("v1_mh_reference"), [stage])
    return _MH_REFERENCE_CACHE["statistics"]


def test_V1_plain_mh_matches_closed_form_posterior(linear_gaussian_problem, mh_reference):
    """V1: one MH stage must reproduce the closed-form Gaussian posterior."""
    no_evaluations = mh_reference.counter_accepted + mh_reference.counter_rejected
    acceptance_rate = mh_reference.counter_accepted / no_evaluations
    assert 0.1 < acceptance_rate < 0.7, f"unreasonable MH acceptance rate {acceptance_rate:.3f}"
    _assert_matches_closed_form(mh_reference, linear_gaussian_problem, "V1 (MH)")


@pytest.mark.parametrize("subchain_length", [1, 5, 20])
def test_V2_damh_with_wrong_fixed_surrogate_matches_mh(linear_gaussian_problem, mh_reference,
                                                       tmp_path, subchain_length):
    """
    V2 - the decisive correctness test.

    DAMH with a deliberately wrong but FIXED surrogate (``A' = 1.3 A``) must sample the exact
    posterior for every sub-chain length K. With ``surrogate_model_updates=False``
    ``Algorithm_DAMH._refresh_surrogate_evaluator_if_needed`` never swaps the evaluator, so the
    sub-chain is a genuine MH kernel reversible w.r.t. the surrogate posterior and
    ``correction_log_ratio`` telescopes exactly (findings 1.1/1.2 concern a surrogate that
    *changes* mid sub-chain and therefore cannot apply here).
    """
    problem = linear_gaussian_problem
    stage = Stage(algorithm="DAMH", proposal=RandomWalk(scale=PROPOSAL_SD),
                  surrogate_model_updates=False, subchain_length=subchain_length,
                  max_evaluations=DAMH_EVALUATIONS[subchain_length])
    statistics = _run_chain(problem, tmp_path, [stage],
                            evaluator=FixedLinearEvaluator(SURROGATE_ERROR_FACTOR * problem.A))

    label = f"V2 (DAMH, wrong fixed surrogate, K={subchain_length})"
    _assert_matches_closed_form(statistics, problem, label)
    _assert_matches_reference_chain(statistics, mh_reference, label)


@pytest.mark.parametrize("subchain_length", [1, 5, 20])
def test_V3_damh_smu_with_retrained_surrogate_matches_mh(linear_gaussian_problem, mh_reference,
                                                         tmp_path, subchain_length):
    """
    V3: as V2 but with a real surrogate that is retrained during the stage
    (``surrogate_model_updates=True``, degree-1 ``PolynomialSklearnUpdater``).

    A short MH stage bootstraps the first snapshots, because DAMH cannot start without an
    evaluator. Only the DAMH-SMU stage is analysed.

    Per library_notes/07_testing_plan.md this is expected to hold; the known deferred subtlety
    (finding 1.2, mid-sub-chain refresh timing, WS3) is not chased here - the test records what
    is actually observed.

    Caveat worth knowing when reading a pass here: a degree-1 polynomial fits this *linear*
    forward model exactly, so after the first few snapshots the surrogate is numerically the
    full model and the outer acceptance ratio collapses to 1 (observed: zero outer rejections).
    V3 therefore confirms that switching the evaluator mid sub-chain does not corrupt the chain
    in this setting, but it does not put finding 1.2 under stress - that is what V3b is for.
    """
    problem = linear_gaussian_problem
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(scale=PROPOSAL_SD),
              max_evaluations=SMU_BOOTSTRAP_EVALUATIONS),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=PROPOSAL_SD),
              surrogate_model_updates=True, subchain_length=subchain_length,
              max_evaluations=DAMH_SMU_EVALUATIONS[subchain_length]),
    ]
    updater = PolynomialSklearnUpdater(no_parameters=problem.no_parameters,
                                       no_observations=problem.no_observations,
                                       max_degree=1)
    statistics = _run_chain(problem, tmp_path, stages, stage_index=1,
                            min_snapshots_initial=20, min_snapshots_to_update=50,
                            updater=updater)

    label = f"V3 (DAMH-SMU, retrained surrogate, K={subchain_length})"
    _assert_matches_closed_form(statistics, problem, label)
    _assert_matches_reference_chain(statistics, mh_reference, label)


@pytest.mark.parametrize("subchain_length", [1, 5, 20])
def test_V3b_damh_smu_with_wrong_changing_kdtree_surrogate_matches_mh(linear_gaussian_problem,
                                                                      mh_reference, tmp_path,
                                                                      subchain_length):
    """
    V3b: DAMH-SMU with a surrogate that stays wrong while it keeps changing.

    A 3-nearest-neighbour ``KDTreeUpdater`` is a piecewise-constant interpolant, so it never
    becomes the linear model (unlike V3's degree-1 polynomial) and every retraining changes it.
    The outer acceptance step therefore has real work to do (asserted: outer rejections > 0),
    and the evaluator is swapped while sub-chains are running. This is the variant that puts
    finding 1.2 (surrogate refreshed inside a sub-chain, WS3) under stress; with the current
    code it is expected to pass because the correction is recomputed from the refreshed
    surrogate at the point of the swap - the test records what is actually observed.
    """
    problem = linear_gaussian_problem
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(scale=PROPOSAL_SD),
              max_evaluations=SMU_BOOTSTRAP_EVALUATIONS),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=PROPOSAL_SD),
              surrogate_model_updates=True, subchain_length=subchain_length,
              max_evaluations=DAMH_SMU_KDTREE_EVALUATIONS[subchain_length]),
    ]
    updater = KDTreeUpdater(no_parameters=problem.no_parameters,
                            no_observations=problem.no_observations,
                            no_nearest_neighbors=3)
    statistics = _run_chain(problem, tmp_path, stages, stage_index=1,
                            min_snapshots_initial=20, min_snapshots_to_update=50,
                            updater=updater)

    assert statistics.counter_rejected > 0, (
        "the kd-tree surrogate should be wrong enough to cause outer rejections; "
        "otherwise this test does not stress the correction term")
    label = f"V3b (DAMH-SMU, wrong changing kd-tree surrogate, K={subchain_length})"
    _assert_matches_closed_form(statistics, problem, label)
    _assert_matches_reference_chain(statistics, mh_reference, label)


@pytest.mark.parametrize("subchain_length", [1, 5, 20])
def test_V3c_damh_smu_with_nn_surrogate_matches_mh(linear_gaussian_problem, mh_reference,
                                                   tmp_path, subchain_length):
    """
    V3c: DAMH-SMU whose surrogate is a **neural network trained on the fly**.

    Same shape as V3/V3b (short MH bootstrap, then one DAMH-SMU stage that is the one analysed),
    but the updater is ``NeuralNetworkUpdater`` - two hidden layers of 16 SiLU units,
    AdamW, 50 optimizer steps per ``train()``, ``weighting="uniform"`` and
    ``output_normalization="likelihood"`` at their defaults - retrained every 200 snapshots.

    That budget is deliberately small. A network that fitted the linear model exactly would
    collapse the outer acceptance ratio to 1 (the V3 caveat), and a network trained for much
    longer between evaluator swaps would make the test slow without testing anything new. What
    is checked is that a *poor, constantly changing, non-interpolating* surrogate still leaves
    the target distribution untouched: outer rejections > 0, and the sampled posterior matches
    both the closed form and the V1 MH reference within the shared tolerance.
    """
    problem = linear_gaussian_problem
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(scale=PROPOSAL_SD),
              max_evaluations=SMU_BOOTSTRAP_EVALUATIONS),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=PROPOSAL_SD),
              surrogate_model_updates=True, subchain_length=subchain_length,
              max_evaluations=DAMH_SMU_NN_EVALUATIONS[subchain_length]),
    ]
    statistics = _run_chain(problem, tmp_path, stages, stage_index=1,
                            min_snapshots_initial=NN_MIN_SNAPSHOTS_INITIAL,
                            min_snapshots_to_update=NN_MIN_SNAPSHOTS_TO_UPDATE,
                            updater=_neural_network_updater(problem))

    label = f"V3c (DAMH-SMU, on-the-fly NN surrogate, K={subchain_length})"
    _report(label, statistics, problem)
    assert statistics.counter_rejected > 0, (
        "the on-the-fly neural-network surrogate should be wrong enough to cause outer "
        "rejections; otherwise this test does not stress the correction term")
    _assert_matches_closed_form(statistics, problem, label)
    _assert_matches_reference_chain(statistics, mh_reference, label)


class _CountingEvaluator(Evaluator):
    """Wraps an evaluator and counts how often the SAMPLER calls it (V3d; see ``_TrackedNNUpdater``).

    Only one-point batches are counted: the sampler always evaluates one point at a time, while the
    surrogate manager's error-model estimation (S1, 2026-10-09) calls every new evaluator once on the
    whole (multi-row) training set, whether or not a stage uses the error model.
    """

    def __init__(self, inner: Evaluator, version: int) -> None:
        super().__init__(inner.no_parameters, inner.no_observations)
        self.inner = inner
        self.version = version
        self.calls = 0

    def __call__(self, datapoints):
        if np.atleast_2d(datapoints).shape[0] == 1:
            self.calls += 1
        return self.inner(datapoints)


class _TrackedNNUpdater(NeuralNetworkUpdater):
    """
    The V3d surrogate updater: records every training and hands out counting evaluators.

    ``LocalSurrogateManager`` calls ``train()`` and ``get_evaluator()`` together (one evaluator
    per training), so ``evaluators[i]`` is the surrogate produced by training ``i``, and
    ``training_snapshot_counts[i]`` is how many snapshots it was trained on. That is enough to
    tell which evaluator a later stage was running with, see the V3d docstring.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.training_snapshot_counts: list[int] = []
        self.evaluators: list[_CountingEvaluator] = []

    def train(self) -> None:
        super().train()
        self.training_snapshot_counts.append(int(self.no_snapshots))

    def get_evaluator(self) -> _CountingEvaluator:
        evaluator = _CountingEvaluator(super().get_evaluator(), len(self.evaluators))
        self.evaluators.append(evaluator)
        return evaluator


def test_V3d_mh_then_damh_smu_then_frozen_damh_with_nn_surrogate(linear_gaussian_problem,
                                                                 mh_reference, tmp_path):
    """
    V3d: the **whole** three-stage chain in process - ``MH -> DAMH-SMU(K=5) -> DAMH(K=5)`` with
    the same on-the-fly neural-network surrogate as V3c, the last stage running with the
    surrogate *frozen* (``surrogate_model_updates=False``).

    V3/V3b/V3c all stop after the SMU stage, so until now no local test ran an NN surrogate
    through the hand-over into a frozen DAMH stage (the shape the NN toy example uses, and the
    in-process twin of the MPI case B9).

    Two things are asserted besides the posterior of both DAMH stages:

    * the SMU stage really retrained the network several times (``trainings_in_smu >= 3``), and
    * the frozen stage ran on the surrogate the SMU stage ended with. Snapshots are sent to
      the surrogate manager in every stage (``send_snapshots_to_collector`` defaults to True),
      so training does *not* stop in the frozen stage - what stops is the evaluator swapping.
      The proof used here is therefore direct: the number of snapshots collected by the end of
      the SMU stage is ``sum(accepted + rejected)`` of the first two stages (one snapshot per
      full-model evaluation), which splits ``training_snapshot_counts`` into trainings that
      happened during/before the SMU stage and trainings that happened after it; the evaluator
      of the last training of the first group must have been called, and **every** evaluator
      built afterwards must have zero calls.
    """
    problem = linear_gaussian_problem
    subchain_length = 5
    evaluations = DAMH_SMU_NN_EVALUATIONS[subchain_length]
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(scale=PROPOSAL_SD),
              max_evaluations=SMU_BOOTSTRAP_EVALUATIONS),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=PROPOSAL_SD),
              surrogate_model_updates=True, subchain_length=subchain_length,
              max_evaluations=evaluations),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=PROPOSAL_SD),
              surrogate_model_updates=False, subchain_length=subchain_length,
              max_evaluations=evaluations),
    ]
    updater = _neural_network_updater(problem, updater_class=_TrackedNNUpdater)
    result = _run_stages(problem, tmp_path, stages,
                         min_snapshots_initial=NN_MIN_SNAPSHOTS_INITIAL,
                         min_snapshots_to_update=NN_MIN_SNAPSHOTS_TO_UPDATE,
                         updater=updater)
    assert [stage_result.name for stage_result in result.stage_results] == \
        ["alg0000_MH", "alg0001_DAMH-SMU", "alg0002_DAMH"], \
        [stage_result.name for stage_result in result.stage_results]

    # one snapshot per full-model evaluation, i.e. per accepted or rejected outer proposal:
    snapshots_after_smu = sum(stage_result.counter_accepted + stage_result.counter_rejected
                              for stage_result in result.stage_results[:2])
    assert updater.no_snapshots == snapshots_after_smu + (
        result.stage_results[2].counter_accepted + result.stage_results[2].counter_rejected), (
        "one snapshot per full-model evaluation is the assumption the stage split below rests "
        f"on; updater holds {updater.no_snapshots} snapshots")
    trainings_in_smu = sum(1 for count in updater.training_snapshot_counts
                           if count <= snapshots_after_smu)
    assert trainings_in_smu >= 3, (
        f"the network was trained only {trainings_in_smu} time(s) up to the end of the "
        f"DAMH-SMU stage: {updater.training_snapshot_counts}")

    frozen_evaluator = updater.evaluators[trainings_in_smu - 1]
    assert frozen_evaluator.calls > 0, (
        "the last surrogate built during the DAMH-SMU stage was never used")
    later_calls = {evaluator.version: evaluator.calls
                   for evaluator in updater.evaluators[trainings_in_smu:]}
    assert all(calls == 0 for calls in later_calls.values()), (
        "the frozen DAMH stage must keep the surrogate the DAMH-SMU stage ended with, but a "
        f"surrogate trained after that stage was evaluated: {later_calls}")

    smu_statistics = _stage_statistics(problem, tmp_path, result.stage_results[1])
    _report("V3d (DAMH-SMU, on-the-fly NN, K=5)", smu_statistics, problem)
    print(f"    trainings up to the end of the DAMH-SMU stage: {trainings_in_smu} "
          f"(of {len(updater.training_snapshot_counts)} in total), frozen-stage surrogate "
          f"version {frozen_evaluator.version} with {frozen_evaluator.calls} calls", flush=True)
    assert smu_statistics.counter_rejected > 0, (
        "the on-the-fly neural-network surrogate should be wrong enough to cause outer "
        "rejections; otherwise this test does not stress the correction term")
    _assert_matches_closed_form(smu_statistics, problem, "V3d (DAMH-SMU, on-the-fly NN, K=5)")
    _assert_matches_reference_chain(smu_statistics, mh_reference,
                                    "V3d (DAMH-SMU, on-the-fly NN, K=5)")

    frozen_statistics = _stage_statistics(problem, tmp_path, result.stage_results[2])
    _report("V3d (DAMH, frozen NN, K=5)", frozen_statistics, problem)
    _assert_matches_closed_form(frozen_statistics, problem, "V3d (DAMH, frozen NN, K=5)")
    _assert_matches_reference_chain(frozen_statistics, mh_reference, "V3d (DAMH, frozen NN, K=5)")
