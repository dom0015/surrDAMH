"""
Tests for the standalone local runner (``Problem.run_sampling_local``, engine ``surrDAMH.runner_local``).

The reproducibility test runs both samplers inside this process; the MPI runner
is called directly (``run_SAMPLER``) with a world of size 1, so no ``mpiexec``
subprocess is spawned (marker ``unit``).
"""

import numpy as np
import pytest

import surrDAMH
from surrDAMH.modules import algorithms as alg
from surrDAMH.modules.algorithm_interfaces_local import (LocalEvaluatorProvider,
                                                         LocalSurrogateManager)
from surrDAMH.modules.continuation import load_carry_over
from surrDAMH.modules.proposal_builder import build_proposal
from surrDAMH.process_SAMPLER import run_SAMPLER
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage
from surrDAMH.surrogates.parent import Evaluator, Updater

pytestmark = pytest.mark.unit


class ToySolver(Solver):
    """Same model as toy_examples SolverSpecExample1 (2 parameters, 1 observation)."""

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = 2
        self.no_observations = 1

    def set_parameters(self, parameters):
        self.x = parameters[0]
        self.y = parameters[1]

    def get_observations(self):
        return np.array([(self.x ** 2 - self.y) * (np.log((self.x - self.y) ** 2 + 1))])


class FixedSample:
    """Degenerate 'distribution' returning a fixed initial sample."""

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
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=50)]
    return conf, prior, likelihood, stages


def _read_samples(output_dir, stage_name="alg0000_MH"):
    path = output_dir / "sampling_output" / "samples" / stage_name / "rank0000.csv"
    # skiprows=1: output format v2 writes the header "multiplicity,par_0,...,log_posterior"
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def test_run_local_reproduces_chain_0_of_mpi_run(tmp_path):
    """A local MH run must reproduce chain 0 of an MPI run with the same configuration."""
    mpi_dir = tmp_path / "mpi"
    local_dir = tmp_path / "local"

    conf_mpi, prior_mpi, likelihood_mpi, stages_mpi = _make_problem(mpi_dir)
    # world size is 1 in a plain pytest process, so this is exactly chain 0 of an MPI run
    run_SAMPLER(conf_mpi, prior_mpi, likelihood_mpi, stages_mpi,
                solver_instance=ToySolver(), surrogate_evaluator=None)

    conf_local, prior_local, likelihood_local, stages_local = _make_problem(local_dir)
    run = surrDAMH.Problem(prior_local, likelihood_local, solver=ToySolver()).run_sampling_local(
        conf_local, stages_local)

    samples_mpi = _read_samples(mpi_dir)
    samples_local = _read_samples(local_dir)
    assert samples_mpi.shape == samples_local.shape
    # columns: weight, parameter 0, parameter 1, log-posterior
    np.testing.assert_array_equal(samples_mpi, samples_local)

    assert run.runner == "local"
    assert len(run.stage_results) == 1
    assert run.stage_results[0].name == "alg0000_MH"
    assert run.stage_results[-1].current is not None
    np.testing.assert_array_equal(run.stage_results[-1].current.parameters, samples_local[-1, 1:3])


def test_run_local_rejects_mpi_roles(tmp_path):
    conf, prior, likelihood, stages = _make_problem(tmp_path)
    conf.use_solvers_pool = True
    with pytest.raises(ValueError):
        surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(conf, stages)


def test_build_proposal_uses_stage_and_previous_covariance(tmp_path, capsys):
    conf, prior, _, _ = _make_problem(tmp_path)

    stage = Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=10)
    proposal = build_proposal(stage=stage, conf=conf, prior=prior, seed=1)
    np.testing.assert_array_equal(proposal.sd_or_cov, np.array([0.5, 0.5]))

    # 2026-09-20: the per-stage hand-over is a dict keyed by Stage field name ("carried"),
    # not the single prev_cov argument
    stage_carried = Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=10)
    prev_cov = np.array([[0.25, 0.0], [0.0, 0.25]])
    proposal_carried = build_proposal(stage=stage_carried, conf=conf, prior=prior, seed=1,
                                      carried={"scale": prev_cov})
    np.testing.assert_array_equal(proposal_carried.sd_or_cov, prev_cov)

    # 2026-09-21: a stage that leaves proposal_sd_or_cov None is adaptive by default ("adapt
    # unless pinned"); an explicitly NON-adaptive one with nothing to carry over gets the
    # prior-derived default plus a printed warning (it used to raise ValueError)
    stage_missing = Stage(algorithm="MH", max_evaluations=10)
    assert stage_missing.adaptive is True
    stage_fixed_guess = Stage(algorithm="MH", proposal=RandomWalk(adaptive=False), max_evaluations=10)
    capsys.readouterr()
    proposal_guess = build_proposal(stage=stage_fixed_guess, conf=conf, prior=prior, seed=1, carried=None,
                                    stage_index=3)
    out = capsys.readouterr().out
    assert "Warning: stage 3: RandomWalk(adaptive=False) without a scale and no earlier adaptive stage provides one" in out
    assert type(proposal_guess) is not type(proposal_carried)  # plain, non-adaptive random walk
    np.testing.assert_allclose(proposal_guess.sd_or_cov, np.sqrt(2.38 ** 2 / 2) * np.ones(2))


class _ConstantEvaluator(Evaluator):
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, parameters):
        parameters = np.atleast_2d(parameters)
        return np.full((parameters.shape[0], 1), self.value)


class _ConstantUpdater(Updater):
    """Minimal updater producing a constant evaluator; counts trainings."""

    def __init__(self):
        self.no_trainings = 0
        self.no_snapshots = 0

    def delayed_init(self, data):
        return None

    def add_data(self, parameters, observations, weights):
        self.no_snapshots += len(parameters)

    def train(self):
        self.no_trainings += 1

    def get_evaluator(self):
        return _ConstantEvaluator(self.no_trainings)

    def get_initial_snapshots(self):
        return None


def test_local_evaluator_provider_reports_only_new_evaluators():
    """``evaluator_is_available()`` means "a NEW evaluator is pending" (findings A27/M25)."""
    provider = LocalEvaluatorProvider(_ConstantEvaluator(1.0))
    assert provider.evaluator_is_available() is False  # an evaluator exists, but no new one is pending

    provider.set_evaluator(_ConstantEvaluator(2.0))
    assert provider.evaluator_is_available() is True
    provider.get_evaluator()
    assert provider.evaluator_is_available() is False


def test_local_surrogate_manager_reports_only_new_evaluators():
    manager = LocalSurrogateManager(updater=_ConstantUpdater(), min_snapshots_initial=1, min_snapshots_to_update=2)
    assert manager.evaluator_is_available() is False  # nothing trained yet

    manager.send_to_collector([np.array([0.0, 0.0]), np.array([0.0]), 1.0])
    assert manager.evaluator_is_available() is True  # first surrogate trained
    manager.get_evaluator()
    assert manager.evaluator_is_available() is False  # picked up, nothing new

    manager.send_to_collector([np.array([1.0, 1.0]), np.array([1.0]), 1.0])
    assert manager.evaluator_is_available() is False  # below min_snapshots_to_update
    manager.send_to_collector([np.array([2.0, 2.0]), np.array([2.0]), 1.0])
    assert manager.evaluator_is_available() is True  # retrained


class _ExactEvaluator(Evaluator):
    """Surrogate that calls the full model itself, so ``G~(x)`` equals ``G(x)`` bit for bit."""

    def __init__(self, solver):
        self.solver = solver

    def __call__(self, datapoints):
        observations = []
        for parameters in np.atleast_2d(datapoints):
            self.solver.set_parameters(parameters)
            observations.append(np.atleast_1d(self.solver.get_observations()))
        return np.array(observations)


@pytest.mark.parametrize("subchain_length", [1, 5, 20])
def test_damh_with_exact_surrogate_never_rejects(linear_gaussian_problem, tmp_path, subchain_length):
    """
    Fast always-on canary for the DAMH delayed-acceptance correction (WS1 acceptance note; the
    statistical version is ``tests/validation/test_gaussian_toy.py``).

    If the surrogate equals the full model, the outer log acceptance ratio
    ``[log L(y) - log L(x)] - correction_log_ratio`` is identically zero, so every sub-chain
    proposal that reaches the outer test must be accepted. A wrong sign or a broken telescoping
    of ``correction_log_ratio`` would double the ratio instead of cancelling it and produce
    rejections here.
    """
    problem = linear_gaussian_problem
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=2, output_dir=str(tmp_path),
                                  use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedSample([0.3, -0.7]))
    stage = Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), surrogate_model_updates=False,
                  subchain_length=subchain_length, max_evaluations=100)
    run = surrDAMH.Problem(problem.prior, problem.likelihood, solver=problem.solver).run_sampling_local(
        conf, [stage], surrogate_evaluator=_ExactEvaluator(problem.solver))

    stage_result = run.stage_results[0]
    assert stage_result.counter_accepted > 0
    assert stage_result.counter_rejected == 0, (
        f"exact surrogate must give a zero outer log acceptance ratio, but "
        f"{stage_result.counter_rejected} of "
        f"{stage_result.counter_accepted + stage_result.counter_rejected} outer proposals were rejected"
    )


def test_run_local_damh_with_local_surrogate_manager(tmp_path):
    """Best-effort DAMH support: an MH stage followed by a DAMH-SMU stage runs in process."""
    conf, prior, likelihood, _ = _make_problem(tmp_path)
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20),
        Stage(algorithm="DAMH", proposal=RandomWalk(scale=0.5), max_evaluations=20, subchain_length=2),
    ]
    updater = _ConstantUpdater()
    run = surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(
        conf, stages, surrogate_updater=updater)

    assert [stage_result.name for stage_result in run.stage_results] == ["alg0000_MH", "alg0001_DAMH-SMU"]
    assert updater.no_snapshots > 0
    assert updater.no_trainings > 0
    assert isinstance(run.stage_results[-1].current, alg.Sample)


class _WrongFixedEvaluator(Evaluator):
    """Deliberately wrong, never-updated surrogate ``G~(x) = 1.3 * A @ x``."""

    def __init__(self, A):
        self.A = 1.3 * np.asarray(A, dtype=float)
        self.no_parameters = self.A.shape[1]

    def __call__(self, datapoints):
        return np.atleast_2d(datapoints) @ self.A.T


def test_stage_after_use_only_surrogate_starts_from_exact_observations(linear_gaussian_problem, tmp_path):
    """
    Finding A11: after an ``MH(use_only_surrogate=True)`` stage, the carried-over sample must be
    re-evaluated with the FULL model before the next stage starts.

    Without the fix (``algorithms.sample_carried_to_next_stage``), the carried sample keeps the
    surrogate observations/log-likelihood produced by the surrogate-only stage, so the next exact
    stage compares an exact ``log L(y)`` against a surrogate ``log L(x)`` in its first acceptance
    ratio and writes a surrogate log-posterior in its first output row.

    Asserted on the third stage's first ``samples`` row: parameters are the state the second stage
    ended in, and the log-posterior equals the EXACT one. The reference is built from the same
    calls ``AlgorithmBase._compute_log_posterior_terms`` makes (``likelihood.logpdf(G(x))`` plus
    ``prior.logpdf(x)``), so additive-constant conventions of ``Distribution.logpdf`` cannot make
    the comparison spurious. A guard asserts the surrogate value would have differed.

    ``no_observations = 2`` here: this path also needs the A20 fix in ``SurrogateAsSolver``.
    """
    problem = linear_gaussian_problem
    conf = surrDAMH.Configuration(no_parameters=problem.no_parameters,
                                  no_observations=problem.no_observations,
                                  output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedSample([0.3, -0.7]))
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30),
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30, use_only_surrogate=True),
        Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30),
    ]
    evaluator = _WrongFixedEvaluator(problem.A)
    run = surrDAMH.Problem(problem.prior, problem.likelihood, solver=problem.solver).run_sampling_local(
        conf, stages, surrogate_evaluator=evaluator)

    surrogate_stage_final = run.stage_results[1].current.parameters
    rows = _read_samples(tmp_path, stage_name=run.stage_results[2].name)
    first_row = rows[0]
    np.testing.assert_allclose(first_row[1:1 + problem.no_parameters], surrogate_stage_final)

    parameters = surrogate_stage_final
    problem.solver.set_parameters(parameters)
    exact_log_posterior = (problem.likelihood.logpdf(problem.solver.get_observations())
                           + problem.prior.logpdf(parameters))
    surrogate_log_posterior = (problem.likelihood.logpdf(evaluator(parameters).reshape(-1))
                               + problem.prior.logpdf(parameters))
    assert not np.isclose(exact_log_posterior, surrogate_log_posterior), (
        "the test surrogate is not wrong enough to distinguish the two conventions")

    assert first_row[-1] == pytest.approx(exact_log_posterior, abs=1e-12), (
        "the first row of the stage following a use_only_surrogate stage must carry the EXACT "
        f"log-posterior {exact_log_posterior} (surrogate value would be {surrogate_log_posterior})")


def test_run_local_writes_last_sample_and_can_continue_from_it(tmp_path):
    """
    ``run_sampling_local`` writes ``last_sample/<stage>/rank0000.npz`` after every stage (like
    ``run_SAMPLER``), and a second local run with ``initial_sample_type="continued"`` starts
    bit-for-bit from that sample (chain 0 of the source run).
    """
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"

    conf_a, prior, likelihood, stages_a = _make_problem(dir_a)
    run_a = surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(conf_a, stages_a)
    last_sample_file = dir_a / "sampling_output" / "last_sample" / "alg0000_MH" / "rank0000.npz"
    assert last_sample_file.is_file()
    saved = np.load(last_sample_file)["parameters"]
    assert saved.dtype == np.float64
    assert run_a.stage_results[-1].current is not None
    np.testing.assert_array_equal(saved, run_a.stage_results[-1].current.parameters)

    conf_b = surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(dir_b),
                                    use_collector=False, use_solvers_pool=False,
                                    initial_sample_type="continued", continued_from_dir=str(dir_a))
    stages_b = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=20)]
    run_b = surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(conf_b, stages_b)

    samples_b = _read_samples(dir_b)
    # the first written state of run B is the carried-over initial sample (written when it is left)
    np.testing.assert_array_equal(samples_b[0, 1:3], saved)
    assert run_b.stage_results[-1].current is not None
    assert (dir_b / "sampling_output" / "last_sample" / "alg0000_MH" / "rank0000.npz").is_file()


# ---------------------------------------------------------------------------
# G4 (decided 2026-09-17): the initial sample is drawn from a per-chain seeded generator.
# ---------------------------------------------------------------------------

def _make_prior_start_problem(output_dir):
    """Same problem as ``_make_problem`` but with ``initial_sample_type="prior"``."""
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(output_dir),
                                  use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="prior")
    prior = surrDAMH.distributions.PriorIndependentComponents([
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    ])
    likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=1.0)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30)]
    return conf, prior, likelihood, stages


def test_prior_initial_sample_is_reproducible_across_runs(tmp_path):
    """
    G4: ``initial_sample_type="prior"`` used to draw from the unseeded global NumPy RNG, so two
    runs of the same configuration started from different samples (finding 1.9/A14). The draw
    now comes from ``np.random.default_rng(initial_sample_seed(no_stages, rank))``, so two runs
    produce the same initial sample and byte-identical sample CSVs.

    The global RNG is deliberately advanced to a different state between the two runs; before
    G4 that alone changed the initial sample.
    """
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"

    np.random.seed(12345)
    conf_a, prior_a, likelihood_a, stages_a = _make_prior_start_problem(dir_a)
    run_a = surrDAMH.Problem(prior_a, likelihood_a, solver=ToySolver()).run_sampling_local(conf_a, stages_a)

    np.random.seed(999)  # different global RNG state for the second run
    conf_b, prior_b, likelihood_b, stages_b = _make_prior_start_problem(dir_b)
    run_b = surrDAMH.Problem(prior_b, likelihood_b, solver=ToySolver()).run_sampling_local(conf_b, stages_b)

    samples_a = _read_samples(dir_a)
    samples_b = _read_samples(dir_b)
    np.testing.assert_array_equal(samples_a, samples_b)
    assert run_a.stage_results[-1].current is not None and run_b.stage_results[-1].current is not None
    np.testing.assert_array_equal(run_a.stage_results[-1].current.parameters,
                                  run_b.stage_results[-1].current.parameters)


def test_prior_initial_sample_differs_between_ranks(tmp_path):
    """G4: the initial-sample seed is a function of the rank, so two chains of the same run
    start from different samples. Exercised through ``_get_initial_sample``'s MPI counterpart
    by calling the shared seed helper for rank 0 and rank 1."""
    from surrDAMH.distributions.parent import rvs_with_generator
    from surrDAMH.modules.seeds import initial_sample_seed

    _, prior, _, stages = _make_prior_start_problem(tmp_path)
    no_stages = len(stages)

    draws = [rvs_with_generator(prior, np.random.default_rng(initial_sample_seed(no_stages, rank)))
             for rank in range(4)]
    for rank in range(1, 4):
        assert not np.allclose(draws[0], draws[rank]), f"rank {rank} started from chain 0's sample"
    # and each rank's draw is itself reproducible
    repeated = rvs_with_generator(prior, np.random.default_rng(initial_sample_seed(no_stages, 2)))
    np.testing.assert_array_equal(draws[2], repeated)


def test_user_specified_initial_sample_accepts_generatorless_rvs(tmp_path):
    """G4 compatibility: a duck-typed ``initial_samples_distribution`` whose ``rvs()`` takes no
    arguments (the ``FixedSample`` style used across the test suite and by user scripts) still
    works -- ``rvs_with_generator`` falls back to the no-argument call."""
    conf, prior, likelihood, stages = _make_problem(tmp_path)
    run = surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(conf, stages)
    samples = _read_samples(tmp_path)
    np.testing.assert_array_equal(samples[0, 1:3], np.array([0.3, -0.7]))
    assert run.stage_results[-1].current is not None


# ---------------------------------------------------------------------------
# A30 (decided 2026-09-17): a state carried across a stage boundary is counted once.
# ---------------------------------------------------------------------------

def _read_notes(output_dir, stage_name):
    """``notes/<stage>/rank0000.csv``: accepted, rejected, pre-rejected, sum, seed, out_of_bounds."""
    path = output_dir / "sampling_output" / "notes" / stage_name / "rank0000.csv"
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)[0]
    return int(values[0]), int(values[1]), int(values[2])


def test_two_stage_weights_count_each_state_once(tmp_path):
    """
    A30: stage 0 (nothing before it) writes its initial state with the ``+1``; stage 1 starts
    from stage 0's final state, which stage 0 already wrote, so stage 1's first row carries
    only the rejections spent there. Hence

    - stage 0: ``sum(weights) == accepted + rejected + prerejected + 1``,
    - stage 1: ``sum(weights) == accepted + rejected + prerejected``    (no ``+1``),
    - concatenated: ``sum(weights) == total iterations + 1``            (one state per iteration
      plus the run's starting state).
    """
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(tmp_path),
                                  use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedSample([0.3, -0.7]))
    prior = surrDAMH.distributions.PriorIndependentComponents([
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    ])
    likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=1.0)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30),
              Stage(algorithm="MH", proposal=RandomWalk(scale=0.3), max_evaluations=25)]
    surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(conf, stages)

    samples_0 = _read_samples(tmp_path, "alg0000_MH")
    samples_1 = _read_samples(tmp_path, "alg0001_MH")
    accepted_0, rejected_0, prerejected_0 = _read_notes(tmp_path, "alg0000_MH")
    accepted_1, rejected_1, prerejected_1 = _read_notes(tmp_path, "alg0001_MH")

    assert samples_0[:, 0].sum() == accepted_0 + rejected_0 + prerejected_0 + 1
    assert samples_1[:, 0].sum() == accepted_1 + rejected_1 + prerejected_1

    total_iterations = accepted_0 + rejected_0 + prerejected_0 + accepted_1 + rejected_1 + prerejected_1
    assert samples_0[:, 0].sum() + samples_1[:, 0].sum() == total_iterations + 1

    # the carried state is the same parameter vector in both files, written once with its
    # stage-0 dwell time and once with only stage 1's leading rejections
    np.testing.assert_array_equal(samples_0[-1, 1:3], samples_1[0, 1:3])


def test_stage_after_is_excluded_stage_also_drops_the_plus_one(tmp_path):
    """
    A30, the ``is_excluded`` case: stage 1 restarts from the sample stage 0 STARTED from, and
    stage 0 already wrote that sample as its own first row -- so stage 1 must not count it
    again either. Its first row can therefore be weight 0 (first proposal accepted at once).
    """
    conf = surrDAMH.Configuration(no_parameters=2, no_observations=1, output_dir=str(tmp_path),
                                  use_collector=False, use_solvers_pool=False,
                                  initial_sample_type="user_specified",
                                  initial_samples_distribution=FixedSample([0.3, -0.7]))
    prior = surrDAMH.distributions.PriorIndependentComponents([
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    ])
    likelihood = surrDAMH.distributions.Normal(mean=5.0, sd=1.0)
    stages = [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=30, is_excluded=True),
              Stage(algorithm="MH", proposal=RandomWalk(scale=0.3), max_evaluations=25)]
    surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(conf, stages)

    samples_0 = _read_samples(tmp_path, "alg0000_MH")
    samples_1 = _read_samples(tmp_path, "alg0001_MH")
    accepted_1, rejected_1, prerejected_1 = _read_notes(tmp_path, "alg0001_MH")

    np.testing.assert_array_equal(samples_0[0, 1:3], samples_1[0, 1:3])  # same restart state
    assert samples_1[:, 0].sum() == accepted_1 + rejected_1 + prerejected_1
    assert samples_1[0, 0] == 0  # this configuration accepts its first proposal immediately


def test_zero_weight_row_survives_decompress(tmp_path):
    """A30 consequence: ``post_processing.decompress`` must drop a weight-0 row instead of
    breaking (``np.cumsum`` of the weights already handles it)."""
    from surrDAMH.post_processing import decompress

    samples_compressed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = np.array([0, 2, 1])
    decompressed = decompress(samples_compressed, weights)
    np.testing.assert_array_equal(decompressed, np.array([[3.0, 4.0], [3.0, 4.0], [5.0, 6.0]]))


def test_run_local_adaptive_mh_without_starting_covariance_then_damh_carries_it(tmp_path):
    """2026-09-21: an adaptive MH stage may leave ``proposal_sd_or_cov=None`` (the start is
    derived from the prior) and the following DAMH stage takes the adapted covariance over."""
    conf, prior, likelihood, _ = _make_problem(tmp_path)
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=150),
        Stage(algorithm="DAMH", proposal=RandomWalk(), max_evaluations=20, subchain_length=2),
    ]
    run = surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(
        conf, stages, surrogate_updater=_ConstantUpdater())
    assert [r.name for r in run.stage_results] == ["alg0000_MH-adaptive", "alg0001_DAMH-SMU"]
    assert run.stage_results[0].counter_accepted > 0
    assert (tmp_path / "sampling_output" / "adaptive_stats" / "alg0000_MH-adaptive").exists() \
        or any(p.name == "adaptive_stats" for p in tmp_path.rglob("adaptive_stats"))


def test_run_local_adaptive_mh_writes_carry_over_file_consumed_by_the_next_stage(tmp_path):
    """2026-09-21: the pooled adaptation state ``run_sampling_local`` hands to the DAMH stage is also
    persisted to ``sampling_output/carry_over/<stage>.npz``, for exactly the adaptive stage --
    and ``summary`` (diagnostic-only) is internally consistent with ``carry_over`` (what
    ``build_proposal`` actually consumed): ``base_cov * exp(2*log_sigma) == proposal_sd_or_cov``."""
    conf, prior, likelihood, _ = _make_problem(tmp_path)
    stages = [
        Stage(algorithm="MH", proposal=RandomWalk(adaptive=True), max_evaluations=150),
        # adaptive=False pins the consumer stage ("adapt unless pinned" default since 2026-09-21)
        Stage(algorithm="DAMH", proposal=RandomWalk(adaptive=False), max_evaluations=20,
              subchain_length=2),
    ]
    surrDAMH.Problem(prior, likelihood, solver=ToySolver()).run_sampling_local(
        conf, stages, surrogate_updater=_ConstantUpdater())

    # the proposal keys for the adaptive stage only; since C1 (2026-10-11) every stage's file
    # carries the pooled posterior statistics (a non-adaptive stage: nothing else)
    loaded = load_carry_over(str(tmp_path), "alg0000_MH-adaptive")
    assert loaded is not None
    frozen = load_carry_over(str(tmp_path), "alg0001_DAMH-SMU")
    assert frozen is not None and set(frozen[0]) == {"posterior_statistics", "posterior_statistics_n_seen"}
    assert set(frozen[1]) == {"posterior_statistics_n"}
    assert frozen[1]["posterior_statistics_n"] > loaded[1]["posterior_statistics_n"] == loaded[1]["n_pooled"]

    carry_over, summary = loaded
    cov = carry_over["scale"]
    assert cov.shape == (2, 2)
    np.testing.assert_allclose(cov, cov.T)  # symmetric
    assert np.all(np.linalg.eigvalsh(cov) > 0)  # positive definite

    reconstructed = summary["base_cov"] * np.exp(2.0 * summary["log_sigma"])
    np.testing.assert_allclose(reconstructed, cov, rtol=1e-10, atol=1e-14)
    assert summary["n_pooled"] > 0
    assert summary["mean"].shape == (2,)
