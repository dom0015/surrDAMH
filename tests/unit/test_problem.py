#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ``surrDAMH.Problem`` (2026-10-08 API split of ``SamplingFramework``): where the
problem sizes come from (explicit argument > solver instance > prior/likelihood dimension),
the mismatch and missing-size errors, the dimension-free ``Normal``, the hand-over of the
sizes to ``Configuration`` (``resolve_problem_sizes``), ``describe()``, ``SamplingRun.load``,
and the observation-shape checks C2 (solvers-pool helper) and C3 (first evaluation).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.stats

import surrDAMH
from surrDAMH.distributions import (FromScipy, GaussianMixture, Normal, NormalComponent,
                                    PriorIndependentComponents, StandardizedNormal,
                                    distribution_dimension)
from surrDAMH.process_SOLVER import check_declared_solver_sizes
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage

pytestmark = pytest.mark.unit


class DeclaringSolver(Solver):
    """Declares its sizes; ``G(x) = sum(x) * ones(no_observations)``."""

    def __init__(self, no_parameters=2, no_observations=3, solver_id=0, output_dir=None):
        super().__init__(solver_id, output_dir, no_parameters=no_parameters, no_observations=no_observations)

    def get_observations(self):
        return np.full(self.no_observations, float(np.sum(self.parameters)))


class SilentSolver(Solver):
    """Declares nothing; returns ``n_out`` values."""

    def __init__(self, n_out=2):
        self.n_out = n_out

    def get_observations(self):
        return np.full(self.n_out, float(np.sum(self.parameters)))


class NoDimensionDistribution(surrDAMH.Distribution):
    """A distribution nothing can read a dimension off."""

    def logpdf(self, sample):
        return 0.0


class AttributeDistribution(surrDAMH.Distribution):
    def __init__(self, **attributes):
        self.__dict__.update(attributes)


def _prior(d):
    return PriorIndependentComponents([NormalComponent(mu=0.0, sigma=1.0) for _ in range(d)])


# ---------------------------------------------------------------------------------------------
# distribution_dimension
# ---------------------------------------------------------------------------------------------

def test_distribution_dimension_of_every_shipped_class():
    assert distribution_dimension(Normal(mean=np.zeros(4), sd=1.0)) == 4
    assert distribution_dimension(Normal(mean=0.0, sd=1.0, dim=3)) == 3
    assert distribution_dimension(StandardizedNormal(Normal(mean=np.zeros(5), sd=2.0))) == 5
    assert distribution_dimension(_prior(6)) == 6
    assert distribution_dimension(GaussianMixture(means=[np.zeros(2), np.ones(2)])) == 2
    assert distribution_dimension(FromScipy(scipy.stats.multivariate_normal(mean=np.zeros(3)))) == 3
    assert distribution_dimension(FromScipy(scipy.stats.norm(0.0, 1.0))) is None  # no 'dim'
    assert distribution_dimension(AttributeDistribution(no_parameters=7)) == 7
    assert distribution_dimension(AttributeDistribution(dim=8)) == 8
    assert distribution_dimension(NoDimensionDistribution()) is None
    # dimension-free Normal: scalar mean, no dim
    assert distribution_dimension(Normal(mean=1.0, sd=1.0)) is None


def test_from_scipy_dimension_never_draws_samples():
    class NoDrawing:
        dim = 2

        def logpdf(self, x):
            return 0.0

        def rvs(self, *args, **kwargs):
            raise AssertionError("distribution_dimension must not draw samples")

    assert distribution_dimension(FromScipy(NoDrawing())) == 2


def test_dimension_free_normal_flag_and_with_dimension():
    free = Normal(mean=5.0, sd=0.5)
    assert free.dimension_free is True
    assert free.n == 1  # direct use unchanged: a 1-D distribution
    broadcast = free.with_dimension(3)
    assert broadcast.dimension_free is False
    np.testing.assert_array_equal(broadcast.mean, np.full(3, 5.0))
    np.testing.assert_array_equal(broadcast.sd, np.full(3, 0.5))
    # not dimension-free: unchanged object
    assert Normal(mean=5.0, sd=0.5, dim=2).dimension_free is False
    assert Normal(mean=np.zeros(2), sd=1.0).dimension_free is False
    assert Normal(mean=0.0, cov=np.eye(1)).dimension_free is False
    fixed = Normal(mean=5.0, sd=0.5, dim=2)
    assert fixed.with_dimension(4) is fixed


# ---------------------------------------------------------------------------------------------
# size resolution: each source, precedence, mismatch, missing
# ---------------------------------------------------------------------------------------------

def test_sizes_from_prior_and_likelihood_dimensions():
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0), solver=SilentSolver(3))
    assert (problem.no_parameters, problem.no_observations) == (2, 3)
    assert "prior" in problem.no_parameters_source
    assert "likelihood" in problem.no_observations_source


def test_sizes_from_the_solver_instance():
    # prior_bound=None: the centre/scale of this stub prior are unknown (S0)
    problem = surrDAMH.Problem(NoDimensionDistribution(), NoDimensionDistribution(),
                               solver=DeclaringSolver(no_parameters=4, no_observations=5), prior_bound=None)
    assert (problem.no_parameters, problem.no_observations) == (4, 5)
    assert problem.no_parameters_source == "solver DeclaringSolver.no_parameters"
    assert problem.no_observations_source == "solver DeclaringSolver.no_observations"


def test_sizes_from_explicit_arguments():
    problem = surrDAMH.Problem(NoDimensionDistribution(), NoDimensionDistribution(), solver=SilentSolver(),
                               no_parameters=6, no_observations=2, prior_bound=None)
    assert (problem.no_parameters, problem.no_observations) == (6, 2)
    assert problem.no_parameters_source == "Problem(no_parameters=...)"


def test_explicit_wins_when_all_sources_agree():
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0),
                               solver=DeclaringSolver(no_parameters=2, no_observations=3),
                               no_parameters=2, no_observations=3)
    assert problem.no_parameters_source == "Problem(no_parameters=...)"
    assert problem.no_observations_source == "Problem(no_observations=...)"
    # without the explicit value the solver comes before the distribution
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0),
                               solver=DeclaringSolver(no_parameters=2, no_observations=3))
    assert problem.no_parameters_source == "solver DeclaringSolver.no_parameters"


def test_mismatch_names_every_source_and_value():
    with pytest.raises(ValueError) as excinfo:
        surrDAMH.Problem(_prior(3), Normal(mean=np.zeros(3), sd=1.0),
                         solver=DeclaringSolver(no_parameters=2, no_observations=3), no_parameters=4)
    message = str(excinfo.value)
    assert "inconsistent no_parameters" in message
    assert "Problem(no_parameters=...) = 4" in message
    assert "solver DeclaringSolver.no_parameters = 2" in message
    assert "prior (PriorIndependentComponents) dimension = 3" in message

    with pytest.raises(ValueError, match=r"inconsistent no_observations: solver DeclaringSolver.no_observations "
                                         r"= 3, likelihood \(Normal\) dimension = 2"):
        surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(2), sd=1.0),
                         solver=DeclaringSolver(no_parameters=2, no_observations=3))


def test_error_when_nothing_supplies_a_size():
    with pytest.raises(ValueError, match=r"cannot determine no_parameters.*give dim= on the prior.*no_parameters="):
        surrDAMH.Problem(NoDimensionDistribution(), Normal(mean=np.zeros(2), sd=1.0), solver=SilentSolver())
    with pytest.raises(ValueError, match=r"cannot determine no_observations.*no_observations="):
        surrDAMH.Problem(_prior(2), NoDimensionDistribution(), solver=SilentSolver())


def test_dimension_free_prior_is_broadcast_to_the_solver_size():
    problem = surrDAMH.Problem(Normal(mean=1.0, sd=2.0), Normal(mean=np.zeros(3), sd=1.0),
                               solver=DeclaringSolver(no_parameters=4, no_observations=3))
    assert problem.no_parameters == 4
    assert isinstance(problem.prior_physical, Normal)
    np.testing.assert_array_equal(problem.prior_physical.mean, np.full(4, 1.0))
    np.testing.assert_array_equal(problem.prior_physical.sd, np.full(4, 2.0))
    assert isinstance(problem.prior, StandardizedNormal) and problem.prior.n == 4
    np.testing.assert_allclose(problem.prior.transform(np.ones(4)), np.full(4, 3.0))
    assert "dimension-free prior broadcast" in problem.no_parameters_source


def test_dimension_free_prior_without_any_size_source_raises_with_both_fixes():
    with pytest.raises(ValueError) as excinfo:
        surrDAMH.Problem(Normal(mean=0.0, sd=1.0), Normal(mean=np.zeros(2), sd=1.0), solver=SilentSolver())
    message = str(excinfo.value)
    assert "Normal(mean=<scalar>) has no dimension" in message
    assert "dim=" in message and "no_parameters=" in message


def test_dimension_free_likelihood_is_broadcast_to_the_explicit_size():
    problem = surrDAMH.Problem(_prior(2), Normal(mean=5.0, sd=0.1), solver=SilentSolver(3), no_observations=3)
    assert problem.no_observations == 3
    np.testing.assert_array_equal(problem.likelihood.mean, np.full(3, 5.0))
    np.testing.assert_array_equal(problem.likelihood.sd, np.full(3, 0.1))
    with pytest.raises(ValueError, match=r"cannot determine no_observations: the likelihood Normal\(mean=<scalar>\)"):
        surrDAMH.Problem(_prior(2), Normal(mean=5.0, sd=0.1), solver=SilentSolver(3))


def test_solver_argument_type_dispatch():
    spec = surrDAMH.SolverSpec(solver_module_path=__file__, solver_module_name="test_problem",
                               solver_class_name="DeclaringSolver", solver_parameters={})
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0), solver=spec)
    assert problem.solver_spec is spec and problem.solver_instance is None
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0), solver=SilentSolver(3))
    assert problem.solver_spec is None and isinstance(problem.solver_instance, SilentSolver)
    with pytest.raises(ValueError, match="give the forward model"):
        surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0), solver=None)
    with pytest.raises(TypeError, match="surrDAMH.Solver instance or a SolverSpec"):
        surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0), solver=object())


def test_prior_physical_and_standardized_prior():
    physical = Normal(mean=np.array([1.0, 2.0]), sd=np.array([0.5, 0.5]))
    problem = surrDAMH.Problem(physical, Normal(mean=np.zeros(2), sd=1.0), solver=SilentSolver())
    assert problem.prior_physical is physical
    assert isinstance(problem.prior, StandardizedNormal) and problem.prior.physical is physical


def test_describe_names_sizes_and_sources():
    problem = surrDAMH.Problem(_prior(2), Normal(mean=5.0, sd=1.0), solver=DeclaringSolver(no_observations=1))
    text = problem.describe()
    assert "no_parameters=2 (from solver DeclaringSolver.no_parameters)" in text
    assert "no_observations=1 (from solver DeclaringSolver.no_observations (dimension-free likelihood" in text
    assert "solver instance DeclaringSolver" in text


# ---------------------------------------------------------------------------------------------
# Configuration hand-over
# ---------------------------------------------------------------------------------------------

def test_configuration_output_dir_is_first_and_sizes_are_optional(tmp_path):
    conf = surrDAMH.Configuration(str(tmp_path), use_collector=False, use_solvers_pool=False)
    assert conf.output_dir == str(tmp_path)
    assert conf.no_parameters is None and conf.no_observations is None


def test_resolve_problem_sizes_sets_checks_and_updates_requested_fields(tmp_path):
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False)
    assert conf.requested_posterior_fields()["no_parameters"] is None
    conf.resolve_problem_sizes(3, 2)
    assert (conf.no_parameters, conf.no_observations) == (3, 2)
    assert conf.requested_posterior_fields()["no_parameters"] == 3
    assert conf.requested_posterior_fields()["no_observations"] == 2
    assert "no_parameters*=3" in conf.describe() and "no_observations*=2" in conf.describe()
    conf.resolve_problem_sizes(3, 2)  # equal: fine, idempotent

    explicit = surrDAMH.Configuration(output_dir=str(tmp_path), no_parameters=3, no_observations=2,
                                      use_collector=False, use_solvers_pool=False)
    with pytest.raises(ValueError, match=r"Configuration\(no_observations=2\) does not match the problem's "
                                         r"no_observations=5"):
        explicit.resolve_problem_sizes(3, 5)


def test_load_continuation_is_a_no_op_unless_continued(tmp_path):
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False)
    conf.load_continuation()
    assert conf.continued_samples is None


def test_run_sampling_local_resolves_sizes_into_the_configuration(tmp_path):
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False)
    problem = surrDAMH.Problem(_prior(2), Normal(mean=1.0, sd=1.0), solver=DeclaringSolver(no_observations=1))
    run = problem.run_sampling_local(conf, [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5),
                                                  max_evaluations=10)])
    assert (conf.no_parameters, conf.no_observations) == (2, 1)
    assert run.runner == "local" and run.conf is conf and run.problem is problem
    assert run.output_dir == str(tmp_path)
    assert len(run.stage_results) == 1 and run.stages[0].name == "alg0000_MH"
    assert isinstance(run.solver_instance, DeclaringSolver)
    with open(run.manifest_path) as f:
        manifest = json.load(f)
    assert manifest["problem"]["no_parameters"] == 2
    assert manifest["problem"]["no_observations_source"].startswith("solver DeclaringSolver.no_observations")


def test_run_sampling_local_rejects_a_configuration_that_disagrees(tmp_path):
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), no_parameters=3, use_collector=False,
                                  use_solvers_pool=False)
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(1), sd=1.0), solver=DeclaringSolver(no_observations=1))
    with pytest.raises(ValueError, match=r"Configuration\(no_parameters=3\) does not match"):
        problem.run_sampling_local(conf, [Stage(algorithm="MH", max_evaluations=5)])


def test_run_sampling_rejects_a_solver_instance_with_the_solvers_pool(tmp_path):
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False)
    conf.use_solvers_pool = True  # a world of size 1 cannot host a pool rank; only the check matters
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0), solver=DeclaringSolver())
    with pytest.raises(ValueError, match=r"use_solvers_pool=True needs solver=SolverSpec"):
        problem.run_sampling(conf, [Stage(algorithm="MH", max_evaluations=5)])


def test_sampling_run_load_rebuilds_a_report_only_run(tmp_path):
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False)
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0), solver=DeclaringSolver())
    problem.run_sampling_local(conf, [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5), max_evaluations=10)])

    loaded = surrDAMH.SamplingRun.load(str(tmp_path), problem)
    assert loaded.runner == "local"
    assert loaded.stages is None and loaded.stage_results is None
    assert (loaded.conf.no_parameters, loaded.conf.no_observations) == (2, 3)
    assert loaded.conf.use_solvers_pool is False
    assert loaded.solver_instance is problem.solver_instance
    with pytest.raises(FileNotFoundError):
        surrDAMH.SamplingRun.load(str(tmp_path / "missing"), problem)


# ---------------------------------------------------------------------------------------------
# C3: the first evaluation must return (no_observations,)
# ---------------------------------------------------------------------------------------------

class WrongCountSolver(Solver):
    """Declares 2 observations but returns 3 (the bug C3 catches)."""

    def __init__(self, solver_id=0, output_dir=None):
        super().__init__(solver_id, output_dir, no_parameters=2)

    def get_observations(self):
        return np.zeros(3)


def test_c3_in_process_solver_returning_the_wrong_number_of_observations(tmp_path):
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False)
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(2), sd=1.0), solver=WrongCountSolver())
    with pytest.raises(ValueError) as excinfo:
        problem.run_sampling_local(conf, [Stage(algorithm="MH", max_evaluations=5)])
    message = str(excinfo.value)
    assert "WrongCountSolver.get_observations() returned shape (3,)" in message
    assert "expected (2,)" in message


def test_c3_check_observations_shape_helper():
    from surrDAMH.solvers import check_observations_shape
    check_observations_shape(np.zeros(2), 2, "X")  # correct shape passes
    check_observations_shape([1.0, 2.0], 2, "X")   # a list of the right length too
    with pytest.raises(ValueError, match=r"X.get_observations\(\) returned shape \(\), expected \(2,\)"):
        check_observations_shape(1.0, 2, "X")
    with pytest.raises(ValueError, match=r"returned shape \(1, 2\), expected \(2,\)"):
        check_observations_shape(np.zeros((1, 2)), 2, "X")


# ---------------------------------------------------------------------------------------------
# C2: the solvers pool compares declared and configured sizes
# ---------------------------------------------------------------------------------------------

def test_c2_check_declared_solver_sizes(tmp_path, capsys):
    import surrDAMH.process_SOLVER as process_solver
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), no_parameters=2, no_observations=3,
                                  use_collector=False, use_solvers_pool=False)
    spec = surrDAMH.SolverSpec(solver_module_path=__file__, solver_module_name="test_problem",
                               solver_class_name="DeclaringSolver", solver_parameters={})
    check_declared_solver_sizes([(2, 3)], conf, 0, spec)  # agree: silent
    with pytest.raises(ValueError, match=r"solver DeclaringSolver \(solver_id=1, child rank 0\) declares "
                                         r"no_observations=4, but the run is configured with no_observations=3"):
        check_declared_solver_sizes([(2, 4)], conf, 1, spec)

    process_solver._undeclared_sizes_warned = False
    capsys.readouterr()
    check_declared_solver_sizes([(None, None), (None, 3)], conf, 0, spec)
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and "does not declare no_parameters" in out


def test_write_report_of_a_local_run_and_of_a_loaded_run(tmp_path):
    """``SamplingRun.write_report`` for ``runner="local"`` (no MPI calls) and after ``load``."""
    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False)
    problem = surrDAMH.Problem(_prior(2), Normal(mean=np.zeros(3), sd=1.0), solver=DeclaringSolver())
    run = problem.run_sampling_local(conf, [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5),
                                                  max_evaluations=40)])
    report = tmp_path / "post_processing_output" / "report_extended.html"
    samples = run.write_report(include_expensive_sections=False)
    assert samples is not None and report.is_file()
    report.unlink()

    samples = surrDAMH.SamplingRun.load(str(tmp_path), problem).write_report(include_expensive_sections=False)
    assert samples is not None and report.is_file()
    assert "sampling_output/run_manifest.json" in report.read_text()  # configuration taken from the manifest


def test_write_report_accepts_a_plain_number_for_a_single_observation(tmp_path):
    """2026-10-14: with one observation, ``write_report(observations=<float>)`` works (it used to fail in the
    observation histogram with ``'float' object is not subscriptable``); a wrong length raises ValueError."""
    import pytest
    from surrDAMH.solvers import Solver

    class OneObservationSolver(Solver):
        def __init__(self, solver_id=0, output_dir=None):
            self.no_parameters, self.no_observations = 2, 1

        def set_parameters(self, parameters):
            self.u = np.asarray(parameters, dtype=float)

        def get_observations(self):
            return np.array([float(np.sum(self.u ** 2))])

    conf = surrDAMH.Configuration(output_dir=str(tmp_path), use_collector=False, use_solvers_pool=False,
                                  save_snapshots_to_file=True)  # raw data -> the observation histogram runs
    problem = surrDAMH.Problem(_prior(2), Normal(mean=1.0, sd=0.5), solver=OneObservationSolver())
    run = problem.run_sampling_local(conf, [Stage(algorithm="MH", proposal=RandomWalk(scale=0.5),
                                                  max_evaluations=60)])
    for value in (1.0, np.float64(1.0), np.array(1.0), [1.0], np.array([1.0])):
        assert run.write_report(observations=value) is not None
    with pytest.raises(ValueError, match="no_observations=1"):
        run.write_report(observations=[1.0, 2.0])
