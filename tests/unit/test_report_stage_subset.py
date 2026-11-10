"""
Regression (2026-10-08): ``write_report`` must work when the FIRST stage is not displayed -- an
explicit ``stages_to_disp=[1]``, or an ``is_excluded`` warm-up stage left out of the posterior by
the default selection mask (since 2026-10-09 it stays in the report, every pooled section skips it). ``_write_report_rank0`` used to replace
``samples.summary`` by the displayed subset, after which ``plot_acceptance_rates``/the CpUS
section indexed it by the original stage position and raised ``IndexError``.
"""
import os

import numpy as np
import pytest

import surrDAMH
from surrDAMH.proposals import RandomWalk
from surrDAMH.solvers import Solver
from surrDAMH.stages import Stage

pytestmark = pytest.mark.unit


class _ToySolver(Solver):
    def __init__(self, solver_id=0, output_dir=None):
        super().__init__(solver_id, output_dir, no_parameters=2, no_observations=1)

    def set_parameters(self, parameters):
        self.x = np.asarray(parameters, dtype=float)

    def get_observations(self):
        return np.array([self.x[0] + 0.5 * self.x[1]])


def _run(tmp_path, stages):
    prior = surrDAMH.distributions.Normal(mean=0.0, sd=1.0, dim=2)
    likelihood = surrDAMH.distributions.Normal(mean=[0.3], sd=0.5)
    problem = surrDAMH.Problem(prior, likelihood, solver=_ToySolver())
    conf = surrDAMH.Configuration(output_dir=str(tmp_path / "run"), use_collector=False, use_solvers_pool=False)
    return problem.run_sampling_local(conf, stages)


def _report_files(run):
    out = os.path.join(run.output_dir, "post_processing_output")
    return os.path.join(out, "report_extended.html"), os.path.join(out, "summary.csv")


def test_report_with_an_explicit_stage_subset_not_starting_at_zero(tmp_path):
    run = _run(tmp_path, [Stage(proposal=RandomWalk(scale=0.5), max_evaluations=40),
                          Stage(proposal=RandomWalk(scale=0.5), max_evaluations=40)])
    samples = run.write_report(stages_to_disp=[1], selection=False)   # all sections, incl. the plots
    html, csv = _report_files(run)
    assert os.path.isfile(html)
    import pandas as pd
    assert list(pd.read_csv(csv, index_col=0).index) == ["alg0001_MH"]
    assert len(samples.summary) == 2, "the in-memory summary must keep every stage"


def test_report_with_an_excluded_first_stage_under_the_default_selection(tmp_path):
    run = _run(tmp_path, [Stage(proposal=RandomWalk(scale=0.5), max_evaluations=40, is_excluded=True),
                          Stage(proposal=RandomWalk(scale=0.5), max_evaluations=40)])
    samples = run.write_report()   # default selection: the is_excluded stage is not in the posterior
    html, csv = _report_files(run)
    assert os.path.isfile(html)
    import pandas as pd
    summary = pd.read_csv(csv, index_col=0)
    assert list(summary.index) == ["alg0000_MH", "alg0001_MH"]
    assert list(summary["chains_in_posterior"]) == ["0/1", "1/1"]
    assert samples.chain_indices == [[0], [0]]
    assert samples.posterior_view().chain_indices == [[], [0]]
