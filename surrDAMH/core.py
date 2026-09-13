#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
from typing import List, Literal
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

import surrDAMH.process_COLLECTOR
import surrDAMH.process_SAMPLER
import surrDAMH.process_SOLVER
from surrDAMH.configuration import Configuration
from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.solver_specification import SolverSpec
from surrDAMH.solvers import Solver, get_solver_from_spec
from surrDAMH.stages import Stage
from surrDAMH.surrogates.parent import Evaluator, Updater
from surrDAMH.modules.test_data import TestData


def identity(sample):
    return sample


class SamplingFramework:
    """
    Created on each MPI rank (except spawned solvers).
    Forward model solver (mapping from parameters fo observations) must be specified,
    if solvers pool is used, solver must be specified using solver_spec,
    if solvers pool is not used, solver can be specified using solver_spec or solver_instance.

    Args:
        conf (Configuration),
        prior (Distribution),
        likelihood (Distribution),
        solver_spec (SolverSpec),
        solver_instance (Solver): only if solvers pool is not used,
        list_of_stages (List[Stage]),
        surrogate_updater (Updater | None)
    """

    def __init__(self, conf: Configuration, prior: Distribution, likelihood: Distribution,
                 list_of_stages: List[Stage], solver_spec: SolverSpec | None = None, solver_instance: Solver | None = None,
                 surrogate_updater: Updater | None = None, surrogate_evaluator: Evaluator | None = None,
                 initial_snapshots: List[npt.NDArray] | None = None,
                 surrogate_test_data: TestData | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray] | None = None):
        self.conf = conf
        self.prior = prior
        self.likelihood = likelihood
        self.solver_spec = solver_spec
        self.list_of_stages = list_of_stages
        self.surrogate_updater = surrogate_updater
        self.surrogate_evaluator = surrogate_evaluator
        self.solver_instance = solver_instance
        self.initial_snapshots = initial_snapshots
        self.surrogate_test_data = surrogate_test_data

        self.comm_world = MPI.COMM_WORLD
        self.rank_world = self.comm_world.Get_rank()

    def _configure_surrogate_gradients(self) -> None:
        should_warn = self.rank_world == 0

        if self.surrogate_updater is not None:
            self.surrogate_updater.set_use_gradients(self.conf.use_surrogate_gradients)
        if self.surrogate_evaluator is not None and hasattr(self.surrogate_evaluator, "set_use_gradients"):
            self.surrogate_evaluator.set_use_gradients(self.conf.use_surrogate_gradients)

        if not self.conf.use_surrogate_gradients:
            return

        if self.conf.transform_before_surrogate:
            if should_warn:
                warnings.warn(
                    "Surrogate gradients require transform_before_surrogate=False. "
                    "Disabling use_surrogate_gradients. The combination with "
                    "transform_before_surrogate=True is deprecated for gradient-based surrogate use.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.conf.use_surrogate_gradients = False
        elif self.surrogate_updater is not None and not self.surrogate_updater.supports_gradients():
            if should_warn:
                warnings.warn(
                    f"Surrogate updater {type(self.surrogate_updater).__name__} does not implement gradients. "
                    "Disabling use_surrogate_gradients.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.conf.use_surrogate_gradients = False
        elif self.surrogate_evaluator is not None and not self.surrogate_evaluator.supports_gradients():
            if should_warn:
                warnings.warn(
                    f"Surrogate evaluator {type(self.surrogate_evaluator).__name__} does not implement gradients. "
                    "Disabling use_surrogate_gradients.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.conf.use_surrogate_gradients = False

        if self.surrogate_updater is not None:
            self.surrogate_updater.set_use_gradients(self.conf.use_surrogate_gradients)
        if self.surrogate_evaluator is not None and hasattr(self.surrogate_evaluator, "set_use_gradients"):
            self.surrogate_evaluator.set_use_gradients(self.conf.use_surrogate_gradients)

    def run(self):
        self._configure_surrogate_gradients()

        # check if prior has the "transform" method:
        if not hasattr(self.prior, "transform"):
            self.prior.transform = identity

        if self.rank_world == self.conf.rank_solvers_pool:
            assert self.solver_spec is not None, "solver_spec must be given"
            optional_output = surrDAMH.process_SOLVER.run_SOLVER(self.conf, self.solver_spec)
        elif self.rank_world == self.conf.rank_collector:
            assert self.surrogate_updater is not None
            if isinstance(self.surrogate_test_data, TestData):
                self.surrogate_test_data = self.surrogate_test_data.as_surrogate_test_data(prior=self.prior, likelihood=self.likelihood)
            initial_snapshots = self.initial_snapshots
            if initial_snapshots is None and self.surrogate_updater is not None:
                initial_snapshots = self.surrogate_updater.get_initial_snapshots()

            optional_output = surrDAMH.process_COLLECTOR.run_COLLECTOR(
                self.conf,
                surrogate_updater=self.surrogate_updater,
                initial_snapshots=initial_snapshots,
                surrogate_test_data=self.surrogate_test_data,
            )
        else:
            if self.conf.use_solvers_pool is False:
                if self.solver_instance is None:
                    assert self.solver_spec is not None, "either solver_spec or solver_instance must be given"
                    solver_output_dir = ensure_dir(os.path.join(self.conf.output_dir, "solver_output", "rank{}".format(self.rank_world)))
                    self.solver_instance = get_solver_from_spec(self.solver_spec, solver_id=self.rank_world, solver_output_dir=solver_output_dir)
            else:
                self.solver_instance = None
            optional_output = surrDAMH.process_SAMPLER.run_SAMPLER(
                self.conf, self.prior, self.likelihood, self.list_of_stages,
                solver_instance=self.solver_instance, surrogate_evaluator=self.surrogate_evaluator)

        self.comm_world.Barrier()

        return optional_output


    def temptemptemp(self, no_observations: int = 0, observations_to_disp: np.ndarray | None = None,
                            grid: np.ndarray | None = None, grid_interp: np.ndarray | None = None, 
                            bins: List[int] | None = None, chains_to_disp: Iterable | None = None,
                            stages_to_disp: List[int] | None = None, observations: np.ndarray | None = None, 
                            cmap="viridis_r", output_file: str = "report_extended.html",
                            bins1d: int = 20, bins2d: int = 20, par_names: List[str] | None = None,
                            parameters_to_disp: Iterable | None = None,
                            prior=None, no_best_fits: int = 0,
                            obs_grid: np.ndarray | None = None,
                            no_sensors: int | None = None,
                            include_expensive_sections: bool = False,
                            field_statistics: List[dict[str, Any]] | None = None,
                            configuration: Any | None = None,
                            ranking_mode: Literal["l2", "posterior", "likelihood"] = "l2"):
        """
        Creates an extended report in HTML format containing all available post-processing tools,
        including visualizations and statistics for combined stages and individual stages separately.

        Args:
            no_observations (int): number of observations (default: 0, set to >0 if observations are available)
            observations_to_disp (ndarray of int of length N): indices forming the time series (otherwise all are used)
            grid (ndarray of float of length N): time values for the time series (otherwise range(N) is used)
            grid_interp (ndarray of float): time grid for horizontal axis (otherwise grid_interp = grid)
            bins (list of int of length 2): [bins_x, bins_y] for observation histograms (optional)
            chains_to_disp (list of int of length N): chains that should be included (otherwise all chains are included)
            stages_to_disp (list of int): stages that should be included (otherwise all stages are included)
            observations (ndarray of shape (no_observations,)): vector of observations (optional)
            cmap (str): colormap for observation histograms (default: "viridis_r")
            output_file (str): name of the output HTML file (default: "report_extended.html")
            bins1d (int): Number of bins in 1d histograms (default: 20)
            bins2d (int): Number of bins in 2d histograms (default: 20)
            par_names (list of str of length N): Parameter names for plots (optional)
            prior (PriorIndependentComponents or None): Prior distribution. If provided, marginal prior PDFs
                are overlaid on 1D histograms in the histogram grids.
            no_best_fits (int): Number of exact-evaluation best fits to include from raw snapshots.
            ranking_mode (str): Ranking metric for best-fit selection. One of 'l2', 'posterior', or 'likelihood'.
            obs_grid (ndarray of float): Grid for plotting best-fit observation trajectories.
            no_sensors (int): Number of sensors for reshaping observations in best-fit plots.
            include_expensive_sections (bool): If true, include slow per-stage diagnostics,
                autocorrelation plots, ESS/R-hat summaries, and observation histograms.
            field_statistics (list of dict): Derived posterior field summaries. Each item should contain
                keys "name", "mean", "std", and optionally "coordinates".
            configuration (Any | None): Run configuration to render near the top of the report.)"""
        return

    def write_report(self, stages_to_disp: list[int] | None = None, 
                     observations: np.ndarray | None = None,
                     par_names: list[str] | None = None, 
                     bins1d: int = 20, bins2d: int = 20, 
                     no_best_fits: int = 10,
                     ranking_mode: Literal["l2", "posterior", "likelihood"] = "l2",
                     parameters_to_disp: list[int] | None = None,
                     observations_to_disp: np.ndarray | None = None,
                     include_expensive_sections = True,
                     grid = None, grid_interp = None, obs_grid = None,
                     no_sensors = None, cmap = "viridis_r", chains_to_disp = None) -> surrDAMH.post_processing.Samples | None:
        """
        Writes a html report to the output directory.
        
        Args:
            stages_to_disp: list of stage indices to include in the report, if None, all stages are included
            observations: reference observations, if None, no reference observations are used
            par_names: list of parameter names, if None, default names are used
            bins1d: number of bins for 1D histograms
            bins2d: number of bins for 2D histograms
            no_best_fits: number of best fit samples to display
            ranking_mode: how to rank the best fit samples, one of "l2", "posterior", "likelihood"
            parameters_to_disp: list of parameter indices to include in the report, if None, first 10 parameters are included
            observations_to_disp: list of observation indices to include in the report, if None, all observations are included
            include_expensive_sections: whether to include sections that are expensive to compute
            grid: grid for 2D histograms, if None, a default grid is used
        """

        if self.rank_world != 0:
            self.comm_world.Barrier()
            return None

        if par_names is None:
            par_names = getattr(self.solver_instance, "par_names", None)
        if stages_to_disp is None:
            stages_to_disp = [stage_idx for stage_idx in range(len(self.list_of_stages))]
        if parameters_to_disp is None:
            parameters_to_disp = list(range(min(self.conf.no_parameters, 10)))
        if not stages_to_disp:
            raise ValueError("No report stages are available.")

        post_processing_dir_path = os.path.join(self.conf.output_dir, "post_processing_output")
        ensure_dir(post_processing_dir_path)

        samples = surrDAMH.post_processing.Samples(self.conf.no_parameters, self.conf.output_dir)
        samples.calculate_CpUS([[i] for i in stages_to_disp])
        setattr(samples, "summary", samples.summary.iloc[stages_to_disp, :].copy())
        samples.get_summary().to_csv(os.path.join(post_processing_dir_path, "summary.csv"))

        output_html_file_path = os.path.join(post_processing_dir_path, "report_extended.html")

        if self.solver_instance and hasattr(self.solver_instance, 'field_builder') and hasattr(self.solver_instance, 'coords') and hasattr(self.solver_instance, 'measurement_points'):
            field_mean, field_std = samples.compute_posterior_field_statistics(self.solver_instance.field_builder)
            observation_field_mean, observation_field_std = samples.compute_posterior_field_statistics(self.solver_instance.set_parameters_and_get_observations)
            field_statistics = [
                {
                    "mean": field_mean,
                    "std": field_std,
                    "coordinates": self.solver_instance.coords,
                    "name": "Posterior diffusion coefficient field",
                },
                {
                    "mean": observation_field_mean,
                    "std": observation_field_std,
                    "coordinates": self.solver_instance.measurement_points[:, :2],
                    "name": "Posterior solution at measurement points",
                },
            ]
        else:
            field_statistics = None

        samples.html_report_extended(
            no_observations=self.conf.no_observations,
            stages_to_disp=stages_to_disp,
            observations=observations,
            output_file=output_html_file_path,
            bins1d=bins1d,
            bins2d=bins2d,
            parameters_to_disp=parameters_to_disp,
            prior=self.prior,
            no_best_fits=no_best_fits,
            include_expensive_sections=include_expensive_sections,
            field_statistics=field_statistics,
            configuration=self.conf,
            ranking_mode=ranking_mode,
            par_names=par_names,
            observations_to_disp=observations_to_disp,
            grid=grid,
            grid_interp=grid_interp,
            obs_grid=obs_grid,
            no_sensors=no_sensors,
            cmap=cmap,
            chains_to_disp=chains_to_disp

        )

        if self.solver_instance is not None:
            best_fit_parameters = np.asarray(getattr(samples, "best_fit_parameters", []), dtype=float)
            if best_fit_parameters.size:
                if best_fit_parameters.ndim == 1:
                    best_fit_parameters = best_fit_parameters.reshape(1, -1)
                self.solver_instance.set_parameters(best_fit_parameters[0])
                self.solver_instance.get_observations()
                try:
                    visualizations = self.solver_instance.visualize_solution(show=False)
                except Exception as exc:
                    print(f"Solver visualization skipped for best fit: {exc}", flush=True)
                    visualizations = []
                for fig_idx, (fig, _) in enumerate(visualizations, start=1):
                    fig_path = os.path.join(post_processing_dir_path, f"best_fit_solver_visualization_{fig_idx}.png")
                    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)

        self.comm_world.Barrier()
        return samples
