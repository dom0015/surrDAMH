#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report writers for one sampling run: the self-contained extended HTML report
(``html_report_extended``) plus the two legacy single-file reports (``pdf_report``,
``html_report``).

Split out of the former single-file ``surrDAMH/post_processing.py`` (WS9b).
``SamplesReports(SamplesPlots)`` is the last link of the mixin chain combined into
:class:`surrDAMH.post_processing.Samples` in ``loading.py``
(``SamplesStatistics`` -> ``SamplesPlots`` -> ``SamplesReports`` -> ``Samples``); it
calls the plotting and statistics methods, and the base attributes/resolvers, through
ordinary inheritance from ``SamplesPlots``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, List, Literal

import matplotlib.pyplot as plt
import numpy as np

from surrDAMH.post_processing.plots import SamplesPlots


class SamplesReports(SamplesPlots):
    """Reporting layer of :class:`surrDAMH.post_processing.Samples` (see module docstring)."""

    def pdf_report(self, no_observations: int, chosen_observations: np.ndarray | None = None, grid: np.ndarray | None = None,
                   grid_interp: np.ndarray | None = None, bins: List[int] | None = None, chains_to_disp: Iterable | None = None,
                   stages_to_disp: List[int] | None = None, observations: np.ndarray | None = None, cmap="viridis_r"):
        """
        Creates a report in pdf format containing all of the above post-processing tools, 
        e.g. summary table, histograms of observations, chain traces.

        Args:
            no_observations (int): number of observations
            chosen_observations (ndarray of int of length N): indices forming the time series (otherwise all are used)
            grid (ndarray of float of length N): time values for the time series (otherwise range(N) is used)
            grid_interp (ndarray of float): time grid for horizontal axis (otherwise grid_inter = grid)
            bins (list of int of length 2): [bins_x, bins_y] (optional)
            chains_to_disp (list of int of length N): chains that should be included (otherwise all chains are included)
            stages_to_disp (list of int): stages that should be included (otherwise all stages are included)
            observations (ndarray of shape (no_observations,)): vector of observations (optional)
        """
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages('report.pdf') as pdf:
            # summary table:
            fig, ax = plt.subplots(figsize=(10, 2))
            try:
                ax.axis('tight')
                ax.axis('off')
                table = ax.table(cellText=self.summary.values, colLabels=self.summary.columns, rowLabels=self.summary.index, loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.2, 1.2)
                pdf.savefig(fig)
            finally:
                plt.close(fig)

            # histograms of observations:
            fig = self.hist_observations(no_observations=no_observations, chosen_observations=chosen_observations,
                                         grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                         stages_to_disp=stages_to_disp, observations=observations, cmap=cmap)
            try:
                pdf.savefig(fig)
            finally:
                plt.close(fig)

            # chain traces:
            fig, _ = self.plot_chains(average=False, parameters_to_disp=None, stages_to_disp=stages_to_disp,
                                      scale=None, par_names=None, burn_in=None,
                                      chains_to_disp=chains_to_disp)
            try:
                pdf.savefig(fig)
            finally:
                plt.close(fig)

    def html_report(self, no_observations: int, chosen_observations: np.ndarray | None = None, grid: np.ndarray | None = None,
                    grid_interp: np.ndarray | None = None, bins: List[int] | None = None, chains_to_disp: Iterable | None = None,
                    stages_to_disp: List[int] | None = None, observations: np.ndarray | None = None, cmap="viridis_r"):
        """
        Creates a report in html format containing all of the above post-processing tools, 
        e.g. summary table, histograms of observations, chain traces.

        Args:
            no_observations (int): number of observations
            chosen_observations (ndarray of int of length N): indices forming the time series (otherwise all are used)
            grid (ndarray of float of length N): time values for the time series (otherwise range(N) is used)
            grid_interp (ndarray of float): time grid for horizontal axis (otherwise grid_inter = grid)
            bins (list of int of length 2): [bins_x, bins_y] (optional)
            chains_to_disp (list of int of length N): chains that should be included (otherwise all chains are included)
            stages_to_disp (list of int): stages that should be included (otherwise all stages are included)
            observations (ndarray of shape (no_observations,)): vector of observations (optional)
        """
        # summary table:
        html = self.summary.to_html()
        with open("report.html", "w") as f:
            f.write(html)
        # histograms of observations:
        fig = self.hist_observations(no_observations=no_observations, chosen_observations=chosen_observations, grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                     stages_to_disp=stages_to_disp, observations=observations, cmap=cmap)
        try:
            fig.savefig("hist_observations.png")
        finally:
            plt.close(fig)
        # chain traces:
        fig, _ = self.plot_chains(average=False, parameters_to_disp=None, stages_to_disp=stages_to_disp,
                                  scale=None, par_names=None, burn_in=None, chains_to_disp=chains_to_disp)
        try:
            fig.savefig("chain_traces.png")
        finally:
            plt.close(fig)
        # add images to html:
        with open("report.html", "a") as f:
            f.write("<h2>Histograms of observations</h2>")
            f.write('<img src="hist_observations.png" alt="hist_observations">')
            f.write("<h2>Chain traces</h2>")
            f.write('<img src="chain_traces.png" alt="chain_traces">')
        # surrogate quality (axes is None when no surrogate_quality.csv was written):
        fig, axes = self.plot_surrogate_quality()
        try:
            if axes is not None:
                fig.savefig("surrogate_quality.png")
                with open("report.html", "a") as f:
                    f.write("<h2>Surrogate Model Quality</h2>")
                    f.write('<img src="surrogate_quality.png" alt="surrogate_quality">')
        finally:
            plt.close(fig)

    def html_report_extended(self, no_observations: int = 0, observations_to_disp: np.ndarray | None = None,
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
            configuration (Any | None): Run configuration to render near the top of the report. If
                None, the effective configuration recorded in ``run_manifest.json`` is used.

        Notes:
            ``chains_to_disp`` restricts every sample-derived section (moments, histograms,
            traces, autocorrelation, ESS, R-hat, correlation, best fits, observation
            histograms). The per-stage counters in section 1 and the acceptance-rate plot in
            4.1 come from ``notes``/``subchain_stats`` and are always whole-stage totals; the
            report says so where it shows them (WS9b, finding P5).
        """
        import base64
        from dataclasses import fields, is_dataclass
        from html import escape
        from io import BytesIO
        # Initialize stages to display
        stages_to_disp = self._resolve_stages(stages_to_disp)
        observation_data_available = self._raw_data_available(stages_to_disp=stages_to_disp)
        if chains_to_disp is None:
            chains_note = ''
        else:
            chains_note = (f' Restricted to chain(s) {list(chains_to_disp)}; the per-stage counters '
                           f'in section 1 and 4.1 remain whole-stage totals.')

        # Helper function to convert matplotlib figure to base64 image
        def fig_to_base64(fig):
            try:
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                return base64.b64encode(buf.read()).decode('utf-8')
            finally:
                plt.close(fig)  # P8: also on the failure path

        def get_configuration_items(config: Any | None) -> list[tuple[str, Any]]:
            if config is None:
                return []
            if is_dataclass(config):
                return [(field.name, getattr(config, field.name)) for field in fields(config)]
            if isinstance(config, dict):
                return list(config.items())
            if hasattr(config, "__dict__"):
                return list(vars(config).items())
            return [("configuration", config)]

        def format_configuration_value(value: Any) -> str:
            if isinstance(value, np.ndarray):
                return np.array2string(value, threshold=20, edgeitems=3)
            return repr(value)

        def key_value_table(rows, key_header: str, value_header: str, formatter=format_configuration_value):
            parts = ['        <table class="summary-table">',
                     f'            <tr><th>{escape(key_header)}</th><th>{escape(value_header)}</th></tr>']
            for key, value in rows:
                parts.append(f'            <tr><td>{escape(str(key))}</td>'
                             f'<td>{escape(formatter(value))}</td></tr>')
            parts.append('        </table>')
            return parts

        manifest = dict(getattr(self.run_data, "manifest", None) or {})
        configuration_items = get_configuration_items(configuration)
        configuration_source = "the configuration object passed to this report"
        if not configuration_items:
            # WS9 bullet 5: without an explicit object, report the EFFECTIVE configuration
            # the run recorded for itself.
            configuration_items = sorted((manifest.get("configuration") or {}).items())
            configuration_source = "sampling_output/run_manifest.json"
        unverified_options = list(manifest.get("unverified_options") or [])
        run_provenance = [(label, manifest.get(key)) for label, key in [
            ("runner", "runner"),
            ("surrDAMH version", "surrdamh_version"),
            ("output format_version", "format_version"),
            ("created at", "created_at"),
            ("finished at", "finished_at"),
        ] if manifest.get(key) is not None]

        # Start building HTML
        html_parts = []
        html_parts.append('<!DOCTYPE html>')
        html_parts.append('<html lang="en">')
        html_parts.append('<head>')
        html_parts.append('    <meta charset="UTF-8">')
        html_parts.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append('    <title>Extended Sampling Report</title>')
        html_parts.append('    <style>')
        html_parts.append('        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }')
        html_parts.append('        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }')
        html_parts.append('        h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 40px; }')
        html_parts.append('        h3 { color: #7f8c8d; margin-top: 30px; }')
        html_parts.append('        .section { background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }')
        html_parts.append('        .description { color: #555; margin: 10px 0; line-height: 1.6; font-style: italic; }')
        html_parts.append('        table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
        html_parts.append('        th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }')
        html_parts.append('        th { background-color: #3498db; color: white; font-weight: bold; }')
        html_parts.append('        tr:nth-child(even) { background-color: #f9f9f9; }')
        html_parts.append('        img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }')
        html_parts.append('        .stats-table { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 15px 0; }')
        html_parts.append('        .toc { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }')
        html_parts.append('        .toc ul { list-style-type: none; padding-left: 20px; }')
        html_parts.append('        .toc a { color: #2980b9; text-decoration: none; }')
        html_parts.append('        .toc a:hover { text-decoration: underline; }')
        html_parts.append('        pre { background-color: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto; }')
        html_parts.append('    </style>')
        html_parts.append('</head>')
        html_parts.append('<body>')
        # Title
        html_parts.append('    <h1>Extended Sampling Report</h1>')
        html_parts.append('    <p class="description">This report contains comprehensive post-processing results including summary statistics, ')
        html_parts.append('    visualizations, and detailed analysis for all stages of the sampling process.</p>')
        html_parts.append('    <div class="section" id="configuration">')
        html_parts.append('        <h2>Run Configuration</h2>')
        html_parts.append(f'        <p class="description">Effective configuration of this sampling run, from '
                          f'{escape(configuration_source)}.</p>')
        if configuration_items:
            html_parts.extend(key_value_table(configuration_items, "option", "value"))
        else:
            html_parts.append('        <p class="description" style="color: orange;">No configuration was '
                              'recorded for this run and none was passed to the report.</p>')
        if run_provenance:
            html_parts.append('        <h3>Run provenance</h3>')
            html_parts.extend(key_value_table(run_provenance, "field", "value", formatter=str))
        html_parts.append('        <h3>Unverified options</h3>')
        if unverified_options:
            html_parts.append('        <p class="description" style="color: orange;">These options were active '
                              'in this run and are not covered by the library\'s verification; read the posterior '
                              'with that in mind.</p>')
            html_parts.extend(key_value_table(list(enumerate(unverified_options, start=1)),
                                              "#", "option", formatter=str))
        else:
            html_parts.append('        <p class="description">None &mdash; this run used no option flagged as '
                              'unverified in its manifest.</p>')
        html_parts.append('    </div>')

        # Table of Contents
        html_parts.append('    <div class="toc">')
        html_parts.append('        <h2>Table of Contents</h2>')
        html_parts.append('        <ul>')
        html_parts.append('            <li><a href="#configuration">Run Configuration</a></li>')
        html_parts.append('            <li><a href="#summary">1. Summary Statistics</a></li>')
        html_parts.append('            <li><a href="#overall">2. Overall Analysis (Combined Stages)</a></li>')
        html_parts.append('            <li><a href="#individual">3. Individual Stage Analysis</a></li>')
        html_parts.append('            <li><a href="#diagnostics">4. Convergence Diagnostics & Autocorrelation</a></li>')
        html_parts.append('            <li><a href="#surrogate_quality">5. Surrogate Model Quality</a></li>')
        if no_observations > 0 and observation_data_available:
            html_parts.append('            <li><a href="#observations">6. Observation Histograms</a></li>')
        if no_best_fits > 0:
            html_parts.append('            <li><a href="#best_fits">Best-fit analysis</a></li>')
        if field_statistics:
            html_parts.append('            <li><a href="#field_statistics">Posterior field statistics</a></li>')
        html_parts.append('        </ul>')
        html_parts.append('    </div>')
        
        # 1. SUMMARY STATISTICS
        html_parts.append('    <div class="section" id="summary">')
        html_parts.append('        <h2>1. Summary Statistics</h2>')
        html_parts.append('        <p class="description">This table summarizes the acceptance and rejection rates for all sampling stages. ')
        html_parts.append('        "Accepted" samples were accepted by the Metropolis-Hastings criterion, "rejected" samples were rejected, ')
        html_parts.append('        and "pre-rejected" samples (if any) were rejected by a surrogate model before evaluation. ')
        html_parts.append('        For DAMH stages, the table also includes the mean within-subchain surrogate acceptance rate, the fraction of subchains that produced a changed proposal, and the exact outer acceptance conditional on a changed proposal. ')
        html_parts.append('        These counters are whole-stage totals over every chain, also when the rest of the report is restricted to a subset of chains.</p>')
        html_parts.append(self.summary.to_html(classes='summary-table'))
        html_parts.append('    </div>')

        # 2. OVERALL ANALYSIS (COMBINED STAGES)
        html_parts.append('    <div class="section" id="overall">')
        html_parts.append('        <h2>2. Overall Analysis (Combined Stages)</h2>')
        html_parts.append(f'        <p class="description">This section presents aggregated results from all selected sampling stages combined.{chains_note}</p>')
        # 2.1 Mean and Covariance
        html_parts.append('        <h3>2.1 Posterior Mean and Covariance Matrix</h3>')
        html_parts.append('        <p class="description">The posterior mean represents the expected value of each parameter, ')
        html_parts.append('        while the covariance matrix shows the variance and correlation structure among parameters.</p>')
        posterior_mean, cov = self.get_mean_and_cov(stages_to_disp=stages_to_disp, chains_to_disp=chains_to_disp)

        html_parts.append('        <div class="stats-table">')
        html_parts.append('            <h4>Posterior Mean:</h4>')
        html_parts.append('            <pre>')
        for i, val in enumerate(posterior_mean):
            param_name = par_names[i] if par_names and i < len(par_names) else f"Parameter {i}"
            html_parts.append(f'{param_name}: {val:.6f}')
        html_parts.append('            </pre>')
        html_parts.append('            <h4>Covariance Matrix:</h4>')
        html_parts.append('            <pre>')
        html_parts.append(np.array2string(cov, precision=6, suppress_small=True))
        html_parts.append('            </pre>')
        html_parts.append('        </div>')
        # 2.2 Histogram Grid / 2.3 Marginals
        if include_expensive_sections:
            try:
                html_parts.append('        <h3>2.2 Parameter Distribution Histograms</h3>')
                html_parts.append('        <p class="description">This grid shows 1D histograms (diagonal) and 2D joint histograms (off-diagonal) ')
                html_parts.append('        for all parameter combinations. 1D histograms show marginal distributions, while 2D histograms ')
                html_parts.append('        reveal correlations between parameter pairs.</p>')
                fig, _ = self.plot_hist_grid(
                    bins1d=bins1d,
                    bins2d=bins2d,
                    parameters_to_disp=parameters_to_disp,
                    stages_to_disp=stages_to_disp,
                    par_names=par_names,
                    prior=prior,
                    chains_to_disp=chains_to_disp,
                )
                img_base64 = fig_to_base64(fig)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Overall Histogram Grid">')
            except Exception as e:
                html_parts.append(f'        <p class="description" style="color: orange;">Histogram grid unavailable: {escape(str(e))}</p>')
            try:
                html_parts.append('        <h3>2.3 One-dimensional Marginal Histograms</h3>')
                html_parts.append('        <p class="description">For high-dimensional problems, all 1D marginals are more informative than a full pairwise grid.</p>')
                fig, _ = self.plot_hist_marginals(
                    bins1d=bins1d,
                    parameters_to_disp=range(self.no_parameters),
                    stages_to_disp=stages_to_disp,
                    par_names=par_names,
                    prior=prior,
                    ncols=4,
                    chains_to_disp=chains_to_disp,
                )
                img_base64 = fig_to_base64(fig)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Overall 1D Marginals">')
            except Exception as e:
                html_parts.append(f'        <p class="description" style="color: orange;">1D marginals unavailable: {str(e)}</p>')
        else:
            html_parts.append('        <h3>2.2 Parameter Distribution Histograms</h3>')
            html_parts.append('        <p class="description">Skipped in the lightweight report because full histograms require concatenating very large sample arrays.</p>')
            html_parts.append('        <h3>2.3 One-dimensional Marginal Histograms</h3>')
            html_parts.append('        <p class="description">Skipped in the lightweight report because all marginals for this run are too memory-intensive.</p>')
        # (a commented-out "2.3 Chain Traces / 2.4 Cumulative Averages" block used to sit here
        #  as a bare string literal; both plots are produced per stage in section 3 and the
        #  dead copy -- which also held two bare `except:` clauses -- was removed in WS9b.)
        # 2.5 Best fits
        html_parts.append('        <div id="best_fits">')
        html_parts.append('        <h3>2.5 Best-fit Analysis</h3>')
        if ranking_mode == "posterior":
            mode_description = "best exact-model evaluations ranked by largest log-posterior value"
        elif ranking_mode == "likelihood":
            mode_description = "best exact-model evaluations ranked by largest log-likelihood value"
        else:
            mode_description = "best exact-model evaluations ranked by smallest L2 misfit to the supplied observations"
        html_parts.append(f'        <p class="description">{mode_description}.</p>')
        if no_best_fits > 0 and no_observations > 0 and observations is not None and observation_data_available:
            try:
                df_best, best_par, best_obs, best_scores = self.find_best_fits(
                    no_observations=no_observations,
                    observations=observations,
                    n_best=no_best_fits,
                    chains_to_disp=chains_to_disp,
                    stages_to_disp=stages_to_disp,
                    par_names=par_names,
                    ranking_mode=ranking_mode,
                )
                self.best_fit_parameters = best_par
                if len(df_best) > 0:
                    html_parts.append(df_best.to_html(index=False, classes='summary-table', float_format=lambda x: f"{x:.6g}"))
                    fig_best, _ = self.plot_best_fits(
                        best_obs=best_obs,
                        best_scores=best_scores,
                        observations=observations,
                        obs_grid=obs_grid if obs_grid is not None else grid,
                        n_sensors=no_sensors,
                        obs_label="Observations",
                        score_label=ranking_mode.upper() if ranking_mode != "l2" else "L2",
                    )
                    img_base64 = fig_to_base64(fig_best)
                    html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Best-fit observations">')
                else:
                    html_parts.append('        <p class="description" style="color: orange;">Best-fit analysis unavailable: no exact snapshots were found.</p>')
            except Exception as e:
                html_parts.append(f'        <p class="description" style="color: orange;">Best-fit analysis unavailable: {str(e)}</p>')
        elif no_best_fits > 0:
            html_parts.append('        <p class="description" style="color: orange;">Best-fit analysis requires observations and raw snapshot files.</p>')
        else:
            html_parts.append('        <p class="description">Best-fit analysis not requested for this report.</p>')
        html_parts.append('        </div>')

        # 2.6 Posterior field statistics
        html_parts.append('        <div id="field_statistics">')
        html_parts.append('        <h3>2.6 Posterior Field Statistics</h3>')
        html_parts.append('        <p class="description">Posterior mean and uncertainty for derived spatial or field-valued quantities.</p>')
        if field_statistics:
            for field_info in field_statistics:
                try:
                    field_name = field_info.get("name", "Field")
                    html_parts.append(f'        <h4>{field_name}</h4>')
                    fig_field, _ = self.plot_posterior_field_statistics(
                        mean_field=field_info["mean"],
                        std_field=field_info["std"],
                        coordinates=field_info.get("coordinates"),
                        field_name=field_name,
                    )
                    img_base64 = fig_to_base64(fig_field)
                    html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="{field_name}">')
                except Exception as e:
                    html_parts.append(f'        <p class="description" style="color: orange;">Field statistics unavailable: {str(e)}</p>')
        else:
            html_parts.append('        <p class="description">No posterior field statistics were supplied for this report.</p>')
        html_parts.append('        </div>')
        
        html_parts.append('    </div>')
        
        # 3. INDIVIDUAL STAGE ANALYSIS / 4. DIAGNOSTICS
        if include_expensive_sections:
            html_parts.append('    <div class="section" id="individual">')
            html_parts.append('        <h2>3. Individual Stage Analysis</h2>')
            html_parts.append('        <p class="description">This section presents detailed analysis for each sampling stage separately. ')
            html_parts.append('        Different stages may use different proposal distributions or algorithm parameters.</p>')

            for stage_idx in stages_to_disp:
                stage_name = self.stage_names[stage_idx]
                html_parts.append(f'        <h3>Stage: {stage_name}</h3>')
                html_parts.append(f'        <p class="description">Analysis results for sampling stage "{stage_name}".</p>')

                html_parts.append('        <h4>Stage Summary:</h4>')
                stage_summary = self.summary.iloc[stage_idx:stage_idx+1]
                html_parts.append(stage_summary.to_html(classes='stage-summary'))

                html_parts.append('        <h4>Posterior Mean and Covariance (This Stage):</h4>')
                stage_mean, stage_cov = self.get_mean_and_cov(stages_to_disp=[stage_idx],
                                                             chains_to_disp=chains_to_disp)
                html_parts.append('        <div class="stats-table">')
                html_parts.append('            <strong>Mean:</strong>')
                html_parts.append('            <pre>')
                for i, val in enumerate(stage_mean):
                    param_name = par_names[i] if par_names and i < len(par_names) else f"Parameter {i}"
                    html_parts.append(f'{param_name}: {val:.6f}')
                html_parts.append('            </pre>')
                html_parts.append('            <strong>Covariance:</strong>')
                html_parts.append('            <pre>')
                html_parts.append(np.array2string(stage_cov, precision=6, suppress_small=True))
                html_parts.append('            </pre>')
                html_parts.append('        </div>')

                html_parts.append('        <h4>Parameter Distribution Histograms:</h4>')
                html_parts.append('        <p class="description">1D and 2D histograms for this stage only.</p>')
                fig, _ = self.plot_hist_grid(
                    bins1d=bins1d,
                    bins2d=bins2d,
                    parameters_to_disp=parameters_to_disp,
                    stages_to_disp=[stage_idx],
                    par_names=par_names,
                    prior=prior,
                    chains_to_disp=chains_to_disp,
                )
                img_base64 = fig_to_base64(fig)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Stage {stage_name} Histogram Grid">')

                html_parts.append('        <h4>Chain Traces:</h4>')
                html_parts.append('        <p class="description">Evolution of parameter values during this stage.</p>')
                fig, _ = self.plot_chains(
                    average=False,
                    stages_to_disp=[stage_idx],
                    parameters_to_disp=parameters_to_disp,
                    par_names=par_names,
                    chains_to_disp=chains_to_disp,
                )
                img_base64 = fig_to_base64(fig)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Stage {stage_name} Chain Traces">')

                html_parts.append('        <h4>Cumulative Averages:</h4>')
                html_parts.append('        <p class="description">Running mean showing convergence behavior for this stage.</p>')
                fig, _ = self.plot_chains(
                    average=True,
                    stages_to_disp=[stage_idx],
                    parameters_to_disp=parameters_to_disp,
                    par_names=par_names,
                    chains_to_disp=chains_to_disp,
                )
                img_base64 = fig_to_base64(fig)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Stage {stage_name} Cumulative Averages">')
            html_parts.append('    </div>')

            html_parts.append('    <div class="section" id="diagnostics">')
            html_parts.append('        <h2>4. Convergence Diagnostics & Autocorrelation Analysis</h2>')
            html_parts.append('        <p class="description">Diagnostic measures to assess mixing quality, convergence, and sampling efficiency. ')
            html_parts.append('        Lower autocorrelation times and higher effective sample sizes indicate better sampling efficiency.</p>')

            html_parts.append('        <h3>4.1 Acceptance Rates</h3>')
            html_parts.append('        <p class="description">Acceptance rate visualized by stage. The optimal acceptance rate for Metropolis-Hastings is ~23.4%. ')
            html_parts.append('        Rates significantly lower (< 10%) or higher (> 50%) suggest issues with proposal distribution. ')
            html_parts.append('        Computed from the whole-stage counters, i.e. over every chain.</p>')
            fig_acc = self.plot_acceptance_rates(stages_to_disp=stages_to_disp)
            img_base64 = fig_to_base64(fig_acc)
            html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Acceptance Rates">')

            html_parts.append('        <h3>4.2 Autocorrelation Functions</h3>')
            html_parts.append('        <p class="description">Autocorrelation functions (ACF) show how correlated samples are at different lags. ')
            html_parts.append('        Rapid decay to near-zero indicates good mixing. The red shaded region (±0.05) represents ')
            html_parts.append('        the approximate 95% confidence interval under independence.</p>')
            try:
                fig_acf, _ = self.plot_autocorr(
                    stages_to_disp=stages_to_disp,
                    parameters_to_disp=range(self.no_parameters),
                    par_names=par_names,
                    max_lag=200,
                    chains_to_disp=chains_to_disp,
                )
                img_base64 = fig_to_base64(fig_acf)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Autocorrelation Functions">')
            except Exception as e:
                html_parts.append(f'        <p class="description" style="color: orange;">Autocorrelation plot unavailable: {str(e)}</p>')

            html_parts.append('        <h3>4.3 Effective Sample Size (ESS)</h3>')
            html_parts.append('        <p class="description">ESS represents the equivalent number of independent samples drawn from the posterior. ')
            html_parts.append('        ESS = Total Samples / Autocorrelation Time. Higher ESS relative to total samples indicates efficient sampling.</p>')
            ess_results = self.calculate_effective_sample_size(stages_to_disp=stages_to_disp,
                                                              chains_to_disp=chains_to_disp)
            if ess_results:
                html_parts.append('        <div class="stats-table">')
                html_parts.append('            <h4>Effective Sample Size Summary:</h4>')
                html_parts.append('            <pre>')
                html_parts.append(f'Total samples (all chains, all stages): {ess_results["total_samples"]:,}\n')
                html_parts.append(f'Overall ESS: {ess_results["ess_overall"]:.1f}\n')
                html_parts.append(f'ESS efficiency: {(ess_results["ess_overall"]/max(1,ess_results["total_samples"])*100):.2f}%\n\n')
                html_parts.append('ESS per parameter:\n')
                for param_idx, ess in sorted(ess_results['ess_per_param'].items()):
                    param_name = par_names[param_idx] if par_names and param_idx < len(par_names) else f"Parameter {param_idx}"
                    autocorr_time = ess_results['autocorr_times'][param_idx]
                    html_parts.append(f'  {param_name}: ESS = {ess:.1f}, Autocorr. Time = {autocorr_time:.2f}\n')
                html_parts.append('            </pre>')
                html_parts.append('        </div>')
            else:
                html_parts.append('        <p class="description" style="color: orange;">ESS calculation not available for this run.</p>')

            html_parts.append('        <h3>4.4 Cost per Uncorrelated Sample (CpUS)</h3>')
            html_parts.append('        <p class="description">CpUS combines autocorrelation with evaluation cost. ')
            html_parts.append('        Lower CpUS indicates a more efficient stage. Values are computed per stage using stage-wise autocorrelation.</p>')
            try:
                cpus_summary = self.calculate_CpUS(list_of_stages_groups=[[i] for i in stages_to_disp],
                                                   surrogate_cost_ratio=0.0, chains_to_disp=chains_to_disp)
                cpus_df = cpus_summary.iloc[stages_to_disp].copy()
                available_columns = [
                    column for column in [
                        "accepted",
                        "rejected",
                        "pre-rejected",
                        "sum",
                        "ratio_eval",
                        "subchain_acc_rate",
                        "subchain_move_rate",
                        "outer_acc_given_move",
                        "autocorr",
                        "CpUS",
                    ] if column in cpus_df.columns
                ]
                cpus_df = cpus_df[available_columns].copy()
                cpus_df = cpus_df.rename(columns={"autocorr": "autocorr_time"})
                html_parts.append(cpus_df.to_html(classes='summary-table', float_format=lambda x: f"{x:.4f}"))
            except Exception as e:
                html_parts.append(f'        <p class="description" style="color: orange;">CpUS calculation unavailable: {str(e)}</p>')

            html_parts.append('        <h3>4.5 Per-Stage Autocorrelation Analysis</h3>')
            html_parts.append('        <p class="description">Detailed autocorrelation analysis for each sampling stage separately. ')
            html_parts.append('        This allows assessment of mixing quality at different stages of the algorithm.</p>')

            for stage_idx in stages_to_disp:
                stage_name = self.stage_names[stage_idx]
                html_parts.append(f'        <h4>Stage: {stage_name}</h4>')

                html_parts.append('        <p class="description"><strong>Autocorrelation Function:</strong></p>')
                try:
                    fig_stage_acf, _ = self.plot_autocorr(
                        stages_to_disp=[stage_idx],
                        parameters_to_disp=range(self.no_parameters),
                        par_names=par_names,
                        max_lag=200,
                        chains_to_disp=chains_to_disp,
                    )
                    img_base64 = fig_to_base64(fig_stage_acf)
                    html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="ACF {stage_name}">')
                except Exception as e:
                    html_parts.append(f'        <p class="description" style="color: orange;">ACF plot unavailable for {stage_name}: {str(e)}</p>')

                html_parts.append('        <p class="description"><strong>Effective Sample Size:</strong></p>')
                try:
                    stage_ess = self.calculate_effective_sample_size(stages_to_disp=[stage_idx],
                                                                    chains_to_disp=chains_to_disp)
                    if stage_ess:
                        html_parts.append('        <div class="stats-table">')
                        html_parts.append('            <pre>')
                        html_parts.append(f'Stage samples: {stage_ess["total_samples"]:,}\n')
                        html_parts.append(f'Stage ESS: {stage_ess["ess_overall"]:.1f}\n')
                        html_parts.append(f'ESS efficiency: {(stage_ess["ess_overall"]/max(1,stage_ess["total_samples"])*100):.2f}%\n\n')
                        html_parts.append('ESS per parameter:\n')
                        for param_idx, ess in sorted(stage_ess['ess_per_param'].items()):
                            param_name = par_names[param_idx] if par_names and param_idx < len(par_names) else f"Parameter {param_idx}"
                            autocorr_time = stage_ess['autocorr_times'][param_idx]
                            html_parts.append(f'  {param_name}: ESS = {ess:.1f}, Autocorr. Time = {autocorr_time:.2f}\n')
                        html_parts.append('            </pre>')
                        html_parts.append('        </div>')
                except Exception as e:
                    html_parts.append(f'        <p class="description" style="color: orange;">ESS calculation unavailable for {stage_name}: {str(e)}</p>')

            html_parts.append('        <h3>4.6 Gelman-Rubin Convergence (R-hat)</h3>')
            html_parts.append('        <p class="description">R-hat compares within-chain and between-chain variance. ')
            html_parts.append('        Values close to 1.0 indicate convergence; values above 1.05 suggest insufficient mixing.</p>')
            rhat_results = self.calculate_gelman_rubin(stages_to_disp=stages_to_disp,
                                                      chains_to_disp=chains_to_disp)
            if rhat_results:
                html_parts.append('        <div class="stats-table">')
                html_parts.append('            <pre>')
                html_parts.append(f'Chains used: {rhat_results["m_chains"]}\n')
                html_parts.append(f'Samples per chain (truncated): {rhat_results["n_per_chain"]}\n')
                html_parts.append(f'Max R-hat: {rhat_results["rhat_max"]:.4f}\n')
                html_parts.append(f'Mean R-hat: {rhat_results["rhat_mean"]:.4f}\n\n')
                html_parts.append('R-hat per parameter:\n')
                for param_idx, rhat in sorted(rhat_results['rhat_per_param'].items()):
                    param_name = par_names[param_idx] if par_names and param_idx < len(par_names) else f"Parameter {param_idx}"
                    html_parts.append(f'  {param_name}: R-hat = {rhat:.4f}\n')
                html_parts.append('            </pre>')
                html_parts.append('        </div>')
            else:
                html_parts.append('        <p class="description" style="color: orange;">Gelman-Rubin analysis unavailable (requires at least two non-empty chains).</p>')

            html_parts.append('        <h3>4.7 Parameter Correlation Heatmap</h3>')
            html_parts.append('        <p class="description">Pairwise linear correlations among posterior parameters. ')
            html_parts.append('        Values near ±1 indicate strong dependency; values near 0 indicate weak linear relationship.</p>')
            fig_corr, _, corr_matrix = self.plot_parameter_correlation_heatmap(
                stages_to_disp=stages_to_disp, par_names=par_names, chains_to_disp=chains_to_disp)
            img_base64 = fig_to_base64(fig_corr)
            html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Parameter Correlation Heatmap">')

            html_parts.append('        <h3>4.8 Diagnostics Summary</h3>')
            html_parts.append('        <p class="description">Compact pass/warn/fail style summary for quick interpretation of chain quality.</p>')
            acceptance_rates = (self.summary.iloc[stages_to_disp]['accepted'] / self.summary.iloc[stages_to_disp]['sum']) * 100.0
            acc_min = float(np.min(acceptance_rates))
            acc_max = float(np.max(acceptance_rates))

            if ess_results:
                ess_eff = float((ess_results['ess_overall'] / max(1, ess_results['total_samples'])) * 100.0)
            else:
                ess_eff = None

            if rhat_results:
                rhat_max = float(rhat_results['rhat_max'])
            else:
                rhat_max = None

            if corr_matrix is not None and corr_matrix.shape[0] > 1:
                corr_tmp = corr_matrix.copy()
                np.fill_diagonal(corr_tmp, 0.0)
                max_abs_corr = float(np.max(np.abs(corr_tmp)))
            else:
                max_abs_corr = None

            cpus_value = None
            if "CpUS" in self.summary.columns:
                cpus_values = self.summary.iloc[stages_to_disp]["CpUS"].values
                cpus_values = cpus_values[cpus_values >= 0]
                if cpus_values.size > 0:
                    cpus_value = float(np.mean(cpus_values))

            def grade(label, value, good_cond, warn_cond, fmt):
                if value is None:
                    return f"{label}: unavailable"
                if good_cond(value):
                    status = "PASS"
                elif warn_cond(value):
                    status = "WARN"
                else:
                    status = "FAIL"
                return f"{label}: {fmt(value)} ({status})"

            html_parts.append('        <div class="stats-table">')
            html_parts.append('            <pre>')
            html_parts.append(grade("Acceptance rate range [%]", (acc_min, acc_max),
                                   lambda v: v[0] >= 10.0 and v[1] <= 50.0,
                                   lambda v: v[0] >= 5.0 and v[1] <= 70.0,
                                   lambda v: f"{v[0]:.2f} .. {v[1]:.2f}") + '\n')
            html_parts.append(grade("ESS efficiency [%]", ess_eff,
                                   lambda v: v >= 10.0,
                                   lambda v: v >= 3.0,
                                   lambda v: f"{v:.2f}") + '\n')
            html_parts.append(grade("Max R-hat", rhat_max,
                                   lambda v: v <= 1.01,
                                   lambda v: v <= 1.05,
                                   lambda v: f"{v:.4f}") + '\n')
            html_parts.append(grade("Max |correlation|", max_abs_corr,
                                   lambda v: v <= 0.8,
                                   lambda v: v <= 0.95,
                                   lambda v: f"{v:.3f}") + '\n')
            html_parts.append(grade("Mean CpUS", cpus_value,
                                   lambda v: v <= 50.0,
                                   lambda v: v <= 200.0,
                                   lambda v: f"{v:.4f}") + '\n')
            html_parts.append('            </pre>')
            html_parts.append('        </div>')
            html_parts.append('    </div>')
        else:
            html_parts.append('    <div class="section" id="individual">')
            html_parts.append('        <h2>3. Individual Stage Analysis</h2>')
            html_parts.append('        <p class="description">Omitted in the lightweight report because per-stage plots and summaries are expensive for high-dimensional problems.</p>')
            html_parts.append('    </div>')

            html_parts.append('    <div class="section" id="diagnostics">')
            html_parts.append('        <h2>4. Convergence Diagnostics & Autocorrelation Analysis</h2>')
            html_parts.append('        <p class="description">Omitted in the lightweight report.</p>')
            html_parts.append('    </div>')

        # 5. SURROGATE MODEL QUALITY
        html_parts.append('    <div class="section" id="surrogate_quality">')
        html_parts.append('        <h2>5. Surrogate Model Quality</h2>')
        html_parts.append('        <p class="description">Out-of-sample quality metrics of the surrogate model, measured on newly arrived snapshots ')
        html_parts.append('        before they are added to the training set. Decreasing error indicates a surrogate model that improves as more data is collected.</p>')
        fig_sq, axes_sq = self.plot_surrogate_quality()
        img_base64 = fig_to_base64(fig_sq)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Surrogate Model Quality">')
        html_parts.append('        <h3>5.1 Fixed Test Data Monitoring</h3>')
        html_parts.append('        <p class="description">These diagnostics track the surrogate on a fixed test set generated before sampling. ')
        html_parts.append('        Because the test data are immutable and excluded from training, this view is useful for monitoring generalization across surrogate updates.</p>')
        fig_sq_test, _ = self.plot_surrogate_quality_test()
        img_base64 = fig_to_base64(fig_sq_test)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Surrogate Model Quality on Fixed Test Data">')
        html_parts.append('        <h3>5.2 Posterior-Weighted Fixed Test Data Monitoring</h3>')
        html_parts.append('        <p class="description">These metrics weight each fixed test point by its posterior mass, emphasizing regions that are more relevant to the posterior distribution.</p>')
        fig_sq_test_weighted, _ = self.plot_surrogate_quality_test_weighted()
        img_base64 = fig_to_base64(fig_sq_test_weighted)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Posterior-Weighted Surrogate Model Quality on Fixed Test Data">')
        html_parts.append('    </div>')
        
        # 6. OBSERVATION HISTOGRAMS (if available)
        if include_expensive_sections and no_observations > 0 and observation_data_available:
            html_parts.append('    <div class="section" id="observations">')
            html_parts.append('        <h2>6. Observation Histograms</h2>')
            html_parts.append('        <p class="description">These histograms show the distribution of model outputs (observations) ')
            html_parts.append('        generated by the sampled parameters. If actual observations are provided, they are overlaid ')
            html_parts.append('        in red for comparison.</p>')
            
            # Overall observation histogram
            html_parts.append('        <h3>6.1 Combined Stages</h3>')
            fig = self.hist_observations(no_observations=no_observations, chosen_observations=observations_to_disp,
                                        grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                        stages_to_disp=stages_to_disp, observations=observations, cmap=cmap)
            img_base64 = fig_to_base64(fig)
            html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Overall Observation Histogram">')
            
            # Per-stage observation histograms
            html_parts.append('        <h3>6.2 Individual Stages</h3>')
            for stage_idx in stages_to_disp:
                stage_name = self.stage_names[stage_idx]
                html_parts.append(f'        <h4>Stage: {stage_name}</h4>')
                fig = self.hist_observations(no_observations=no_observations, chosen_observations=observations_to_disp,
                                            grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                            stages_to_disp=[stage_idx], observations=observations, cmap=cmap)
                img_base64 = fig_to_base64(fig)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Stage {stage_name} Observation Histogram">')
            
            html_parts.append('    </div>')
        elif no_observations > 0:
            html_parts.append('    <div class="section" id="observations">')
            html_parts.append('        <h2>6. Observation Histograms</h2>')
            if include_expensive_sections:
                html_parts.append('        <p class="description">Observation histograms are not available for this run because raw snapshots were not saved. ')
                html_parts.append('        Enable <code>save_snapshots_to_file=True</code> in configuration to include this section.</p>')
            else:
                html_parts.append('        <p class="description">Observation histograms were skipped in the lightweight report to keep generation time reasonable.</p>')
            html_parts.append('    </div>')
        
        # Footer
        html_parts.append('    <div class="section">')
        html_parts.append('        <p style="text-align: center; color: #7f8c8d; margin-top: 40px;">')
        html_parts.append('        Report generated by surrDAMH post-processing module')
        html_parts.append('        </p>')
        html_parts.append('    </div>')
        
        html_parts.append('</body>')
        html_parts.append('</html>')
        
        # Write to file
        output_dirname = os.path.dirname(output_file)
        if output_dirname:
            os.makedirs(output_dirname, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(html_parts))
        
        print(f"Extended HTML report saved to: {output_file}")
        return output_file

