#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report writers for one sampling run: the self-contained extended HTML report
(``html_report_extended``) plus the legacy single-file report (``html_report``).

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
from surrDAMH.stages import POSTERIOR_AFFECTING_FIELDS, Stage


class SamplesReports(SamplesPlots):
    """Reporting layer of :class:`surrDAMH.post_processing.Samples` (see module docstring)."""

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
                            ranking_mode: Literal["l2", "posterior", "likelihood"] = "l2",
                            pool_mode_note: str | None = None,
                            stages: List[Any] | None = None):
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
            pool_mode_note (str | None): finding 2.7 -- when the caller has no live Solver
                on the reporting rank (``use_solvers_pool=True``), a short explanatory
                note to show in place of the sections that need one (parameter names,
                posterior field statistics) instead of silently falling back to generic
                labels / an unexplained "not supplied" message.
            stages (list | None): the run's ``Stage`` objects (or dicts), rendered in the
                "Sampling Stages" section. If None, the ``stages`` list recorded in
                ``run_manifest.json`` is used (2026-09-21).

        Layout (2026-09-21): every top-level section and every per-stage block is a
        ``<details>`` element that is COLLAPSED when the file is opened; the "Expand all" /
        "Collapse all" buttons under the title and the table-of-contents links (which open the
        section they point into) are plain HTML + a few lines of inline JavaScript, the report
        stays a single self-contained file. The "Proposal Adaptation" section (5) plots, for
        every displayed stage with an adaptive proposal, the per-period acceptance probability
        the proposal was fed and its adapted parameters (``plot_adaptation``).

        Notes:
            ``chains_to_disp`` restricts every sample-derived section (moments, histograms,
            traces, autocorrelation, ESS, R-hat, correlation, best fits, observation
            histograms). The per-stage counters in section 1 and the acceptance-rate plot in
            4.1 come from ``notes``/``subchain_stats`` and are always whole-stage totals; the
            report says so where it shows them (WS9b, finding P5).
        """
        import base64
        import json
        import math
        import sys
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

        def section_open(section_id: str, title: str, open_by_default: bool = False) -> list[str]:
            """A top-level report section: a ``<details>`` block, collapsed unless asked otherwise."""
            open_attr = ' open' if open_by_default else ''
            return [f'    <details class="section" id="{section_id}"{open_attr}>',
                    f'        <summary><h2>{title}</h2></summary>']

        def stage_open(block_id: str, title: str, level: str = "h3") -> list[str]:
            """A collapsed per-stage block nested inside a section."""
            return [f'        <details class="stage" id="{escape(block_id)}">',
                    f'            <summary><{level}>{escape(title)}</{level}></summary>']

        stage_close = '        </details>'

        def get_stage_items(spec: Any) -> dict[str, Any]:
            if is_dataclass(spec):
                return {field.name: getattr(spec, field.name) for field in fields(spec)}
            if isinstance(spec, dict):
                return dict(spec)
            if hasattr(spec, "__dict__"):
                return dict(vars(spec))
            return {"stage": spec}

        def format_stage_value(value: Any) -> str:
            # the manifest stores Stage fields through manifest._json_safe: scalars as they are,
            # arrays as {"shape", "dtype", "values"}, objects (Proposal, ...) as their class name
            if value is None:
                return "None"
            if isinstance(value, bool):
                return str(value)
            if isinstance(value, (int, np.integer)) and int(value) == sys.maxsize:
                return "unbounded"
            if isinstance(value, (float, np.floating)) and math.isinf(float(value)):
                return "unbounded" if value > 0 else "-inf"
            if isinstance(value, np.ndarray):
                return np.array2string(value, threshold=20, edgeitems=3)
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, default=str)
            if isinstance(value, (str, int, float, np.generic)):
                return str(value)
            return type(value).__name__

        def default_target_rate(proposal_type: Any) -> float | None:
            # the proposal classes' own defaults (Stage.adaptive_target_rate docstring)
            if proposal_type in ("Hamiltonian", "HamiltonianInfinite"):
                return 0.8
            if proposal_type in ("RWMH", "pCN"):
                return 0.234
            return None

        # -- carry-over subsection of "5. Proposal Adaptation" (2026-09-21) ------------------
        def param_label(index: int) -> str:
            return par_names[index] if par_names and index < len(par_names) else f"p{index}"

        def scalar_table(rows: list[tuple[str, Any]]) -> list[str]:
            parts = ['            <table class="summary-table">',
                     '                <tr><th>quantity</th><th>value</th></tr>']
            for label, value in rows:
                text = str(value) if isinstance(value, (int, np.integer)) else f"{float(value):.6g}"
                parts.append(f'                <tr><td>{escape(label)}</td><td>{escape(text)}</td></tr>')
            parts.append('            </table>')
            return parts

        def vector_table(header: str, values: np.ndarray) -> list[str]:
            parts = ['            <table class="summary-table">',
                     f'                <tr><th>parameter</th><th>{escape(header)}</th></tr>']
            for i, v in enumerate(values):
                parts.append(f'                <tr><td>{escape(param_label(i))}</td><td>{float(v):.4g}</td></tr>')
            parts.append('            </table>')
            return parts

        def matrix_table(matrix: np.ndarray) -> list[str]:
            d = matrix.shape[0]
            labels = [param_label(i) for i in range(d)]
            parts = ['            <table class="summary-table">',
                     '                <tr><th></th>' + ''.join(f'<th>{escape(l)}</th>' for l in labels) + '</tr>']
            for i in range(d):
                cells = ''.join(f'<td>{float(matrix[i, j]):.4g}</td>' for j in range(d))
                parts.append(f'                <tr><th>{escape(labels[i])}</th>{cells}</tr>')
            parts.append('            </table>')
            return parts

        def carry_over_block(stage_idx: int, stage_name: str,
                             carry: tuple[dict, dict] | None) -> list[str]:
            """"Carried over to the next stage" subsection: the ``Proposal.carry_over()`` /
            ``adapted_summary()`` pair ``modules.continuation.save_carry_over`` wrote for this
            stage, or an explanatory sentence if the file does not exist. Shown for every
            adaptive stage regardless of whether its ``adaptive_stats`` trace is present."""
            parts: list[str] = []
            if carry is None:
                parts.append('            <p class="description" style="color: orange;">no carry_over file '
                             '(the proposal of this stage did not adapt, or the run predates carry_over/)</p>')
                return parts
            carried_stage, summary = carry
            next_stage_name = (self.stage_names[stage_idx + 1]
                               if stage_idx + 1 < len(self.stage_names) else None)
            if next_stage_name is not None:
                consumer_sentence = f'consumed by stage {escape(next_stage_name)}.'
            else:
                consumer_sentence = 'this was the last stage; nothing consumed it.'
            parts.append(f'            <p class="description"><b>Carried over to the next stage</b> '
                         f'{consumer_sentence}</p>')
            if "proposal_sd_or_cov" in carried_stage:
                cov = np.asarray(carried_stage["proposal_sd_or_cov"], dtype=float)
                d = cov.shape[0]
                log_sigma = summary.get("log_sigma")
                rows = []
                if log_sigma is not None:
                    rows.append(("log_sigma", log_sigma))
                    rows.append(("scale factor sigma = exp(log_sigma)", float(np.exp(log_sigma))))
                if "n_pooled" in summary:
                    rows.append(("n_pooled", summary["n_pooled"]))
                rows.append(("trace(sd_or_cov)/d", float(np.trace(cov)) / d))
                parts.extend(scalar_table(rows))
                sd = np.sqrt(np.diag(cov))
                parts.append('            <p class="description">Proposal standard deviations:</p>')
                parts.extend(vector_table("sd", sd))
                if d <= 20:
                    parts.append('            <p class="description">Full covariance matrix:</p>')
                    parts.extend(matrix_table(cov))
                    corr = cov / np.outer(sd, sd)
                    parts.append('            <p class="description">Correlation matrix:</p>')
                    parts.extend(matrix_table(corr))
                else:
                    corr = cov / np.outer(sd, sd)
                    iu = np.triu_indices(d, k=1)
                    order = np.argsort(-np.abs(corr[iu]))[:10]
                    pair_rows = ['            <table class="summary-table">',
                                 '                <tr><th>i</th><th>j</th><th>correlation</th></tr>']
                    for idx in order:
                        i, j = int(iu[0][idx]), int(iu[1][idx])
                        pair_rows.append(f'                <tr><td>{escape(param_label(i))}</td>'
                                         f'<td>{escape(param_label(j))}</td>'
                                         f'<td>{corr[i, j]:.4g}</td></tr>')
                    pair_rows.append('            </table>')
                    parts.append(f'            <p class="description">d={d} &gt; 20: showing the standard '
                                 'deviations and the 10 largest |correlations| only; the full '
                                 f'covariance matrix is in sampling_output/carry_over/{escape(stage_name)}.npz.</p>')
                    parts.extend(pair_rows)
            elif "pcn_beta" in carried_stage:
                parts.extend(scalar_table([("beta", carried_stage["pcn_beta"])]))
            elif "hamiltonian_step_size" in carried_stage:
                parts.extend(scalar_table([("step_size", carried_stage["hamiltonian_step_size"])]))
            else:
                # a proposal family added after this report code (forward compatibility):
                # show whatever scalar-shaped items came back rather than nothing
                parts.extend(scalar_table([(key, value) for key, value in carried_stage.items()
                                          if np.asarray(value).ndim == 0]))
            return parts

        manifest = dict(getattr(self.run_data, "manifest", None) or {})
        stage_specs = [get_stage_items(spec) for spec in (stages or [])]
        stages_source = "the stage list passed to this report"
        if not stage_specs:
            stage_specs = [dict(spec) for spec in (manifest.get("stages") or []) if isinstance(spec, dict)]
            stages_source = "sampling_output/run_manifest.json"
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
        html_parts.append('        .section { background-color: white; padding: 10px 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }')
        html_parts.append('        details > summary { cursor: pointer; list-style: none; }')
        html_parts.append('        details > summary::-webkit-details-marker { display: none; }')
        html_parts.append('        details > summary::before { content: "\\25B8"; display: inline-block; width: 1.1em; color: #3498db; }')
        html_parts.append('        details[open] > summary::before { content: "\\25BE"; }')
        html_parts.append('        details.section > summary { padding: 10px 0; }')
        html_parts.append('        details.section[open] > summary { border-bottom: 2px solid #95a5a6; margin-bottom: 15px; }')
        html_parts.append('        details.section > summary h2 { display: inline; border-bottom: none; margin: 0; padding: 0; }')
        html_parts.append('        details.stage { background-color: #fafafa; border-left: 3px solid #bdc3c7; padding: 6px 15px; margin: 15px 0; }')
        html_parts.append('        details.stage > summary { padding: 6px 0; }')
        html_parts.append('        details.stage > summary h3, details.stage > summary h4 { display: inline; margin: 0; color: #34495e; }')
        html_parts.append('        details.toc > summary h2 { display: inline; border-bottom: none; margin: 0; padding: 0; }')
        html_parts.append('        .controls { margin: 10px 0 20px 0; }')
        html_parts.append('        .controls button { background-color: #3498db; color: white; border: none; border-radius: 3px; padding: 6px 12px; margin-right: 8px; cursor: pointer; }')
        html_parts.append('        .controls button:hover { background-color: #2980b9; }')
        html_parts.append('        .footer { text-align: center; color: #7f8c8d; margin-top: 40px; }')
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
        html_parts.append('    visualizations, and detailed analysis for all stages of the sampling process. ')
        html_parts.append('    Every section and every per-stage block is collapsed when the file is opened; click a heading to unfold it.</p>')
        html_parts.append('    <div class="controls"><button type="button" data-toggle-all="open">Expand all</button> '
                          '<button type="button" data-toggle-all="close">Collapse all</button></div>')
        html_parts.extend(section_open("configuration", "Run Configuration"))
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
        html_parts.append('    </details>')

        # Sampling stages specification (2026-09-21): one column per stage of the RUN, one row
        # per Stage field, in Stage's own field order; '*' marks the posterior-/acceptance-rate-
        # affecting fields exactly as Stage.describe() does.
        html_parts.extend(section_open("stages", "Sampling Stages"))
        html_parts.append(f'        <p class="description">Effective settings of every stage of this run, from '
                          f'{escape(stages_source)}. Fields marked with * change the sampled distribution or the '
                          'acceptance rate; "unbounded" is a stopping condition that was not set. '
                          'The last row says whether the stage is included in the analysis sections below.</p>')
        if stage_specs:
            stage_field_order = [field.name for field in fields(Stage)]
            present_keys = {key for spec in stage_specs for key in spec}
            stage_rows = [key for key in stage_field_order if key in present_keys] \
                + sorted(key for key in present_keys if key not in stage_field_order)
            stage_headers = []
            for index, spec in enumerate(stage_specs):
                name = spec.get("name") or (self.stage_names[index] if index < len(self.stage_names) else None)
                stage_headers.append(f"stage {index}" if name is None else f"{index}: {name}")
            html_parts.append('        <div style="overflow-x: auto;">')
            html_parts.append('        <table class="summary-table stages-table">')
            html_parts.append('            <tr><th>field</th>' + ''.join(f'<th>{escape(h)}</th>' for h in stage_headers) + '</tr>')
            for key in stage_rows:
                label = f"{key} *" if key in POSTERIOR_AFFECTING_FIELDS else key
                cells = ''.join(f'<td>{escape(format_stage_value(spec.get(key)))}</td>' if key in spec else '<td></td>'
                                for spec in stage_specs)
                html_parts.append(f'            <tr><td>{escape(label)}</td>{cells}</tr>')
            included = ''.join('<td>yes</td>' if index in stages_to_disp else '<td>no</td>'
                               for index in range(len(stage_specs)))
            html_parts.append(f'            <tr><td>included in this report</td>{included}</tr>')
            html_parts.append('        </table>')
            html_parts.append('        </div>')
            if len(stage_specs) != self.no_stages:
                html_parts.append(f'        <p class="description" style="color: orange;">The specification lists '
                                  f'{len(stage_specs)} stage(s) but the output directory holds {self.no_stages}; '
                                  'the columns above are matched to stage directories by position.</p>')
        else:
            html_parts.append('        <p class="description" style="color: orange;">No stage specification was '
                              'recorded in the run manifest and none was passed to the report.</p>')
        html_parts.append('    </details>')

        # Table of Contents
        html_parts.append('    <details class="toc" open>')
        html_parts.append('        <summary><h2>Table of Contents</h2></summary>')
        html_parts.append('        <ul>')
        html_parts.append('            <li><a href="#configuration">Run Configuration</a></li>')
        html_parts.append('            <li><a href="#stages">Sampling Stages</a></li>')
        html_parts.append('            <li><a href="#summary">1. Summary Statistics</a></li>')
        html_parts.append('            <li><a href="#overall">2. Overall Analysis (Combined Stages)</a></li>')
        html_parts.append('            <li><a href="#individual">3. Individual Stage Analysis</a></li>')
        html_parts.append('            <li><a href="#diagnostics">4. Convergence Diagnostics & Autocorrelation</a></li>')
        html_parts.append('            <li><a href="#adaptation">5. Proposal Adaptation</a></li>')
        html_parts.append('            <li><a href="#surrogate_quality">6. Surrogate Model Quality</a></li>')
        if no_observations > 0 and observation_data_available:
            html_parts.append('            <li><a href="#observations">7. Observation Histograms</a></li>')
        if no_best_fits > 0:
            html_parts.append('            <li><a href="#best_fits">Best-fit analysis</a></li>')
        if field_statistics:
            html_parts.append('            <li><a href="#field_statistics">Posterior field statistics</a></li>')
        html_parts.append('        </ul>')
        html_parts.append('    </details>')
        
        # 1. SUMMARY STATISTICS
        html_parts.extend(section_open("summary", "1. Summary Statistics"))
        html_parts.append('        <p class="description">This table summarizes the acceptance and rejection rates for all sampling stages. ')
        html_parts.append('        "Accepted" samples were accepted by the Metropolis-Hastings criterion, "rejected" samples were rejected, ')
        html_parts.append('        and "pre-rejected" samples (if any) were rejected by a surrogate model before evaluation. ')
        html_parts.append('        For DAMH stages, the table also includes the mean within-subchain surrogate acceptance rate, the fraction of subchains that produced a changed proposal, and the exact outer acceptance conditional on a changed proposal. ')
        html_parts.append('        These counters are whole-stage totals over every chain, also when the rest of the report is restricted to a subset of chains.</p>')
        html_parts.append(self.summary.to_html(classes='summary-table'))
        html_parts.append('    </details>')

        # 2. OVERALL ANALYSIS (COMBINED STAGES)
        html_parts.extend(section_open("overall", "2. Overall Analysis (Combined Stages)"))
        html_parts.append(f'        <p class="description">This section presents aggregated results from all selected sampling stages combined.{chains_note}</p>')
        # 2.1 Mean and Covariance
        html_parts.append('        <h3>2.1 Posterior Mean and Covariance Matrix</h3>')
        html_parts.append('        <p class="description">The posterior mean represents the expected value of each parameter, ')
        html_parts.append('        while the covariance matrix shows the variance and correlation structure among parameters.</p>')
        if par_names is None and pool_mode_note:
            # finding 2.7: par_names normally comes from the live Solver; explain the
            # generic "Parameter N" labels below instead of leaving it unexplained.
            html_parts.append(f'        <p class="description" style="color: orange;">{escape(pool_mode_note)} '
                              'Parameter names below fall back to generic labels ("Parameter N").</p>')
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
        elif pool_mode_note:
            # finding 2.7: distinguish "no live Solver to compute these from" from the
            # generic "nothing was supplied" case below.
            html_parts.append(f'        <p class="description" style="color: orange;">{escape(pool_mode_note)}</p>')
        else:
            html_parts.append('        <p class="description">No posterior field statistics were supplied for this report.</p>')
        html_parts.append('        </div>')
        
        html_parts.append('    </details>')
        
        # 3. INDIVIDUAL STAGE ANALYSIS / 4. DIAGNOSTICS
        if include_expensive_sections:
            html_parts.extend(section_open("individual", "3. Individual Stage Analysis"))
            html_parts.append('        <p class="description">This section presents detailed analysis for each sampling stage separately. ')
            html_parts.append('        Different stages may use different proposal distributions or algorithm parameters.</p>')

            for stage_idx in stages_to_disp:
                stage_name = self.stage_names[stage_idx]
                html_parts.extend(stage_open(f"individual_{stage_name}", f"Stage: {stage_name}"))
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
                html_parts.append(stage_close)
            html_parts.append('    </details>')

            html_parts.extend(section_open("diagnostics", "4. Convergence Diagnostics & Autocorrelation Analysis"))
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
                html_parts.extend(stage_open(f"autocorr_{stage_name}", f"Stage: {stage_name}", level="h4"))

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
                html_parts.append(stage_close)

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
            html_parts.append('    </details>')
        else:
            html_parts.extend(section_open("individual", "3. Individual Stage Analysis"))
            html_parts.append('        <p class="description">Omitted in the lightweight report because per-stage plots and summaries are expensive for high-dimensional problems.</p>')
            html_parts.append('    </details>')

            html_parts.extend(section_open("diagnostics", "4. Convergence Diagnostics & Autocorrelation Analysis"))
            html_parts.append('        <p class="description">Omitted in the lightweight report.</p>')
            html_parts.append('    </details>')

        # 5. PROPOSAL ADAPTATION (2026-09-21): per adaptive stage, the per-period trace written to
        # adaptive_stats/<stage>/rank%04d.csv (docs/outputs.md). Cheap (one row per period), so it
        # is produced in the lightweight report too.
        html_parts.extend(section_open("adaptation", "5. Proposal Adaptation"))
        html_parts.append('        <p class="description">For every displayed stage whose proposal adapted, the per-period trace of the '
                          'adaptation: the mean acceptance probability the proposal was fed in each period (top panel, with the target '
                          'rate it drives towards as a dashed line) and every adapted parameter it logged, one line per chain, followed '
                          'by the proposal actually CARRIED OVER to the next stage (the pooled covariance/beta/step size that stage '
                          'started from). Random walk: log_sigma (Robbins-Monro log-scale), trace_C_over_d and shrinkage_delta of the '
                          'installed covariance estimate (NaN before the warm-up count is reached); pCN: beta; Hamiltonian family: the '
                          'leapfrog log_step_size and its dual-averaged log_step_size_bar. In a DAMH stage the random walk and pCN are '
                          'fed the OVERALL outer acceptance probability (a pre-rejected iteration counts as 0), the Hamiltonian step '
                          'size the sub-chain\'s own acceptance against the surrogate. Values are shown as logged, i.e. on the log scale '
                          f'where the proposal adapts on the log scale.{escape(chains_note)}</p>')
        adaptive_blocks = 0
        for stage_idx in stages_to_disp:
            stage_name = self.stage_names[stage_idx]
            spec = stage_specs[stage_idx] if stage_idx < len(stage_specs) else {}
            trace = self.adaptive_stats[stage_idx] if stage_idx < len(self.adaptive_stats) else None
            has_trace = trace is not None and not trace.empty
            carry = self.carry_over[stage_idx] if stage_idx < len(self.carry_over) else None
            if not has_trace and not spec.get("adaptive", False) and carry is None:
                continue
            adaptive_blocks += 1
            html_parts.extend(stage_open(f"adaptation_{stage_name}", f"Stage: {stage_name}"))
            target_rate = spec.get("adaptive_target_rate")
            target_source = "adaptive_target_rate"
            if target_rate is None:
                target_rate = default_target_rate(spec.get("proposal_type"))
                target_source = "the proposal's default"
            if not has_trace:
                html_parts.append('            <p class="description" style="color: orange;">This stage is declared '
                                  'adaptive=True but no adaptive_stats trace was found (save_to_file=False, or the run '
                                  'predates the adaptive_stats file).</p>')
            else:
                parts = [f'proposal_type={format_stage_value(spec.get("proposal_type"))}' if "proposal_type" in spec else None,
                         f'target rate {target_rate:g} ({target_source})' if target_rate is not None else 'target rate unknown',
                         f'{len(trace)} logged period(s) over '
                         f'{trace["rank_world"].nunique() if "rank_world" in trace.columns else 1} chain(s)']
                html_parts.append('            <p class="description">' + escape('; '.join(p for p in parts if p)) + '</p>')
                try:
                    fig_adapt, _ = self.plot_adaptation(stage_idx, chains_to_disp=chains_to_disp, target_rate=target_rate)
                    if fig_adapt is None:
                        html_parts.append('            <p class="description" style="color: orange;">No adaptation trace '
                                          'for the selected chain(s).</p>')
                    else:
                        img_base64 = fig_to_base64(fig_adapt)
                        html_parts.append(f'            <img src="data:image/png;base64,{img_base64}" alt="Proposal adaptation {escape(stage_name)}">')
                        # last logged period of every chain: the values the stage ended with
                        counter = next((c for c in ("n", "m") if c in trace.columns), None)
                        if "rank_world" in trace.columns:
                            ranks = sorted(int(r) for r in trace["rank_world"].unique())
                            chain_of_rank = {rank: position for position, rank in enumerate(ranks)}
                            last_rows = trace.groupby("rank_world", sort=True).tail(1).copy()
                            last_rows.insert(0, "chain", [chain_of_rank[int(r)] for r in last_rows["rank_world"]])
                            last_rows = last_rows.drop(columns=["rank_world"]).set_index("chain")
                            if chains_to_disp is not None:
                                wanted = {int(c) for c in chains_to_disp}
                                last_rows = last_rows.loc[[c for c in last_rows.index if c in wanted]]
                            html_parts.append('            <p class="description">Last logged period of every chain'
                                              + (f' (column {escape(counter)} = adaptation counter)' if counter else '') + ':</p>')
                            html_parts.append(last_rows.to_html(classes='summary-table', float_format=lambda x: f"{x:.6g}"))
                except Exception as e:
                    html_parts.append(f'            <p class="description" style="color: orange;">Adaptation plot unavailable for {escape(stage_name)}: {escape(str(e))}</p>')
            html_parts.extend(carry_over_block(stage_idx, stage_name, carry))
            html_parts.append(stage_close)
        if adaptive_blocks == 0:
            html_parts.append('        <p class="description">No displayed stage used an adaptive proposal.</p>')
        html_parts.append('    </details>')

        # 6. SURROGATE MODEL QUALITY
        html_parts.extend(section_open("surrogate_quality", "6. Surrogate Model Quality"))
        html_parts.append('        <p class="description">Out-of-sample quality metrics of the surrogate model, measured on newly arrived snapshots ')
        html_parts.append('        before they are added to the training set. Decreasing error indicates a surrogate model that improves as more data is collected.</p>')
        fig_sq, axes_sq = self.plot_surrogate_quality()
        img_base64 = fig_to_base64(fig_sq)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Surrogate Model Quality">')
        html_parts.append('        <h3>6.1 Fixed Test Data Monitoring</h3>')
        html_parts.append('        <p class="description">These diagnostics track the surrogate on a fixed test set generated before sampling. ')
        html_parts.append('        Because the test data are immutable and excluded from training, this view is useful for monitoring generalization across surrogate updates.</p>')
        fig_sq_test, _ = self.plot_surrogate_quality_test()
        img_base64 = fig_to_base64(fig_sq_test)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Surrogate Model Quality on Fixed Test Data">')
        html_parts.append('        <h3>6.2 Posterior-Weighted Fixed Test Data Monitoring</h3>')
        html_parts.append('        <p class="description">These metrics weight each fixed test point by its posterior mass, emphasizing regions that are more relevant to the posterior distribution.</p>')
        fig_sq_test_weighted, _ = self.plot_surrogate_quality_test_weighted()
        img_base64 = fig_to_base64(fig_sq_test_weighted)
        html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Posterior-Weighted Surrogate Model Quality on Fixed Test Data">')
        html_parts.append('    </details>')
        
        # 7. OBSERVATION HISTOGRAMS (if available)
        if include_expensive_sections and no_observations > 0 and observation_data_available:
            html_parts.extend(section_open("observations", "7. Observation Histograms"))
            html_parts.append('        <p class="description">These histograms show the distribution of model outputs (observations) ')
            html_parts.append('        generated by the sampled parameters. If actual observations are provided, they are overlaid ')
            html_parts.append('        in red for comparison.</p>')
            
            # Overall observation histogram
            html_parts.append('        <h3>7.1 Combined Stages</h3>')
            fig = self.hist_observations(no_observations=no_observations, chosen_observations=observations_to_disp,
                                        grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                        stages_to_disp=stages_to_disp, observations=observations, cmap=cmap)
            img_base64 = fig_to_base64(fig)
            html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Overall Observation Histogram">')
            
            # Per-stage observation histograms
            html_parts.append('        <h3>7.2 Individual Stages</h3>')
            for stage_idx in stages_to_disp:
                stage_name = self.stage_names[stage_idx]
                html_parts.extend(stage_open(f"observations_{stage_name}", f"Stage: {stage_name}", level="h4"))
                fig = self.hist_observations(no_observations=no_observations, chosen_observations=observations_to_disp,
                                            grid=grid, grid_interp=grid_interp, bins=bins, chains_to_disp=chains_to_disp,
                                            stages_to_disp=[stage_idx], observations=observations, cmap=cmap)
                img_base64 = fig_to_base64(fig)
                html_parts.append(f'        <img src="data:image/png;base64,{img_base64}" alt="Stage {stage_name} Observation Histogram">')
                html_parts.append(stage_close)
            
            html_parts.append('    </details>')
        elif no_observations > 0:
            html_parts.extend(section_open("observations", "7. Observation Histograms"))
            if include_expensive_sections:
                html_parts.append('        <p class="description">Observation histograms are not available for this run because raw snapshots were not saved. ')
                html_parts.append('        Enable <code>save_snapshots_to_file=True</code> in configuration to include this section.</p>')
            else:
                html_parts.append('        <p class="description">Observation histograms were skipped in the lightweight report to keep generation time reasonable.</p>')
            html_parts.append('    </details>')
        
        # Footer
        html_parts.append('    <p class="footer">Report generated by surrDAMH post-processing module</p>')
        # collapsible sections: open every <details> ancestor of a fragment target (table of
        # contents links into collapsed sections), and the Expand all / Collapse all buttons
        html_parts.append('    <script>')
        html_parts.append('    (function () {')
        html_parts.append('        function revealHash() {')
        html_parts.append('            if (!location.hash) { return; }')
        html_parts.append('            var target = document.getElementById(decodeURIComponent(location.hash.slice(1)));')
        html_parts.append('            for (var el = target; el; el = el.parentElement) { if (el.tagName === "DETAILS") { el.open = true; } }')
        html_parts.append('            if (target) { target.scrollIntoView(); }')
        html_parts.append('        }')
        html_parts.append('        window.addEventListener("hashchange", revealHash);')
        html_parts.append('        revealHash();')
        html_parts.append('        document.querySelectorAll("[data-toggle-all]").forEach(function (button) {')
        html_parts.append('            button.addEventListener("click", function () {')
        html_parts.append('                var open = button.getAttribute("data-toggle-all") === "open";')
        html_parts.append('                document.querySelectorAll("details").forEach(function (d) { d.open = open; });')
        html_parts.append('            });')
        html_parts.append('        });')
        html_parts.append('    })();')
        html_parts.append('    </script>')
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

