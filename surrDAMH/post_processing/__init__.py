#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing of one sampling run (output format v2, see ``docs/outputs.md``).

``surrDAMH.post_processing`` used to be a single 2400-line module; WS9b split it into
``base`` (the shared attribute contract and stage/chain/burn-in resolvers), ``statistics``
(moments, ESS, R-hat, autocorrelation, best fits), ``plots`` (matplotlib figures),
``html_report`` (the extended HTML report) and ``loading`` (construction from
:func:`surrDAMH.modules.run_data.read_run`). ``Samples`` is a linear chain of these
layers -- ``SamplesStatistics(SamplesBase)`` -> ``SamplesPlots`` -> ``SamplesReports`` ->
``Samples`` -- so each layer sees what it uses from the layer(s) below it through
ordinary inheritance. This package is the facade: every name that was importable from
``surrDAMH.post_processing`` before is importable from it still.

    from surrDAMH.post_processing import Samples
    samples = Samples(no_parameters, output_dir)   # output_dir CONTAINS sampling_output/

``Samples.find_best_fits`` and ``Samples.calculate_gelman_rubin`` are methods of
``Samples`` (defined in ``statistics.py``), not module-level functions.
"""

# Report generation must work without a display; set before pyplot is imported anywhere
# in the package (this module runs first for every ``surrDAMH.post_processing`` import).
import matplotlib

matplotlib.use("Agg")

from surrDAMH.modules.manifest import RunFormatError  # noqa: E402
from surrDAMH.modules.run_data import (RunData, raw_data_columns,  # noqa: E402
                                       read_run, sampling_output_dir)
from surrDAMH.post_processing.base import SamplesBase  # noqa: E402
from surrDAMH.post_processing.loading import Samples, StageSamples  # noqa: E402
from surrDAMH.post_processing.plots import (SamplesPlots,  # noqa: E402
                                            add_normal_dist_grid)
from surrDAMH.post_processing.html_report import SamplesReports  # noqa: E402
from surrDAMH.post_processing.statistics import (Autocorrelation,  # noqa: E402
                                                 SamplesStatistics,
                                                 auto_window, autocorr_FM,
                                                 decompress,
                                                 rank_best_fit_candidates)

__all__ = [
    # facade
    "Samples", "StageSamples",
    # format-v2 reader, re-exported so user scripts need only this import
    "read_run", "RunData", "RunFormatError", "raw_data_columns", "sampling_output_dir",
    # statistics
    "Autocorrelation", "rank_best_fit_candidates", "decompress",
    "autocorr_FM", "auto_window",
    # plots
    "add_normal_dist_grid",
    # layers of Samples's mixin chain (SamplesStatistics(SamplesBase) -> SamplesPlots ->
    # SamplesReports -> Samples); exported for subclassing
    "SamplesBase", "SamplesStatistics", "SamplesPlots", "SamplesReports",
]
